"""ms-swift `TreeWeaveDistillTrainer`: High-Efficiency Stage 1 / Stage 2 / Stage 3 Self-Distillation."""

from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from swift.trainers import Seq2SeqTrainer
from swift.utils import get_logger

from ..modeling_treeweave import Qwen3_5TreeWeaveForConditionalGeneration
from ..volume_bias import install_volume_bias_on_language_model

logger = get_logger()


@dataclass
class TreeWeaveDistillConfig:
    """Runtime hyperparameters for TreeWeave 3-stage self-distillation inside `ms-swift`."""

    stage: int = 1                                  # 1, 2, or 3
    distill_topk: int = 512
    distill_temperature: float = 1.0
    distill_chunk_size: int = 1024                  # Bounds peak logits memory on 248K vocab to <0.5GB!
    teacher_mode: str = "auto"                      # "auto" | "shared_frozen" | "external_teacher" | "offline_topk_cache"
    teacher_model_path: Optional[str] = None        # Required when `teacher_mode == "external_teacher"` (Stage 3 without offline topk)
    topk_cache_dir: Optional[str] = None            # Directory of `<uuid>.safetensors` shards for `offline_topk_cache`
    lambda_rate: float = 0.05
    lambda_bal: float = 0.01
    rho_s1: float = 0.75
    rho_s2: float = 0.50
    rho_t1: float = 0.65
    rho_t2: float = 0.50
    train_volume_bias_in_stage1: bool = True
    freeze_merger_in_stage3: bool = False
    stage1_uniform_scale: Optional[Tuple[int, int]] = None  # e.g. `(1, 0)` for Ablation Q5 single-scale warmup!

    @classmethod
    def from_env(cls) -> "TreeWeaveDistillConfig":
        """Read overrides from `TREEWEAVE_*` environment variables when invoked via plain `swift sft` CLI."""
        def _int(k: str, d: int) -> int:
            v = os.environ.get(k)
            return int(v) if v is not None and v.strip() else d

        def _float(k: str, d: float) -> float:
            v = os.environ.get(k)
            return float(v) if v is not None and v.strip() else d

        def _bool(k: str, d: bool) -> bool:
            v = os.environ.get(k)
            if v is None or not v.strip():
                return d
            return v.strip().lower() in ("1", "true", "yes", "on")

        u_scale_raw = os.environ.get("TREEWEAVE_STAGE1_UNIFORM_SCALE", "").strip()
        u_scale = None
        if u_scale_raw:
            parts = [int(x) for x in u_scale_raw.split(",")]
            if len(parts) != 2:
                raise ValueError(f"TREEWEAVE_STAGE1_UNIFORM_SCALE must be 's,t', got {u_scale_raw!r}.")
            u_scale = (parts[0], parts[1])

        return cls(
            stage=_int("TREEWEAVE_STAGE", 1),
            distill_topk=_int("TREEWEAVE_DISTILL_TOPK", 512),
            distill_temperature=_float("TREEWEAVE_DISTILL_TEMP", 1.0),
            distill_chunk_size=_int("TREEWEAVE_DISTILL_CHUNK_SIZE", 1024),
            teacher_mode=os.environ.get("TREEWEAVE_TEACHER_MODE", "auto").strip(),
            teacher_model_path=os.environ.get("TREEWEAVE_TEACHER_MODEL_PATH") or None,
            topk_cache_dir=os.environ.get("TREEWEAVE_TOPK_CACHE_DIR") or None,
            lambda_rate=_float("TREEWEAVE_LAMBDA_RATE", 0.05),
            lambda_bal=_float("TREEWEAVE_LAMBDA_BAL", 0.01),
            rho_s1=_float("TREEWEAVE_RHO_S1", 0.75),
            rho_s2=_float("TREEWEAVE_RHO_S2", 0.50),
            rho_t1=_float("TREEWEAVE_RHO_T1", 0.65),
            rho_t2=_float("TREEWEAVE_RHO_T2", 0.50),
            train_volume_bias_in_stage1=_bool("TREEWEAVE_STAGE1_TRAIN_VOL_BIAS", True),
            freeze_merger_in_stage3=_bool("TREEWEAVE_STAGE3_FREEZE_MERGER", False),
            stage1_uniform_scale=u_scale,
        )


ACTIVE_DISTILL_CONFIG: TreeWeaveDistillConfig = TreeWeaveDistillConfig()


def set_active_distill_config(cfg: TreeWeaveDistillConfig) -> None:
    global ACTIVE_DISTILL_CONFIG
    ACTIVE_DISTILL_CONFIG = cfg
    if cfg.topk_cache_dir:
        os.environ["TREEWEAVE_TOPK_CACHE_DIR"] = str(cfg.topk_cache_dir)
    else:
        os.environ.pop("TREEWEAVE_TOPK_CACHE_DIR", None)


def get_active_distill_config() -> TreeWeaveDistillConfig:
    return ACTIVE_DISTILL_CONFIG


def update_last_checkpoint_symlink(root_output_dir: Path | str, checkpoint_dir: Path | str) -> None:
    """Create/refresh `<root_output_dir>/last` symlink pointing to `checkpoint_dir` for seamless stage chaining."""
    ckpt = Path(checkpoint_dir).resolve()
    if not ckpt.is_dir():
        return
    for parent in {Path(root_output_dir).resolve(), ckpt.parent}:
        if not parent.is_dir() or parent == ckpt:
            continue
        link = parent / "last"
        tmp_link = parent / f".last.{os.getpid()}.tmp"
        try:
            if tmp_link.exists() or tmp_link.is_symlink():
                tmp_link.unlink()
            tmp_link.symlink_to(ckpt)
            tmp_link.replace(link)
        except OSError:
            if tmp_link.exists() or tmp_link.is_symlink():
                tmp_link.unlink(missing_ok=True)


def _extract_sample_costs_no_decode(
    base_ds: Any,
    sample_indices: Sequence[int],
    fallback_lengths: Dict[int, int],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Extract per-sample `(tok_len, vis_cost)` strictly from lightweight metadata without decoding images/videos."""
    tok_lens = dict(fallback_lengths)
    vis_costs: Dict[int, int] = {int(i): 0 for i in sample_indices}

    arrow_ds = base_ds
    for _ in range(3):
        if arrow_ds is None:
            break
        if hasattr(arrow_ds, "column_names"):
            break
        arrow_ds = getattr(arrow_ds, "dataset", None)

    if arrow_ds is None or not hasattr(arrow_ds, "column_names"):
        return tok_lens, vis_costs

    cols = set(arrow_ds.column_names)
    if "lengths" in cols:
        raw_lens = arrow_ds["lengths"]
        for i in sample_indices:
            v = raw_lens[i]
            tok_lens[i] = int(max(v) if isinstance(v, list) else v)

    # Prefer exact encoded visual grid token counts if present; else count media metadata paths (zero decode!)
    if "video_grid_thw" in cols or "image_grid_thw" in cols:
        v_col = arrow_ds["video_grid_thw"] if "video_grid_thw" in cols else None
        i_col = arrow_ds["image_grid_thw"] if "image_grid_thw" in cols else None
        for i in sample_indices:
            vc = 0
            if v_col is not None and v_col[i]:
                for g in v_col[i]:
                    if len(g) == 3:
                        vc += int((int(g[0]) * int(g[1]) * int(g[2])) // 4 * 1.5)
            if i_col is not None and i_col[i]:
                for g in i_col[i]:
                    if len(g) == 3:
                        vc += int((int(g[0]) * int(g[1]) * int(g[2])) // 4)
            vis_costs[i] = vc
    elif "videos" in cols or "images" in cols:
        v_col = arrow_ds["videos"] if "videos" in cols else None
        i_col = arrow_ds["images"] if "images" in cols else None
        for i in sample_indices:
            vc = 0
            if v_col is not None and v_col[i]:
                vc += 4000 * len(v_col[i])
            if i_col is not None and i_col[i]:
                vc += 400 * len(i_col[i])
            vis_costs[i] = vc
    return tok_lens, vis_costs


def repack_balanced_indices(
    sample_indices: Sequence[int],
    tok_lens: Dict[int, int],
    vis_costs: Dict[int, int],
    packing_length: int,
    initial_num_bins: int = 1,
) -> Tuple[List[List[int]], List[int], List[int]]:
    """Vision- and segment-balanced LPT bin-packing to eliminate multi-rank stragglers.

    Places heavy video/visual items first into the valid bin with lowest `(vis_cost, num_segments, tok_len)`,
    then distributes shorter image/text samples to equalize both visual compute and segment counts across bins.
    """
    items = [
        (int(vis_costs.get(i, 0)), int(tok_lens.get(i, 1)), int(i))
        for i in sample_indices
    ]
    if not items:
        return [], [], []

    items.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    cap = max(int(packing_length), max(x[1] for x in items))
    total_toks = sum(x[1] for x in items)
    num_init = max(1, int(initial_num_bins), int(math.ceil(total_toks / max(cap * 0.96, 1.0))))

    bins_idx: List[List[int]] = [[] for _ in range(num_init)]
    bins_tok: List[int] = [0] * num_init
    bins_vis: List[int] = [0] * num_init
    # Min-heap keyed by `(vis_cost, num_segments, tok_len, bin_id)`
    heap: List[Tuple[int, int, int, int]] = [(0, 0, 0, b_id) for b_id in range(num_init)]
    heapq.heapify(heap)

    max_probe = 64
    for v_cost, t_len, s_idx in items:
        popped: List[Tuple[int, int, int, int]] = []
        chosen_bin: Optional[int] = None
        while heap and len(popped) < max_probe:
            entry = heapq.heappop(heap)
            b_id = entry[3]
            if bins_tok[b_id] + t_len <= cap:
                chosen_bin = b_id
                break
            popped.append(entry)

        if chosen_bin is None:
            # Scan remaining heap entries only if needed before opening a new bin
            best_pos = -1
            best_key = None
            for pos, entry in enumerate(heap):
                b_id = entry[3]
                if bins_tok[b_id] + t_len <= cap:
                    if best_key is None or entry < best_key:
                        best_key = entry
                        best_pos = pos
            if best_pos >= 0:
                entry = heap.pop(best_pos)
                heapq.heapify(heap)
                chosen_bin = entry[3]
            else:
                chosen_bin = len(bins_idx)
                bins_idx.append([])
                bins_tok.append(0)
                bins_vis.append(0)

        bins_idx[chosen_bin].append(s_idx)
        bins_tok[chosen_bin] += t_len
        bins_vis[chosen_bin] += v_cost
        heapq.heappush(
            heap,
            (bins_vis[chosen_bin], len(bins_idx[chosen_bin]), bins_tok[chosen_bin], chosen_bin),
        )
        for entry in popped:
            heapq.heappush(heap, entry)

    out_idx: List[List[int]] = []
    out_len: List[int] = []
    out_work: List[int] = []
    for b_items, b_tok, b_vis in zip(bins_idx, bins_tok, bins_vis):
        if not b_items:
            continue
        out_idx.append(b_items)
        out_len.append(int(b_tok))
        out_work.append(int(b_tok + b_vis + 300 * len(b_items)))
    return out_idx, out_len, out_work


def rebalance_packing_dataset(train_dataset: Any) -> Optional[List[int]]:
    """Re-pack a `PackingDataset` for uniform vision/segment load across DP ranks and return per-bin work costs."""
    if train_dataset is None or not hasattr(train_dataset, "packed_idx") or not hasattr(train_dataset, "packed_length"):
        return None
    orig_packed = getattr(train_dataset, "packed_idx", None)
    if not orig_packed or len(orig_packed) <= 1:
        return None

    is_dist_active = dist.is_available() and dist.is_initialized()
    is_rank0 = (not is_dist_active) or (dist.get_rank() == 0)

    payload: List[Any] = [None]
    if is_rank0:
        packing_len = int(getattr(train_dataset, "packing_length", 0) or max(train_dataset.packed_length))
        flat_indices: List[int] = []
        fallback_lens: Dict[int, int] = {}
        orig_seg_counts: List[int] = []
        for grp, grp_len in zip(orig_packed, train_dataset.packed_length):
            orig_seg_counts.append(len(grp))
            per_item_est = max(1, int(grp_len) // max(len(grp), 1))
            for idx in grp:
                ii = int(idx)
                flat_indices.append(ii)
                fallback_lens[ii] = per_item_est

        tok_lens, vis_costs = _extract_sample_costs_no_decode(
            getattr(train_dataset, "dataset", None), flat_indices, fallback_lens
        )
        new_idx, new_len, work_costs = repack_balanced_indices(
            flat_indices,
            tok_lens,
            vis_costs,
            packing_length=packing_len,
            initial_num_bins=len(orig_packed),
        )
        if new_idx:
            orig_max_segs = max(orig_seg_counts)
            orig_min_segs = max(1, min(orig_seg_counts))
            new_seg_counts = [len(g) for g in new_idx]
            logger.info(
                f"[TreeWeave] Balanced packing enabled: {len(orig_packed)} -> {len(new_idx)} bins | "
                f"segments/bin min..max: {orig_min_segs}..{orig_max_segs} -> "
                f"{min(new_seg_counts)}..{max(new_seg_counts)} | "
                f"work_cost min..max: {min(work_costs)}..{max(work_costs)}"
            )
            payload[0] = (new_idx, new_len, work_costs)

    if is_dist_active:
        dist.broadcast_object_list(payload, src=0)

    if payload[0] is None:
        return None
    new_idx, new_len, work_costs = payload[0]
    train_dataset.packed_idx = new_idx
    train_dataset.packed_length = new_len
    return work_costs


class _LightweightStage3Teacher(nn.Module):
    """Holds ONLY `merger` + `language_model` + `lm_head` of the native Teacher (drops duplicate ViT blocks!)."""

    def __init__(self, native_model: Qwen3_5ForConditionalGeneration):
        super().__init__()
        # Drop the 24-layer ViT blocks & patch_embed since Student's frozen ViT produces identical raw features!
        native_model.model.visual.blocks = nn.ModuleList()
        self.model = native_model.model
        # Enable single-pass varlen SDPA on Teacher LLM too when `attn_impl='sdpa'`
        install_volume_bias_on_language_model(self.model.language_model, mode="none")
        self.lm_head = native_model.lm_head
        self.requires_grad_(False)
        self.eval()


class TreeWeaveDistillTrainer(Seq2SeqTrainer):
    """`ms-swift` `Seq2SeqTrainer` subclass executing Stage 1, Stage 2, and Stage 3 self-distillation."""

    def __init__(self, *args, distill_config: Optional[TreeWeaveDistillConfig] = None, **kwargs):
        self.distill_cfg = distill_config or get_active_distill_config()
        t_args = kwargs.get("args") if "args" in kwargs else (args[1] if len(args) > 1 else None)
        if t_args is not None and getattr(t_args, "gradient_checkpointing_kwargs", None) is None:
            t_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        super().__init__(*args, **kwargs)

        unwrapped = self.accelerator.unwrap_model(self.model)
        if not isinstance(unwrapped, Qwen3_5TreeWeaveForConditionalGeneration):
            raise TypeError(
                f"TreeWeaveDistillTrainer expects Qwen3_5TreeWeaveForConditionalGeneration, got {type(unwrapped).__name__}."
            )

        # Deterministic per-rank RNG for Stage 1 Stochastic Pyramid Weaving
        rank = getattr(self.accelerator, "process_index", 0)
        seed = int(getattr(self.args, "seed", 42)) + 10007 * int(rank)
        self._stochastic_rng = np.random.default_rng(seed)

        # Resolve Teacher Mode
        mode = self.distill_cfg.teacher_mode
        if mode == "auto":
            if self.distill_cfg.stage in (1, 2):
                mode = "shared_frozen"
            elif self.distill_cfg.teacher_model_path:
                mode = "external_teacher"
            else:
                mode = "offline_topk_cache"
        self._resolved_teacher_mode = mode

        self.external_teacher: Optional[nn.Module] = None
        if mode == "external_teacher":
            t_path = self.distill_cfg.teacher_model_path
            if not t_path:
                raise ValueError(
                    "Stage 3 with `teacher_mode='external_teacher'` requires `teacher_model_path` (native Qwen3.5 checkpoint)."
                )
            logger.info(f"[TreeWeave] Loading lightweight Stage-3 Teacher (Merger + LLM only) from: {t_path}")
            t_cfg = AutoConfig.from_pretrained(t_path)
            # Load in student's dtype on current device
            param_dtype = next(unwrapped.model.language_model.parameters()).dtype
            attn_impl = getattr(unwrapped.model.language_model.config, "_attn_implementation", "sdpa")
            native = Qwen3_5ForConditionalGeneration.from_pretrained(
                t_path,
                config=t_cfg,
                dtype=param_dtype,
                attn_implementation=attn_impl,
            )
            self.external_teacher = _LightweightStage3Teacher(native).to(self.accelerator.device)
            self.external_teacher.eval()

        # Vision- and segment-balanced re-packing + matched-cost rank grouping (eliminates multi-rank stragglers)
        self._balanced_work_costs: Optional[List[int]] = rebalance_packing_dataset(
            getattr(self, "train_dataset", None)
        )

        logger.info(
            f"[TreeWeave] Trainer initialized: stage={self.distill_cfg.stage}, "
            f"teacher_mode={self._resolved_teacher_mode}, topk={self.distill_cfg.distill_topk}, "
            f"lambda_rate={self.distill_cfg.lambda_rate}, lambda_bal={self.distill_cfg.lambda_bal}"
        )

    def get_train_dataloader(self, skip_batches: int = 0):
        dataloader = super().get_train_dataloader(skip_batches=skip_batches)
        if (
            self._balanced_work_costs is not None
            and getattr(self.args, "train_dataloader_shuffle", True)
            and dataloader is not None
        ):
            bs = getattr(dataloader, "batch_sampler", None)
            # Unwrap accelerate `SkipBatchSampler` if resuming mid-epoch
            inner_bs = getattr(bs, "batch_sampler", bs)
            if inner_bs is not None and hasattr(inner_bs, "group_by_length") and hasattr(inner_bs, "lengths"):
                if len(self._balanced_work_costs) == len(self.train_dataset):
                    inner_bs.group_by_length = True
                    inner_bs.lengths = self._balanced_work_costs
        return dataloader

    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Bypass parent's `prepare_logits_to_keep` because Stage 1 & Stage 3 dynamically shorten visual prefix!
        prepared = super(Seq2SeqTrainer, self)._prepare_inputs(inputs)
        return prepared

    def training_step(self, model: nn.Module, inputs: Dict[str, Any], *args, **kwargs) -> torch.Tensor:
        unwrapped = self.accelerator.unwrap_model(model)
        if isinstance(unwrapped, Qwen3_5TreeWeaveForConditionalGeneration):
            unwrapped.enforce_frozen_eval_mode(self.distill_cfg.stage)
        return super().training_step(model, inputs, *args, **kwargs)

    def compute_loss(self, model: nn.Module, inputs: Dict[str, Any], return_outputs: bool = False, num_items_in_batch=None):
        cfg = self.distill_cfg
        # Pop non-forward keys added by ms-swift Seq2SeqTrainer
        inputs.pop("compute_loss_func", None)
        inputs.pop("loss_scale", None)
        inputs.pop("channel", None)

        if self._resolved_teacher_mode == "offline_topk_cache":
            if "teacher_topk_indices" not in inputs or "teacher_topk_logprobs" not in inputs:
                raise ValueError(
                    "Stage 3 configured for `offline_topk_cache`, but batch has no `teacher_topk_indices`. "
                    "Either run `treeweave-cache-topk` on the rollout dataset or pass `--teacher_model_path /path/to/Qwen3.5-4B`."
                )

        # Call THROUGH `model(...)` (`__call__`) so DDP / DeepSpeed ZeRO-2 / ZeRO-3 / FSDP hooks execute properly!
        outputs = model(
            distill_stage=cfg.stage,
            external_teacher=self.external_teacher,
            distill_topk=cfg.distill_topk,
            distill_temperature=cfg.distill_temperature,
            distill_chunk_size=cfg.distill_chunk_size,
            lambda_rate=cfg.lambda_rate,
            lambda_bal=cfg.lambda_bal,
            rho_s1=cfg.rho_s1,
            rho_s2=cfg.rho_s2,
            rho_t1=cfg.rho_t1,
            rho_t2=cfg.rho_t2,
            stochastic_rng=self._stochastic_rng if model.training else np.random.default_rng(0),
            stage1_uniform_scale=cfg.stage1_uniform_scale,
            **inputs,
        )

        loss = outputs.loss
        mode = "train" if model.training else "eval"
        if hasattr(self, "custom_metrics") and mode in self.custom_metrics:
            cm = self.custom_metrics[mode]
            if outputs.distill_loss is not None:
                cm["distill_kl"].update(outputs.distill_loss.detach())
            if cfg.stage == 2 and outputs.rate_loss is not None and outputs.balance_loss is not None:
                cm["rate_retention"].update(outputs.rate_loss.detach())
                cm["moe_balance"].update(outputs.balance_loss.detach())
            if outputs.treeweave_metrics:
                for m_key, m_val in outputs.treeweave_metrics.items():
                    # Log concise summary keys to avoid flooding progress bars while preserving key diagnostics
                    if m_key in (
                        "actual_token_savings",
                        "expected_savings",
                        "img_expected_savings",
                        "vid_expected_savings",
                        "router_entropy",
                        "p_s1_mean",
                        "p_s2_mean",
                        "p_t1_mean",
                        "p_t2_mean",
                        "share_s0_t0",
                        "share_s0_t2",
                        "share_s1_t0",
                        "share_s2_t0",
                        "share_s2_t2",
                    ):
                        cm[m_key].update(m_val)

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        unwrapped = self.accelerator.unwrap_model(self.model)
        if isinstance(unwrapped, Qwen3_5TreeWeaveForConditionalGeneration):
            unwrapped.sync_config_to_hf()
        res = super()._save(output_dir=output_dir, state_dict=state_dict)
        if self.is_world_process_zero():
            saved_dir = Path(output_dir or self.args.output_dir)
            if saved_dir.name.startswith("checkpoint-") and (saved_dir / "config.json").is_file():
                update_last_checkpoint_symlink(self.args.output_dir, saved_dir)
        return res

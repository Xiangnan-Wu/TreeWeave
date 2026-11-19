"""ms-swift `TreeWeaveDistillTrainer`: High-Efficiency Stage 1 / Stage 2 / Stage 3 Self-Distillation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from swift.trainers import Seq2SeqTrainer
from swift.utils import get_logger

from ..modeling_treeweave import Qwen3_5TreeWeaveForConditionalGeneration

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


class _LightweightStage3Teacher(nn.Module):
    """Holds ONLY `merger` + `language_model` + `lm_head` of the native Teacher (drops duplicate ViT blocks!)."""

    def __init__(self, native_model: Qwen3_5ForConditionalGeneration):
        super().__init__()
        # Drop the 24-layer ViT blocks & patch_embed since Student's frozen ViT produces identical raw features!
        native_model.model.visual.blocks = nn.ModuleList()
        self.model = native_model.model
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

        logger.info(
            f"[TreeWeave] Trainer initialized: stage={self.distill_cfg.stage}, "
            f"teacher_mode={self._resolved_teacher_mode}, topk={self.distill_cfg.distill_topk}, "
            f"lambda_rate={self.distill_cfg.lambda_rate}, lambda_bal={self.distill_cfg.lambda_bal}"
        )

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

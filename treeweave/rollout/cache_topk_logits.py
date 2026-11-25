"""8-GPU Data-Parallel (`DP=8`) Offline Teacher Top-K (`K=512`) Logprob Extractor.

Precomputes reusable `(topk_indices int32, topk_logprobs float16)` `.safetensors` shards per sample so
Stage 1, Stage 2, and especially Stage 3 (`--distill_teacher_mode offline_topk_cache`) can run with
ZERO Teacher LLM forward FLOPs and ZERO extra Teacher VRAM across sweeps and ablations.

Key features:
- Multi-GPU `DP=8` sharding (`--gpus 0,1,2,3,4,5,6,7`) with UUID deduplication before sharding.
- Crash-safe atomic `.safetensors` writes (`tmp -> os.replace`) + header-verified resume skipping.
- Automatic `flash_attention_2` selection on CUDA (`sdpa` fallback on CPU / MPS / non-FA2 setups).
- Multi-sample length-sorted micro-batching (`--batch-size`, `--max-batch-tokens`) AND optional
  padding-free sequence packing (`--pack-sequences`) per Teacher forward pass.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from swift.model import get_model_processor
from swift.template import get_template
from swift.utils import get_logger

from ..losses import compute_teacher_topk_from_hidden, extract_active_positions
from ..modeling_treeweave import _maybe_install_qwen3_5_sp_linear_patch, _run_lm_varlen_safe
from ..swift_plugin.register import register_treeweave_with_swift
from ..swift_plugin.template import (
    QWEN3_5_TREEWEAVE_TEMPLATE_TYPE,
    Qwen3_5TreeWeaveTemplate,
    install_treeweave_resize_patch,
)
from ..volume_bias import install_volume_bias_on_language_model
from .offline_rollout import (
    _verify_and_log_gpu_pin,
    compute_sample_uuid,
    compute_topk_cache_uuid,
    enable_subprocess_resize_bootstrap,
)

logger = get_logger()


def _resolve_attn_impl(requested: str, dev: torch.device) -> str:
    if requested != "auto":
        if dev.type != "cuda" and "flash" in requested:
            return "sdpa"
        return requested
    if dev.type == "cuda" and importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def _is_valid_cached_safetensors(path: Path) -> bool:
    """Fast header-only integrity check for crash-safe resume skipping."""
    if not path.is_file():
        return False
    try:
        if path.stat().st_size == 0:
            path.unlink(missing_ok=True)
            return False
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            if not {"topk_indices", "topk_logprobs"}.issubset(keys):
                path.unlink(missing_ok=True)
                return False
            s_idx = list(f.get_slice("topk_indices").get_shape())
            s_lp = list(f.get_slice("topk_logprobs").get_shape())
            if len(s_idx) != 2 or s_idx != s_lp:
                path.unlink(missing_ok=True)
                return False
            if "target_labels" in keys:
                s_tgt = list(f.get_slice("target_labels").get_shape())
                if s_tgt != [s_idx[0]]:
                    path.unlink(missing_ok=True)
                    return False
        return True
    except Exception:
        path.unlink(missing_ok=True)
        return False


def _atomic_save_topk(
    target_st: Path,
    idx_t: torch.Tensor,
    lp_t: torch.Tensor,
    target_labels: Optional[torch.Tensor] = None,
    source_labels: Optional[Sequence[int] | torch.Tensor] = None,
    loss_scale: str = "default+ignore_empty_think",
) -> None:
    """Write `.safetensors` to a process-private temp file and `os.replace` atomically."""
    tmp_st = target_st.parent / f".{target_st.stem}.{os.getpid()}.tmp.safetensors"
    tensors: Dict[str, torch.Tensor] = {
        "topk_indices": idx_t.contiguous().cpu(),
        "topk_logprobs": lp_t.contiguous().cpu(),
    }
    if target_labels is not None:
        tensors["target_labels"] = target_labels.to(torch.int32).contiguous().cpu()
    if source_labels is not None:
        src_t = (
            source_labels.detach().to(torch.int32).contiguous().cpu()
            if isinstance(source_labels, torch.Tensor)
            else torch.tensor(list(source_labels), dtype=torch.int32)
        )
        tensors["source_labels"] = src_t
    try:
        save_file(
            tensors,
            str(tmp_st),
            metadata={"format": "pt", "loss_scale": str(loss_scale)},
        )
        os.replace(tmp_st, target_st)
    except BaseException:
        tmp_st.unlink(missing_ok=True)
        raise


def _unwrap_pooler_features(vision_out: Any) -> torch.Tensor:
    pooler = getattr(vision_out, "pooler_output", vision_out)
    if isinstance(pooler, (list, tuple)):
        return torch.cat(list(pooler), dim=0)
    return pooler


def _form_microbatches(
    encoded_items: List[Dict[str, Any]],
    *,
    batch_size: int,
    max_batch_tokens: int,
    pack_sequences: bool,
) -> List[List[Dict[str, Any]]]:
    """Group length-sorted encoded items into memory-bounded microbatches or packed rows."""
    eff_bs = max(1, int(batch_size))
    eff_tok_cap = max(1, int(max_batch_tokens))
    ordered = sorted(encoded_items, key=lambda x: x["seq_len"])

    batches: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_sum = 0
    cur_max = 0

    for item in ordered:
        slen = int(item["seq_len"])
        if not cur:
            cur = [item]
            cur_sum = slen
            cur_max = slen
            continue

        next_len = len(cur) + 1
        next_sum = cur_sum + slen
        next_max = max(cur_max, slen)
        cost = next_sum if pack_sequences else (next_len * next_max)

        if next_len <= eff_bs and cost <= max(eff_tok_cap, slen):
            cur.append(item)
            cur_sum = next_sum
            cur_max = next_max
        else:
            batches.append(cur)
            cur = [item]
            cur_sum = slen
            cur_max = slen

    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def _run_padded_microbatch(
    teacher: Qwen3_5ForConditionalGeneration,
    template: Any,
    sub_batch: List[Dict[str, Any]],
    *,
    dev: torch.device,
    topk: int,
    temperature: float,
    loss_scale: str = "default+ignore_empty_think",
) -> None:
    template.padding_free = False
    template.packing = False
    collated = template.data_collator([copy.deepcopy(it["encoded"]) for it in sub_batch])
    batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in collated.items()}

    labels = batch.pop("labels")
    tpos = batch.pop("text_position_ids", None)
    pos_ids = batch.get("position_ids")
    if isinstance(pos_ids, torch.Tensor) and pos_ids.ndim == 3 and pos_ids.shape[0] == 3 and tpos is not None:
        batch["position_ids"] = torch.cat((tpos.to(pos_ids.dtype).unsqueeze(0), pos_ids), dim=0)
    for k in (
        "loss_scale",
        "channel",
        "teacher_topk_indices",
        "teacher_topk_logprobs",
        "teacher_topk_valid_mask",
        "cu_seq_lens_q",
        "cu_seq_lens_k",
        "max_length_q",
        "max_length_k",
    ):
        batch.pop(k, None)

    model_out = teacher.model(**batch, use_cache=False)
    active = extract_active_positions(labels, text_position_ids=None)
    act_hidden = model_out.last_hidden_state[active.batch_indices, active.seq_indices]
    all_idx, all_lp = compute_teacher_topk_from_hidden(
        act_hidden, teacher.lm_head, topk=topk, temperature=temperature
    )

    for b_idx, item in enumerate(sub_batch):
        mask_b = active.batch_indices == b_idx
        idx_b = all_idx[mask_b]
        lp_b = all_lp[mask_b]
        tgt_b = active.target_labels[mask_b]
        if int(idx_b.shape[0]) != int(item["exp_active"]):
            raise RuntimeError(
                f"Active token count mismatch in padded Top-K extraction for {item['uid']}: "
                f"got {idx_b.shape[0]}, expected {item['exp_active']}."
            )
        _atomic_save_topk(
            item["target_st"],
            idx_b,
            lp_b,
            target_labels=tgt_b,
            source_labels=item["encoded"].get("labels"),
            loss_scale=loss_scale,
        )


@torch.no_grad()
def _run_packed_microbatch(
    teacher: Qwen3_5ForConditionalGeneration,
    template: Any,
    sub_batch: List[Dict[str, Any]],
    *,
    dev: torch.device,
    topk: int,
    temperature: float,
    loss_scale: str = "default+ignore_empty_think",
) -> None:
    if len(sub_batch) == 1:
        _run_padded_microbatch(
            teacher, template, sub_batch, dev=dev, topk=topk, temperature=temperature, loss_scale=loss_scale
        )
        return

    prev_pf = template.padding_free
    prev_pack = template.packing
    try:
        template.padding_free = True
        template.packing = False
        collated = template.data_collator([copy.deepcopy(it["encoded"]) for it in sub_batch])
    finally:
        template.padding_free = prev_pf
        template.packing = prev_pack

    batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in collated.items()}
    labels = batch.pop("labels")
    tpos = batch.pop("text_position_ids", None)
    for k in ("loss_scale", "channel", "teacher_topk_indices", "teacher_topk_logprobs", "teacher_topk_valid_mask"):
        batch.pop(k, None)

    pos_ids = batch.get("position_ids")
    if tpos is None and isinstance(pos_ids, torch.Tensor) and pos_ids.ndim == 3 and pos_ids.shape[0] == 4:
        tpos = pos_ids[0].long()
    elif isinstance(pos_ids, torch.Tensor) and pos_ids.ndim == 3 and pos_ids.shape[0] == 3 and tpos is not None:
        pos_ids = torch.cat((tpos.to(pos_ids.dtype).unsqueeze(0), pos_ids), dim=0)

    input_ids = batch["input_ids"]
    inputs_embeds = teacher.model.get_input_embeddings()(input_ids)
    if batch.get("pixel_values") is not None:
        img_feat = _unwrap_pooler_features(
            teacher.model.get_image_features(batch["pixel_values"], batch.get("image_grid_thw"))
        ).to(inputs_embeds.device, inputs_embeds.dtype)
        img_mask, _ = teacher.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=img_feat
        )
        inputs_embeds = inputs_embeds.masked_scatter(img_mask, img_feat)
    if batch.get("pixel_values_videos") is not None:
        vid_feat = _unwrap_pooler_features(
            teacher.model.get_video_features(batch["pixel_values_videos"], batch.get("video_grid_thw"))
        ).to(inputs_embeds.device, inputs_embeds.dtype)
        _, vid_mask = teacher.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=vid_feat
        )
        inputs_embeds = inputs_embeds.masked_scatter(vid_mask, vid_feat)

    lm_out = _run_lm_varlen_safe(
        teacher.model.language_model,
        inputs_embeds=inputs_embeds,
        position_ids=pos_ids,
        attention_mask=None,
        use_cache=False,
        cu_seq_lens_q=batch.get("cu_seq_lens_q"),
        cu_seq_lens_k=batch.get("cu_seq_lens_k"),
        max_length_q=batch.get("max_length_q"),
        max_length_k=batch.get("max_length_k"),
    )
    active = extract_active_positions(labels, text_position_ids=tpos)
    act_hidden = lm_out.last_hidden_state[active.batch_indices, active.seq_indices]
    all_idx, all_lp = compute_teacher_topk_from_hidden(
        act_hidden, teacher.lm_head, topk=topk, temperature=temperature
    )

    exp_counts = [int(it["exp_active"]) for it in sub_batch]
    if sum(exp_counts) != int(all_idx.shape[0]):
        raise RuntimeError(
            f"Packed Top-K active token total mismatch: expected {sum(exp_counts)}, got {all_idx.shape[0]}."
        )
    idx_splits = torch.split(all_idx, exp_counts, dim=0)
    lp_splits = torch.split(all_lp, exp_counts, dim=0)
    tgt_splits = torch.split(active.target_labels, exp_counts, dim=0)
    for item, idx_s, lp_s, tgt_s in zip(sub_batch, idx_splits, lp_splits, tgt_splits):
        _atomic_save_topk(
            item["target_st"],
            idx_s,
            lp_s,
            target_labels=tgt_s,
            source_labels=item["encoded"].get("labels"),
            loss_scale=loss_scale,
        )


@torch.no_grad()
def _topk_worker_shard(
    rank: int,
    gpu_id: Optional[str],
    shard_tasks: List[Tuple[str, Dict[str, Any]]],
    worker_cfg: Dict[str, Any],
    *,
    is_isolated_subprocess: bool = False,
) -> int:
    if gpu_id is not None and not is_isolated_subprocess:
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if str(gpu_id).lower() == "cpu" else str(gpu_id)

    os.environ["IMAGE_MAX_TOKEN_NUM"] = str(worker_cfg["image_max_tokens"])
    os.environ["VIDEO_MAX_TOKEN_NUM"] = str(worker_cfg["video_max_tokens"])
    os.environ["FPS_MAX_FRAMES"] = str(worker_cfg["max_frames"])
    # Ensure we do not load partially-built topk cache inside `template.encode` while generating it!
    os.environ.pop("TREEWEAVE_TOPK_CACHE_DIR", None)

    pixel_factor = int(worker_cfg["pixel_factor"])
    enable_subprocess_resize_bootstrap(pixel_factor)
    register_treeweave_with_swift(activate_distill_trainer=False)
    install_treeweave_resize_patch(pixel_factor)
    _maybe_install_qwen3_5_sp_linear_patch()

    cache_dir = Path(worker_cfg["cache_dir"]).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    pending: List[Tuple[str, Dict[str, Any], Path]] = []
    for uid, row in shard_tasks:
        target_st = cache_dir / f"{uid}.safetensors"
        if not _is_valid_cached_safetensors(target_st):
            clean_row = dict(row)
            clean_row.pop("topk_cache_path", None)
            pending.append((uid, clean_row, target_st))

    if not pending:
        logger.info(
            f"[TopK Worker {rank}] All {len(shard_tasks)} unique samples already cached in {cache_dir}."
        )
        return 0

    if is_isolated_subprocess:
        _verify_and_log_gpu_pin(rank, gpu_id)

    req_dev = worker_cfg.get("device")
    if gpu_id is not None and str(gpu_id).lower() != "cpu":
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif gpu_id is not None and str(gpu_id).lower() == "cpu":
        dev = torch.device("cpu")
    else:
        dev = torch.device(req_dev or ("cuda" if torch.cuda.is_available() else "cpu"))

    dtype_str = worker_cfg.get("dtype")
    if dtype_str is not None:
        dt = getattr(torch, dtype_str) if isinstance(dtype_str, str) else dtype_str
    else:
        dt = torch.bfloat16 if dev.type == "cuda" else torch.float32

    loss_scale = str(worker_cfg.get("loss_scale") or "default+ignore_empty_think")
    attn_impl = _resolve_attn_impl(str(worker_cfg["attn_impl"]), dev)
    logger.info(
        f"[TopK Worker {rank}] GPU={gpu_id} device={dev} dtype={dt} attn={attn_impl} "
        f"loss_scale={loss_scale} batch_size={worker_cfg['batch_size']} pack={worker_cfg['pack_sequences']}: "
        f"extracting {len(pending)}/{len(shard_tasks)} unique samples..."
    )

    teacher_model_path = str(worker_cfg["teacher_model_path"])
    _, processor = get_model_processor(teacher_model_path, model_type="qwen3_5", load_model=False)
    template = get_template(
        processor,
        template_type=QWEN3_5_TREEWEAVE_TEMPLATE_TYPE,
        loss_scale=loss_scale,
    )
    template.set_mode("train")
    template.topk_cache_dir = None

    teacher = Qwen3_5ForConditionalGeneration.from_pretrained(
        teacher_model_path,
        dtype=dt,
        attn_implementation=attn_impl,
    ).to(dev)
    # Enable single-pass varlen SDPA on Teacher too when `attn_impl='sdpa'`
    install_volume_bias_on_language_model(teacher.model.language_model, mode="none")
    teacher.eval()
    template.model = teacher

    batch_size = max(1, int(worker_cfg["batch_size"]))
    max_batch_tokens = max(1, int(worker_cfg["max_batch_tokens"]))
    pack_sequences = bool(worker_cfg["pack_sequences"])
    topk = int(worker_cfg["topk"])
    temperature = float(worker_cfg["temperature"])

    window_size = max(batch_size * 8, 32)
    computed = 0
    for win_start in range(0, len(pending), window_size):
        win_slice = pending[win_start : win_start + window_size]
        encoded_window: List[Dict[str, Any]] = []
        for uid, row, target_st in win_slice:
            enc = template.encode(row)
            enc.pop("topk_cache_path", None)
            enc.pop("topk_active_tokens", None)
            enc.pop("topk_active_targets", None)
            enc.pop("topk_truncation_side", None)
            exp_active = Qwen3_5TreeWeaveTemplate._count_active_supervised_tokens(enc.get("labels")) or 0
            encoded_window.append(
                {
                    "uid": uid,
                    "encoded": enc,
                    "target_st": target_st,
                    "seq_len": len(enc["input_ids"]),
                    "exp_active": exp_active,
                }
            )

        microbatches = _form_microbatches(
            encoded_window,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
            pack_sequences=pack_sequences,
        )
        for sub_batch in microbatches:
            if pack_sequences and len(sub_batch) > 1:
                _run_packed_microbatch(
                    teacher,
                    template,
                    sub_batch,
                    dev=dev,
                    topk=topk,
                    temperature=temperature,
                    loss_scale=loss_scale,
                )
            else:
                _run_padded_microbatch(
                    teacher,
                    template,
                    sub_batch,
                    dev=dev,
                    topk=topk,
                    temperature=temperature,
                    loss_scale=loss_scale,
                )
            computed += len(sub_batch)

    return computed


@torch.no_grad()
def cache_dataset_topk(
    *,
    teacher_model_path: str,
    rollout_jsonl: Path,
    output_jsonl: Path,
    cache_dir: Path,
    topk: int = 512,
    temperature: float = 1.0,
    pixel_factor: int = 128,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    gpus: Optional[Sequence[str]] = None,
    batch_size: int = 4,
    max_batch_tokens: int = 16384,
    pack_sequences: bool = False,
    attn_impl: str = "auto",
    loss_scale: str = "default+ignore_empty_think",
    image_max_tokens: int = 4096,
    video_max_tokens: int = 512,
    max_frames: int = 64,
    max_samples: Optional[int] = None,
) -> Path:
    """Precompute per-sample `(topk_indices int32, topk_logprobs float16, target_labels, source_labels)` shards."""
    enable_subprocess_resize_bootstrap(int(pixel_factor))
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Remove any leftover process-private temp files from interrupted runs
    for stale_tmp in cache_dir.glob(".*.tmp.safetensors"):
        stale_tmp.unlink(missing_ok=True)

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    unique_by_uid: Dict[str, Dict[str, Any]] = {}
    with Path(rollout_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rollout_uid = row.get("_rollout_uuid") or compute_sample_uuid(row)
            topk_uid = compute_topk_cache_uuid(row)
            row_with_uid = dict(row)
            row_with_uid["_rollout_uuid"] = rollout_uid
            row_with_uid["_topk_cache_uuid"] = topk_uid
            rows.append(row_with_uid)
            if topk_uid not in unique_by_uid:
                unique_by_uid[topk_uid] = row_with_uid
            if max_samples is not None and len(rows) >= max_samples:
                break

    if not rows:
        raise ValueError(f"No valid samples found in {rollout_jsonl}.")

    unique_tasks = list(unique_by_uid.items())
    dtype_name = str(dtype).removeprefix("torch.") if dtype is not None else None
    worker_cfg: Dict[str, Any] = {
        "teacher_model_path": str(teacher_model_path),
        "cache_dir": str(cache_dir),
        "topk": int(topk),
        "temperature": float(temperature),
        "pixel_factor": int(pixel_factor),
        "device": device,
        "dtype": dtype_name,
        "batch_size": int(batch_size),
        "max_batch_tokens": int(max_batch_tokens),
        "pack_sequences": bool(pack_sequences),
        "attn_impl": str(attn_impl),
        "loss_scale": str(loss_scale),
        "image_max_tokens": int(image_max_tokens),
        "video_max_tokens": int(video_max_tokens),
        "max_frames": int(max_frames),
    }

    gpu_list = list(gpus) if gpus else [None]
    num_shards = len(gpu_list)
    if num_shards == 1:
        _topk_worker_shard(0, gpu_list[0], unique_tasks, worker_cfg)
    else:
        # Use `subprocess.Popen` with per-worker `CUDA_VISIBLE_DEVICES` set BEFORE child Python/CUDA init!
        work_meta_dir = cache_dir / ".worker_manifests"
        work_meta_dir.mkdir(parents=True, exist_ok=True)
        cfg_file = work_meta_dir / f"worker_cfg_{os.getpid()}.json"
        cfg_file.write_text(json.dumps(worker_cfg, ensure_ascii=False), encoding="utf-8")
        repo_root = str(Path(__file__).resolve().parents[2])
        procs: List[Tuple[int, subprocess.Popen]] = []
        try:
            for r, gpu_id in enumerate(gpu_list):
                tasks_file = work_meta_dir / f"worker_{os.getpid()}_{r:02d}_tasks.jsonl"
                with tasks_file.open("w", encoding="utf-8") as tf:
                    for uid, item_row in unique_tasks[r::num_shards]:
                        tf.write(json.dumps({"uid": uid, "row": item_row}, ensure_ascii=False) + "\n")

                child_env = dict(os.environ)
                existing_pp = child_env.get("PYTHONPATH", "")
                if repo_root not in [p for p in existing_pp.split(os.pathsep) if p]:
                    child_env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else repo_root
                if gpu_id is not None:
                    child_env["CUDA_VISIBLE_DEVICES"] = "" if str(gpu_id).lower() == "cpu" else str(gpu_id)

                cmd = [
                    sys.executable,
                    "-m",
                    "treeweave.rollout.cache_topk_logits",
                    "--_worker-rank",
                    str(r),
                    "--_worker-gpu",
                    "" if gpu_id is None else str(gpu_id),
                    "--_worker-cfg",
                    str(cfg_file),
                    "--_worker-tasks",
                    str(tasks_file),
                ]
                procs.append((r, subprocess.Popen(cmd, env=child_env)))

            failed: List[Tuple[int, int]] = []
            for r, proc in procs:
                rc = proc.wait()
                if rc != 0:
                    failed.append((r, rc))
            if failed:
                raise RuntimeError(f"Top-K cache worker(s) exited with non-zero code: {failed}.")
        except BaseException:
            for _, proc in procs:
                if proc.poll() is None:
                    proc.terminate()
            for _, proc in procs:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            raise

    written = 0
    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for row in rows:
            topk_uid = row["_topk_cache_uuid"]
            target_st = cache_dir / f"{topk_uid}.safetensors"
            if not target_st.is_file():
                raise RuntimeError(f"Missing expected Top-K shard after extraction: {target_st}")
            out_row = dict(row)
            out_row["topk_cache_path"] = str(target_st)
            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1

    logger.info(
        f"[TreeWeave TopK Cache] Cached {len(unique_tasks)} unique shards / {written} rows "
        f"(K={topk}, DP={num_shards}, loss_scale={loss_scale}) -> {output_jsonl}"
    )
    return output_jsonl


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # Internal isolated worker flags (used by `subprocess.Popen` for strict per-GPU `CUDA_VISIBLE_DEVICES` isolation)
    p.add_argument("--_worker-rank", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-gpu", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-cfg", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-tasks", type=Path, default=None, help=argparse.SUPPRESS)

    p.add_argument("--teacher-model", required=False, help="Path or HF ID of native Qwen/Qwen3.5-4B teacher")
    p.add_argument("--rollout-jsonl", required=False, type=Path, help="Input teacher rollout JSONL")
    p.add_argument("--output-jsonl", required=False, type=Path, help="Output JSONL annotated with topk_cache_path")
    p.add_argument("--cache-dir", required=False, type=Path, help="Directory for <uuid>.safetensors Top-K shards")
    p.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated GPU IDs for DP=N sharding (use 'cpu' or '' for single-process default device)",
    )
    p.add_argument("--batch-size", type=int, default=4, help="Max samples per Teacher forward microbatch")
    p.add_argument("--max-batch-tokens", type=int, default=16384, help="Token budget cap per microbatch")
    p.add_argument("--pack-sequences", action="store_true", help="Pack multiple samples padding-free per forward")
    p.add_argument("--attn-impl", choices=("auto", "flash_attention_2", "sdpa", "eager"), default="auto")
    p.add_argument(
        "--loss-scale",
        default="default+ignore_empty_think",
        help="ms-swift template loss_scale matching training (`default+ignore_empty_think` prevents BPE boundary drift)",
    )
    p.add_argument("--topk", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--pixel-factor", type=int, default=128)
    p.add_argument("--image-max-tokens", type=int, default=4096)
    p.add_argument("--video-max-tokens", type=int, default=512)
    p.add_argument("--max-frames", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args(argv)

    if args._worker_rank is not None:
        cfg = json.loads(args._worker_cfg.read_text(encoding="utf-8"))
        shard_tasks: List[Tuple[str, Dict[str, Any]]] = []
        with args._worker_tasks.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    shard_tasks.append((rec["uid"], rec["row"]))
        gpu_arg = args._worker_gpu if (args._worker_gpu is not None and args._worker_gpu != "") else None
        _topk_worker_shard(
            args._worker_rank,
            gpu_arg,
            shard_tasks,
            cfg,
            is_isolated_subprocess=True,
        )
        return

    if not args.teacher_model or not args.rollout_jsonl or not args.output_jsonl or not args.cache_dir:
        p.error("--teacher-model, --rollout-jsonl, --output-jsonl, and --cache-dir are required.")

    raw_gpus = args.gpus.strip()
    if raw_gpus.lower() in ("", "none"):
        gpus = None
    elif raw_gpus.lower() == "cpu":
        gpus = ["cpu"]
    else:
        gpus = [g.strip() for g in raw_gpus.split(",") if g.strip()]

    cache_dataset_topk(
        teacher_model_path=args.teacher_model,
        rollout_jsonl=args.rollout_jsonl,
        output_jsonl=args.output_jsonl,
        cache_dir=args.cache_dir,
        topk=args.topk,
        temperature=args.temperature,
        pixel_factor=args.pixel_factor,
        gpus=gpus,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        pack_sequences=args.pack_sequences,
        attn_impl=args.attn_impl,
        loss_scale=args.loss_scale,
        image_max_tokens=args.image_max_tokens,
        video_max_tokens=args.video_max_tokens,
        max_frames=args.max_frames,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

"""8-GPU Data-Parallel (`DP=8`) Offline Self-Distillation Rollout Pipeline (SGLang FA3 / vLLM / Transformers).

Why Offline DP=8 Rollout Instead of Online Self-Distillation:
1. **100% Data Reuse**: Sample native `Qwen3.5-4B` trajectories ONCE and reuse `teacher_rollouts.jsonl`
   across Stage 1, Stage 2, Stage 3, and all Q1--Q5 ablations (saving 5x--8x total GPU hours).
2. **Linear 8x Scaling (`TP=1, DP=8`)**: `Qwen3.5-4B` in BF16 occupies only ~8.5 GB VRAM, leaving ~70 GB
   per H100-80GB purely for continuous batching KV cache with zero cross-GPU NCCL all-reduce overhead.
3. **Engine Choice (`sglang` vs `vllm`)**:
   - `sglang` (Recommended on H100 `sm90`): FlashAttention-3 (`fa3`) + fused Triton GatedDeltaNet kernels
     deliver ~20--30% faster prefill+decode on 512--4096 visual-token multimodal prompts.
   - `vllm`: High-throughput offline `VllmEngine` with `enable_prefix_caching=False` (avoids hashing overhead
     on unique images/videos) and `max_num_batched_tokens=32768`.
   - `transformers`: Portable local fallback for CPU / MPS verification.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from swift.utils import get_logger

from ..swift_plugin.template import install_treeweave_resize_patch

logger = get_logger()


def enable_subprocess_resize_bootstrap(pixel_factor: int = 128) -> str:
    """Install in-process resize patch AND configure `PYTHONPATH` `sitecustomize.py` bootstrap
    so spawned `vLLM` (`EngineCore`) and `SGLang` (`TokenizerManager` / `Scheduler`) OS child
    processes enforce the exact same `128px` / `64px` `smart_resize` visual grid as training.
    """
    install_treeweave_resize_patch(int(pixel_factor))
    bootstrap_dir = str(Path(__file__).resolve().parent / "_bootstrap")
    repo_root = str(Path(__file__).resolve().parents[2])
    existing = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    merged: List[str] = []
    for candidate in [bootstrap_dir, repo_root] + existing:
        if candidate not in merged:
            merged.append(candidate)
    new_pythonpath = os.pathsep.join(merged)
    os.environ["PYTHONPATH"] = new_pythonpath
    return new_pythonpath


def _canonical_media_list(items: Any) -> List[str]:
    out: List[str] = []
    for x in items or []:
        if isinstance(x, dict):
            raw = x.get("path") or x.get("image") or x.get("video") or ""
        else:
            raw = str(x)
        raw_str = str(raw)
        if raw_str and os.path.exists(raw_str):
            raw_str = str(Path(raw_str).resolve())
        out.append(raw_str)
    return out


def _canonicalize_turn(role: str, content: str) -> Dict[str, str]:
    """Normalize turn content identically to `Qwen3_5Template._swift_prepare_inputs` & `AutoPreprocessor`."""
    stripped = content.strip()
    if role == "assistant" and "</think>" in stripped and "<think>" in stripped:
        before, _, after = stripped.partition("</think>")
        reasoning = before.rstrip("\n").rsplit("<think>", 1)[-1].lstrip("\n").strip()
        rest = after.lstrip("\n")
        stripped = f"<think>\n{reasoning}\n</think>\n\n{rest}"
    return {"role": role, "content": stripped}


def compute_sample_uuid(row: Dict[str, Any]) -> str:
    """Deterministic SHA-1 over canonicalized PROMPT turns + media paths (for prompt-level rollout resume)."""
    msgs = list(row.get("messages") or [])
    if msgs and msgs[-1].get("role") == "assistant":
        msgs = msgs[:-1]
    canon_msgs = [
        _canonicalize_turn(str(m.get("role", "")), str(m.get("content", "")))
        for m in msgs
    ]
    payload = {
        "messages": canon_msgs,
        "images": _canonical_media_list(row.get("images")),
        "videos": _canonical_media_list(row.get("videos")),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def compute_topk_cache_uuid(row: Dict[str, Any]) -> str:
    """Deterministic SHA-1 over ALL turns (prompt + assistant response) + media paths.

    Invariant to `ms-swift`'s `AutoPreprocessor` (`MessagesPreprocessor`), which strips extra keys like
    `_rollout_uuid` / `topk_cache_path` and converts media strings to `{'bytes': None, 'path': ...}`,
    while strictly distinguishing distinct assistant responses for the same prompt!
    """
    msgs = list(row.get("messages") or [])
    canon_msgs = [
        _canonicalize_turn(str(m.get("role", "")), str(m.get("content", "")))
        for m in msgs
    ]
    sys_prompt = row.get("system")
    if isinstance(sys_prompt, str) and sys_prompt.strip():
        if not canon_msgs or canon_msgs[0]["role"] != "system":
            canon_msgs = [_canonicalize_turn("system", sys_prompt)] + canon_msgs
    payload = {
        "messages": canon_msgs,
        "images": _canonical_media_list(row.get("images")),
        "videos": _canonical_media_list(row.get("videos")),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def strip_trailing_assistant(row: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(row)
    msgs = list(out.get("messages") or [])
    if msgs and msgs[-1].get("role") == "assistant":
        msgs = msgs[:-1]
    out["messages"] = msgs
    return out


def resolve_rollout_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    import importlib.util
    import torch

    if not torch.cuda.is_available():
        return "transformers"
    # Prefer vLLM first in `auto` mode because `ms-swift`'s `VllmEngine` shares the exact
    # `qwen_vl_utils` `do_resize=False` + `(video_tensor, video_metadata)` timestamp pipeline
    # with `Qwen3_5TreeWeaveTemplate`; fall back to SGLang (`fa3`), then Transformers.
    if importlib.util.find_spec("vllm") is not None:
        return "vllm"
    if importlib.util.find_spec("sglang") is not None:
        return "sglang"
    return "transformers"


def _translate_sglang_engine_inputs(
    inputs: Dict[str, Any],
    orig_videos: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Translate `ms-swift` `Template` multimodal keys (`images`/`videos`) into `sgl.Engine.async_generate` schema."""
    engine_inputs = {
        k: v
        for k, v in inputs.items()
        if k not in ("template_inputs", "images", "videos", "audios", "mm_processor_kwargs")
    }
    images = inputs.get("images")
    if images:
        engine_inputs["image_data"] = images[0] if len(images) == 1 else list(images)
    videos = orig_videos if orig_videos else inputs.get("videos")
    if videos:
        engine_inputs["video_data"] = videos[0] if len(videos) == 1 else list(videos)
    audios = inputs.get("audios")
    if audios:
        engine_inputs["audio_data"] = audios[0] if len(audios) == 1 else list(audios)
    return engine_inputs


def _patch_sglang_engine_multimodal(engine: Any) -> Any:
    """Fix `ms-swift==4.6.x` `SglangEngine` passing unexpected `images=`/`videos=` kwargs to `sgl.Engine.async_generate`."""
    import types

    orig_infer_async = engine.infer_async

    async def _patched_infer_async(self, infer_request, request_config=None, *, pre_infer_hook=None, **kwargs):
        raw_videos = copy.deepcopy(getattr(infer_request, "videos", None) or [])

        def _combined_hook(hook_kwargs: Dict[str, Any]) -> Dict[str, Any]:
            hook_kwargs["inputs"]["_orig_raw_videos"] = raw_videos
            return pre_infer_hook(hook_kwargs) if pre_infer_hook is not None else hook_kwargs

        return await orig_infer_async(
            infer_request,
            request_config=request_config,
            pre_infer_hook=_combined_hook,
            **kwargs,
        )

    async def _patched_infer_full_async(self, inputs: Dict[str, Any], generation_config: Dict[str, Any], request_config):
        orig_vids = inputs.pop("_orig_raw_videos", None)
        engine_inputs = _translate_sglang_engine_inputs(inputs, orig_videos=orig_vids)
        output = await self.engine.async_generate(**engine_inputs, sampling_params=generation_config)
        output["prompt_token_ids"] = inputs["input_ids"]
        return self._create_chat_completion_response(output, inputs, request_config.return_details)

    engine.infer_async = types.MethodType(_patched_infer_async, engine)
    engine._infer_full_async = types.MethodType(_patched_infer_full_async, engine)
    return engine


def _resolve_sglang_attention_backend(requested: str = "auto") -> Optional[str]:
    """Select an architecture-safe SGLang attention backend (`fa3` is Hopper sm90-only; Blackwell B200/B300 uses `flashinfer`)."""
    if requested and requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major == 9:
            return "fa3"
        if major >= 10:
            return "flashinfer"
    return None


def _build_engine(
    *,
    model_path: str,
    backend: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_batch_size: int,
    enable_thinking: bool,
    image_max_tokens: int,
    video_max_tokens: int,
    max_frames: int,
    pixel_factor: int,
    sglang_attention_backend: str = "auto",
):
    # Ensure 128px (or 64px) grid alignment matches TreeWeave training across parent & spawned child workers!
    os.environ["IMAGE_MAX_TOKEN_NUM"] = str(image_max_tokens)
    os.environ["VIDEO_MAX_TOKEN_NUM"] = str(video_max_tokens)
    os.environ["FPS_MAX_FRAMES"] = str(max_frames)
    enable_subprocess_resize_bootstrap(pixel_factor)

    if backend == "sglang":
        from swift.infer_engine import SglangEngine

        sgl_attn = _resolve_sglang_attention_backend(sglang_attention_backend)
        sgl_kwargs = {"attention_backend": sgl_attn} if sgl_attn else {}
        engine = SglangEngine(
            model_id_or_path=model_path,
            model_type="qwen3_5",
            template_type="qwen3_5",
            context_length=max_model_len,
            mem_fraction_static=gpu_memory_utilization,
            tp_size=1,
            engine_kwargs=sgl_kwargs,
        )
        engine = _patch_sglang_engine_multimodal(engine)
    elif backend == "vllm":
        from swift.infer_engine import VllmEngine

        engine = VllmEngine(
            model_id_or_path=model_path,
            model_type="qwen3_5",
            template_type="qwen3_5",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            enable_prefix_caching=False,
            max_num_seqs=max_batch_size,
            engine_kwargs={"max_num_batched_tokens": max(max_model_len, 32768)},
        )
    elif backend == "transformers":
        import torch
        from swift.infer_engine import TransformersEngine

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        engine = TransformersEngine(
            model_path,
            model_type="qwen3_5",
            template_type="qwen3_5",
            torch_dtype=dtype,
            device_map=device_map,
            max_batch_size=max_batch_size,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend!r}. Expected sglang, vllm, or transformers.")

    for tmpl_attr in ("template", "default_template"):
        tmpl = getattr(engine, tmpl_attr, None)
        if tmpl is not None:
            if hasattr(tmpl, "enable_thinking"):
                tmpl.enable_thinking = enable_thinking
            if hasattr(tmpl, "max_length") and max_model_len:
                tmpl.max_length = max_model_len
            tok = getattr(tmpl, "tokenizer", None)
            if tok is not None and hasattr(tok, "convert_tokens_to_ids"):
                unk_id = getattr(tok, "unk_token_id", None)
                im_id = tok.convert_tokens_to_ids("<|image_pad|>")
                vid_id = tok.convert_tokens_to_ids("<|video_pad|>")
                if isinstance(im_id, int) and im_id >= 0 and im_id != unk_id:
                    tmpl.image_token_id = im_id
                if isinstance(vid_id, int) and vid_id >= 0 and vid_id != unk_id:
                    tmpl.video_token_id = vid_id
    return engine


def _format_completion(choice) -> str:
    msg = choice.message
    reasoning = getattr(msg, "reasoning_content", None)
    content = msg.content or ""
    if reasoning:
        return f"<think>\n{reasoning.strip()}\n</think>\n\n{content.lstrip()}"
    return content


def _worker_process_shard(
    rank: int,
    gpu_id: Optional[str],
    shard_rows: List[Dict[str, Any]],
    shard_output_path: str,
    worker_cfg: Dict[str, Any],
) -> int:
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if str(gpu_id).lower() == "cpu" else str(gpu_id)

    enable_subprocess_resize_bootstrap(int(worker_cfg["pixel_factor"]))
    out_file = Path(shard_output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Load already-completed UUIDs for crash-safe resume
    completed_uuids = set()
    if out_file.is_file():
        with out_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "_rollout_uuid" in rec:
                        completed_uuids.add(rec["_rollout_uuid"])
                except json.JSONDecodeError:
                    continue

    pending: List[Dict[str, Any]] = []
    for row in shard_rows:
        uid = row.get("_rollout_uuid") or compute_sample_uuid(row)
        if uid not in completed_uuids:
            r_copy = copy.deepcopy(row)
            r_copy["_rollout_uuid"] = uid
            pending.append(r_copy)

    if not pending:
        logger.info(f"[Rollout Worker {rank}] All {len(shard_rows)} samples already cached in {out_file}.")
        return 0

    backend = resolve_rollout_backend(worker_cfg["backend"])
    logger.info(
        f"[Rollout Worker {rank}] GPU={gpu_id} backend={backend}: generating {len(pending)}/{len(shard_rows)} samples..."
    )
    engine = _build_engine(
        model_path=worker_cfg["model_path"],
        backend=backend,
        max_model_len=worker_cfg["max_model_len"],
        gpu_memory_utilization=worker_cfg["gpu_memory_utilization"],
        max_batch_size=worker_cfg["max_batch_size"],
        enable_thinking=worker_cfg["enable_thinking"],
        image_max_tokens=worker_cfg["image_max_tokens"],
        video_max_tokens=worker_cfg["video_max_tokens"],
        max_frames=worker_cfg["max_frames"],
        pixel_factor=worker_cfg["pixel_factor"],
    )

    from swift.infer_engine import InferRequest, RequestConfig

    req_cfg = RequestConfig(
        max_tokens=worker_cfg["max_new_tokens"],
        temperature=worker_cfg["temperature"],
        top_p=worker_cfg["top_p"],
    )

    chunk_size = int(worker_cfg["chunk_size"])
    written = 0
    with out_file.open("a", encoding="utf-8") as sink:
        for start in range(0, len(pending), chunk_size):
            batch_rows = pending[start : start + chunk_size]
            requests = []
            for r in batch_rows:
                clean = strip_trailing_assistant(r)
                requests.append(
                    InferRequest(
                        messages=clean["messages"],
                        images=clean.get("images") or [],
                        videos=clean.get("videos") or [],
                    )
                )
            responses = engine.infer(requests, request_config=req_cfg, use_tqdm=(rank == 0))
            for orig_row, resp in zip(batch_rows, responses):
                if isinstance(resp, Exception):
                    logger.warning(f"[Rollout Worker {rank}] Skipping failed sample {orig_row['_rollout_uuid']}: {resp}")
                    continue
                text = _format_completion(resp.choices[0]).strip()
                if not text:
                    continue
                out_row = strip_trailing_assistant(orig_row)
                out_row["messages"].append({"role": "assistant", "content": text})
                out_row["finish_reason"] = resp.choices[0].finish_reason
                sink.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                written += 1
            sink.flush()

    return written


def load_input_jsonl(input_path: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("_rollout_uuid", compute_sample_uuid(row))
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def run_sharded_rollout(
    *,
    model_path: str,
    input_jsonl: Path,
    output_jsonl: Path,
    gpus: Optional[Sequence[str]] = None,
    backend: str = "auto",
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.88,
    max_batch_size: int = 64,
    chunk_size: int = 256,
    enable_thinking: bool = False,
    image_max_tokens: int = 4096,
    video_max_tokens: int = 512,
    max_frames: int = 64,
    pixel_factor: int = 128,
    max_samples: Optional[int] = None,
) -> Path:
    """Run resumable single- or multi-GPU (`DP=N`) offline teacher rollout and merge into `output_jsonl`."""
    rows = load_input_jsonl(Path(input_jsonl), max_samples=max_samples)
    if not rows:
        raise ValueError(f"No valid samples found in {input_jsonl}.")

    output_jsonl = Path(output_jsonl)
    shard_dir = output_jsonl.parent / f".{output_jsonl.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    worker_cfg = {
        "model_path": str(model_path),
        "backend": backend,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_model_len": int(max_model_len),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "max_batch_size": int(max_batch_size),
        "chunk_size": int(chunk_size),
        "enable_thinking": bool(enable_thinking),
        "image_max_tokens": int(image_max_tokens),
        "video_max_tokens": int(video_max_tokens),
        "max_frames": int(max_frames),
        "pixel_factor": int(pixel_factor),
    }

    enable_subprocess_resize_bootstrap(int(pixel_factor))
    gpu_list = list(gpus) if gpus else [None]
    num_shards = len(gpu_list)
    shard_paths = [shard_dir / f"shard_{r:02d}.jsonl" for r in range(num_shards)]

    if num_shards == 1:
        _worker_process_shard(0, gpu_list[0], rows, str(shard_paths[0]), worker_cfg)
    else:
        ctx = mp.get_context("spawn")
        procs = []
        for r, gpu_id in enumerate(gpu_list):
            p = ctx.Process(
                target=_worker_process_shard,
                args=(r, gpu_id, rows[r::num_shards], str(shard_paths[r]), worker_cfg),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Rollout worker exited with non-zero code {p.exitcode}.")

    # Deterministically merge shards in original input order
    by_uuid: Dict[str, Dict[str, Any]] = {}
    for sp in shard_paths:
        if not sp.is_file():
            continue
        with sp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    by_uuid[rec["_rollout_uuid"]] = rec

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    merged_count = 0
    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for row in rows:
            uid = row["_rollout_uuid"]
            if uid in by_uuid:
                out_f.write(json.dumps(by_uuid[uid], ensure_ascii=False) + "\n")
                merged_count += 1

    logger.info(f"[TreeWeave Rollout] Wrote {merged_count}/{len(rows)} reusable rollouts -> {output_jsonl}")
    return output_jsonl


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Path or HF ID of native Qwen/Qwen3.5-4B teacher")
    p.add_argument("--input-jsonl", required=True, type=Path, help="Input multimodal JSONL dataset")
    p.add_argument("--output-jsonl", required=True, type=Path, help="Output reusable teacher rollout JSONL")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU IDs for DP=N sharding (use 'cpu' on CPU)")
    p.add_argument("--backend", choices=("auto", "sglang", "vllm", "transformers"), default="auto")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy off-policy teacher rollout (VisionWeave default)")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    p.add_argument("--max-batch-size", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--enable-thinking", action="store_true", help="Enable <think> reasoning traces during rollout")
    p.add_argument("--image-max-tokens", type=int, default=4096)
    p.add_argument("--video-max-tokens", type=int, default=512)
    p.add_argument("--max-frames", type=int, default=64)
    p.add_argument("--pixel-factor", type=int, default=128)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args(argv)

    gpus = None if args.gpus.strip().lower() in ("", "cpu", "none") else [g.strip() for g in args.gpus.split(",") if g.strip()]
    run_sharded_rollout(
        model_path=args.model,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        gpus=gpus,
        backend=args.backend,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_batch_size=args.max_batch_size,
        chunk_size=args.chunk_size,
        enable_thinking=args.enable_thinking,
        image_max_tokens=args.image_max_tokens,
        video_max_tokens=args.video_max_tokens,
        max_frames=args.max_frames,
        pixel_factor=args.pixel_factor,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

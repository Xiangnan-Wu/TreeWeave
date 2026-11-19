# Portions derived from SGLang and VisionWeave (Apache-2.0).
# Adapted for Qwen3.5-TreeWeave 9-scale spatiotemporal tree routing & 4 FP32 router heads.
"""Transport `TreeRoutePayload` metadata between SGLang disaggregated encoder and tokenizer processes."""

from __future__ import annotations

import ast
import inspect
import logging
import math
import os
import textwrap
from contextvars import ContextVar
from typing import Optional

from .. import ARCHITECTURE
from .sglang_route import ROUTE_KWARG, ROUTE_META_KEY

logger = logging.getLogger(__name__)

__all__ = [
    "ROUTE_KWARG",
    "ROUTE_META_KEY",
    "install_arch_allowlist",
    "install_encoder_admission",
    "install_encoder_decode_offload",
    "install_encoder_embedding_dtype",
    "install_encoder_error_cleanup",
    "install_encoder_metadata",
    "install_encoder_video_factor",
    "install_encoder_video_timestamps",
    "install_receiver_fail_closed",
    "install_receiver_inflight_gate",
    "install_receiver_metadata",
    "installed_targets",
    "publish_route",
]

_MARK = "_treeweave"
_installed: set = set()
_ROUTE_SLOT: ContextVar[Optional[dict]] = ContextVar("treeweave_route_slot", default=None)
_ROUTE_BATCH: ContextVar[Optional[tuple]] = ContextVar("treeweave_route_batch", default=None)

_FP32_ROUTER_ISLAND = (
    "visual.router.head_s1.",
    "visual.router.head_s2.",
    "visual.router.head_t1.",
    "visual.router.head_t2.",
    "visual.router.bias_s1",
    "visual.router.bias_s2",
    "visual.router.bias_t1",
    "visual.router.bias_t2",
)
_DTYPE_ADOPTED = "_treeweave_embedding_dtype_adopted"


def publish_route(payload: dict) -> None:
    """Called once per ViT+Router encode with serialized `TreeRoutePayload.to_dict()`."""
    _ROUTE_SLOT.set(payload)


def _take_route() -> Optional[dict]:
    payload = _ROUTE_SLOT.get()
    _ROUTE_SLOT.set(None)
    return payload


def _take_route_batch() -> Optional[tuple]:
    batch = _ROUTE_BATCH.get()
    _ROUTE_BATCH.set(None)
    return batch


def _require(owner, *markers) -> None:
    src = inspect.getsource(owner)
    missing = [m for m in markers if m not in src]
    if missing:
        raise RuntimeError(
            f"treeweave.serving.core_patches expected lines in {owner.__qualname__} "
            f"that are missing in this sglang version: {missing}."
        )


def _require_absent(owner, *markers) -> None:
    src = inspect.getsource(owner)
    present = [m for m in markers if m in src]
    if present:
        raise RuntimeError(
            f"treeweave.serving.core_patches found unexpected lines in {owner.__qualname__}: {present}."
        )


def _require_trailing_raise(owner, needle: str) -> None:
    body = ast.parse(textwrap.dedent(inspect.getsource(owner))).body[0].body  # type: ignore[attr-defined]
    last = body[-1]
    ok = (
        isinstance(last, ast.If)
        and len(last.body) == 1
        and isinstance(last.body[0], ast.Raise)
        and (needle in ast.unparse(last.body[0]))
    )
    if not ok:
        raise RuntimeError(
            f"treeweave.serving.core_patches needs {owner.__qualname__} to end with raise {needle!r}."
        )


def _first_time(name: str) -> bool:
    if name in _installed:
        return False
    _installed.add(name)
    return True


def _item_routed_length(item_dict: dict) -> int:
    return sum(int(x) for x in item_dict["frame_lengths"])


def install_arch_allowlist() -> None:
    """Allow `Qwen3_5TreeWeaveForConditionalGeneration` through SGLang's encoder-disaggregation gate."""
    if not _first_time("arch_allowlist"):
        return
    from sglang.srt.arg_groups import pd_disaggregation_hook as hook
    from sglang.srt.arg_groups import pipeline

    if getattr(hook.handle_encoder_disaggregation, _MARK, False):
        return
    _require(
        hook.handle_encoder_disaggregation,
        "def handle_encoder_disaggregation(server_args: Any):",
        "model_arch = hf_config.architectures[0]",
        'f"Model type {model_arch} is not supported for encoder disaggregation. "',
    )
    _require_trailing_raise(
        hook.handle_encoder_disaggregation, "is not supported for encoder disaggregation"
    )
    _require(pipeline.run_resolution_pipeline, "handle_encoder_disaggregation(server_args)")
    original = hook.handle_encoder_disaggregation
    unsupported = f"Model type {ARCHITECTURE} is not supported for encoder disaggregation"

    def treeweave_handle_encoder_disaggregation(server_args):
        try:
            return original(server_args)
        except ValueError as exc:
            if not str(exc).startswith(unsupported):
                raise
            logger.info("[treeweave] %s allowed for encoder disaggregation", ARCHITECTURE)

    setattr(treeweave_handle_encoder_disaggregation, _MARK, True)
    hook.handle_encoder_disaggregation = treeweave_handle_encoder_disaggregation


def install_encoder_metadata() -> None:
    """Reconcile routed embedding counts on `MMEncoder` and attach `TreeRoutePayload` per request."""
    if not _first_time("encoder_metadata"):
        return
    from sglang.srt.disaggregation.encoder import runtime as encoder_runtime
    from sglang.srt.disaggregation.encoder import server as encoder_server

    MMEncoder = encoder_server.MMEncoder
    if getattr(MMEncoder._stage_embedding_batch, _MARK, False):
        return
    _require(
        MMEncoder._validate_embedding_token_count,
        "expected_tokens = sum(ctx.preprocess_result.token_counts)",
        "if mm_embedding.shape[0] != expected_tokens:",
    )
    _require(
        MMEncoder._stage_embeddings,
        "embedding = mm_embedding[token_offset : token_offset + num_tokens]",
        "self._stage_embedding_batch(staged_embeddings)",
    )
    _require(
        MMEncoder._stage_embedding_batch,
        "states = [self._embedding_state_for_stage(mm_data) for mm_data in embeddings]",
        "state.embedding_data = mm_data",
        "state.embedding_ready.set()",
    )
    _require(
        encoder_runtime.execute_encode_pipeline,
        "req_id, nbytes, embedding_len, embedding_dim",
    )

    original_validate = MMEncoder._validate_embedding_token_count
    original_stage_batch = MMEncoder._stage_embedding_batch

    def _reconcile_preforward_shape(encoder, mm_data) -> None:
        state = encoder.req_states.get(mm_data.req_id)
        meta = getattr(state, "embedding_data", None) if state is not None else None
        if meta is None or meta.embedding is not None or mm_data.embedding is None:
            return
        want = list(mm_data.shape or [])
        have = list(meta.shape or [])
        if len(have) != 2 or len(want) != 2 or have[1] != want[1]:
            raise RuntimeError(
                f"Pre-forward embedding metadata for {mm_data.req_id} is {have} vs staged {want}."
            )
        meta.shape = want

    def treeweave_validate_embedding_token_count(ctx, mm_embedding):
        payload = _take_route()
        if payload is None:
            raise RuntimeError(
                f"TreeWeave vision tower did not publish a route payload for modality={ctx.modality}."
            )
        lengths = [_item_routed_length(it) for it in payload["items"]]
        if len(lengths) != ctx.num_items:
            raise RuntimeError(
                f"TreeRoutePayload describes {len(lengths)} items, encode context has {ctx.num_items}."
            )
        rows = int(mm_embedding.shape[0])
        if sum(lengths) != rows:
            raise RuntimeError(
                f"TreeRoutePayload describes {sum(lengths)} routed tokens, encoder produced {rows} rows."
            )
        ctx.preprocess_result.token_counts[:] = lengths
        original_validate(ctx, mm_embedding)
        _ROUTE_BATCH.set((payload, tuple(int(n) for n in ctx.items_per_req)))

    def treeweave_stage_embedding_batch(self, embeddings):
        batch = _take_route_batch()
        if batch is None:
            raise RuntimeError("No TreeWeave route payload stashed for staged batch.")
        payload, items_per_req = batch
        if len(embeddings) != len(items_per_req):
            raise RuntimeError(
                f"Staging {len(embeddings)} request(s), encode context had {len(items_per_req)}."
            )
        items = payload["items"]
        item_offset = 0
        for mm_data, num_items in zip(embeddings, items_per_req):
            item_end = item_offset + num_items
            per_req = dict(payload)
            per_req["items"] = items[item_offset:item_end]
            rows = sum(_item_routed_length(it) for it in per_req["items"])
            staged_rows = int(mm_data.shape[0]) if mm_data.shape else 0
            if rows != staged_rows:
                raise RuntimeError(
                    f"Route slice for {mm_data.req_id} has {rows} tokens, staged embedding has {staged_rows}."
                )
            setattr(mm_data, ROUTE_META_KEY, [per_req])
            _reconcile_preforward_shape(self, mm_data)
            item_offset = item_end
        return original_stage_batch(self, embeddings)

    setattr(treeweave_validate_embedding_token_count, _MARK, True)
    setattr(treeweave_stage_embedding_batch, _MARK, True)
    MMEncoder._validate_embedding_token_count = staticmethod(treeweave_validate_embedding_token_count)
    MMEncoder._stage_embedding_batch = treeweave_stage_embedding_batch


def install_encoder_video_timestamps() -> None:
    """Verify SGLang `EncoderPreprocessor` emits per-frame `video_timestamps` for `qwen3_5`."""
    if not _first_time("encoder_video_timestamps"):
        return
    from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessor

    if getattr(EncoderPreprocessor.__init__, _MARK, False):
        return
    _require(
        EncoderPreprocessor._process_video_items,
        'processor_input["video_timestamps"] = video_timestamps',
        "timestamps = self._calculate_timestamps(",
    )
    covered = _timestamp_gate_model_types(EncoderPreprocessor)
    original_init = EncoderPreprocessor.__init__

    def treeweave_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.model_type not in covered:
            raise RuntimeError(
                f"Encoder model_type {self.model_type!r} is not in SGLang timestamp allowlist {covered}."
            )

    setattr(treeweave_init, _MARK, True)
    EncoderPreprocessor.__init__ = treeweave_init


def _timestamp_gate_model_types(preprocessor_cls) -> list:
    src = textwrap.dedent(inspect.getsource(preprocessor_cls._process_video_items))
    func = ast.parse(src).body[0]
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        assigns = [
            st
            for st in ast.walk(node)
            if isinstance(st, ast.Assign)
            and len(st.targets) == 1
            and isinstance(st.targets[0], ast.Subscript)
            and isinstance(st.targets[0].slice, ast.Constant)
            and st.targets[0].slice.value == "video_timestamps"
        ]
        if not assigns:
            continue
        for cmp in ast.walk(node.test):
            if (
                isinstance(cmp, ast.Compare)
                and len(cmp.ops) == 1
                and isinstance(cmp.ops[0], ast.In)
                and isinstance(cmp.left, ast.Attribute)
                and cmp.left.attr == "model_type"
                and isinstance(cmp.comparators[0], ast.List)
                and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in cmp.comparators[0].elts)
            ):
                return [e.value for e in cmp.comparators[0].elts]
    raise RuntimeError(
        "Could not locate `self.model_type in [...]` guarding `video_timestamps` in EncoderPreprocessor."
    )


def install_encoder_video_factor() -> None:
    """Rebind `encoder.preprocessor.preprocess_video` to TreeWeave's `128px` / `64px` video factor."""
    if not _first_time("encoder_video_factor"):
        return
    from sglang.srt.disaggregation.encoder import preprocessor
    from sglang.srt.multimodal.processors import qwen_vl

    if not getattr(qwen_vl.preprocess_video, "_treeweave_video", False):
        raise RuntimeError(
            "qwen_vl.preprocess_video was not aligned by `treeweave.serving.video`."
        )
    preprocessor.preprocess_video = qwen_vl.preprocess_video


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    val = int(raw)
    if val <= 0:
        raise RuntimeError(f"{name}={val} must be a positive integer.")
    return val


def install_encoder_decode_offload() -> None:
    """Offload video decoding to a background thread pool off the encoder's asyncio loop."""
    if not _first_time("encoder_decode_offload"):
        return
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from sglang.srt.disaggregation.encoder import preprocessor

    original = preprocessor.preprocess_video
    if getattr(original, _MARK, False):
        return
    target = getattr(original, "func", original)
    if not asyncio.iscoroutinefunction(target):
        raise RuntimeError(f"encoder.preprocessor.preprocess_video ({target!r}) is not async.")
    threads = _int_env("TREEWEAVE_ENCODER_DECODE_THREADS", 8)
    pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="treeweave-decode")

    async def offloaded_preprocess_video(vr, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(pool, lambda: asyncio.run(original(vr, *args, **kwargs)))

    setattr(offloaded_preprocess_video, _MARK, True)
    setattr(offloaded_preprocess_video, "_treeweave_video", True)
    preprocessor.preprocess_video = offloaded_preprocess_video


def install_encoder_admission() -> None:
    if not _first_time("encoder_admission"):
        return
    from sglang.srt.disaggregation.encoder import runtime

    if runtime.Modality.VIDEO in runtime._BATCHABLE_MODALITIES:
        raise RuntimeError("VIDEO in _BATCHABLE_MODALITIES requires re-verifying batch video slicing.")


def install_encoder_error_cleanup() -> None:
    if not _first_time("encoder_error_cleanup"):
        return
    from sglang.srt.disaggregation.encoder import runtime
    from sglang.srt.disaggregation.encoder.server import MMEncoder

    _require(runtime.execute_encode_pipeline, "nbytes, embedding_len, embedding_dim, error_msg, error_code = result")
    _require_absent(MMEncoder, "embedding_to_send")


def _adopt_tower_embedding_dtype(encoder) -> None:
    """Adopt `visual.merger` dtype (`bfloat16`/`float16`) rather than the 4 FP32 router scoring heads."""
    if getattr(encoder, _DTYPE_ADOPTED, False):
        return
    import torch

    want = encoder.model_config.dtype
    visual = getattr(encoder.model, "visual", None)
    merger = getattr(visual, "merger", None)
    if visual is None or merger is None:
        raise RuntimeError(f"{type(encoder.model).__name__} has no `visual` / `visual.merger`.")
    tower_dtype = getattr(visual, "dtype", None)
    merger_param = next(merger.parameters(), None)
    merger_dtype = None if merger_param is None else merger_param.dtype
    disagree = {
        name: dt
        for name, dt in (("visual.dtype", tower_dtype), ("visual.merger", merger_dtype))
        if dt is not None and dt != want
    }
    if disagree:
        raise RuntimeError(f"Encoder dtype {want} disagrees with {disagree}.")

    strays, island = [], {}
    for name, param in encoder.model.named_parameters():
        if name.startswith(_FP32_ROUTER_ISLAND):
            island[name] = param.dtype
        elif param.dtype != want:
            strays.append(f"{name}/{param.dtype}")
    if strays:
        raise RuntimeError(
            f"Expected non-head parameters to have dtype {want}, got: {sorted(strays)[:8]}."
        )
    bad_island = sorted(f"{n}/{d}" for n, d in island.items() if d is not torch.float32)
    if not island or bad_island:
        raise RuntimeError(
            f"Expected TreeWeave FP32 router heads {_FP32_ROUTER_ISLAND} to be float32, found {island or 'none'}."
        )
    encoder._embedding_dtype = want
    encoder._element_size = torch.tensor([], dtype=want).element_size()
    setattr(encoder, _DTYPE_ADOPTED, True)


def install_encoder_embedding_dtype() -> None:
    if not _first_time("encoder_embedding_dtype"):
        return
    from sglang.srt.disaggregation.encoder.server import MMEncoder

    if getattr(MMEncoder._publish_preprocess_metadata, _MARK, False):
        return
    _require(MMEncoder.__init__, "self._embedding_dtype = next(self.model.parameters()).dtype")
    original_publish = MMEncoder._publish_preprocess_metadata

    async def treeweave_publish_preprocess_metadata(self, ctx, requests):
        _adopt_tower_embedding_dtype(self)
        return await original_publish(self, ctx, requests)

    setattr(treeweave_publish_preprocess_metadata, _MARK, True)
    MMEncoder._publish_preprocess_metadata = treeweave_publish_preprocess_metadata


def install_receiver_metadata() -> None:
    """Register `treeweave_route` on both image and video SGLang receiver metadata channels."""
    if not _first_time("receiver_metadata"):
        return
    from sglang.srt.disaggregation.encoder import receiver

    if ROUTE_META_KEY in receiver._GENERAL_IMAGE_META_ATTRS or getattr(
        receiver.video_meta_attrs_for, _MARK, False
    ):
        return
    if ROUTE_META_KEY in receiver._VIDEO_META_TENSOR_ATTRS:
        raise RuntimeError(f"{ROUTE_META_KEY!r} must not be in _VIDEO_META_TENSOR_ATTRS.")

    original_video_attrs = receiver.video_meta_attrs_for

    def treeweave_video_meta_attrs_for(model_type=None):
        return original_video_attrs(model_type) + (ROUTE_META_KEY,)

    setattr(treeweave_video_meta_attrs_for, _MARK, True)
    receiver.video_meta_attrs_for = treeweave_video_meta_attrs_for
    receiver._GENERAL_IMAGE_META_ATTRS = receiver._GENERAL_IMAGE_META_ATTRS + (ROUTE_META_KEY,)


def install_receiver_fail_closed() -> None:
    if not _first_time("receiver_fail_closed"):
        return
    from sglang.srt.managers import tokenizer_manager as tm

    _require(
        tm._reject_missing_dispatched_encoder_embedding,
        "mm_inputs is None",
        'and disagg.encoder_transfer_backend == "zmq_to_tokenizer"',
    )


def install_receiver_inflight_gate() -> None:
    """Bound concurrent in-flight ZMQ encodes per tokenizer worker."""
    if not _first_time("receiver_inflight_gate"):
        return
    import asyncio

    from sglang.srt.disaggregation.encoder import runtime as encoder_runtime
    from sglang.srt.disaggregation.encoder.receiver import MMReceiverBase
    from sglang.srt.disaggregation.encoder.server import ENCODER_MAX_BATCH_SIZE as encoder_max_batch

    mark = _MARK + "_inflight"
    if getattr(MMReceiverBase.recv_mm_data, mark, False):
        return
    if encoder_runtime.Modality.VIDEO in encoder_runtime._BATCHABLE_MODALITIES:
        raise RuntimeError("VIDEO in _BATCHABLE_MODALITIES requires updating video inflight gate.")

    original = MMReceiverBase.recv_mm_data
    override = os.environ.get("TREEWEAVE_MAX_INFLIGHT_ENCODES")

    def _limit(self, slots_per_encoder: int) -> int:
        if override is not None and override.strip():
            return _int_env("TREEWEAVE_MAX_INFLIGHT_ENCODES", 1)
        from sglang.srt.server_args import get_global_server_args

        args = get_global_server_args()
        workers = max(1, getattr(args, "tokenizer_worker_num", 1) or 1)
        return max(1, math.ceil(len(self.encode_urls) * slots_per_encoder / workers))

    def _gate_for(self, request_obj):
        has_video = bool(getattr(request_obj, "video_data", None))
        attr = "_treeweave_inflight_video" if has_video else "_treeweave_inflight_batch"
        gate = getattr(self, attr, None)
        if gate is None:
            slots = 1 if has_video else int(encoder_max_batch)
            gate = asyncio.Semaphore(_limit(self, slots))
            setattr(self, attr, gate)
        return gate

    async def gated_recv_mm_data(self, request_obj, *args, **kwargs):
        async with _gate_for(self, request_obj):
            return await original(self, request_obj, *args, **kwargs)

    setattr(gated_recv_mm_data, mark, True)
    setattr(gated_recv_mm_data, _MARK, True)
    MMReceiverBase.recv_mm_data = gated_recv_mm_data


def installed_targets() -> list:
    from sglang.srt.arg_groups import pd_disaggregation_hook as hook
    from sglang.srt.disaggregation.encoder import receiver
    from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessor
    from sglang.srt.disaggregation.encoder.preprocessor import (
        preprocess_video as encoder_preprocess_video,
    )
    from sglang.srt.disaggregation.encoder.server import MMEncoder

    return [
        ("pd_disaggregation_hook.handle_encoder_disaggregation", hook.handle_encoder_disaggregation),
        ("MMEncoder._validate_embedding_token_count", MMEncoder._validate_embedding_token_count),
        ("MMEncoder._stage_embedding_batch", MMEncoder._stage_embedding_batch),
        ("MMEncoder._publish_preprocess_metadata", MMEncoder._publish_preprocess_metadata),
        ("EncoderPreprocessor.__init__", EncoderPreprocessor.__init__),
        ("encoder/preprocessor.preprocess_video", encoder_preprocess_video),
        ("receiver.video_meta_attrs_for", receiver.video_meta_attrs_for),
        ("MMReceiverBase.recv_mm_data", receiver.MMReceiverBase.recv_mm_data),
    ]

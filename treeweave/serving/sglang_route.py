"""SGLang Disaggregated Encoder <-> Language Server Wire Protocol, Routed Prompt Builder,
Scaled Integer MRoPE (`POSITION_SCALE = 2`), & Language-Side Volume Compensation Helpers.

Extends VisionWeave's `route.py` and `mrope.py` to support all 9 `(s, t)` TreeWeave granularities,
`128px` / `64px` spatial resize factors (`max_spatial_level in {1, 2}`), cross-frame folded tokens
`(tau_start < tau_end)`, `0`-token folded video slices (`frame_lengths[tau] == 0`), and `log_volumes`
propagation for `TreeWeaveAttentionVolumeAdapter` on the 8 `Qwen3_5Attention` full-attention layers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .. import (
    ARCHITECTURE,
    GRANULARITY_LOG_VOLUMES,
    NUM_GRANULARITIES,
    POSITION_SCALE,
    ROUTE_THRESHOLD_ENV,
    TreeWeaveConfig,
)
from ..layout import TreeRouteLayout

SCHEMA_VERSION: int = 2
ROUTE_META_KEY: str = "treeweave_route"
ROUTE_KWARG: str = ROUTE_META_KEY
BUDGET_SHIFT_ENV: str = "TREEWEAVE_BUDGET_SHIFT"
PIXEL_FACTOR_ENV: str = "TREEWEAVE_PIXEL_FACTOR"

__all__ = [
    "BUDGET_SHIFT_ENV",
    "PIXEL_FACTOR_ENV",
    "ROUTE_KWARG",
    "ROUTE_META_KEY",
    "RoutedPromptBuildResult",
    "RoutedSpanChunk",
    "SCHEMA_VERSION",
    "TreeRouteItem",
    "TreeRoutePayload",
    "TreeVisualRun",
    "apply_sglang_key_direction_volume_bias",
    "budget_logit_shift_from_env",
    "build_routed_input_ids_and_runs",
    "build_sglang_extend_log_volumes",
    "config_fingerprint",
    "derive_serving_geometry",
    "route_threshold_from_env",
    "tree_routed_mrope_positions",
    "validate_serving_geometry",
]


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def derive_serving_geometry(hf_config: Any) -> Tuple[TreeWeaveConfig, int, int, int, int]:
    """Derive `(tw_cfg, fine, pool, coarse, pixel_factor)` from a HF config object or JSON dict."""
    vision = _cfg_get(hf_config, "vision_config", {}) or {}
    v_depth = int(_cfg_get(vision, "depth", 24) or 24)
    v_heads = int(_cfg_get(vision, "num_heads", 16) or 16)
    raw_tw = _cfg_get(hf_config, "treeweave", None)
    if isinstance(raw_tw, TreeWeaveConfig):
        tw_cfg = raw_tw
    else:
        tw_cfg = TreeWeaveConfig.from_dict(raw_tw, vision_depth=v_depth, num_heads=v_heads)

    fine = int(_cfg_get(vision, "spatial_merge_size", tw_cfg.spatial_merge_size) or 2)
    patch_size = int(_cfg_get(vision, "patch_size", 16) or 16)
    coarse = 1 << (int(tw_cfg.max_spatial_level) + 1)  # 8 for s<=2 (128px), 4 for s<=1 (64px)
    pool = max(1, coarse // max(fine, 1))
    pixel_factor = patch_size * coarse
    return tw_cfg, fine, pool, coarse, pixel_factor


def validate_serving_geometry(hf_config: Any) -> Tuple[TreeWeaveConfig, int, int, int, int]:
    """Validate `hf_config` geometry for disaggregated SGLang serving and return derived geometry."""
    vision = _cfg_get(hf_config, "vision_config", None)
    if vision is None:
        raise ValueError("Missing `vision_config` in checkpoint configuration.")
    tw_cfg, fine, pool, coarse, pixel_factor = derive_serving_geometry(hf_config)

    patch_size = int(_cfg_get(vision, "patch_size", 0) or 0)
    if patch_size != 16 or fine != 2:
        raise ValueError(
            f"{ARCHITECTURE} requires patch_size == 16 and spatial_merge_size == 2, "
            f"got patch_size={patch_size}, spatial_merge_size={fine}."
        )
    if tw_cfg.max_spatial_level not in (1, 2):
        raise ValueError(
            f"{ARCHITECTURE} requires max_spatial_level in (1, 2), got {tw_cfg.max_spatial_level}."
        )
    if tw_cfg.max_temporal_level not in (0, 1, 2):
        raise ValueError(
            f"{ARCHITECTURE} requires max_temporal_level in (0, 1, 2), got {tw_cfg.max_temporal_level}."
        )
    if tw_cfg.router_cross_attn_depth != 6 or len(tw_cfg.router_sample_depths) != 3:
        raise ValueError(
            f"{ARCHITECTURE} requires router_cross_attn_depth == 6 over 3 sampled depths, "
            f"got depth={tw_cfg.router_cross_attn_depth}, sample_depths={tw_cfg.router_sample_depths}."
        )
    num_blocks = int(_cfg_get(vision, "depth", 0) or 0)
    depths = [int(d) for d in tw_cfg.router_sample_depths]
    if depths != sorted(depths) or any(d < 0 or d > num_blocks for d in depths):
        raise ValueError(
            f"router_sample_depths must be sorted within [0, {num_blocks}], got {depths}."
        )
    deepstack = list(_cfg_get(vision, "deepstack_visual_indexes", []) or [])
    if deepstack:
        raise ValueError(
            f"{ARCHITECTURE} does not support deepstack; got deepstack_visual_indexes={deepstack!r}."
        )
    if tw_cfg.use_volume_attn_bias and tw_cfg.volume_bias_mode not in ("key_direction", "none"):
        raise ValueError(
            f"{ARCHITECTURE} SGLang disaggregated serving supports volume_bias_mode in "
            f"('key_direction', 'none'), got {tw_cfg.volume_bias_mode!r} (paged FA3/FlashInfer "
            "kernels do not accept dense additive attention masks)."
        )
    return tw_cfg, fine, pool, coarse, pixel_factor


def config_fingerprint(hf_config: Any) -> str:
    """Short deterministic SHA-256 hex digest agreed upon by the SGLang encoder and language server."""
    vision = _cfg_get(hf_config, "vision_config", {}) or {}
    tw_cfg, fine, pool, coarse, pixel_factor = derive_serving_geometry(hf_config)
    payload = {
        "arch": list(_cfg_get(hf_config, "architectures", []) or []),
        "position_scale": POSITION_SCALE,
        "fine": fine,
        "pool": pool,
        "coarse": coarse,
        "pixel_factor": pixel_factor,
        "max_spatial_level": int(tw_cfg.max_spatial_level),
        "max_temporal_level": int(tw_cfg.max_temporal_level),
        "allow_fine_temporal_merge": bool(tw_cfg.allow_fine_temporal_merge),
        "use_volume_attn_bias": bool(tw_cfg.use_volume_attn_bias),
        "volume_bias_mode": str(tw_cfg.volume_bias_mode if tw_cfg.use_volume_attn_bias else "none"),
        "patch_size": int(_cfg_get(vision, "patch_size", 0) or 0),
        "temporal_patch_size": int(_cfg_get(vision, "temporal_patch_size", 0) or 0),
        "vision_hidden": int(_cfg_get(vision, "hidden_size", 0) or 0),
        "out_hidden": int(_cfg_get(vision, "out_hidden_size", 0) or 0),
        "vision_depth": int(_cfg_get(vision, "depth", 0) or 0),
        "router_depth": int(tw_cfg.router_cross_attn_depth),
        "router_sample_depths": [int(d) for d in tw_cfg.router_sample_depths],
        "router_heads": int(tw_cfg.router_num_heads),
        "router_mlp_ratio": int(tw_cfg.router_mlp_ratio),
        "image_token_id": int(_cfg_get(hf_config, "image_token_id", -1) or -1),
        "video_token_id": int(_cfg_get(hf_config, "video_token_id", -1) or -1),
        "vision_start_token_id": int(_cfg_get(hf_config, "vision_start_token_id", -1) or -1),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def route_threshold_from_env(default: Optional[float] = None) -> float:
    """Read and validate `TREEWEAVE_ROUTE_THRESHOLD` from the environment."""
    raw = os.environ.get(ROUTE_THRESHOLD_ENV)
    if raw is None or not raw.strip():
        if default is not None:
            return float(default)
        raise RuntimeError(
            f"{ROUTE_THRESHOLD_ENV} is not set. Launch through `python -m treeweave.serving.serve`, "
            "which exports the route threshold for both encoder and language processes."
        )
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{ROUTE_THRESHOLD_ENV}={raw!r} is not a valid float.") from exc
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(f"{ROUTE_THRESHOLD_ENV}={value} must be finite and non-negative.")
    return value


def budget_logit_shift_from_env(default: float = 0.0) -> float:
    """Read optional `TREEWEAVE_BUDGET_SHIFT` from the environment."""
    raw = os.environ.get(BUDGET_SHIFT_ENV)
    if raw is None or not raw.strip():
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{BUDGET_SHIFT_ENV}={raw!r} is not a valid float.") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"{BUDGET_SHIFT_ENV}={value} must be finite.")
    return value


@dataclass(frozen=True)
class TreeVisualRun:
    """One `<|vision_start|>`-delimited placeholder run in SGLang (an image, or one slice of a video).

    Note: Unlike VisionWeave, `length` (`spatial_pos_x2.shape[0]`) CAN BE `0` when all blocks in slice `tau`
    folded into a later completion slice `tau_end > tau`! Even when `length == 0`, `native_span` still advances
    the sequence position cursor so timestamps and following text tokens have 100% native-identical positions!
    """

    start: int                          # Index in `input_ids` where this run's placeholders start
    spatial_pos_x2: np.ndarray          # `(n, 2)` int32 `(2 * row_center, 2 * col_center)`
    tau_start: np.ndarray               # `(n,)` int32 start slice index (`0` for images)
    tau_end: np.ndarray                 # `(n,)` int32 completion slice index (`0` for images)
    slice_index: int                    # Which slice `0..T-1` of the parent media item this run belongs to
    media_id: int                       # Parent media item ID (for looking up `slice_base_pos_x2[tau_start]`)
    native_span: int                    # Unscaled `max(H, W) // 2`


def tree_routed_mrope_positions(
    total_length: int,
    runs: Sequence[TreeVisualRun],
) -> Tuple[np.ndarray, int]:
    """Compute exact integer-scaled `(3, total_length)` MRoPE positions and `mrope_position_delta` for SGLang!

    Theorem (Dyadic Half-Integer Exactness):
    - Every slice `tau` starts at an EVEN scaled coordinate `c_x2[tau] = 2 * c_unscaled[tau]`!
    - Therefore, for ANY temporally folded token spanning `[tau_start, tau_end]`, the midpoint base is:
          `c_mid_x2 = (c_x2[tau_start] + c_x2[tau_end]) // 2 = c_unscaled[tau_start] + c_unscaled[tau_end]`
      which is **ALWAYS AN EXACT INTEGER**!
    - Adding `spatial_pos_x2 = 2 * cell_start + (2^s - 1)` yields exact integer coordinates for all 9 scales!
    """
    ps = POSITION_SCALE
    positions = np.zeros((3, total_length), dtype=np.int64)
    cur_x2 = 0
    cursor = 0
    slice_bases_by_media: Dict[Tuple[int, int], int] = {}

    for run in runs:
        if run.start < cursor:
            raise ValueError(f"Overlapping visual run at {run.start} < cursor {cursor}.")
        text_len = run.start - cursor
        if text_len > 0:
            positions[:, cursor : run.start] = np.arange(text_len, dtype=np.int64) * ps + cur_x2
            cur_x2 += ps * text_len

        slice_bases_by_media[(run.media_id, run.slice_index)] = cur_x2
        n = int(run.spatial_pos_x2.shape[0])
        if n > 0:
            t0_base = np.array(
                [slice_bases_by_media[(run.media_id, int(t0))] for t0 in run.tau_start],
                dtype=np.int64,
            )
            t1_base = np.array(
                [slice_bases_by_media[(run.media_id, int(t1))] for t1 in run.tau_end],
                dtype=np.int64,
            )
            # Both `t0_base` and `t1_base` are multiples of `ps = 2`, so `(t0_base + t1_base) // 2` is EXACT!
            c_mid_x2 = (t0_base + t1_base) // 2
            hw_x2 = run.spatial_pos_x2.astype(np.int64)
            positions[0, run.start : run.start + n] = c_mid_x2
            positions[1, run.start : run.start + n] = c_mid_x2 + hw_x2[:, 0]
            positions[2, run.start : run.start + n] = c_mid_x2 + hw_x2[:, 1]

        cur_x2 += ps * int(run.native_span)
        cursor = run.start + n

    if cursor > total_length:
        raise ValueError(f"Visual runs reach {cursor} > total_length {total_length}.")
    if cursor < total_length:
        rem = total_length - cursor
        positions[:, cursor:] = np.arange(rem, dtype=np.int64) * ps + cur_x2

    delta = int(positions.max()) + ps * (1 - total_length) if total_length > 0 else 0
    return positions, delta


@dataclass(frozen=True)
class TreeRouteItem:
    """Compact ZMQ-serializable metadata for one routed image or video item in SGLang."""

    modality: str
    grid: Tuple[int, int, int]
    scale_ids: np.ndarray               # `(N_out,)` int8 `0..8`
    tau_start: np.ndarray               # `(N_out,)` int16
    tau_end: np.ndarray                 # `(N_out,)` int16
    spatial_pos_x2: np.ndarray          # `(N_out, 2)` int32
    frame_lengths: Tuple[int, ...]

    @property
    def output_length(self) -> int:
        return int(self.scale_ids.shape[0])

    @property
    def native_tokens(self) -> int:
        t, h, w = self.grid
        return int(t * (h // 2) * (w // 2))

    @property
    def log_volumes(self) -> np.ndarray:
        lut = np.asarray(GRANULARITY_LOG_VOLUMES, dtype=np.float32)
        return lut[self.scale_ids]

    def validate(self, fine: int = 2) -> None:
        if self.modality not in ("image", "video"):
            raise ValueError(f"Invalid modality {self.modality!r}.")
        t, h, w = int(self.grid[0]), int(self.grid[1]), int(self.grid[2])
        if t <= 0 or h <= 0 or w <= 0 or h % 4 != 0 or w % 4 != 0:
            raise ValueError(f"Invalid routed grid {self.grid}: expected positive (T, H, W) with H, W % 4 == 0.")
        if self.modality == "image" and t != 1:
            raise ValueError(f"Image route item must have T == 1, got grid={self.grid}.")
        if len(self.frame_lengths) != t:
            raise ValueError(
                f"frame_lengths has {len(self.frame_lengths)} entries, expected T={t} for grid={self.grid}."
            )
        n = sum(self.frame_lengths)
        if n != self.output_length:
            raise ValueError(f"sum(frame_lengths)={n} != output_length={self.output_length}.")
        max_per_slice = (h // fine) * (w // fine)
        for tau, fl in enumerate(self.frame_lengths):
            if not (0 <= int(fl) <= max_per_slice):
                raise ValueError(
                    f"Slice {tau} frame_length={fl} violates [0, {max_per_slice}] bound for grid={self.grid}."
                )
        if (
            self.tau_start.shape != (n,)
            or self.tau_end.shape != (n,)
            or self.spatial_pos_x2.shape != (n, 2)
        ):
            raise ValueError("Mismatched array shapes inside TreeRouteItem.")
        if n > 0:
            if int(self.scale_ids.min()) < 0 or int(self.scale_ids.max()) >= NUM_GRANULARITIES:
                raise ValueError("TreeRouteItem scale_ids out of [0, 9) range.")
            if np.any(self.tau_start < 0) or np.any(self.tau_end >= t) or np.any(self.tau_start > self.tau_end):
                raise ValueError("TreeRouteItem [tau_start, tau_end] bounds invalid.")

    @classmethod
    def from_layout(cls, modality: str, layout: TreeRouteLayout) -> "TreeRouteItem":
        if modality not in ("image", "video"):
            raise ValueError(f"Invalid modality {modality!r}.")
        return cls(
            modality=modality,
            grid=layout.grid,
            scale_ids=layout.scale_ids.astype(np.int8, copy=False),
            tau_start=layout.tau_start.astype(np.int16, copy=False),
            tau_end=layout.tau_end.astype(np.int16, copy=False),
            spatial_pos_x2=layout.spatial_pos_x2.astype(np.int32, copy=False),
            frame_lengths=layout.frame_lengths,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "grid": list(self.grid),
            "scale_ids": self.scale_ids.tobytes(),
            "tau_start": self.tau_start.tobytes(),
            "tau_end": self.tau_end.tobytes(),
            "spatial_pos_x2": self.spatial_pos_x2.tobytes(),
            "frame_lengths": list(self.frame_lengths),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TreeRouteItem":
        frame_lengths = tuple(int(x) for x in raw["frame_lengths"])
        n = sum(frame_lengths)
        return cls(
            modality=str(raw["modality"]),
            grid=tuple(int(x) for x in raw["grid"]),  # type: ignore[arg-type]
            scale_ids=np.frombuffer(raw["scale_ids"], dtype=np.int8, count=n).copy(),
            tau_start=np.frombuffer(raw["tau_start"], dtype=np.int16, count=n).copy(),
            tau_end=np.frombuffer(raw["tau_end"], dtype=np.int16, count=n).copy(),
            spatial_pos_x2=np.frombuffer(raw["spatial_pos_x2"], dtype=np.int32, count=2 * n).reshape(n, 2).copy(),
            frame_lengths=frame_lengths,
        )


@dataclass(frozen=True)
class TreeRoutePayload:
    """Envelope sent over SGLang `zmq_to_tokenizer` carrying `TreeRouteItem`s."""

    fingerprint: str
    items: Tuple[TreeRouteItem, ...]
    schema_version: int = SCHEMA_VERSION
    threshold: Optional[float] = None
    budget_logit_shift: float = 0.0

    @property
    def output_length(self) -> int:
        return sum(it.output_length for it in self.items)

    def validate(
        self,
        *,
        fingerprint: Optional[str] = None,
        threshold: Optional[float] = None,
        embedding_rows: Optional[int] = None,
        fine: int = 2,
    ) -> None:
        if int(self.schema_version) != SCHEMA_VERSION:
            raise ValueError(f"Expected schema_version={SCHEMA_VERSION}, got {self.schema_version}.")
        if not self.items:
            raise ValueError("TreeRoutePayload contains no media items.")
        if fingerprint is not None and self.fingerprint != fingerprint:
            raise ValueError(
                f"Checkpoint fingerprint skew between encoder ({self.fingerprint!r}) "
                f"and language processor ({fingerprint!r})."
            )
        if (
            threshold is not None
            and self.threshold is not None
            and not math.isclose(float(self.threshold), float(threshold), rel_tol=0.0, abs_tol=1e-6)
        ):
            raise ValueError(
                f"Route threshold skew between encoder ({self.threshold}) and language processor ({threshold})."
            )
        for item in self.items:
            item.validate(fine=fine)
        if embedding_rows is not None and self.output_length != int(embedding_rows):
            raise ValueError(
                f"TreeRoutePayload describes {self.output_length} tokens, "
                f"but encoder produced {embedding_rows} embedding rows."
            )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema_version": int(self.schema_version),
            "fingerprint": str(self.fingerprint),
            "items": [it.to_dict() for it in self.items],
            "budget_logit_shift": float(self.budget_logit_shift),
        }
        if self.threshold is not None:
            out["threshold"] = float(self.threshold)
        return out

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TreeRoutePayload":
        if int(raw.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"Expected schema_version={SCHEMA_VERSION}, got {raw.get('schema_version')}.")
        thr_raw = raw.get("threshold")
        return cls(
            fingerprint=str(raw["fingerprint"]),
            items=tuple(TreeRouteItem.from_dict(x) for x in raw["items"]),
            schema_version=SCHEMA_VERSION,
            threshold=float(thr_raw) if thr_raw is not None else None,
            budget_logit_shift=float(raw.get("budget_logit_shift", 0.0)),
        )


@dataclass(frozen=True)
class RoutedSpanChunk:
    """Non-empty placeholder span `(offset_start, offset_end)` (inclusive) for one SGLang `MultimodalDataItem`."""

    modality: str                       # `"image"` or `"video"`
    offset: Tuple[int, int]             # inclusive `(start, end)` in `input_ids` with `end >= start`
    embedding_slice: Tuple[int, int]    # `[emb_start, emb_end)` slice into `embeddings[modality]`
    log_volumes: np.ndarray             # `(num_tokens,)` float32 `ln(4^s * 2^t)`


@dataclass(frozen=True)
class RoutedPromptBuildResult:
    """Result of expanding/shrinking a prompt with routed `TreeRouteItem`s and per-frame timestamps."""

    input_ids: List[int]
    chunks: List[RoutedSpanChunk]
    runs: List[TreeVisualRun]
    token_log_volumes: np.ndarray       # `(len(input_ids),)` float32


def build_routed_input_ids_and_runs(
    prompt_ids: Sequence[int],
    *,
    image_items: Sequence[TreeRouteItem],
    video_items: Sequence[TreeRouteItem],
    video_timestamps: Optional[Sequence[Sequence[float]]],
    encode_timestamp_fn: Callable[[str], List[int]],
    img_token_id: int,
    video_token_id: int,
    vision_start_token_id: Optional[int],
    vision_end_token_id: Optional[int],
    fine: int = 2,
) -> RoutedPromptBuildResult:
    """Build routed `input_ids`, non-empty `MultimodalDataItem` chunks, and `TreeVisualRun`s (including 0-token slices).

    Mirrors SGLang's `QwenVLImageProcessor.build_input_ids_with_timestamps`, extended for TreeWeave:
    - Image placeholders (`[vision_start, img_token_id, vision_end]`) expand to `item.output_length` `img_token_id`s.
    - Video placeholders (`[video_token_id]`) expand into `T` slices `<{t:.1f} seconds><|vision_start|>[video_token_id * k_tau]<|vision_end|>`.
    - When `k_tau == 0` (entire slice `tau` folded into a later completion slice `tau_end > tau`), emits
      `<timestamp><|vision_start|><|vision_end|>` with ZERO `video_token_id`s, omits empty `RoutedSpanChunk`s
      so SGLang never sees inverted `(start, start - 1)` offsets, and STILL appends a 0-length `TreeVisualRun`
      so `tree_routed_mrope_positions` records `slice_bases_by_media[(media_id, tau)]` and advances `cur_x2`
      by `POSITION_SCALE * native_span`!
    """
    prompt = list(prompt_ids)
    placeholder_locs: List[Tuple[int, str]] = []
    for tok_idx, tok in enumerate(prompt):
        if tok == img_token_id:
            placeholder_locs.append((tok_idx, "image"))
        elif tok == video_token_id:
            placeholder_locs.append((tok_idx, "video"))

    input_ids: List[int] = []
    chunks: List[RoutedSpanChunk] = []
    runs: List[TreeVisualRun] = []

    cur_idx = 0
    img_idx = 0
    vid_idx = 0
    media_id_counter = 0
    emb_cursors = {"image": 0, "video": 0}

    for tok_idx, mod_name in placeholder_locs:
        if tok_idx < cur_idx:
            raise ValueError(f"Overlapping placeholder token at index {tok_idx} < cur_idx {cur_idx}.")

        # Copy preceding text (for images, includes `<|vision_start|>` right at `tok_idx - 1`)
        input_ids.extend(prompt[cur_idx:tok_idx])
        media_id = media_id_counter
        media_id_counter += 1

        if mod_name == "image":
            if img_idx >= len(image_items):
                raise ValueError(
                    f"Prompt has more image placeholders than route payload items ({len(image_items)})."
                )
            item = image_items[img_idx]
            img_idx += 1
            mm_token_num = item.output_length
            native_span = max(int(item.grid[1]), int(item.grid[2])) // fine
            mm_offset_start = len(input_ids)
            input_ids.extend([img_token_id] * mm_token_num)

            if mm_token_num > 0:
                e0 = emb_cursors["image"]
                emb_cursors["image"] = e0 + mm_token_num
                chunks.append(
                    RoutedSpanChunk(
                        modality="image",
                        offset=(mm_offset_start, mm_offset_start + mm_token_num - 1),
                        embedding_slice=(e0, e0 + mm_token_num),
                        log_volumes=item.log_volumes.copy(),
                    )
                )
            runs.append(
                TreeVisualRun(
                    start=mm_offset_start,
                    spatial_pos_x2=item.spatial_pos_x2,
                    tau_start=item.tau_start.astype(np.int32, copy=False),
                    tau_end=item.tau_end.astype(np.int32, copy=False),
                    slice_index=0,
                    media_id=media_id,
                    native_span=native_span,
                )
            )
        else:
            if vid_idx >= len(video_items):
                raise ValueError(
                    f"Prompt has more video placeholders than route payload items ({len(video_items)})."
                )
            item = video_items[vid_idx]
            if video_timestamps is None or vid_idx >= len(video_timestamps):
                raise ValueError(
                    f"TreeWeave requires per-frame timestamps for every video, missing for video {vid_idx}."
                )
            curr_timestamps = video_timestamps[vid_idx]
            vid_idx += 1

            num_frames = int(item.grid[0])
            if len(curr_timestamps) < num_frames:
                raise ValueError(
                    f"Video {vid_idx - 1} has {num_frames} frames, but only {len(curr_timestamps)} timestamps."
                )
            native_span = max(int(item.grid[1]), int(item.grid[2])) // fine
            item_log_vols = item.log_volumes
            frame_cursor = 0

            for frame_idx in range(num_frames):
                frame_seqlen = int(item.frame_lengths[frame_idx])
                curr_time = float(curr_timestamps[frame_idx])
                ts_tokens = list(encode_timestamp_fn(f"<{curr_time:.1f} seconds>"))
                input_ids.extend(ts_tokens)
                if vision_start_token_id is not None:
                    input_ids.append(vision_start_token_id)

                run_start = len(input_ids)
                if frame_seqlen > 0:
                    input_ids.extend([video_token_id] * frame_seqlen)
                    e0 = emb_cursors["video"]
                    emb_cursors["video"] = e0 + frame_seqlen
                    sl = slice(frame_cursor, frame_cursor + frame_seqlen)
                    chunks.append(
                        RoutedSpanChunk(
                            modality="video",
                            offset=(run_start, run_start + frame_seqlen - 1),
                            embedding_slice=(e0, e0 + frame_seqlen),
                            log_volumes=item_log_vols[sl].copy(),
                        )
                    )
                    runs.append(
                        TreeVisualRun(
                            start=run_start,
                            spatial_pos_x2=item.spatial_pos_x2[sl],
                            tau_start=item.tau_start[sl].astype(np.int32, copy=False),
                            tau_end=item.tau_end[sl].astype(np.int32, copy=False),
                            slice_index=frame_idx,
                            media_id=media_id,
                            native_span=native_span,
                        )
                    )
                    frame_cursor += frame_seqlen
                else:
                    # 0-token folded video slice: `<|vision_start|><|vision_end|>` with zero `<|video_pad|>` tokens!
                    runs.append(
                        TreeVisualRun(
                            start=run_start,
                            spatial_pos_x2=np.zeros((0, 2), dtype=np.int32),
                            tau_start=np.zeros((0,), dtype=np.int32),
                            tau_end=np.zeros((0,), dtype=np.int32),
                            slice_index=frame_idx,
                            media_id=media_id,
                            native_span=native_span,
                        )
                    )

                if vision_end_token_id is not None:
                    input_ids.append(vision_end_token_id)

        cur_idx = tok_idx + 1

    input_ids.extend(prompt[cur_idx:])
    if img_idx != len(image_items) or vid_idx != len(video_items):
        raise ValueError(
            f"Prompt consumed {img_idx}/{len(image_items)} image items and {vid_idx}/{len(video_items)} video items."
        )

    token_log_volumes = np.zeros(len(input_ids), dtype=np.float32)
    for ch in chunks:
        s_off, e_off = ch.offset
        token_log_volumes[s_off : e_off + 1] = ch.log_volumes

    return RoutedPromptBuildResult(
        input_ids=input_ids,
        chunks=chunks,
        runs=runs,
        token_log_volumes=token_log_volumes,
    )


def build_sglang_extend_log_volumes(
    forward_batch: Any,
    total_tokens: int,
    device: torch.device | str,
) -> Optional[torch.Tensor]:
    """Assemble flat `(total_tokens,)` `ln(M_j)` tensor for an SGLang prefill / chunked-prefill `ForwardBatch`.

    Returns `None` during decode or when no multimodal request in the batch carries non-zero log-volumes.
    """
    fmode = getattr(forward_batch, "forward_mode", None)
    if fmode is not None and hasattr(fmode, "is_decode") and fmode.is_decode():
        return None
    mm_list = getattr(forward_batch, "mm_inputs", None)
    if not mm_list or total_tokens <= 0:
        return None

    prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if prefix_lens is None:
        raw_p = getattr(forward_batch, "extend_prefix_lens", None)
        if isinstance(raw_p, torch.Tensor):
            prefix_lens = raw_p.detach().cpu().tolist()
    extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if extend_lens is None:
        raw_e = getattr(forward_batch, "extend_seq_lens", None)
        if isinstance(raw_e, torch.Tensor):
            extend_lens = raw_e.detach().cpu().tolist()
    if prefix_lens is None or extend_lens is None:
        return None
    if isinstance(prefix_lens, torch.Tensor):
        prefix_lens = prefix_lens.detach().cpu().tolist()
    if isinstance(extend_lens, torch.Tensor):
        extend_lens = extend_lens.detach().cpu().tolist()
    sum_extend = sum(int(x) for x in extend_lens)
    if len(mm_list) != len(extend_lens) or sum_extend <= 0 or sum_extend > int(total_tokens):
        return None

    buf = np.zeros(int(total_tokens), dtype=np.float32)
    has_any = False
    cursor = 0
    for mm, p_raw, e_raw in zip(mm_list, prefix_lens, extend_lens):
        p_len = int(p_raw)
        e_len = int(e_raw)
        if mm is not None and e_len > 0:
            for item in getattr(mm, "mm_items", None) or ():
                msd = getattr(item, "model_specific_data", None) or {}
                lv_raw = msd.get("treeweave_log_volumes")
                if lv_raw is None:
                    continue
                lv = (
                    lv_raw.detach().cpu().float().numpy()
                    if isinstance(lv_raw, torch.Tensor)
                    else np.asarray(lv_raw, dtype=np.float32)
                )
                lv_off = 0
                for s_off, e_off in getattr(item, "offsets", None) or ():
                    span_len = int(e_off) - int(s_off) + 1
                    if span_len <= 0:
                        continue
                    ov_s = max(int(s_off), p_len)
                    ov_e = min(int(e_off) + 1, p_len + e_len)
                    if ov_e > ov_s:
                        sub_lv = lv[lv_off + (ov_s - int(s_off)) : lv_off + (ov_e - int(s_off))]
                        if np.any(sub_lv != 0.0):
                            buf[cursor + (ov_s - p_len) : cursor + (ov_e - p_len)] = sub_lv
                            has_any = True
                    lv_off += span_len
        cursor += e_len

    if not has_any:
        return None
    return torch.from_numpy(buf).to(device=device, dtype=torch.float32)


def apply_sglang_key_direction_volume_bias(
    key_states: torch.Tensor,
    log_volumes_flat: Optional[torch.Tensor],
    key_volume_vec_local: torch.Tensor,
    *,
    num_kv_heads_local: int,
    head_dim: int,
    rotary_dim: int,
) -> torch.Tensor:
    """Apply `TreeWeaveAttentionVolumeAdapter` post-`k_norm` non-rotary key offset to flat SGLang keys.

    Supports both `(N_tokens, H_kv_local * head_dim)` and `(N_tokens, H_kv_local, head_dim)` SGLang layouts,
    preserving exact `head_dim` and writing volume-compensated keys straight into SGLang's KV cache pool!
    """
    if log_volumes_flat is None:
        return key_states
    n_tok = int(key_states.shape[0])
    if int(log_volumes_flat.shape[0]) != n_tok or n_tok == 0:
        return key_states

    orig_shape = key_states.shape
    k_3d = key_states.view(n_tok, num_kv_heads_local, head_dim)
    lv = log_volumes_flat.to(device=k_3d.device, dtype=k_3d.dtype)
    w_vec = key_volume_vec_local.to(device=k_3d.device, dtype=k_3d.dtype)
    delta = lv[:, None, None] * w_vec[None, :, :]
    k_rot = k_3d[:, :, :rotary_dim]
    k_pass = k_3d[:, :, rotary_dim:] + delta
    return torch.cat((k_rot, k_pass), dim=-1).reshape(orig_shape)

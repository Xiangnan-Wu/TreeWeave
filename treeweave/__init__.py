"""TreeWeave: Orthogonal Spatiotemporal Quad-Binary Tree Weaving for Native MLLMs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

POSITION_SCALE: int = 2
ARCHITECTURE: str = "Qwen3_5TreeWeaveForConditionalGeneration"
ROUTE_THRESHOLD_ENV: str = "TREEWEAVE_ROUTE_THRESHOLD"

SPATIAL_SCALES: Tuple[int, int, int] = (0, 1, 2)
TEMPORAL_SCALES: Tuple[int, int, int] = (0, 1, 2)
NUM_GRANULARITIES: int = len(SPATIAL_SCALES) * len(TEMPORAL_SCALES)  # 9


def granularity_index(s: int, t: int) -> int:
    """Map `(s, t) in {0,1,2} x {0,1,2}` to linear index `0..8`."""
    if s not in SPATIAL_SCALES or t not in TEMPORAL_SCALES:
        raise ValueError(f"Invalid granularity (s={s}, t={t}); expected s,t in {{0,1,2}}.")
    return s * 3 + t


def granularity_from_index(idx: int) -> Tuple[int, int]:
    """Inverse of `granularity_index`."""
    if not 0 <= idx < NUM_GRANULARITIES:
        raise ValueError(f"Granularity index must be in [0, {NUM_GRANULARITIES}), got {idx}.")
    return (idx // 3, idx % 3)


def granularity_volume(s: int, t: int) -> int:
    """Number of native post-merger visual tokens covered by a `(s, t)` node: `4^s * 2^t`."""
    return (4**s) * (2**t)


def granularity_log_volume(s: int, t: int) -> float:
    """Natural log of covered native token volume `ln(4^s * 2^t)`."""
    return math.log(float(granularity_volume(s, t)))


GRANULARITY_VOLUMES: Tuple[int, ...] = tuple(
    granularity_volume(s, t) for s in SPATIAL_SCALES for t in TEMPORAL_SCALES
)
GRANULARITY_LOG_VOLUMES: Tuple[float, ...] = tuple(
    granularity_log_volume(s, t) for s in SPATIAL_SCALES for t in TEMPORAL_SCALES
)


@dataclass
class TreeWeaveConfig:
    """Serializable configuration attached to `hf_config.treeweave`."""

    # Geometry & tree depth controls (also used directly for Q1 / Q2 ablations)
    spatial_merge_size: int = 2
    max_spatial_level: int = 2          # 1 -> 2D binary (VisionWeave), 2 -> 3-level quadtree
    max_temporal_level: int = 2         # 0 -> frame-independent, 2 -> up to 4-slice binary tree
    allow_fine_temporal_merge: bool = True  # False -> Rigid Octree ablation (Q2)
    temporal_window_size: int = 4

    # Router architecture
    router_cross_attn_depth: int = 6
    router_sample_depths: Tuple[int, int, int] = (0, 12, 24)
    router_num_heads: int = 16
    router_mlp_ratio: int = 2
    router_use_temporal_attn: bool = True

    # Hard-routing thresholds & inference-time single-knob temperature/shift
    theta_s1: float = 0.5
    theta_s2: float = 0.5
    theta_t1: float = 0.5
    theta_t2: float = 0.5
    budget_logit_shift: float = 0.0     # >0 compresses more aggressively, <0 retains more detail

    # Volume-preserving attention compensation (Q4 ablation toggle)
    use_volume_attn_bias: bool = True
    volume_bias_mode: str = "key_direction"  # "key_direction" | "exact_logit_bias" | "none"
    volume_gamma_init: float = 1.0

    # Initialization metadata
    initial_compression_prob: float = 0.1
    seed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def image_resize_factor(self) -> int:
        """Pixel alignment factor: `patch_size(16) * 2^(max_spatial_level + 1)` -> 128 for s<=2, 64 for s<=1."""
        return 16 * (2 ** (self.max_spatial_level + 1))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["router_sample_depths"] = list(self.router_sample_depths)
        return d

    @classmethod
    def from_dict(cls, raw: Dict[str, Any] | None, vision_depth: int = 24, num_heads: int = 16) -> "TreeWeaveConfig":
        if not raw:
            return cls(
                router_sample_depths=(0, vision_depth // 2, vision_depth),
                router_num_heads=num_heads,
            )
        data = dict(raw)
        if "router_sample_depths" in data:
            depths = tuple(int(x) for x in data["router_sample_depths"])
            data["router_sample_depths"] = depths  # type: ignore[assignment]
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        extra = {k: data.pop(k) for k in list(data.keys()) if k not in known}
        extra.update(data.pop("extra", {}) or {})
        return cls(**data, extra=extra)


__all__ = [
    "ARCHITECTURE",
    "GRANULARITY_LOG_VOLUMES",
    "GRANULARITY_VOLUMES",
    "NUM_GRANULARITIES",
    "POSITION_SCALE",
    "ROUTE_THRESHOLD_ENV",
    "SPATIAL_SCALES",
    "TEMPORAL_SCALES",
    "TreeWeaveConfig",
    "granularity_from_index",
    "granularity_index",
    "granularity_log_volume",
    "granularity_volume",
]

"""Image/video `128px` / `64px` grid alignment and routed MRoPE tripwires for SGLang serving."""

from __future__ import annotations

import functools
import importlib
import logging
import os
from typing import Any, Optional

from .. import ARCHITECTURE
from .sglang_route import PIXEL_FACTOR_ENV, derive_serving_geometry

logger = logging.getLogger(__name__)

__all__ = [
    "get_pixel_factor",
    "get_pool_factor",
    "install_all",
    "install_resize_factor_patch",
    "install_rope_index_tripwire",
    "set_geometry",
]

_MARK = "_treeweave"
_PATCH_ATTR = "_treeweave_patched_factor"
_installed: set = set()
_POOL: Optional[int] = None
_PIXEL_FACTOR: Optional[int] = None


def set_geometry(hf_config: Any) -> None:
    """Publish `(pool, pixel_factor)` derived from `hf_config.treeweave.max_spatial_level`."""
    global _POOL, _PIXEL_FACTOR
    _, _, pool, _, pixel_factor = derive_serving_geometry(hf_config)
    if _PIXEL_FACTOR is not None and _PIXEL_FACTOR != pixel_factor:
        raise RuntimeError(
            f"treeweave.serving.patches already configured for pixel_factor={_PIXEL_FACTOR}, "
            f"cannot reconfigure to {pixel_factor} in the same process."
        )
    _POOL = pool
    _PIXEL_FACTOR = pixel_factor
    os.environ[PIXEL_FACTOR_ENV] = str(pixel_factor)


def get_pool_factor() -> int:
    if _POOL is not None:
        return _POOL
    raw = os.environ.get(PIXEL_FACTOR_ENV, "").strip()
    if raw:
        return max(1, int(raw) // 32)
    raise RuntimeError(
        "treeweave.serving.patches.set_geometry() was not called and TREEWEAVE_PIXEL_FACTOR is unset."
    )


def get_pixel_factor() -> int:
    if _PIXEL_FACTOR is not None:
        return _PIXEL_FACTOR
    raw = os.environ.get(PIXEL_FACTOR_ENV, "").strip()
    if raw:
        val = int(raw)
        if val > 0 and val % 64 == 0:
            return val
    raise RuntimeError(
        "treeweave.serving.patches.set_geometry() was not called and TREEWEAVE_PIXEL_FACTOR is unset."
    )


def _first_time(name: str) -> bool:
    if name in _installed:
        return False
    _installed.add(name)
    return True


def _wrap_smart_resize(orig_fn: Any) -> Any:
    if getattr(orig_fn, _PATCH_ATTR, False) or getattr(orig_fn, _MARK, False):
        return orig_fn

    @functools.wraps(orig_fn)
    def treeweave_smart_resize(
        height: int,
        width: int,
        factor: int = 32,
        min_pixels=None,
        max_pixels=None,
        *args,
        **kwargs,
    ):
        px_factor = get_pixel_factor()
        eff_factor = max(int(factor), px_factor)
        if eff_factor % px_factor != 0:
            eff_factor = ((eff_factor + px_factor - 1) // px_factor) * px_factor
        min_floor = eff_factor * eff_factor
        if min_pixels is not None:
            min_pixels = max(int(min_pixels), min_floor)
        else:
            min_pixels = min_floor
        if max_pixels is not None:
            max_pixels = max(int(max_pixels), min_pixels)
        h_bar, w_bar = orig_fn(
            height,
            width,
            factor=eff_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            *args,
            **kwargs,
        )
        return max(eff_factor, int(h_bar)), max(eff_factor, int(w_bar))

    setattr(treeweave_smart_resize, _PATCH_ATTR, True)
    setattr(treeweave_smart_resize, _MARK, True)
    return treeweave_smart_resize


def install_resize_factor_patch() -> None:
    """Align `transformers` and `qwen_vl_utils` `smart_resize` bindings to `128px` / `64px`."""
    if not _first_time("resize_factor"):
        return
    patched = []
    for mod_path in (
        "transformers.models.qwen2_vl.image_processing_qwen2_vl",
        "transformers.models.qwen2_vl.image_processing_pil_qwen2_vl",
        "transformers.models.qwen2_vl.image_processing_qwen2_vl_fast",
        "transformers.models.qwen2_vl.video_processing_qwen2_vl",
        "transformers.models.qwen3_vl.video_processing_qwen3_vl",
        "qwen_vl_utils.vision_process",
    ):
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        fn = getattr(mod, "smart_resize", None)
        if callable(fn):
            setattr(mod, "smart_resize", _wrap_smart_resize(fn))
            patched.append(mod_path.rsplit(".", 1)[-1])
    logger.info("[treeweave] smart_resize aligned to pixel_factor in %s", ", ".join(patched))


def install_rope_index_tripwire() -> None:
    """Ensure `MRotaryEmbedding.get_rope_index` never silently returns unrouted native positions."""
    if not _first_time("rope_index_tripwire"):
        return
    from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

    if getattr(MRotaryEmbedding.get_rope_index, _MARK, False):
        return

    def treeweave_get_rope_index(*args, **kwargs):
        raise RuntimeError(
            f"MRotaryEmbedding.get_rope_index was called while serving {ARCHITECTURE}. "
            "TreeWeave positions come exclusively from `tree_routed_mrope_positions` in "
            "`treeweave/serving/processors/processor.py`."
        )

    setattr(treeweave_get_rope_index, _MARK, True)
    MRotaryEmbedding.get_rope_index = staticmethod(treeweave_get_rope_index)
    logger.info("[treeweave] MRotaryEmbedding.get_rope_index -> tripwire installed")


def install_all() -> None:
    """Install all serving patches idempotently in the current worker process."""
    from . import position_patches, video

    video.apply_sglang_video_factor()
    install_resize_factor_patch()
    install_rope_index_tripwire()
    position_patches.install_position_scale_rope_patch()
    position_patches.install_mrope_arithmetic_patches()
    position_patches.install_mrope_fastpath_guard()

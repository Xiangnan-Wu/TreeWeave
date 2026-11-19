"""Align SGLang video preprocessing dimensions to TreeWeave's `128px` (`s<=2`) or `64px` (`s<=1`) grid."""

from __future__ import annotations

import functools
import inspect
import logging
import os

from sglang.srt.multimodal.processors import qwen_vl

from .sglang_route import PIXEL_FACTOR_ENV

logger = logging.getLogger(__name__)
UPSTREAM_IMAGE_FACTOR = 28
_MARKER = "[treeweave] video resize factor"


def get_video_pixel_factor() -> int:
    raw = os.environ.get(PIXEL_FACTOR_ENV, "128").strip()
    factor = int(raw) if raw else 128
    if factor <= 0 or factor % 64 != 0:
        raise ValueError(f"{PIXEL_FACTOR_ENV}={factor} must be a positive multiple of 64.")
    return factor


def apply_sglang_video_factor() -> None:
    video_factor = get_video_pixel_factor()
    if getattr(qwen_vl.preprocess_video, "_treeweave_video", False):
        return
    if qwen_vl.IMAGE_FACTOR != UPSTREAM_IMAGE_FACTOR:
        raise RuntimeError(
            f"Expected upstream qwen_vl.IMAGE_FACTOR={UPSTREAM_IMAGE_FACTOR}, found {qwen_vl.IMAGE_FACTOR}; "
            "re-verify video factor patch against this SGLang build."
        )
    orig_fn = qwen_vl.preprocess_video
    sig = inspect.signature(orig_fn)
    if "image_factor" not in sig.parameters:
        raise RuntimeError("sglang qwen_vl.preprocess_video no longer accepts `image_factor`.")

    def _bind(*args, **kwargs):
        px = get_video_pixel_factor()
        bound = sig.bind_partial(*args, **kwargs)
        cur = bound.arguments.get("image_factor")
        if cur is None or int(cur) < px:
            eff = px
        elif int(cur) % px != 0:
            eff = ((int(cur) + px - 1) // px) * px
        else:
            eff = int(cur)
        bound.arguments["image_factor"] = eff
        return bound.args, bound.kwargs

    if inspect.iscoroutinefunction(orig_fn):
        @functools.wraps(orig_fn)
        async def patched(*args, **kwargs):
            b_args, b_kwargs = _bind(*args, **kwargs)
            return await orig_fn(*b_args, **b_kwargs)
    else:
        @functools.wraps(orig_fn)
        def patched(*args, **kwargs):
            b_args, b_kwargs = _bind(*args, **kwargs)
            return orig_fn(*b_args, **b_kwargs)

    patched._treeweave_video = True  # type: ignore[attr-defined]
    qwen_vl.preprocess_video = patched
    msg = f"{_MARKER} {UPSTREAM_IMAGE_FACTOR} -> {video_factor}"
    logger.info(msg)


apply_sglang_video_factor()

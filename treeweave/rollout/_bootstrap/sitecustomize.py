"""Auto-loaded Python startup hook (`PYTHONPATH=.../treeweave/rollout/_bootstrap`) for vLLM/SGLang child processes.

Both `vLLM` (V1 `EngineCore` subprocesses) and `SGLang` (`TokenizerManager` / `Scheduler` subprocesses) use
`multiprocessing` `spawn`, which starts fresh Python interpreters that do not inherit in-memory monkey-patches
from the parent worker. Placing this directory on `PYTHONPATH` ensures every spawned child interpreter applies
the exact `TREEWEAVE_PIXEL_FACTOR` (`128` for `s<=2`, `64` for `s<=1`) `smart_resize` alignment at import time
with zero eager `torch`/`transformers` import overhead when those modules are unused.
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Set

_PATCH_ATTR = "_treeweave_patched_factor"
_HOOK_ATTR = "_treeweave_resize_meta_hook"
_THIS_DIR = str(Path(__file__).resolve().parent)

_TARGET_MODULES: Set[str] = {
    "qwen_vl_utils.vision_process",
    "transformers.models.qwen2_vl.image_processing_qwen2_vl",
    "transformers.models.qwen2_vl.image_processing_qwen2_vl_fast",
    "transformers.models.qwen2_vl.image_processing_pil_qwen2_vl",
    "transformers.models.qwen2_vl.video_processing_qwen2_vl",
    "transformers.models.qwen3_vl.video_processing_qwen3_vl",
    "sglang.srt.multimodal.processors.qwen_vl",
    "vllm.model_executor.models.qwen2_vl",
    "vllm.model_executor.models.qwen3_vl",
}


def _chain_downstream_sitecustomize() -> None:
    """Execute any pre-existing `sitecustomize.py` later on `sys.path` so user/env hooks are preserved."""
    other_paths = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            if str(Path(entry).resolve()) == _THIS_DIR:
                continue
        except OSError:
            pass
        other_paths.append(entry)
    if not other_paths:
        return
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", other_paths)
    if spec is not None and spec.origin and Path(spec.origin).is_file():
        try:
            code = compile(Path(spec.origin).read_bytes(), spec.origin, "exec")
            downstream_ns = {
                "__name__": "sitecustomize",
                "__file__": spec.origin,
                "__package__": None,
                "__cached__": getattr(spec, "cached", None),
                "__builtins__": __builtins__,
            }
            exec(code, downstream_ns)
        except Exception:
            pass


def _active_pixel_factor(default_factor: int) -> int:
    raw = os.environ.get("TREEWEAVE_PIXEL_FACTOR", "")
    if raw.strip():
        try:
            val = int(raw.strip())
            if val > 0 and val % 64 == 0:
                return val
        except ValueError:
            pass
    return default_factor


def _wrap_smart_resize(orig_fn: Any, default_factor: int) -> Any:
    if getattr(orig_fn, _PATCH_ATTR, False):
        return orig_fn

    @functools.wraps(orig_fn)
    def wrapped_smart_resize(
        height: int,
        width: int,
        factor: int = 32,
        min_pixels=None,
        max_pixels=None,
        *args,
        **kwargs,
    ):
        px_factor = _active_pixel_factor(default_factor)
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

    setattr(wrapped_smart_resize, _PATCH_ATTR, True)
    return wrapped_smart_resize


def _wrap_sglang_preprocess_video(orig_fn: Any, default_factor: int) -> Any:
    if getattr(orig_fn, _PATCH_ATTR, False):
        return orig_fn
    sig = inspect.signature(orig_fn)
    if "image_factor" not in sig.parameters:
        return orig_fn

    def _bind_aligned_args(*args, **kwargs):
        px_factor = _active_pixel_factor(default_factor)
        bound = sig.bind_partial(*args, **kwargs)
        cur = bound.arguments.get("image_factor")
        if cur is None or int(cur) < px_factor:
            eff = px_factor
        elif int(cur) % px_factor != 0:
            eff = ((int(cur) + px_factor - 1) // px_factor) * px_factor
        else:
            eff = int(cur)
        bound.arguments["image_factor"] = eff
        return bound.args, bound.kwargs

    if inspect.iscoroutinefunction(orig_fn):
        @functools.wraps(orig_fn)
        async def async_wrapped(*args, **kwargs):
            b_args, b_kwargs = _bind_aligned_args(*args, **kwargs)
            return await orig_fn(*b_args, **b_kwargs)

        setattr(async_wrapped, _PATCH_ATTR, True)
        setattr(async_wrapped, "_treeweave_video", True)
        return async_wrapped

    @functools.wraps(orig_fn)
    def sync_wrapped(*args, **kwargs):
        b_args, b_kwargs = _bind_aligned_args(*args, **kwargs)
        return orig_fn(*b_args, **b_kwargs)

    setattr(sync_wrapped, _PATCH_ATTR, True)
    setattr(sync_wrapped, "_treeweave_video", True)
    return sync_wrapped


def _patch_module(module: Any, default_factor: int) -> None:
    fn = getattr(module, "smart_resize", None)
    if callable(fn):
        setattr(module, "smart_resize", _wrap_smart_resize(fn, default_factor))
    if getattr(module, "__name__", "") == "sglang.srt.multimodal.processors.qwen_vl":
        pv = getattr(module, "preprocess_video", None)
        if callable(pv):
            setattr(module, "preprocess_video", _wrap_sglang_preprocess_video(pv, default_factor))


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, orig_loader: Any, default_factor: int):
        self._orig_loader = orig_loader
        self._default_factor = default_factor

    def create_module(self, spec):
        if hasattr(self._orig_loader, "create_module"):
            return self._orig_loader.create_module(spec)
        return None

    def exec_module(self, module):
        self._orig_loader.exec_module(module)
        _patch_module(module, self._default_factor)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig_loader, name)


class _TreeWeaveResizeImportHook(importlib.abc.MetaPathFinder):
    _treeweave_resize_meta_hook = True

    def __init__(self, default_factor: int):
        self.default_factor = default_factor
        self._in_progress: Set[str] = set()

    def find_spec(self, fullname: str, path=None, target=None):
        if fullname not in _TARGET_MODULES or fullname in self._in_progress:
            return None
        self._in_progress.add(fullname)
        try:
            for finder in sys.meta_path:
                if finder is self or getattr(finder, _HOOK_ATTR, False):
                    continue
                find_spec = getattr(finder, "find_spec", None)
                if find_spec is None:
                    continue
                spec = find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = _PatchingLoader(spec.loader, self.default_factor)
                    return spec
            return None
        finally:
            self._in_progress.discard(fullname)


def _install_bootstrap_if_configured() -> None:
    raw = os.environ.get("TREEWEAVE_PIXEL_FACTOR", "").strip()
    if not raw:
        return
    try:
        factor = int(raw)
    except ValueError:
        return
    if factor <= 0 or factor % 64 != 0:
        return

    if not any(getattr(f, _HOOK_ATTR, False) for f in sys.meta_path):
        sys.meta_path.insert(0, _TreeWeaveResizeImportHook(factor))

    for mod_name in list(_TARGET_MODULES):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            _patch_module(mod, factor)


_chain_downstream_sitecustomize()
_install_bootstrap_if_configured()

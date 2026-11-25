"""Pytest bootstrap ensuring CPU unit tests pass cleanly on GPU dev machines with CUDA-only `fla`/`causal_conv1d`.

On NVIDIA B300 / H100 dev machines where `causal-conv1d` and `flash-linear-attention` (`fla`) are installed,
`transformers.models.qwen3_5.modeling_qwen3_5` binds `causal_conv1d_fn`, `causal_conv1d_update`,
`chunk_gated_delta_rule`, and `fused_recurrent_gated_delta_rule` to CUDA-only Triton/C++ kernels, which
crash when unit tests instantiate tiny Qwen3.5 models on CPU.

This hook wraps those bindings so CPU tensors cleanly execute the pure-PyTorch reference implementations,
while CUDA tensors still invoke the original accelerated kernels unmodified.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _install_qwen3_5_cpu_kernel_fallbacks() -> None:
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as m
    except ImportError:
        return

    if getattr(m, "_treeweave_pytest_cpu_fallback_installed", False):
        return

    act2fn = getattr(m, "ACT2FN", None)

    def _apply_act(x: torch.Tensor, activation: str | None) -> torch.Tensor:
        if activation is None:
            return x
        if act2fn is not None and activation in act2fn:
            return act2fn[activation](x)
        if activation in ("silu", "swish"):
            return F.silu(x)
        raise ValueError(f"Unsupported activation in CPU fallback: {activation!r}")

    def _torch_causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        _, hidden_size, seq_len = hidden_states.shape
        padding = weight.shape[-1] - 1
        out = F.conv1d(
            hidden_states.to(weight.dtype),
            weight=weight.unsqueeze(1),
            bias=bias,
            padding=padding,
            groups=hidden_size,
        )[:, :, :seq_len]
        out = _apply_act(out, activation)
        return out.to(hidden_states.dtype)

    def _torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None, **kwargs):
        _, hidden_size, seq_len = hidden_states.shape
        state_len = conv_state.shape[-1]
        hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
        conv_state.copy_(hidden_states_new[:, :, -state_len:])
        out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
        out = out[:, :, -seq_len:]
        out = _apply_act(out, activation)
        return out.to(hidden_states.dtype)

    orig_conv_fn = getattr(m, "causal_conv1d_fn", None)
    if callable(orig_conv_fn):
        def _safe_conv_fn(hidden_states, *args, **kwargs):
            if isinstance(hidden_states, torch.Tensor) and not hidden_states.is_cuda:
                return _torch_causal_conv1d_fn(hidden_states, *args, **kwargs)
            return orig_conv_fn(hidden_states, *args, **kwargs)

        m.causal_conv1d_fn = _safe_conv_fn

    orig_conv_upd = getattr(m, "causal_conv1d_update", None)
    if callable(orig_conv_upd):
        def _safe_conv_upd(hidden_states, *args, **kwargs):
            if isinstance(hidden_states, torch.Tensor) and not hidden_states.is_cuda:
                return _torch_causal_conv1d_update(hidden_states, *args, **kwargs)
            return orig_conv_upd(hidden_states, *args, **kwargs)

        m.causal_conv1d_update = _safe_conv_upd

    # Unwrap pure-PyTorch GatedDeltaNet fallbacks in case they carry `@use_kernel_func_from_hub_with_fallback`
    orig_torch_chunk = getattr(m, "torch_chunk_gated_delta_rule", None)
    pure_torch_chunk = getattr(orig_torch_chunk, "__wrapped__", orig_torch_chunk)
    orig_torch_recur = getattr(m, "torch_recurrent_gated_delta_rule", None)
    pure_torch_recur = getattr(orig_torch_recur, "__wrapped__", orig_torch_recur)

    def _wrap_delta_rule(orig_fn, pure_fn):
        if not callable(orig_fn) or not callable(pure_fn):
            return orig_fn

        def _safe_delta(query, *args, **kwargs):
            if isinstance(query, torch.Tensor) and not query.is_cuda:
                # Strip CUDA-only kwargs (e.g. `cu_seqlens`) not accepted by pure-PyTorch reference
                clean_kw = dict(kwargs)
                clean_kw.pop("cu_seqlens", None)
                clean_kw.pop("cu_seq_lens_q", None)
                try:
                    return pure_fn(query, *args, **clean_kw)
                except TypeError:
                    return pure_fn(query, *args)
            return orig_fn(query, *args, **kwargs)

        return _safe_delta

    if callable(orig_torch_chunk):
        m.torch_chunk_gated_delta_rule = _wrap_delta_rule(orig_torch_chunk, pure_torch_chunk)
    if callable(orig_torch_recur):
        m.torch_recurrent_gated_delta_rule = _wrap_delta_rule(orig_torch_recur, pure_torch_recur)

    orig_fast_chunk = getattr(m, "chunk_gated_delta_rule", None)
    if callable(orig_fast_chunk) and callable(pure_torch_chunk):
        m.chunk_gated_delta_rule = _wrap_delta_rule(orig_fast_chunk, pure_torch_chunk)

    orig_fast_recur = getattr(m, "fused_recurrent_gated_delta_rule", None)
    if callable(orig_fast_recur) and callable(pure_torch_recur):
        m.fused_recurrent_gated_delta_rule = _wrap_delta_rule(orig_fast_recur, pure_torch_recur)

    m._treeweave_pytest_cpu_fallback_installed = True


def pytest_configure(config) -> None:  # noqa: ARG001
    _install_qwen3_5_cpu_kernel_fallbacks()

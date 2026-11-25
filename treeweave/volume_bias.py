"""Volume-Preserving Attention Compensation for Qwen3.5 Full-Attention Layers.

Supports two backends plus ablation-off mode:
1. `key_direction` (Default): Post-`k_norm` non-rotary key subspace offset
       `k'_{j, d_rot:} = k_{j, d_rot:} + ln(M_j) * w_{l, h_kv}`
   Preserves `head_dim = 256` exactly so FlashAttention-2 (`varlen` packed training),
   FlashAttention-3 (`fa3` on H100), PyTorch SDPA, and KV-cached decoding all work natively
   with ZERO kernel modifications and ZERO O(L^2) bias matrices.
2. `exact_logit_bias`: Exact additive scalar `gamma_{l, h} * ln(M_j)` broadcast `[B, H, 1, S_kv]`
   bias for PyTorch `sdpa` / `eager` attention (and zero-shot `gamma = 1` Q4 ablations).
3. `none`: Disabled (Ablation Q4 `w/o Volume Bias`).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

__all__ = [
    "TreeWeaveAttentionVolumeAdapter",
    "active_token_log_volumes",
    "active_varlen_bounds",
    "install_volume_bias_on_language_model",
    "varlen_bounds_context",
    "volume_bias_context",
]

_ACTIVE_LOG_VOLUMES: ContextVar[Optional[torch.Tensor]] = ContextVar(
    "treeweave_active_token_log_volumes", default=None
)
_ACTIVE_VARLEN_BOUNDS: ContextVar[Optional[Tuple[int, ...]]] = ContextVar(
    "treeweave_active_varlen_bounds", default=None
)


def active_token_log_volumes() -> Optional[torch.Tensor]:
    return _ACTIVE_LOG_VOLUMES.get()


def active_varlen_bounds() -> Optional[Tuple[int, ...]]:
    return _ACTIVE_VARLEN_BOUNDS.get()


@contextmanager
def volume_bias_context(token_log_volumes: Optional[torch.Tensor]):
    """Bind `(B, L)` per-token `ln(4^s * 2^t)` tensor for the duration of a language-model forward pass."""
    prev = _ACTIVE_LOG_VOLUMES.get()
    _ACTIVE_LOG_VOLUMES.set(token_log_volumes)
    try:
        yield
    finally:
        _ACTIVE_LOG_VOLUMES.set(prev)


@contextmanager
def varlen_bounds_context(bounds: Optional[Sequence[int]]):
    """Bind pre-fetched CPU segment boundaries `(0, L1, L1+L2, ...)` once per forward pass (0 GPU->CPU syncs per layer)."""
    prev = _ACTIVE_VARLEN_BOUNDS.get()
    _ACTIVE_VARLEN_BOUNDS.set(tuple(int(x) for x in bounds) if bounds is not None else None)
    try:
        yield
    finally:
        _ACTIVE_VARLEN_BOUNDS.set(prev)


class TreeWeaveAttentionVolumeAdapter(nn.Module):
    """Lightweight per-full-attention-layer volume compensation module."""

    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rotary_dim: int,
        mode: str = "key_direction",
        gamma_init: float = 1.0,
    ):
        super().__init__()
        if mode not in ("key_direction", "exact_logit_bias", "none"):
            raise ValueError(f"Unsupported volume_bias_mode: {mode!r}.")
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.non_rotary_dim = head_dim - rotary_dim
        if self.non_rotary_dim <= 0:
            raise ValueError(f"Expected rotary_dim ({rotary_dim}) < head_dim ({head_dim}).")
        self.mode = mode

        # Learned key-subspace vector `w_{l, h_kv} in R^{head_dim - rotary_dim}` (zero-init -> exact native at step 0)
        self.key_volume_vec = nn.Parameter(torch.zeros(num_key_value_heads, self.non_rotary_dim))
        # Per-query-head scalar `gamma_{l, h}` for `exact_logit_bias` mode
        self.logit_gamma = nn.Parameter(torch.full((num_attention_heads,), float(gamma_init)))
        self._cached_kv_log_volumes: Optional[torch.Tensor] = None

    @torch.no_grad()
    def reset_parameters(self, gamma_init: float = 1.0) -> None:
        nn.init.zeros_(self.key_volume_vec)
        self.logit_gamma.fill_(float(gamma_init))

    def configure_trainable(self, enabled: bool) -> None:
        """Enable gradients ONLY on the parameter active under `self.mode` (avoiding unused DDP parameters)."""
        self.key_volume_vec.requires_grad_(bool(enabled and self.mode == "key_direction"))
        self.logit_gamma.requires_grad_(bool(enabled and self.mode == "exact_logit_bias"))

    def apply_to_keys(self, key_states: torch.Tensor, log_volumes: torch.Tensor) -> torch.Tensor:
        """`key_states`: `(B, H_kv, S, D_head)`, `log_volumes`: `(B, S)`."""
        if self.mode == "none":
            return key_states
        lv = log_volumes.to(device=key_states.device, dtype=key_states.dtype)
        # `(B, 1, S, 1) * (1, H_kv, 1, D_non_rot)` -> `(B, H_kv, S, D_non_rot)`
        delta = lv[:, None, :, None] * self.key_volume_vec.to(dtype=key_states.dtype)[None, :, None, :]
        k_rot = key_states[..., : self.rotary_dim]
        k_pass = key_states[..., self.rotary_dim :] + delta
        return torch.cat((k_rot, k_pass), dim=-1)

    def augment_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        log_volumes_kv: torch.Tensor,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        """Add `gamma_{l, h} * ln(M_j)` as `[B, H_q, 1, S_kv]` additive bias for SDPA / eager."""
        lv = log_volumes_kv.to(device=query_states.device, dtype=query_states.dtype)
        gamma = self.logit_gamma.to(device=query_states.device, dtype=query_states.dtype)
        bias = gamma[None, :, None, None] * lv[:, None, None, :]  # (B, H_q, 1, S_kv)
        if attention_mask is None:
            # Must also preserve causal masking when `attention_mask` was None in SDPA/eager!
            q_len = query_states.shape[2]
            kv_len = lv.shape[1]
            if q_len > 1:
                causal = torch.zeros((q_len, kv_len), device=query_states.device, dtype=query_states.dtype)
                upper = torch.ones((q_len, kv_len), device=query_states.device, dtype=torch.bool).triu(
                    diagonal=1 + (kv_len - q_len)
                )
                causal = causal.masked_fill(upper, torch.finfo(query_states.dtype).min)
                return bias + causal[None, None, :, :]
            return bias
        if attention_mask.dtype == torch.bool:
            float_mask = torch.zeros(attention_mask.shape, device=query_states.device, dtype=query_states.dtype)
            float_mask = float_mask.masked_fill(~attention_mask, torch.finfo(query_states.dtype).min)
            return float_mask + bias
        return attention_mask + bias


def _make_patched_qwen3_5_attn_forward(attn_mod: Qwen3_5Attention) -> Callable:
    def forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ):
        adapter: Optional[TreeWeaveAttentionVolumeAdapter] = getattr(
            attn_mod, "treeweave_volume_bias", None
        )
        if "treeweave_token_log_volumes" in kwargs:
            log_vols = kwargs.pop("treeweave_token_log_volumes")
        else:
            log_vols = active_token_log_volumes()

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn_mod.head_dim)

        query_states, gate = torch.chunk(
            attn_mod.q_proj(hidden_states).view(*input_shape, -1, attn_mod.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = attn_mod.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = attn_mod.k_norm(attn_mod.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        use_adapter = adapter is not None and adapter.mode != "none"
        attn_impl = str(getattr(attn_mod.config, "_attn_implementation", "sdpa"))
        cu_q = kwargs.get("cu_seq_lens_q")
        is_varlen_sdpa = (
            past_key_values is None
            and "flash" not in attn_impl
            and cu_q is not None
            and isinstance(cu_q, torch.Tensor)
            and int(cu_q.numel()) > 2
            and query_states.shape[0] == 1
        )

        if use_adapter and log_vols is not None and log_vols.shape[:2] == input_shape[:2]:
            # If key_direction mode OR flash_attention_* varlen is active, inject into post-k_norm keys BEFORE KV cache!
            if adapter.mode == "key_direction" or "flash" in attn_impl:
                key_states = adapter.apply_to_keys(key_states, log_vols)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn_mod.layer_idx
            )

        if use_adapter and adapter.mode == "exact_logit_bias" and "flash" not in attn_impl and not is_varlen_sdpa:
            past_len_before = 0
            if past_key_values is not None:
                past_len_before = key_states.shape[2] - query_states.shape[2]
            if log_vols is not None and log_vols.shape[1] == key_states.shape[2]:
                adapter._cached_kv_log_volumes = log_vols
                attention_mask = adapter.augment_attention_mask(attention_mask, log_vols, query_states)
            elif log_vols is None and past_len_before == 0:
                adapter._cached_kv_log_volumes = None
            elif (
                past_key_values is not None
                and past_len_before > 0
                and adapter._cached_kv_log_volumes is not None
            ):
                new_toks = key_states.shape[2] - adapter._cached_kv_log_volumes.shape[1]
                if new_toks > 0 and adapter._cached_kv_log_volumes.shape[0] == input_shape[0]:
                    pad_zero = adapter._cached_kv_log_volumes.new_zeros((input_shape[0], new_toks))
                    kv_lv = torch.cat((adapter._cached_kv_log_volumes, pad_zero), dim=1)
                else:
                    kv_lv = adapter._cached_kv_log_volumes
                if kv_lv.shape[1] == key_states.shape[2] and kv_lv.shape[0] == input_shape[0]:
                    adapter._cached_kv_log_volumes = kv_lv
                    attention_mask = adapter.augment_attention_mask(attention_mask, kv_lv, query_states)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn_impl, eager_attention_forward
        )
        drop_p = 0.0 if not attn_mod.training else attn_mod.attention_dropout
        if is_varlen_sdpa:
            # Single-pass packed varlen SDPA: ALL 24 GatedDeltaNet layers + all 32 QKV/MLP/Norm/RoPE layers
            # run ONCE on the full `[1, L_packed, D]` tensor; only the inner attention op isolates packed segments!
            bounds = active_varlen_bounds()
            if bounds is None or len(bounds) != int(cu_q.numel()):
                bounds = tuple(int(x) for x in cu_q.detach().cpu().tolist())
            seg_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
            }
            use_exact_seg = (
                use_adapter
                and adapter.mode == "exact_logit_bias"
                and log_vols is not None
                and log_vols.shape[:2] == input_shape[:2]
            )
            out_parts: List[torch.Tensor] = []
            for s_i, e_i in zip(bounds[:-1], bounds[1:]):
                if e_i <= s_i:
                    continue
                q_s = query_states[:, :, s_i:e_i, :]
                k_s = key_states[:, :, s_i:e_i, :]
                v_s = value_states[:, :, s_i:e_i, :]
                mask_s = (
                    adapter.augment_attention_mask(None, log_vols[:, s_i:e_i], q_s)
                    if use_exact_seg
                    else None
                )
                seg_out, _ = attention_interface(
                    attn_mod,
                    q_s,
                    k_s,
                    v_s,
                    mask_s,
                    dropout=drop_p,
                    scaling=attn_mod.scaling,
                    **seg_kwargs,
                )
                out_parts.append(seg_out)
            attn_output = out_parts[0] if len(out_parts) == 1 else torch.cat(out_parts, dim=1)
            attn_weights = None
        else:
            attn_output, attn_weights = attention_interface(
                attn_mod,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=drop_p,
                scaling=attn_mod.scaling,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = attn_mod.o_proj(attn_output)
        return attn_output, attn_weights

    return forward


def _wrap_linear_attn_strip_volume_kwarg(linear_attn: nn.Module) -> None:
    """Prevent `treeweave_token_log_volumes` in `**kwargs` from reaching raw `causal_conv1d_fn(**kwargs)`."""
    if getattr(linear_attn, "_treeweave_kwarg_stripped", False):
        return
    orig_fwd = linear_attn.forward

    def _safe_linear_forward(*args, **kwargs):
        kwargs.pop("treeweave_token_log_volumes", None)
        return orig_fwd(*args, **kwargs)

    linear_attn.forward = _safe_linear_forward  # type: ignore[method-assign]
    linear_attn._treeweave_kwarg_stripped = True


def install_volume_bias_on_language_model(
    language_model: nn.Module,
    *,
    mode: str = "key_direction",
    gamma_init: float = 1.0,
) -> int:
    """Attach `TreeWeaveAttentionVolumeAdapter` + varlen-safe SDPA to every `Qwen3_5Attention` layer."""
    text_cfg = language_model.config
    rope_params = getattr(text_cfg, "rope_parameters", None) or {}
    partial_factor = float(rope_params.get("partial_rotary_factor", 0.25))
    head_dim = int(getattr(text_cfg, "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads))
    rotary_dim = int(head_dim * partial_factor)

    installed = 0
    for layer in language_model.layers:
        lin_attn = getattr(layer, "linear_attn", None)
        if lin_attn is not None:
            _wrap_linear_attn_strip_volume_kwarg(lin_attn)
        attn = getattr(layer, "self_attn", None)
        if not isinstance(attn, Qwen3_5Attention):
            continue
        if not hasattr(attn, "treeweave_volume_bias"):
            attn.treeweave_volume_bias = TreeWeaveAttentionVolumeAdapter(
                num_attention_heads=text_cfg.num_attention_heads,
                num_key_value_heads=text_cfg.num_key_value_heads,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                mode=mode,
                gamma_init=gamma_init,
            )
            attn.forward = _make_patched_qwen3_5_attn_forward(attn)  # type: ignore[method-assign]
        else:
            attn.treeweave_volume_bias.mode = mode
        installed += 1
    language_model._treeweave_varlen_sdpa_patched = installed > 0
    return installed

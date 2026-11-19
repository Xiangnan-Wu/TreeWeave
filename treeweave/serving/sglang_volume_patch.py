"""SGLang Language-Server Volume Compensation Patch (`TreeWeaveAttentionVolumeAdapter` + `RadixAttention`).

Injects `ln(4^s * 2^t) * w_{l, h_kv}` into the post-`k_norm`, post-RoPE non-rotary key channels
(`dims rotary_dim : head_dim`) right before `RadixAttention` writes `k, v` into SGLang's KV cache pool:
- Preserves `head_dim = 256` for FlashAttention-3 (`fa3`), FlashInfer, and Triton kernels.
- Zero decode-step overhead and 100% CUDA-graph compatible (cached visual keys already carry the offset).
- Supports Tensor Parallelism (`TP=1` and `TP=2`) head sharding seamlessly.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Optional, Set

import torch
from torch import nn

from ..volume_bias import TreeWeaveAttentionVolumeAdapter
from .sglang_route import apply_sglang_key_direction_volume_bias, derive_serving_geometry

__all__ = [
    "_SGLangVolumeBiasedRadixAttention",
    "active_sglang_flat_log_volumes",
    "finalize_sglang_volume_bias_loaded_names",
    "install_sglang_volume_bias_on_causal_lm",
    "load_sglang_volume_bias_tensor",
    "sglang_volume_bias_context",
]

_SGLANG_FLAT_LOG_VOLUMES: ContextVar[Optional[torch.Tensor]] = ContextVar(
    "treeweave_sglang_flat_log_volumes", default=None
)


def active_sglang_flat_log_volumes() -> Optional[torch.Tensor]:
    return _SGLANG_FLAT_LOG_VOLUMES.get()


@contextmanager
def sglang_volume_bias_context(flat_log_volumes: Optional[torch.Tensor]):
    prev = _SGLANG_FLAT_LOG_VOLUMES.get()
    _SGLANG_FLAT_LOG_VOLUMES.set(flat_log_volumes)
    try:
        yield
    finally:
        _SGLANG_FLAT_LOG_VOLUMES.set(prev)


class _SGLangVolumeBiasedRadixAttention(nn.Module):
    """Transparent wrapper around SGLang's `RadixAttention` applying post-`k_norm` non-rotary key offsets."""

    def __init__(self, orig_attn: nn.Module, adapter: TreeWeaveAttentionVolumeAdapter):
        super().__init__()
        self._orig_attn = orig_attn
        # Avoid registering `adapter` twice under `named_parameters()` (it is owned by `self_attn.treeweave_volume_bias`)
        object.__setattr__(self, "_adapter", adapter)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs):
        lv = active_sglang_flat_log_volumes()
        if lv is not None and self._adapter.mode != "none":
            k = apply_sglang_key_direction_volume_bias(
                k,
                lv,
                self._adapter.key_volume_vec,
                num_kv_heads_local=self._adapter.num_key_value_heads,
                head_dim=self._adapter.head_dim,
                rotary_dim=self._adapter.rotary_dim,
            )
        return self._orig_attn(q, k, v, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._orig_attn, name)


def _resolve_decoder_layers(causal_lm: nn.Module):
    if hasattr(causal_lm, "language_model"):
        lm = causal_lm.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers
    if hasattr(causal_lm, "model") and hasattr(causal_lm.model, "layers"):
        return causal_lm.model.layers
    if hasattr(causal_lm, "layers"):
        return causal_lm.layers
    raise ValueError(f"Could not locate decoder layers on {type(causal_lm).__name__}.")


def install_sglang_volume_bias_on_causal_lm(
    causal_lm: nn.Module,
    hf_config: Any,
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    dtype: Optional[torch.dtype] = None,
) -> Dict[int, TreeWeaveAttentionVolumeAdapter]:
    """Attach `TreeWeaveAttentionVolumeAdapter` & wrap `self_attn.attn` on all full-attention layers."""
    tw_cfg, _, _, _, _ = derive_serving_geometry(hf_config)
    text_cfg = hf_config["text_config"] if isinstance(hf_config, dict) else hf_config.text_config
    get_t = (lambda k, d=None: text_cfg.get(k, d)) if isinstance(text_cfg, dict) else (lambda k, d=None: getattr(text_cfg, k, d))

    num_layers = int(get_t("num_hidden_layers"))
    layer_types = get_t("layer_types")
    if layer_types is None:
        interval = int(get_t("full_attention_interval", 4) or 4)
        layer_types = [
            "linear_attention" if bool((i + 1) % interval) else "full_attention"
            for i in range(num_layers)
        ]

    num_q_heads = int(get_t("num_attention_heads"))
    num_kv_heads = int(get_t("num_key_value_heads", num_q_heads) or num_q_heads)
    head_dim = int(get_t("head_dim", int(get_t("hidden_size")) // num_q_heads))
    rope_params = get_t("rope_parameters") or {}
    partial_factor = float(rope_params.get("partial_rotary_factor", 0.25))
    rotary_dim = int(head_dim * partial_factor)

    if num_q_heads % tp_size != 0 or num_kv_heads % tp_size != 0:
        raise ValueError(
            f"TP size {tp_size} must divide num_attention_heads={num_q_heads} and num_key_value_heads={num_kv_heads}."
        )
    num_q_local = num_q_heads // tp_size
    num_kv_local = num_kv_heads // tp_size
    vol_mode = tw_cfg.volume_bias_mode if tw_cfg.use_volume_attn_bias else "none"
    if vol_mode not in ("key_direction", "none"):
        raise ValueError(
            f"SGLang volume compensation supports mode in ('key_direction', 'none'), got {vol_mode!r}."
        )

    layers = _resolve_decoder_layers(causal_lm)
    installed: Dict[int, TreeWeaveAttentionVolumeAdapter] = {}
    for layer_idx, layer in enumerate(layers):
        if layer_idx < len(layer_types) and layer_types[layer_idx] != "full_attention":
            continue
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None or not hasattr(self_attn, "attn"):
            continue

        if not hasattr(self_attn, "treeweave_volume_bias"):
            adapter = TreeWeaveAttentionVolumeAdapter(
                num_attention_heads=num_q_local,
                num_key_value_heads=num_kv_local,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                mode=vol_mode,
                gamma_init=tw_cfg.volume_gamma_init,
            )
            ref_param = next(self_attn.parameters(), None)
            target_dev = ref_param.device if ref_param is not None else None
            target_dt = dtype if dtype is not None else (ref_param.dtype if ref_param is not None else None)
            if target_dev is not None or target_dt is not None:
                adapter = adapter.to(device=target_dev, dtype=target_dt)
            self_attn.treeweave_volume_bias = adapter
            if not isinstance(self_attn.attn, _SGLangVolumeBiasedRadixAttention):
                self_attn.attn = _SGLangVolumeBiasedRadixAttention(self_attn.attn, adapter)
        else:
            adapter = self_attn.treeweave_volume_bias
            adapter.mode = vol_mode
        installed[layer_idx] = adapter

    return installed


@torch.no_grad()
def load_sglang_volume_bias_tensor(
    adapters_by_layer: Dict[int, TreeWeaveAttentionVolumeAdapter],
    name: str,
    tensor: torch.Tensor,
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    loaded_record: Optional[Set[str]] = None,
) -> bool:
    """Intercept `*.layers.{idx}.self_attn.treeweave_volume_bias.{param}` and shard across TP ranks."""
    marker = ".self_attn.treeweave_volume_bias."
    if marker not in name:
        return False
    prefix, param_name = name.split(marker, 1)
    layer_str = prefix.rsplit(".layers.", 1)[-1]
    layer_idx = int(layer_str)
    adapter = adapters_by_layer.get(layer_idx)
    if adapter is None:
        raise RuntimeError(
            f"Checkpoint contains volume-bias weight {name!r}, but layer {layer_idx} has no adapter."
        )
    if param_name == "key_volume_vec":
        total_kv = int(tensor.shape[0])
        local_kv = total_kv // tp_size
        sl = tensor[tp_rank * local_kv : (tp_rank + 1) * local_kv]
        adapter.key_volume_vec.data.copy_(
            sl.to(device=adapter.key_volume_vec.device, dtype=adapter.key_volume_vec.dtype)
        )
    elif param_name == "logit_gamma":
        total_q = int(tensor.shape[0])
        local_q = total_q // tp_size
        sl = tensor[tp_rank * local_q : (tp_rank + 1) * local_q]
        adapter.logit_gamma.data.copy_(
            sl.to(device=adapter.logit_gamma.device, dtype=adapter.logit_gamma.dtype)
        )
    else:
        raise RuntimeError(f"Unexpected volume-bias parameter in checkpoint: {name!r}.")

    if loaded_record is not None:
        loaded_record.add(f"{layer_idx}:{param_name}")
    return True


def finalize_sglang_volume_bias_loaded_names(
    model: nn.Module,
    adapters_by_layer: Dict[int, TreeWeaveAttentionVolumeAdapter],
    loaded_vol_keys: Set[str],
    *,
    use_volume_attn_bias: bool,
    volume_bias_mode: str = "key_direction",
) -> Set[str]:
    """Validate required `key_volume_vec` tensors and return exact `model.named_parameters()` names
    for SGLang's `DefaultModelLoader` uninitialized-parameter verification.
    """
    eff_mode = volume_bias_mode if use_volume_attn_bias else "none"
    if eff_mode == "key_direction" and adapters_by_layer:
        missing_layers = [
            idx
            for idx in sorted(adapters_by_layer)
            if f"{idx}:key_volume_vec" not in loaded_vol_keys
        ]
        if missing_layers:
            raise RuntimeError(
                f"Checkpoint did not supply `treeweave_volume_bias.key_volume_vec` "
                f"for full-attention layer(s): {missing_layers}."
            )
    marker = ".self_attn.treeweave_volume_bias."
    return {
        name
        for name, _ in model.named_parameters(remove_duplicate=False)
        if marker in name
    }

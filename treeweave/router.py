"""Hierarchical Spatiotemporal Router: Multi-Depth Cross-Attn + Windowed Temporal Self-Attn + 4 FP32 Heads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "ROUTER_SPATIAL_DEPTH",
    "GlobalCrossAttnRouterLayer",
    "LocalCrossAttnRouterLayer",
    "RouterOutputs",
    "SpatiotemporalTreeRouter",
    "WindowedTemporalSelfAttnLayer",
    "apply_cross_rope",
    "rotate_half",
]

ROUTER_SPATIAL_DEPTH = 6


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_cross_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    query_cos: torch.Tensor,
    query_sin: torch.Tensor,
    key_cos: torch.Tensor,
    key_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE with separate coordinate grids for queries and keys in FP32."""
    q_dtype, k_dtype = query.dtype, key.dtype
    qf, kf = query.float(), key.float()
    q_cos = query_cos.unsqueeze(-2).float()
    q_sin = query_sin.unsqueeze(-2).float()
    k_cos = key_cos.unsqueeze(-2).float()
    k_sin = key_sin.unsqueeze(-2).float()
    q_out = qf * q_cos + rotate_half(qf) * q_sin
    k_out = kf * k_cos + rotate_half(kf) * k_sin
    return q_out.to(q_dtype), k_out.to(k_dtype)


class LocalCrossAttnRouterLayer(nn.Module):
    """Each quadrant query `(I_1, J_1)` attends to the `4 x 4 = 16` ViT patches in its own block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 2, use_kv_norm: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6) if use_kv_norm else nn.Identity()
        self.norm_attn = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )
        self.norm_mlp = nn.LayerNorm(dim, eps=1e-6)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_rope: Tuple[torch.Tensor, torch.Tensor],
        key_rope: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        num_blocks = query.shape[0]
        kv = self.norm_kv(key_value)
        q = self.q_proj(query).view(num_blocks, self.num_heads, self.head_dim)
        k = self.k_proj(kv).view(num_blocks, 16, self.num_heads, self.head_dim)
        v = self.v_proj(kv).view(num_blocks, 16, self.num_heads, self.head_dim)
        q, k = apply_cross_rope(q, k, *query_rope, *key_rope)
        attn = F.scaled_dot_product_attention(
            q.unsqueeze(2), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), is_causal=False
        )
        attn = self.out_proj(attn.squeeze(2).reshape(num_blocks, -1))
        query = self.norm_attn(query + attn)
        return self.norm_mlp(query + self.mlp(query))


class GlobalCrossAttnRouterLayer(nn.Module):
    """Every quadrant query in a frame attends to all ViT patches in that same frame.

    Vectorized per-item across all `T` frames simultaneously (`(T, num_heads, Q_frame, head_dim)`),
    avoiding O(T) Python kernel-launch loops on 64~128 frame videos.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 2, use_kv_norm: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6) if use_kv_norm else nn.Identity()
        self.norm_attn = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )
        self.norm_mlp = nn.LayerNorm(dim, eps=1e-6)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_rope: Tuple[torch.Tensor, torch.Tensor],
        key_rope: Tuple[torch.Tensor, torch.Tensor],
        grid_thw_list: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        kv = self.norm_kv(key_value)
        q = self.q_proj(query).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(kv).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(kv).view(-1, self.num_heads, self.head_dim)
        q, k = apply_cross_rope(q, k, *query_rope, *key_rope)

        outputs: List[torch.Tensor] = []
        q_offset = 0
        k_offset = 0
        for grid in grid_thw_list:
            t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
            q_per_frame = (h // 4) * (w // 4)
            k_per_frame = h * w
            q_total = t * q_per_frame
            k_total = t * k_per_frame

            item_q = q[q_offset : q_offset + q_total].view(t, q_per_frame, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            item_k = k[k_offset : k_offset + k_total].view(t, k_per_frame, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            item_v = v[k_offset : k_offset + k_total].view(t, k_per_frame, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            item_out = F.scaled_dot_product_attention(item_q, item_k, item_v, is_causal=False)
            outputs.append(item_out.permute(0, 2, 1, 3).reshape(q_total, -1))
            q_offset += q_total
            k_offset += k_total

        attn = self.out_proj(outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0))
        query = self.norm_attn(query + attn)
        return self.norm_mlp(query + self.mlp(query))


class WindowedTemporalSelfAttnLayer(nn.Module):
    """Lightweight 4-slice windowed temporal self-attention across `(I_1, J_1)` trajectories when `T > 1`.

    Output projections are zero-initialized so step-0 preserves single-frame router behavior seamlessly.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def apply_transparent_init(self, std: float = 0.02) -> None:
        nn.init.normal_(self.qkv.weight, std=std)
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.mlp[0].weight, std=std)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def _apply_1d_rope(self, q: torch.Tensor, k: torch.Tensor, win_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(win_len, device=q.device, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq.to(q.device)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :].to(q.dtype)
        sin = emb.sin()[None, None, :, :].to(q.dtype)
        q_out = q * cos + rotate_half(q) * sin
        k_out = k * cos + rotate_half(k) * sin
        return q_out, k_out

    def forward_item(self, h_item: torch.Tensor) -> torch.Tensor:
        """`h_item`: `(T, num_blocks_per_frame, d)` -> `(T, num_blocks_per_frame, d)`."""
        t_len, b_len, d = h_item.shape
        if t_len <= 1:
            return h_item

        # Transpose to `(B, T, d)` and split into windows of size <= `window_size`
        x = h_item.transpose(0, 1).contiguous()  # (B, T, d)
        normed = self.norm1(x)
        chunks: List[torch.Tensor] = []
        w = self.window_size
        full_t = (t_len // w) * w
        if full_t > 0:
            win = normed[:, :full_t, :].reshape(-1, w, d)
            qkv = self.qkv(win).view(-1, w, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            q, k = self._apply_1d_rope(q, k, w)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            attn = self.out_proj(attn.transpose(1, 2).reshape(b_len, full_t, d))
            chunks.append(attn)
        if full_t < t_len:
            rem_len = t_len - full_t
            rem = normed[:, full_t:, :]
            qkv = self.qkv(rem).view(b_len, rem_len, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            q, k = self._apply_1d_rope(q, k, rem_len)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            attn = self.out_proj(attn.transpose(1, 2).reshape(b_len, rem_len, d))
            chunks.append(attn)

        attn_all = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=1)
        x = x + attn_all
        x = x + self.mlp(self.norm2(x))
        return x.transpose(0, 1).contiguous()


@dataclass
class RouterOutputs:
    """Per-item hierarchical routing probabilities in FP32."""

    p_s1: List[torch.Tensor]  # each: `(T, B_h, B_w)`
    p_s2: List[torch.Tensor]  # each: `(T, B_h // 2, B_w // 2)`
    p_t1: List[torch.Tensor]  # each: `(T // 2, B_h, B_w)`
    p_t2: List[torch.Tensor]  # each: `(T // 4, B_h, B_w)`


def _build_fp32_binary_head(hidden_size: int) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(hidden_size, eps=1e-6), nn.Linear(hidden_size, 2, bias=False))


def _binary_compress_prob(
    head: nn.Module,
    last_bias: torch.Tensor,
    calib_bias: torch.Tensor,
    budget_shift: float,
    features: torch.Tensor,
) -> torch.Tensor:
    """Compute `P(compress)` in FP32 from `features (..., d)`."""
    if features.numel() == 0:
        return torch.zeros(features.shape[:-1], dtype=torch.float32, device=features.device)
    orig_shape = features.shape[:-1]
    logits = head(features.float().reshape(-1, features.shape[-1])) + last_bias.float()
    total_shift = calib_bias.float() + float(budget_shift)
    logits = logits + torch.stack((torch.zeros_like(total_shift), total_shift)).unsqueeze(0)
    probs = torch.softmax(logits.float(), dim=-1)[:, 1]
    return probs.view(orig_shape)


class SpatiotemporalTreeRouter(nn.Module):
    """Multi-depth spatiotemporal router predicting `(p_s1, p_s2, p_t1, p_t2)` for TreeWeave."""

    _FP32_HEAD_NAMES = ("head_s1", "head_s2", "head_t1", "head_t2")

    def __init__(
        self,
        hidden_size: int,
        depth: int = ROUTER_SPATIAL_DEPTH,
        num_heads: int = 16,
        mlp_ratio: int = 2,
        temporal_window_size: int = 4,
        use_temporal_attn: bool = True,
    ):
        super().__init__()
        if depth != ROUTER_SPATIAL_DEPTH:
            raise ValueError(f"Router spatial depth must be {ROUTER_SPATIAL_DEPTH}, got {depth}.")
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")
        head_dim = hidden_size // num_heads
        if head_dim % 4 != 0:
            raise ValueError(f"Router head_dim ({head_dim}) must be divisible by 4 for 2D axial RoPE.")

        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_temporal_attn = use_temporal_attn

        self.query = nn.Parameter(torch.zeros(1, hidden_size))
        self.layers = nn.ModuleList(
            [
                (
                    LocalCrossAttnRouterLayer(hidden_size, num_heads, mlp_ratio, use_kv_norm=True)
                    if idx % 2 == 0
                    else GlobalCrossAttnRouterLayer(hidden_size, num_heads, mlp_ratio, use_kv_norm=True)
                )
                for idx in range(depth)
            ]
        )
        self.temporal_attn = WindowedTemporalSelfAttnLayer(
            hidden_size, num_heads=num_heads, window_size=temporal_window_size, mlp_ratio=mlp_ratio
        )

        # Macroblock (`2 x 2` quadrants -> 1 `s=2` node) & dyadic temporal pair aggregators
        self.macro_agg = nn.Sequential(
            nn.LayerNorm(4 * hidden_size, eps=1e-6),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.t1_agg = nn.Sequential(
            nn.LayerNorm(3 * hidden_size, eps=1e-6),
            nn.Linear(3 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.t2_agg = nn.Sequential(
            nn.LayerNorm(3 * hidden_size, eps=1e-6),
            nn.Linear(3 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Keep scoring heads & biases strictly in FP32 to prevent BF16 boundary flips
        self.head_s1 = _build_fp32_binary_head(hidden_size).float()
        self.head_s2 = _build_fp32_binary_head(hidden_size).float()
        self.head_t1 = _build_fp32_binary_head(hidden_size).float()
        self.head_t2 = _build_fp32_binary_head(hidden_size).float()

        self.bias_s1 = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.bias_s2 = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.bias_t1 = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.bias_t2 = nn.Parameter(torch.zeros(2, dtype=torch.float32))

        # Offline-calibrated scalar logit biases for `[s1, s2, t1, t2]` (held fixed in Stage 3 / Serving)
        self.register_buffer("router_calib_bias", torch.zeros(4, dtype=torch.float32), persistent=True)

        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, head_dim // 2, 2, dtype=torch.float32) / (head_dim // 2))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def apply_initial_weights(self, std: float = 0.02, initial_prob: float = 0.1) -> None:
        if not (0.0 < initial_prob < 1.0):
            raise ValueError(f"initial_prob must be in (0, 1), got {initial_prob}.")
        nn.init.zeros_(self.query)
        for layer in self.layers:
            for mod in layer.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.normal_(mod.weight, std=std)
                    nn.init.zeros_(mod.bias)
        self.temporal_attn.apply_transparent_init(std=std)
        for agg in (self.macro_agg, self.t1_agg, self.t2_agg):
            nn.init.normal_(agg[1].weight, std=std)
            nn.init.zeros_(agg[1].bias)
            nn.init.zeros_(agg[3].weight)
            nn.init.zeros_(agg[3].bias)

        logit = math.log(initial_prob) - math.log1p(-initial_prob)
        init_bias = torch.tensor([0.0, logit], dtype=torch.float32)
        for head, bias_param in (
            (self.head_s1, self.bias_s1),
            (self.head_s2, self.bias_s2),
            (self.head_t1, self.bias_t1),
            (self.head_t2, self.bias_t2),
        ):
            nn.init.ones_(head[0].weight)
            nn.init.zeros_(head[0].bias)
            nn.init.zeros_(head[1].weight)
            bias_param.copy_(init_bias)
        self.router_calib_bias.zero_()

    def enforce_fp32_heads(self) -> None:
        """Ensure classification heads, biases & RoPE frequencies stay FP32 even after `.to(bfloat16)`."""
        for name in self._FP32_HEAD_NAMES:
            getattr(self, name).float()
        for b_name in ("bias_s1", "bias_s2", "bias_t1", "bias_t2"):
            param = getattr(self, b_name)
            if param.dtype != torch.float32:
                param.data = param.data.float()
        if self.router_calib_bias.dtype != torch.float32:
            self.router_calib_bias = self.router_calib_bias.float()
        if self.inv_freq.dtype != torch.float32:
            self.inv_freq = self.inv_freq.float()
        if self.temporal_attn.inv_freq.dtype != torch.float32:
            self.temporal_attn.inv_freq = self.temporal_attn.inv_freq.float()

    @staticmethod
    def validate_grids(grid_thw_list: Sequence[Sequence[int]]) -> None:
        for grid in grid_thw_list:
            t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
            if t <= 0 or h <= 0 or w <= 0:
                raise ValueError(f"Grid dimensions must be positive, got {(t, h, w)}.")
            if h % 4 != 0 or w % 4 != 0:
                raise ValueError(
                    f"TreeWeave router requires ViT patch grid h, w divisible by 4, got {(t, h, w)}."
                )

    def _rope_2d(self, positions: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = (positions[..., None] * self.inv_freq.to(positions.device)).flatten(1)
        emb = torch.cat((emb, emb), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def init_state(
        self,
        grid_thw_list: Sequence[Sequence[int]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        self.validate_grids(grid_thw_list)
        q_pos_list: List[torch.Tensor] = []
        k_pos_list: List[torch.Tensor] = []
        for grid in grid_thw_list:
            t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
            b_rows = torch.arange(h // 4, dtype=torch.float32, device=device) * 4.0 + 1.5
            b_cols = torch.arange(w // 4, dtype=torch.float32, device=device) * 4.0 + 1.5
            q_frame = torch.stack(torch.meshgrid(b_rows, b_cols, indexing="ij"), dim=-1).reshape(-1, 2)
            p_rows = torch.arange(h, dtype=torch.float32, device=device)
            p_cols = torch.arange(w, dtype=torch.float32, device=device)
            k_frame = torch.stack(torch.meshgrid(p_rows, p_cols, indexing="ij"), dim=-1).reshape(-1, 2)
            q_pos_list.append(q_frame.repeat(t, 1))
            k_pos_list.append(k_frame.repeat(t, 1))

        q_pos = torch.cat(q_pos_list, dim=0)
        k_pos = torch.cat(k_pos_list, dim=0)
        loc_q = torch.tensor([[1.5, 1.5]], dtype=torch.float32, device=device)
        coords4 = torch.arange(4, dtype=torch.float32, device=device)
        loc_k = torch.stack(torch.meshgrid(coords4, coords4, indexing="ij"), dim=-1).reshape(16, 2)

        ctx = {
            "global_query_rope": self._rope_2d(q_pos, dtype),
            "global_key_rope": self._rope_2d(k_pos, dtype),
            "local_query_rope": self._rope_2d(loc_q, dtype),
            "local_key_rope": self._rope_2d(loc_k, dtype),
        }
        return self.query.expand(q_pos.shape[0], -1).to(dtype), ctx

    @staticmethod
    def _local_kv_from_rasters(raster_list: Sequence[torch.Tensor]) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for r in raster_list:
            t, h, w, d = r.shape
            parts.append(
                r.view(t, h // 4, 4, w // 4, 4, d)
                .permute(0, 1, 3, 2, 4, 5)
                .contiguous()
                .view(-1, 16, d)
            )
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def run_tap(
        self,
        query: torch.Tensor,
        context: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        raster_list: Sequence[torch.Tensor],
        grid_thw_list: Sequence[Sequence[int]],
        pair_start_index: int,
    ) -> torch.Tensor:
        """Run the `(LocalCrossAttn, GlobalCrossAttn)` pair at one sampled ViT depth."""
        local_layer = self.layers[pair_start_index]
        global_layer = self.layers[pair_start_index + 1]
        local_kv = self._local_kv_from_rasters(raster_list)
        global_kv = (
            raster_list[0].view(-1, self.hidden_size)
            if len(raster_list) == 1
            else torch.cat([r.view(-1, self.hidden_size) for r in raster_list], dim=0)
        )
        query = local_layer(query, local_kv, context["local_query_rope"], context["local_key_rope"])
        query = global_layer(
            query, global_kv, context["global_query_rope"], context["global_key_rope"], grid_thw_list
        )
        return query

    def score_queries(
        self,
        query: torch.Tensor,
        grid_thw_list: Sequence[Sequence[int]],
        budget_logit_shift: float = 0.0,
        max_spatial_level: int = 2,
        max_temporal_level: int = 2,
    ) -> RouterOutputs:
        """Apply optional windowed temporal self-attention and compute `(p_s1, p_s2, p_t1, p_t2)` per item."""
        p_s1_list: List[torch.Tensor] = []
        p_s2_list: List[torch.Tensor] = []
        p_t1_list: List[torch.Tensor] = []
        p_t2_list: List[torch.Tensor] = []

        offset = 0
        d = self.hidden_size
        for grid in grid_thw_list:
            t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
            bh, bw = h // 4, w // 4
            n_q = t * bh * bw
            h_item = query[offset : offset + n_q].view(t, bh * bw, d)
            offset += n_q

            if self.use_temporal_attn and t > 1 and max_temporal_level >= 1:
                h_item = self.temporal_attn.forward_item(h_item)

            h_grid = h_item.view(t, bh, bw, d)

            # 1) s1 probability per quadrant `(T, bh, bw)`
            if max_spatial_level >= 1:
                p_s1 = _binary_compress_prob(
                    self.head_s1, self.bias_s1, self.router_calib_bias[0], budget_logit_shift, h_grid
                )
            else:
                p_s1 = torch.zeros((t, bh, bw), dtype=torch.float32, device=query.device)
            p_s1_list.append(p_s1)

            # 2) s2 probability per 2x2 macroblock `(T, bh // 2, bw // 2)`
            mh, mw = bh // 2, bw // 2
            if max_spatial_level >= 2 and mh > 0 and mw > 0:
                quads = (
                    h_grid[:, : 2 * mh, : 2 * mw, :]
                    .view(t, mh, 2, mw, 2, d)
                    .permute(0, 1, 3, 2, 4, 5)
                    .contiguous()
                    .view(t, mh, mw, 4, d)
                )
                h_macro = quads.mean(dim=-2) + self.macro_agg(quads.flatten(-2))
                p_s2 = _binary_compress_prob(
                    self.head_s2, self.bias_s2, self.router_calib_bias[1], budget_logit_shift, h_macro
                )
            else:
                p_s2 = torch.zeros((t, mh, mw), dtype=torch.float32, device=query.device)
            p_s2_list.append(p_s2)

            # 3) t1 probability per dyadic slice pair `(T // 2, bh, bw)`
            t1_len = t // 2
            if max_temporal_level >= 1 and t1_len > 0:
                h_even = h_grid[0 : 2 * t1_len : 2]
                h_odd = h_grid[1 : 2 * t1_len : 2]
                h_pair = 0.5 * (h_even + h_odd) + self.t1_agg(
                    torch.cat((h_even, h_odd, h_odd - h_even), dim=-1)
                )
                p_t1 = _binary_compress_prob(
                    self.head_t1, self.bias_t1, self.router_calib_bias[2], budget_logit_shift, h_pair
                )
            else:
                p_t1 = torch.zeros((t1_len, bh, bw), dtype=torch.float32, device=query.device)
            p_t1_list.append(p_t1)

            # 4) t2 probability per 4-slice window `(T // 4, bh, bw)`
            t2_len = t // 4
            if max_temporal_level >= 2 and t2_len > 0:
                # Re-use dyadic pair representations when available
                h_even = h_grid[0 : 4 * t2_len : 2]
                h_odd = h_grid[1 : 4 * t2_len : 2]
                h_pair_4 = 0.5 * (h_even + h_odd)
                h_left = h_pair_4[0 : 2 * t2_len : 2]
                h_right = h_pair_4[1 : 2 * t2_len : 2]
                h_quad_t = 0.5 * (h_left + h_right) + self.t2_agg(
                    torch.cat((h_left, h_right, h_right - h_left), dim=-1)
                )
                p_t2 = _binary_compress_prob(
                    self.head_t2, self.bias_t2, self.router_calib_bias[3], budget_logit_shift, h_quad_t
                )
            else:
                p_t2 = torch.zeros((t2_len, bh, bw), dtype=torch.float32, device=query.device)
            p_t2_list.append(p_t2)

        return RouterOutputs(p_s1=p_s1_list, p_s2=p_s2_list, p_t1=p_t1_list, p_t2=p_t2_list)

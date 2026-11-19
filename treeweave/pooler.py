"""Cascaded Spatial & Difference-Aware Temporal Gated Poolers in ViT Latent Space."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn

from . import NUM_GRANULARITIES, granularity_index

__all__ = [
    "DiffAwareGatedTemporalPooler",
    "GatedSpatialPooler",
    "TreeWeavePyramid",
    "patchify_to_raster",
    "raster_to_merger_blocks",
    "raster_to_merger_input",
]


def patchify_to_raster(
    hidden_states: torch.Tensor,
    grid_thw_list: Sequence[Sequence[int]],
    fine: int = 2,
) -> List[torch.Tensor]:
    """Convert `(total_patches, d)` in `fine x fine` block-major order into per-item `(T, H, W, d)` raster tensors."""
    d = hidden_states.shape[-1]
    out: List[torch.Tensor] = []
    offset = 0
    for grid in grid_thw_list:
        t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
        count = t * h * w
        chunk = hidden_states[offset : offset + count]
        if chunk.shape[0] != count:
            raise ValueError(
                f"Expected {count} ViT patches for grid {(t, h, w)}, got {chunk.shape[0]}."
            )
        raster = (
            chunk.view(t, h // fine, w // fine, fine, fine, d)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(t, h, w, d)
        )
        out.append(raster)
        offset += count
    if offset != hidden_states.shape[0]:
        raise ValueError(
            f"grid_thw_list accounts for {offset} patches, but hidden_states has {hidden_states.shape[0]}."
        )
    return out


def raster_to_merger_blocks(x_raster: torch.Tensor, fine: int = 2) -> torch.Tensor:
    """View `(T_t, H_s, W_s, d)` raster features as `(T_t * (H_s//fine) * (W_s//fine), fine*fine, d)` merger blocks."""
    t_len, h_len, w_len, d = x_raster.shape
    h_aligned = (h_len // fine) * fine
    w_aligned = (w_len // fine) * fine
    if h_aligned == 0 or w_aligned == 0:
        return x_raster.new_zeros((0, fine * fine, d))
    cropped = x_raster[:, :h_aligned, :w_aligned, :]
    return (
        cropped.view(t_len, h_aligned // fine, fine, w_aligned // fine, fine, d)
        .permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, fine * fine, d)
    )


def raster_to_merger_input(x_raster: torch.Tensor, fine: int = 2) -> torch.Tensor:
    """Flatten `(T_t, H_s, W_s, d)` raster features into `(N_merged * fine^2, d)` for `Qwen3_5VisionPatchMerger`."""
    return raster_to_merger_blocks(x_raster, fine=fine).view(-1, x_raster.shape[-1])


class GatedSpatialPooler(nn.Module):
    """Channel-wise softmax-gated `pool x pool` spatial pooler with transparent mean-pooling initialization.

    Matches VisionWeave's `VisionWeaveCompressionProjector` equations:
        u_k = LN(x_k)
        g_k = f_glb([u_1; ...; u_4]) + b_k
        alpha_k = softmax_{k=1..4} f_gate([u_k; g_k])
        out = sum_k alpha_k * (u_k + f_val([u_k; g_k]))
    """

    def __init__(self, hidden_size: int, pool: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.pool = pool
        self.spatial = pool * pool
        h, spatial = hidden_size, self.spatial
        self.norm_resid = nn.LayerNorm(h, eps=1e-6)
        self.global_map = nn.Sequential(nn.Linear(spatial * h, h), nn.GELU(), nn.Linear(h, h))
        self.position_bias = nn.Parameter(torch.zeros(1, spatial, h))
        self.gate_mlp = nn.Sequential(nn.Linear(2 * h, 4 * h), nn.GELU(), nn.Linear(4 * h, h))
        self.trans_mlp = nn.Sequential(nn.Linear(2 * h, 4 * h), nn.GELU(), nn.Linear(4 * h, h))

    @torch.no_grad()
    def apply_transparent_init(self) -> None:
        """Zero final projections of `gate_mlp` and `trans_mlp` so step-0 equals normalized mean pooling."""
        for mlp in (self.gate_mlp, self.trans_mlp):
            nn.init.xavier_uniform_(mlp[0].weight)
            nn.init.zeros_(mlp[0].bias)
            nn.init.zeros_(mlp[2].weight)
            nn.init.zeros_(mlp[2].bias)
        for layer in (self.global_map[0], self.global_map[2]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.position_bias)

    def forward_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        """`(N, pool^2, d) -> (N, d)`."""
        if blocks.shape[0] == 0:
            return blocks.new_zeros((0, self.hidden_size))
        resid = self.norm_resid(blocks)
        global_feat = self.global_map(resid.flatten(1)).unsqueeze(1) + self.position_bias
        paired = torch.cat((resid, global_feat.expand_as(resid)), dim=-1)
        gate = torch.softmax(self.gate_mlp(paired), dim=1)
        trans = resid + self.trans_mlp(paired)
        return (gate * trans).sum(dim=1)

    def forward(self, x_raster: torch.Tensor) -> torch.Tensor:
        """Pool `(T, H, W, d)` -> `(T, H // pool, W // pool, d)` over aligned `pool x pool` neighborhoods."""
        t_len, h_len, w_len, d = x_raster.shape
        out_h, out_w = h_len // self.pool, w_len // self.pool
        if out_h == 0 or out_w == 0:
            return x_raster.new_zeros((t_len, 0, 0, d))
        blocks = raster_to_merger_blocks(x_raster, fine=self.pool)
        pooled = self.forward_blocks(blocks)
        return pooled.view(t_len, out_h, out_w, d)


class DiffAwareGatedTemporalPooler(nn.Module):
    """Difference-aware gated dyadic temporal pooler over `(x_{tau_1}, x_{tau_2})`.

    Equations:
        u_1 = LN(x_1), u_2 = LN(x_2), du = u_2 - u_1
        g_m = f_glb([u_1; u_2; du]) + b_m,  m in {1, 2}
        beta_m = softmax_{m in {1,2}} f_gate([u_m; g_m])
        out = sum_{m=1}^2 beta_m * (u_m + f_val([u_m; g_m]))
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        h = hidden_size
        self.norm_resid = nn.LayerNorm(h, eps=1e-6)
        self.global_map = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, h))
        self.temporal_bias = nn.Parameter(torch.zeros(1, 2, h))
        self.gate_mlp = nn.Sequential(nn.Linear(2 * h, 2 * h), nn.GELU(), nn.Linear(2 * h, h))
        self.trans_mlp = nn.Sequential(nn.Linear(2 * h, 2 * h), nn.GELU(), nn.Linear(2 * h, h))

    @torch.no_grad()
    def apply_transparent_init(self) -> None:
        """Zero final projections so step-0 reduces strictly to `0.5 * (LN(x_1) + LN(x_2))`."""
        for mlp in (self.gate_mlp, self.trans_mlp):
            nn.init.xavier_uniform_(mlp[0].weight)
            nn.init.zeros_(mlp[0].bias)
            nn.init.zeros_(mlp[2].weight)
            nn.init.zeros_(mlp[2].bias)
        for layer in (self.global_map[0], self.global_map[2]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.temporal_bias)

    def forward_pairs(self, pairs: torch.Tensor) -> torch.Tensor:
        """`(N, 2, d) -> (N, d)`."""
        if pairs.shape[0] == 0:
            return pairs.new_zeros((0, self.hidden_size))
        u = self.norm_resid(pairs)
        u1, u2 = u[:, 0], u[:, 1]
        diff = u2 - u1
        g_shared = self.global_map(torch.cat((u1, u2, diff), dim=-1)).unsqueeze(1)
        g = g_shared + self.temporal_bias
        paired = torch.cat((u, g), dim=-1)
        beta = torch.softmax(self.gate_mlp(paired), dim=1)
        trans = u + self.trans_mlp(paired)
        return (beta * trans).sum(dim=1)

    def forward(self, x_raster: torch.Tensor) -> torch.Tensor:
        """Pool `(T, H_s, W_s, d)` along dyadic time pairs -> `(T // 2, H_s, W_s, d)`."""
        t_len, h_len, w_len, d = x_raster.shape
        out_t = t_len // 2
        if out_t == 0 or h_len == 0 or w_len == 0:
            return x_raster.new_zeros((0, h_len, w_len, d))
        cropped = x_raster[: 2 * out_t]
        pairs = (
            cropped.view(out_t, 2, h_len, w_len, d)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
            .view(-1, 2, d)
        )
        pooled = self.forward_pairs(pairs)
        return pooled.view(out_t, h_len, w_len, d)


class TreeWeavePyramid(nn.Module):
    """Builds the 9-granularity `(s, t) in {0,1,2} x {0,1,2}` pyramid in ViT space and projects through shared Merger."""

    def __init__(self, vision_hidden_size: int, llm_hidden_size: int, fine: int = 2):
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.fine = fine
        self.spatial_pooler_1 = GatedSpatialPooler(vision_hidden_size, pool=2)
        self.spatial_pooler_2 = GatedSpatialPooler(vision_hidden_size, pool=2)
        self.temporal_pooler_1 = DiffAwareGatedTemporalPooler(vision_hidden_size)
        self.temporal_pooler_2 = DiffAwareGatedTemporalPooler(vision_hidden_size)
        self.scale_embeddings = nn.Parameter(torch.zeros(NUM_GRANULARITIES, llm_hidden_size))

    @torch.no_grad()
    def apply_transparent_init(self) -> None:
        self.spatial_pooler_1.apply_transparent_init()
        self.spatial_pooler_2.apply_transparent_init()
        self.temporal_pooler_1.apply_transparent_init()
        self.temporal_pooler_2.apply_transparent_init()
        nn.init.zeros_(self.scale_embeddings)

    def build_item_pyramid(
        self,
        x_raster: torch.Tensor,
        max_spatial_level: int = 2,
        max_temporal_level: int = 2,
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """Construct active `(s, t)` ViT feature grids for one media item `(T, H, W, d)`.

        Each `X_{s, t}` has shape `(T // 2^t, H_s, W_s, d)` where post-merger grid is `(H_s // 2, W_s // 2)`.
        """
        pyramid: Dict[Tuple[int, int], torch.Tensor] = {(0, 0): x_raster}
        if max_spatial_level >= 1:
            pyramid[(1, 0)] = self.spatial_pooler_1(pyramid[(0, 0)])
        if max_spatial_level >= 2 and (1, 0) in pyramid:
            # Only pool full 2x2 macroblocks (post-merger requires 2x2 of X_{2,0}, i.e. 4x4 of X_{1,0})
            x10 = pyramid[(1, 0)]
            h1_macro = (x10.shape[1] // 4) * 4
            w1_macro = (x10.shape[2] // 4) * 4
            if h1_macro > 0 and w1_macro > 0:
                pyramid[(2, 0)] = self.spatial_pooler_2(x10[:, :h1_macro, :w1_macro, :])

        t_len = x_raster.shape[0]
        if max_temporal_level >= 1 and t_len >= 2:
            for s in range(max_spatial_level + 1):
                if (s, 0) in pyramid and pyramid[(s, 0)].shape[1] >= self.fine and pyramid[(s, 0)].shape[2] >= self.fine:
                    pyramid[(s, 1)] = self.temporal_pooler_1(pyramid[(s, 0)])
        if max_temporal_level >= 2 and t_len >= 4:
            for s in range(max_spatial_level + 1):
                if (s, 1) in pyramid and pyramid[(s, 1)].shape[0] >= 2:
                    pyramid[(s, 2)] = self.temporal_pooler_2(pyramid[(s, 1)])
        return pyramid

    def project_all_scales(
        self,
        pyramids: Sequence[Dict[Tuple[int, int], torch.Tensor]],
        merger: nn.Module,
    ) -> List[Dict[Tuple[int, int], torch.Tensor]]:
        """Batch-project all `(s, t)` ViT pyramids through the shared Merger in a SINGLE kernel launch."""
        flat_blocks: List[torch.Tensor] = []
        meta: List[Tuple[int, Tuple[int, int], Tuple[int, int, int], int]] = []
        for item_idx, pyr in enumerate(pyramids):
            for (s, t), x_st in pyr.items():
                t_st, h_st, w_st, _ = x_st.shape
                out_h, out_w = h_st // self.fine, w_st // self.fine
                if t_st == 0 or out_h == 0 or out_w == 0:
                    continue
                blocks = raster_to_merger_blocks(x_st, fine=self.fine)
                flat_blocks.append(blocks)
                meta.append((item_idx, (s, t), (t_st, out_h, out_w), blocks.shape[0]))

        projected_items: List[Dict[Tuple[int, int], torch.Tensor]] = [{} for _ in pyramids]
        if not flat_blocks:
            return projected_items

        cat_blocks = torch.cat(flat_blocks, dim=0)
        merged_all = merger(cat_blocks.view(-1, cat_blocks.shape[-1]))
        scale_emb = self.scale_embeddings.to(dtype=merged_all.dtype, device=merged_all.device)

        cursor = 0
        for item_idx, (s, t), shape_3d, token_count in meta:
            chunk = merged_all[cursor : cursor + token_count]
            # Note: (0, 0) scale_embedding stays zero at init; for complete native equivalence at t=1.0,
            # scale_embeddings[0] can also be trained or kept zero in Stage 1/2.
            g_idx = granularity_index(s, t)
            if (s, t) != (0, 0):
                chunk = chunk + scale_emb[g_idx]
            projected_items[item_idx][(s, t)] = chunk.view(*shape_3d, -1)
            cursor += token_count
        return projected_items

    def project_selected_nodes(
        self,
        pyramids: Sequence[Dict[Tuple[int, int], torch.Tensor]],
        layouts: Sequence["TreeRouteLayout"],  # noqa: F821
        merger: nn.Module,
    ) -> List[torch.Tensor]:
        """Gather ONLY routed `(s, t)` pre-merger blocks and run shared `merger` ONCE on surviving tokens."""
        selected_blocks: List[torch.Tensor] = []
        selected_scale_ids: List[torch.Tensor] = []
        item_counts: List[int] = []

        for pyr, layout in zip(pyramids, layouts):
            n_out = layout.output_length
            item_counts.append(n_out)
            if n_out == 0:
                continue
            device = pyr[(0, 0)].device
            d = pyr[(0, 0)].shape[-1]
            item_blocks = pyr[(0, 0)].new_empty((n_out, self.fine * self.fine, d))

            scale_ids_np = layout.scale_ids
            scale_ids_t = torch.from_numpy(scale_ids_np).to(device=device, dtype=torch.long)
            selected_scale_ids.append(scale_ids_t)

            # Group by active (s, t) — at most 9 small tensor gathers per media item
            for s in (0, 1, 2):
                for t in (0, 1, 2):
                    g_id = granularity_index(s, t)
                    mask_np = scale_ids_np == g_id
                    if not mask_np.any():
                        continue
                    x_st = pyr[(s, t)]
                    blocks_st = raster_to_merger_blocks(x_st, fine=self.fine)
                    out_h = x_st.shape[1] // self.fine
                    out_w = x_st.shape[2] // self.fine

                    tau_t = layout.tau_start[mask_np] >> t
                    i_s = layout.cell_row_start[mask_np] >> s
                    j_s = layout.cell_col_start[mask_np] >> s
                    flat_idx = torch.from_numpy(
                        tau_t * (out_h * out_w) + i_s * out_w + j_s
                    ).to(device=device, dtype=torch.long)
                    pos_idx = torch.from_numpy(mask_np.nonzero()[0]).to(device=device, dtype=torch.long)
                    item_blocks.index_copy_(0, pos_idx, blocks_st.index_select(0, flat_idx))

            selected_blocks.append(item_blocks)

        if not selected_blocks:
            d_out = self.llm_hidden_size
            dev = pyramids[0][(0, 0)].device if pyramids else torch.device("cpu")
            return [torch.zeros((0, d_out), device=dev) for _ in layouts]

        cat_blocks = torch.cat(selected_blocks, dim=0)
        cat_scales = torch.cat(selected_scale_ids, dim=0)
        merged = merger(cat_blocks.view(-1, cat_blocks.shape[-1]))

        scale_emb = self.scale_embeddings.to(dtype=merged.dtype, device=merged.device)
        # Keep (0, 0) strictly native (zero offset) so threshold -> inf reproduces native model bit-for-bit
        non_native = cat_scales != 0
        if non_native.any():
            merged = merged.clone()
            merged[non_native] = merged[non_native] + scale_emb[cat_scales[non_native]]

        return list(torch.split(merged, item_counts, dim=0))

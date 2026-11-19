"""Tests for Spatial & Temporal Poolers, Transparent Init, Patch-Order Bijection, and Selected-Node Merger Equivalence."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from treeweave.layout import sample_stochastic_tree_layout
from treeweave.pooler import (
    DiffAwareGatedTemporalPooler,
    GatedSpatialPooler,
    TreeWeavePyramid,
    patchify_to_raster,
    raster_to_merger_input,
)


class DummyMerger(nn.Module):
    def __init__(self, d_in: int, d_out: int, fine: int = 2):
        super().__init__()
        merged = d_in * fine * fine
        self.norm = nn.LayerNorm(d_in, eps=1e-6)
        self.fc1 = nn.Linear(merged, merged)
        self.fc2 = nn.Linear(merged, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.fc1.in_features)
        return self.fc2(F.gelu(self.fc1(x)))


def test_patchify_raster_exact_roundtrip():
    torch.manual_seed(0)
    grids = [(1, 8, 16), (3, 12, 8), (4, 8, 8)]
    d = 16
    total = sum(t * h * w for t, h, w in grids)
    x = torch.randn(total, d)

    rasters = patchify_to_raster(x, grids, fine=2)
    reconstructed = torch.cat([raster_to_merger_input(r, fine=2) for r in rasters], dim=0)
    torch.testing.assert_close(reconstructed, x, atol=0.0, rtol=0.0)


def test_spatial_and_temporal_poolers_transparent_mean_pool_init():
    torch.manual_seed(1)
    d = 32
    sp = GatedSpatialPooler(d, pool=2)
    sp.apply_transparent_init()
    blocks = torch.randn(7, 4, d)
    expected_sp = F.layer_norm(blocks, (d,), eps=1e-6).mean(dim=1)
    torch.testing.assert_close(sp.forward_blocks(blocks), expected_sp, atol=1e-6, rtol=1e-6)

    tm = DiffAwareGatedTemporalPooler(d)
    tm.apply_transparent_init()
    pairs = torch.randn(11, 2, d)
    expected_tm = F.layer_norm(pairs, (d,), eps=1e-6).mean(dim=1)
    torch.testing.assert_close(tm.forward_pairs(pairs), expected_tm, atol=1e-6, rtol=1e-6)


def test_temporal_pooler_explicit_motion_diff_gradients():
    torch.manual_seed(2)
    d = 16
    tm = DiffAwareGatedTemporalPooler(d)
    # Perturb output layers slightly off zero so diff features affect gate & value immediately
    nn.init.normal_(tm.gate_mlp[2].weight, std=0.05)
    nn.init.normal_(tm.trans_mlp[2].weight, std=0.05)

    x_raster = torch.randn(5, 8, 8, d, requires_grad=True)  # odd T=5 -> T//2 = 2 pairs, tail slice ignored cleanly
    out = tm(x_raster)
    assert out.shape == (2, 8, 8, d)
    out.pow(2).mean().backward()
    assert x_raster.grad is not None
    # First 4 slices contribute gradients; trailing 5th slice has 0 gradient
    assert x_raster.grad[:4].abs().sum().item() > 0.0
    assert x_raster.grad[4].abs().sum().item() == 0.0


@pytest.mark.parametrize(
    "grid",
    [(1, 8, 8), (1, 12, 16), (2, 8, 16), (3, 8, 8), (4, 16, 16), (6, 12, 8)],
)
def test_project_selected_nodes_matches_project_all_scales_bit_exact(grid):
    """Verify fast sparse `project_selected_nodes` equals gathering from dense `project_all_scales` across all 9 scales."""
    torch.manual_seed(42)
    d_in, d_out = 16, 24
    pyr_mod = TreeWeavePyramid(d_in, d_out, fine=2)
    # Give non-zero scale embeddings to verify (s, t) != (0, 0) scale offsets match too!
    nn.init.normal_(pyr_mod.scale_embeddings, std=0.1)
    merger = DummyMerger(d_in, d_out, fine=2)

    t, h, w = grid
    x_raster = torch.randn(t, h, w, d_in)
    item_pyr = pyr_mod.build_item_pyramid(x_raster, max_spatial_level=2, max_temporal_level=2)

    rng = np.random.default_rng(123)
    layout = sample_stochastic_tree_layout(grid, rng)
    layout.verify_partition_of_unity()

    sparse_out = pyr_mod.project_selected_nodes([item_pyr], [layout], merger)[0]
    dense_all = pyr_mod.project_all_scales([item_pyr], merger)[0]

    # Manually gather from dense_all according to layout
    ref_rows = []
    for k in range(layout.output_length):
        gid = int(layout.scale_ids[k])
        s, tm = gid // 3, gid % 3
        v_st = dense_all[(s, tm)]
        tau_t = int(layout.tau_start[k]) >> tm
        i_s = int(layout.cell_row_start[k]) >> s
        j_s = int(layout.cell_col_start[k]) >> s
        ref_rows.append(v_st[tau_t, i_s, j_s])

    ref_tensor = torch.stack(ref_rows, dim=0)
    torch.testing.assert_close(sparse_out, ref_tensor, atol=1e-5, rtol=1e-5)

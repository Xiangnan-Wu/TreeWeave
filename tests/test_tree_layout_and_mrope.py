"""Tests for Partition-of-Unity Routing, Causal Completion Order, Half-Integer MRoPE, 0-Token Slices, & Ablation Modes."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from treeweave import POSITION_SCALE, granularity_index
from treeweave.layout import (
    build_routed_sequence_batch,
    hard_tree_route_layout,
    sample_stochastic_tree_layout,
)
from treeweave.serving.sglang_route import (
    TreeRouteItem,
    TreeRoutePayload,
    TreeVisualRun,
    tree_routed_mrope_positions,
)


@pytest.mark.parametrize(
    "grid",
    [
        (1, 4, 4),      # minimal 64px single-quadrant image (no s=2 macroblocks!)
        (1, 8, 8),      # 128px single image (1 s=2 macroblock)
        (1, 12, 16),    # odd quadrant height (bh=3, mh=1 -> mixed boundary quadrant + macroblock!)
        (2, 8, 8),      # 2-slice video (t<=1)
        (3, 8, 16),     # odd 3-slice video (pair [0,1] + tail slice 2)
        (4, 16, 16),    # full 4-slice video (all 9 (s, t) scales active!)
        (5, 12, 12),    # odd T=5 AND odd bh=bw=3 simultaneously!
        (8, 16, 8),     # 2 windows of 4 slices
    ],
)
@pytest.mark.parametrize("seed", ["none", "all", 0, 1, 2, 3, 4])
def test_partition_of_unity_and_causal_order_across_all_grids(grid, seed):
    t, h, w = grid
    bh, bw = h // 4, w // 4
    mh, mw = bh // 2, bw // 2

    if seed == "none":
        p_s1 = np.zeros((t, bh, bw), dtype=np.float32)
        p_s2 = np.zeros((t, mh, mw), dtype=np.float32)
        p_t1 = np.zeros((t // 2, bh, bw), dtype=np.float32)
        p_t2 = np.zeros((t // 4, bh, bw), dtype=np.float32)
    elif seed == "all":
        p_s1 = np.ones((t, bh, bw), dtype=np.float32)
        p_s2 = np.ones((t, mh, mw), dtype=np.float32)
        p_t1 = np.ones((t // 2, bh, bw), dtype=np.float32)
        p_t2 = np.ones((t // 4, bh, bw), dtype=np.float32)
    else:
        rng = np.random.default_rng(seed)
        layout_stoch = sample_stochastic_tree_layout(grid, rng)
        layout_stoch.verify_partition_of_unity()
        p_s1 = rng.random((t, bh, bw), dtype=np.float32)
        p_s2 = rng.random((t, mh, mw), dtype=np.float32)
        p_t1 = rng.random((t // 2, bh, bw), dtype=np.float32)
        p_t2 = rng.random((t // 4, bh, bw), dtype=np.float32)

    layout = hard_tree_route_layout(grid, p_s1, p_s2, p_t1, p_t2)
    layout.verify_partition_of_unity()

    if seed == "none":
        assert layout.output_length == layout.native_tokens
        assert np.all(layout.scale_ids == 0)
    elif seed == "all" and h % 8 == 0 and w % 8 == 0 and t == 4:
        # 64 -> 1 maximum spatiotemporal compression (98.4375% savings!)
        assert layout.output_length * 64 == layout.native_tokens
        assert np.all(layout.scale_ids == granularity_index(2, 2))
        # Slices 0, 1, 2 emit ZERO tokens; all tokens emit at completion slice 3!
        assert layout.frame_lengths[:3] == (0, 0, 0)
        assert layout.frame_lengths[3] == layout.output_length


def test_orthogonal_tree_vs_rigid_octree_static_fine_text():
    """Verify Q2 core claim: Orthogonal Tree folds (s=0, t=2) static fine text (75% savings), whereas Rigid Octree cannot."""
    grid = (4, 8, 8)  # 4 slices, 4x4=16 native tokens/slice = 64 native tokens
    bh, bw = 2, 2
    mh, mw = 1, 1
    # Dense small text everywhere (p_s1 = 0 -> s=0), completely static across 4 frames (p_t1 = p_t2 = 1 -> t=2)
    p_s1 = np.zeros((4, bh, bw), dtype=np.float32)
    p_s2 = np.zeros((4, mh, mw), dtype=np.float32)
    p_t1 = np.ones((2, bh, bw), dtype=np.float32)
    p_t2 = np.ones((1, bh, bw), dtype=np.float32)

    ortho = hard_tree_route_layout(grid, p_s1, p_s2, p_t1, p_t2, allow_fine_temporal_merge=True)
    rigid = hard_tree_route_layout(grid, p_s1, p_s2, p_t1, p_t2, allow_fine_temporal_merge=False)

    ortho.verify_partition_of_unity()
    rigid.verify_partition_of_unity()

    # Orthogonal tree emits 16 (s=0, t=2) tokens (75% savings while keeping full s=0 spatial resolution!)
    assert ortho.output_length == 16
    assert np.all(ortho.scale_ids == granularity_index(0, 2))
    assert np.isclose(ortho.savings_fraction, 0.75)

    # Rigid octree is forced to emit all 64 (s=0, t=0) tokens (0% savings!)
    assert rigid.output_length == 64
    assert np.all(rigid.scale_ids == granularity_index(0, 0))
    assert np.isclose(rigid.savings_fraction, 0.0)


def test_temporal_discontinuity_breaks_cleanly_on_non_isomorphic_frame():
    """Verify Tip in Spec 3.1: Slices 0,1 are static wall (s=2), Slice 2 has a person walk in (s=0)."""
    grid = (4, 8, 8)
    p_s1 = np.ones((4, 2, 2), dtype=np.float32)
    p_s2 = np.ones((4, 1, 1), dtype=np.float32)
    # Person enters at slice 2 -> s=0
    p_s1[2] = 0.0
    p_s2[2] = 0.0
    p_t1 = np.ones((2, 2, 2), dtype=np.float32)
    p_t2 = np.ones((1, 2, 2), dtype=np.float32)

    layout = hard_tree_route_layout(grid, p_s1, p_s2, p_t1, p_t2)
    layout.verify_partition_of_unity()

    # Slices 0 & 1 merge into 1 (s=2, t=1) token at slice 1;
    # Slice 2 (s=0) & Slice 3 (s=2) have mismatched spatial scales so they DO NOT merge across time:
    # Slice 2 emits 16 (s=0, t=0) tokens, Slice 3 emits 1 (s=2, t=0) token!
    assert layout.frame_lengths == (0, 1, 16, 1)
    assert layout.scale_ids[0] == granularity_index(2, 1)
    assert np.all(layout.scale_ids[1:17] == granularity_index(0, 0))
    assert layout.scale_ids[17] == granularity_index(2, 0)


def test_float_mrope_and_sglang_x2_scaled_mrope_are_ieee754_exact_and_preserve_text_positions():
    """Prove that HF float half-integer MRoPE and SGLang x2 integer MRoPE produce identical cos/sin AND identical text positions to Native."""
    grid = (4, 8, 8)  # 4 slices, 16 native tokens/slice, native_span = 4
    # All compressed into a single (s=2, t=2) token! Slices 0,1,2 have 0 tokens, Slice 3 has 1 token!
    full_comp = hard_tree_route_layout(
        grid,
        np.ones((4, 2, 2), dtype=np.float32),
        np.ones((4, 1, 1), dtype=np.float32),
        np.ones((2, 2, 2), dtype=np.float32),
        np.ones((1, 2, 2), dtype=np.float32),
    )
    full_nat = hard_tree_route_layout(
        grid,
        np.zeros((4, 2, 2), dtype=np.float32),
        np.zeros((4, 1, 1), dtype=np.float32),
        np.zeros((2, 2, 2), dtype=np.float32),
        np.zeros((1, 2, 2), dtype=np.float32),
    )

    # Construct a realistic Qwen3.5 video sequence:
    # [2 prefix text] + 4 * ([3 ts+start text] + [16 video pads] + [1 end text]) + [4 suffix text]
    parts_ids = [10, 11]
    parts_mm = [0, 0]
    for _ in range(4):
        parts_ids.extend([20, 21, 248053] + [248057] * 16 + [248054])
        parts_mm.extend([0, 0, 0] + [2] * 16 + [0])
    parts_ids.extend([30, 31, 32, 33])
    parts_mm.extend([0, 0, 0, 0])

    input_ids = torch.tensor([parts_ids], dtype=torch.long)
    mm_types = torch.tensor([parts_mm], dtype=torch.long)
    base_embeds = torch.randn(1, len(parts_ids), 32)

    pack_nat = build_routed_sequence_batch(
        input_ids,
        base_embeds,
        mm_types,
        video_layouts=[full_nat],
        video_routed_embeds=[torch.randn(64, 32)],
    )
    pack_comp = build_routed_sequence_batch(
        input_ids,
        base_embeds,
        mm_types,
        video_layouts=[full_comp],
        video_routed_embeds=[torch.randn(1, 32)],
    )

    # Suffix text tokens (last 4 positions) MUST have the exact same 3D MRoPE positions in Native and 64->1 Compressed!
    torch.testing.assert_close(
        pack_comp.position_ids_4d[1:, 0, -4:],
        pack_nat.position_ids_4d[1:, 0, -4:],
        atol=0.0,
        rtol=0.0,
    )

    # Also build SGLang `TreeVisualRun`s for the compressed sequence (including 3 zero-length runs!) and compare x2 positions!
    sgl_runs = []
    # Positions in compacted sequence:
    # prefix=2 -> F0 ts+start=3 (start=5, n=0) -> F0 end + F1 ts+start=4 (start=9, n=0)
    # -> F1 end + F2 ts+start=4 (start=13, n=0) -> F2 end + F3 ts+start=4 (start=17, n=1) -> F3 end + suffix=5 (total=23)
    starts = [5, 9, 13, 17]
    emb_off = 0
    for tau in range(4):
        k_t = full_comp.frame_lengths[tau]
        sl = slice(emb_off, emb_off + k_t)
        sgl_runs.append(
            TreeVisualRun(
                start=starts[tau],
                spatial_pos_x2=full_comp.spatial_pos_x2[sl],
                tau_start=full_comp.tau_start[sl],
                tau_end=full_comp.tau_end[sl],
                slice_index=tau,
                media_id=0,
                native_span=4,
            )
        )
        emb_off += k_t

    pos_x2, delta_x2 = tree_routed_mrope_positions(pack_comp.inputs_embeds.shape[1], sgl_runs)
    hf_mrope_x2 = (pack_comp.position_ids_4d[1:, 0, :].numpy() * POSITION_SCALE)
    np.testing.assert_array_equal(pos_x2, hf_mrope_x2.astype(np.int64))
    assert float(delta_x2) == float(pack_comp.rope_deltas.item()) * POSITION_SCALE

    # Verify IEEE-754 exactness between HF RotaryEmbedding(pos_float, inv_freq) and SGLang RotaryEmbedding(pos_x2, inv_freq / 2)!
    txt_cfg = Qwen3_5TextConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 1.0,
            "mrope_section": [11, 11, 10],
        },
    )
    rot_hf = Qwen3_5TextRotaryEmbedding(txt_cfg)
    rot_sgl = Qwen3_5TextRotaryEmbedding(txt_cfg)
    rot_sgl.inv_freq = rot_sgl.inv_freq / float(POSITION_SCALE)

    dummy_x = torch.randn(1, pos_x2.shape[1], 64)
    cos_hf, sin_hf = rot_hf(dummy_x, pack_comp.position_ids_4d[1:])
    cos_sgl, sin_sgl = rot_sgl(dummy_x, torch.from_numpy(pos_x2).unsqueeze(1))
    torch.testing.assert_close(cos_hf, cos_sgl, atol=0.0, rtol=0.0)
    torch.testing.assert_close(sin_hf, sin_sgl, atol=0.0, rtol=0.0)

    # Wire serialization roundtrip check
    item = TreeRouteItem.from_layout("video", full_comp)
    payload = TreeRoutePayload(fingerprint="abc123", items=(item,))
    restored = TreeRoutePayload.from_dict(payload.to_dict())
    assert restored.output_length == 1
    np.testing.assert_array_equal(restored.items[0].scale_ids, full_comp.scale_ids)

"""Tests for Soft-Tree Interpolation, Hypercube Vertex Consistency of `r_total`, MoE Balance Loss, & Chunked Top-512 KL."""

from __future__ import annotations

import itertools

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from treeweave.layout import (
    hard_tree_route_layout,
    moe_binary_balance_loss,
    norm_direction_interpolate,
    soft_tree_mix,
)
from treeweave.losses import chunked_topk_kl_distill_loss, compute_teacher_topk_from_hidden, extract_active_positions
from treeweave.router import RouterOutputs


def test_norm_direction_interpolate_decouples_magnitude_and_direction():
    u = torch.tensor([[1.0, 0.0]])
    v = torch.tensor([[0.0, 100.0]])  # 100x larger norm!
    mid = norm_direction_interpolate(u, v, torch.tensor([0.5]))
    # Direction must be exact 45 degrees (1/sqrt(2), 1/sqrt(2)), NOT dominated by v's 100x norm!
    unit = mid / mid.norm(dim=-1, keepdim=True)
    expected_dir = torch.tensor([[2**-0.5, 2**-0.5]])
    torch.testing.assert_close(unit, expected_dir, atol=1e-5, rtol=1e-5)
    # Magnitude must be arithmetic mean 0.5 * 1 + 0.5 * 100 = 50.5
    torch.testing.assert_close(mid.norm(dim=-1), torch.tensor([50.5]), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "b_s1,b_s2,b_t1,b_t2",
    list(itertools.product([0.0, 1.0], repeat=4)),
)
def test_soft_tree_retention_matches_hard_routing_at_all_16_binary_hypercube_vertices(b_s1, b_s2, b_t1, b_t2):
    """Verify Spec Section 4 theorem: continuous `r_total` equals exact discrete retention at every hypercube corner."""
    grid = (4, 8, 8)  # 4 slices x 4x4 native cells = 64 native tokens
    d = 16
    scales = {
        (s, t): torch.randn(4 >> t, 4 >> s, 4 >> s, d)
        for s in (0, 1, 2)
        for t in (0, 1, 2)
    }
    p_s1 = torch.full((4, 2, 2), b_s1)
    p_s2 = torch.full((4, 1, 1), b_s2)
    p_t1 = torch.full((2, 2, 2), b_t1)
    p_t2 = torch.full((1, 2, 2), b_t2)

    r_out = RouterOutputs(p_s1=[p_s1], p_s2=[p_s2], p_t1=[p_t1], p_t2=[p_t2])
    soft = soft_tree_mix([scales], r_out, [grid], eps_gate=1e-12)
    hard = hard_tree_route_layout(grid, p_s1, p_s2, p_t1, p_t2)

    hard_retention = hard.output_length / float(hard.native_tokens)
    assert abs(float(soft.expected_retention.item()) - hard_retention) < 1e-5

    # Also verify that at binary corners, soft-mixed tokens equal the exact selected scale's broadcast tensor!
    gid = int(hard.scale_ids[0])
    s_sel, t_sel = gid // 3, gid % 3
    expected_v = (
        scales[(s_sel, t_sel)]
        .repeat_interleave(1 << t_sel, dim=0)
        .repeat_interleave(1 << s_sel, dim=1)
        .repeat_interleave(1 << s_sel, dim=2)
        .reshape(64, d)
    )
    torch.testing.assert_close(soft.mixed_tokens[0], expected_v, atol=1e-5, rtol=1e-5)


def test_soft_tree_gradients_are_finite_even_near_zero_gates():
    """Ensure `pow(..., 0.25)` and `sqrt(...)` in geometric-mean tree gates never produce NaN/Inf gradients at p=0."""
    grid = (4, 8, 8)
    d = 8
    scales = {(s, t): torch.randn(4 >> t, 4 >> s, 4 >> s, d) for s in (0, 1, 2) for t in (0, 1, 2)}
    p_s1 = torch.zeros((4, 2, 2), requires_grad=True)
    p_s2 = torch.full((4, 1, 1), 0.5, requires_grad=True)
    p_t1 = torch.zeros((2, 2, 2), requires_grad=True)
    p_t2 = torch.full((1, 2, 2), 0.5, requires_grad=True)

    r_out = RouterOutputs(p_s1=[p_s1], p_s2=[p_s2], p_t1=[p_t1], p_t2=[p_t2])
    soft = soft_tree_mix([scales], r_out, [grid])
    loss = soft.mixed_tokens[0].sum() + soft.expected_retention + soft.balance_loss
    loss.backward()
    for g in (p_s1.grad, p_s2.grad, p_t1.grad, p_t2.grad):
        assert g is not None
        assert torch.isfinite(g).all()


def test_moe_balance_loss_pushes_toward_target_rho():
    p_low = torch.full((16,), 0.2, requires_grad=True)
    loss_low = moe_binary_balance_loss(p_low, target_rho=0.75)
    loss_low.backward()
    # F = 0 < rho = 0.75 => gradient wrt p should be negative (encouraging p to increase!)
    assert p_low.grad is not None and (p_low.grad < 0).all()

    p_high = torch.full((16,), 0.9, requires_grad=True)
    loss_high = moe_binary_balance_loss(p_high, target_rho=0.75)
    loss_high.backward()
    # F = 1 > rho = 0.75 => gradient wrt p should be positive (encouraging p to decrease!)
    assert p_high.grad is not None and (p_high.grad > 0).all()


def test_chunked_topk_kl_matches_dense_reference_and_packed_sqrt_len_weighting():
    torch.manual_seed(7)
    d, vocab = 32, 128
    topk = 24
    s_head = nn.Linear(d, vocab, bias=False)
    t_head = nn.Linear(d, vocab, bias=False)

    # Simulate a packed batch of 2 sub-samples:
    # Sample 0: length 6, 4 supervised response tokens; Sample 1: length 5, 1 supervised response token!
    # Also test packed boundary safety: token 0 of Sample 1 has label != -100 artificially to verify text_position_ids==0 masks it out!
    labels = torch.tensor([[-100, -100, 11, 12, 13, 14, 99, -100, -100, -100, 21]], dtype=torch.long)
    tpos = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4]], dtype=torch.long)

    active = extract_active_positions(labels, text_position_ids=tpos)
    # Boundary label `99` at index 6 (where tpos == 0) MUST NOT be predicted by index 5!
    assert active.count == 5  # 4 tokens in Sample 0 + 1 token in Sample 1
    assert active.sample_ids.tolist() == [0, 0, 0, 0, 1]

    s_hid = torch.randn(1, 11, d, requires_grad=True)
    t_hid = torch.randn(1, 11, d)
    s_act = s_hid[active.batch_indices, active.seq_indices]
    t_act = t_hid[active.batch_indices, active.seq_indices]

    # Reference unchunked computation
    with torch.no_grad():
        t_logits = t_head(t_act).float()
        t_vals, t_idx = torch.topk(t_logits, k=topk, dim=-1)
        t_lp = F.log_softmax(t_vals, dim=-1)
    s_logits = s_head(s_act).float()
    s_lp = F.log_softmax(torch.gather(s_logits, -1, t_idx), dim=-1)
    kl_per_tok = (t_lp.exp() * (t_lp - s_lp)).sum(dim=-1)

    ref_mean = kl_per_tok.mean()
    # Sample 0 has 4 tokens -> weight sqrt(4)=2.0; Sample 1 has 1 token -> weight sqrt(1)=1.0
    ref_sqrt = (2.0 * kl_per_tok[:4].mean() + 1.0 * kl_per_tok[4:].mean()) / 3.0

    # Test tiny chunk_size=2 + both live teacher AND offline cached topk!
    loss_live_mean = chunked_topk_kl_distill_loss(
        s_act, s_head, sample_ids=active.sample_ids, teacher_active_hidden=t_act, teacher_lm_head=t_head,
        topk=topk, reduction="token_mean", chunk_size=2,
    )
    loss_live_sqrt = chunked_topk_kl_distill_loss(
        s_act, s_head, sample_ids=active.sample_ids, teacher_active_hidden=t_act, teacher_lm_head=t_head,
        topk=topk, reduction="sample_sqrt_len", chunk_size=2,
    )
    torch.testing.assert_close(loss_live_mean, ref_mean, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(loss_live_sqrt, ref_sqrt, atol=1e-6, rtol=1e-6)

    cached_idx, cached_lp = compute_teacher_topk_from_hidden(t_act, t_head, topk=topk, chunk_size=2)
    loss_cached = chunked_topk_kl_distill_loss(
        s_act, s_head, sample_ids=active.sample_ids, teacher_topk_indices=cached_idx, teacher_topk_logprobs=cached_lp,
        topk=topk, reduction="token_mean", chunk_size=2,
    )
    # fp16 cached logprobs match within ~1e-3
    torch.testing.assert_close(loss_cached, ref_mean, atol=2e-3, rtol=2e-3)


def test_chunked_topk_kl_offline_topk_shape_and_token_count_guards():
    """Ensure `chunked_topk_kl_distill_loss` rejects every silent token-count or shape misalignment."""
    d, vocab, topk = 16, 64, 12
    s_head = nn.Linear(d, vocab, bias=False)
    s_act = torch.randn(5, d, requires_grad=True)
    good_idx = torch.randint(0, vocab, (5, topk), dtype=torch.int32)
    good_lp = torch.randn(5, topk, dtype=torch.float16)

    # Half-specified offline topk must raise
    with pytest.raises(ValueError, match="must be provided together"):
        chunked_topk_kl_distill_loss(s_act, s_head, teacher_topk_indices=good_idx, topk=topk)
    with pytest.raises(ValueError, match="must be provided together"):
        chunked_topk_kl_distill_loss(s_act, s_head, teacher_topk_logprobs=good_lp, topk=topk)

    # Row count > n_tok (e.g. untruncated cache paired with truncated student!) must raise immediately
    with pytest.raises(ValueError, match="mismatch"):
        chunked_topk_kl_distill_loss(
            s_act,
            s_head,
            teacher_topk_indices=torch.randint(0, vocab, (7, topk), dtype=torch.int32),
            teacher_topk_logprobs=torch.randn(7, topk, dtype=torch.float16),
            topk=topk,
        )

    # Row count < n_tok or mismatched indices vs logprobs shapes must raise
    with pytest.raises(ValueError, match="mismatch"):
        chunked_topk_kl_distill_loss(
            s_act,
            s_head,
            teacher_topk_indices=good_idx[:3],
            teacher_topk_logprobs=good_lp[:3],
            topk=topk,
        )
    with pytest.raises(ValueError, match="mismatch"):
        chunked_topk_kl_distill_loss(
            s_act,
            s_head,
            teacher_topk_indices=good_idx,
            teacher_topk_logprobs=torch.randn(5, topk - 2, dtype=torch.float16),
            topk=topk,
        )
    with pytest.raises(ValueError, match="2D"):
        chunked_topk_kl_distill_loss(
            s_act,
            s_head,
            teacher_topk_indices=good_idx.reshape(-1),
            teacher_topk_logprobs=good_lp.reshape(-1),
            topk=topk,
        )

    # Empty active batch (n_tok == 0): non-empty cache must still raise, while matching (0, K) succeeds!
    s_empty = torch.zeros((0, d), requires_grad=True)
    with pytest.raises(ValueError, match="mismatch"):
        chunked_topk_kl_distill_loss(
            s_empty,
            s_head,
            teacher_topk_indices=good_idx,
            teacher_topk_logprobs=good_lp,
            topk=topk,
        )
    empty_idx, empty_lp = compute_teacher_topk_from_hidden(torch.zeros((0, d)), s_head, topk=topk)
    assert empty_idx.shape == (0, topk) and empty_lp.shape == (0, topk)
    zero_loss = chunked_topk_kl_distill_loss(
        s_empty,
        s_head,
        teacher_topk_indices=empty_idx,
        teacher_topk_logprobs=empty_lp,
        topk=topk,
    )
    assert float(zero_loss.item()) == 0.0
    zero_loss.backward()
    assert s_head.weight.grad is not None


def test_soft_tree_non_isomorphic_and_heterogeneous_binary_trees_match_hard_routing_exact():
    """Verify Cross-Frame Spatial Isomorphism & Macroblock Temporal Agreement on non-uniform binary states."""
    import numpy as np

    # 1. Explicit non-isomorphic cross-frame check: Frame 0 has s=0, Frame 1 has s=1, p_t1 = 1 -> NO cross-frame fold!
    grid2 = (2, 8, 8)
    d = 12
    scales2 = {
        (0, 0): torch.randn(2, 4, 4, d),
        (1, 0): torch.randn(2, 2, 2, d),
        (2, 0): torch.randn(2, 1, 1, d),
        (0, 1): torch.randn(1, 4, 4, d),
        (1, 1): torch.randn(1, 2, 2, d),
        (2, 1): torch.randn(1, 1, 1, d),
    }
    p_s1 = torch.tensor([[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]])
    p_s2 = torch.tensor([[[0.0]], [[0.0]]])
    p_t1 = torch.ones((1, 2, 2))
    p_t2 = torch.zeros((0, 2, 2))

    hard2 = hard_tree_route_layout(grid2, p_s1, p_s2, p_t1, p_t2)
    soft2 = soft_tree_mix([scales2], RouterOutputs([p_s1], [p_s2], [p_t1], [p_t2]), [grid2], eps_gate=1e-8)
    assert abs(float(soft2.expected_retention.item()) - 0.625) < 1e-5
    assert hard2.output_length / float(hard2.native_tokens) == 0.625
    v_mix2 = soft2.mixed_tokens[0].view(2, 4, 4, d)
    torch.testing.assert_close(v_mix2[0], scales2[(0, 0)][0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        v_mix2[1], scales2[(1, 0)][1].repeat_interleave(2, 0).repeat_interleave(2, 1), atol=1e-4, rtol=1e-4
    )

    # 2. 40 random spatially & temporally heterogeneous binary trees (both Orthogonal and Q2 Rigid Octree!)
    rng = np.random.default_rng(2026)
    grid4 = (4, 8, 8)
    for trial in range(40):
        scales4 = {(s, t): torch.randn(4 >> t, 4 >> s, 4 >> s, d) for s in (0, 1, 2) for t in (0, 1, 2)}
        ps1 = torch.from_numpy((rng.random((4, 2, 2)) > 0.5).astype(np.float32))
        ps2 = torch.from_numpy((rng.random((4, 1, 1)) > 0.5).astype(np.float32))
        pt1 = torch.from_numpy((rng.random((2, 2, 2)) > 0.5).astype(np.float32))
        pt2 = torch.from_numpy((rng.random((1, 2, 2)) > 0.5).astype(np.float32))
        allow_fine = bool(trial % 2 == 0)

        hard4 = hard_tree_route_layout(grid4, ps1, ps2, pt1, pt2, allow_fine_temporal_merge=allow_fine)
        soft4 = soft_tree_mix(
            [scales4],
            RouterOutputs([ps1], [ps2], [pt1], [pt2]),
            [grid4],
            allow_fine_temporal_merge=allow_fine,
            eps_gate=1e-8,
        )
        hard_ret = hard4.output_length / float(hard4.native_tokens)
        assert abs(float(soft4.expected_retention.item()) - hard_ret) < 1e-4

        canvas = torch.zeros((4, 4, 4, d))
        for k in range(hard4.output_length):
            gid = int(hard4.scale_ids[k])
            s, tm = gid // 3, gid % 3
            t0, r0, c0 = int(hard4.tau_start[k]), int(hard4.cell_row_start[k]), int(hard4.cell_col_start[k])
            vec = scales4[(s, tm)][t0 >> tm, r0 >> s, c0 >> s]
            canvas[t0 : t0 + (1 << tm), r0 : r0 + (1 << s), c0 : c0 + (1 << s)] = vec
        torch.testing.assert_close(soft4.mixed_tokens[0].view(4, 4, 4, d), canvas, atol=2e-4, rtol=2e-4)

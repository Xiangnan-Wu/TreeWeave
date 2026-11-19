"""Orthogonal Spatiotemporal Tree Layout, Soft-Tree Interpolation, MRoPE & Sequence Compaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import (
    GRANULARITY_LOG_VOLUMES,
    GRANULARITY_VOLUMES,
    NUM_GRANULARITIES,
    POSITION_SCALE,
)
from .router import RouterOutputs

__all__ = [
    "SoftTreeResult",
    "TreeRouteLayout",
    "build_routed_sequence_batch",
    "compute_heuristic_router_outputs",
    "hard_tree_route_layout",
    "moe_binary_balance_loss",
    "norm_direction_interpolate",
    "sample_stochastic_tree_layout",
    "shuffle_router_outputs",
    "soft_tree_mix",
]


@dataclass(frozen=True)
class TreeRouteLayout:
    """Exact Partition-of-Unity routed layout for one image (`T=1`) or video (`T>=1`) item.

    Tokens are ordered strictly in **Causal Completion Order** `(tau_end, cell_row_start, cell_col_start)`.
    """

    grid: Tuple[int, int, int]          # Pre-merger `(T, H, W)` ViT patch grid
    scale_ids: np.ndarray               # `(N_out,)` int8 in `0..8` (`3*s + t`)
    tau_start: np.ndarray               # `(N_out,)` int32 slice start in `0..T-1`
    tau_end: np.ndarray                 # `(N_out,)` int32 slice end in `0..T-1`
    cell_row_start: np.ndarray          # `(N_out,)` int32 post-merger row start in `0..H//2-1`
    cell_col_start: np.ndarray          # `(N_out,)` int32 post-merger col start in `0..W//2-1`
    spatial_pos_x2: np.ndarray          # `(N_out, 2)` int32 `(2 * row_center, 2 * col_center)`
    log_volumes: np.ndarray             # `(N_out,)` float32 `ln(4^s * 2^t)`
    frame_lengths: Tuple[int, ...]      # length `T`, tokens emitted at each `tau_end == tau`

    @property
    def output_length(self) -> int:
        return int(self.scale_ids.shape[0])

    @property
    def native_tokens(self) -> int:
        t, h, w = self.grid
        return int(t * (h // 2) * (w // 2))

    @property
    def savings_fraction(self) -> float:
        nat = self.native_tokens
        return 0.0 if nat == 0 else 1.0 - (self.output_length / float(nat))

    def granularity_counts(self) -> np.ndarray:
        """Counts of emitted tokens across the 9 `(s, t)` bins."""
        return np.bincount(self.scale_ids.astype(np.int64), minlength=NUM_GRANULARITIES)

    def granularity_covered_shares(self) -> np.ndarray:
        """Share of NATIVE token volume covered by each of the 9 `(s, t)` bins (sums to 1.0)."""
        counts = self.granularity_counts().astype(np.float64)
        vols = np.asarray(GRANULARITY_VOLUMES, dtype=np.float64)
        covered = counts * vols
        total = max(float(self.native_tokens), 1.0)
        return (covered / total).astype(np.float32)

    def verify_partition_of_unity(self) -> None:
        """Assert that the routed nodes partition `[0, T) x [0, H//2) x [0, W//2)` with zero overlap or gap."""
        t, h, w = self.grid
        nh, nw = h // 2, w // 2
        canvas = np.zeros((t, nh, nw), dtype=np.int32)
        for idx in range(self.output_length):
            gid = int(self.scale_ids[idx])
            s, tm = gid // 3, gid % 3
            span_s = 1 << s
            span_t = 1 << tm
            t0, t1 = int(self.tau_start[idx]), int(self.tau_end[idx])
            r0, c0 = int(self.cell_row_start[idx]), int(self.cell_col_start[idx])
            if t1 - t0 + 1 != span_t:
                raise AssertionError(f"Token {idx} has t={tm} (span {span_t}) but [{t0}, {t1}].")
            canvas[t0 : t0 + span_t, r0 : r0 + span_s, c0 : c0 + span_s] += 1
        if not np.all(canvas == 1):
            holes = int(np.sum(canvas == 0))
            overlaps = int(np.sum(canvas > 1))
            raise AssertionError(
                f"Partition-of-Unity violated on grid {self.grid}: {holes} holes, {overlaps} overlaps."
            )
        # Verify Causal Completion Order `(tau_end, cell_row_start, cell_col_start)`
        if self.output_length > 1:
            keys = np.stack((self.tau_end, self.cell_row_start, self.cell_col_start), axis=1)
            for k in range(self.output_length - 1):
                if tuple(keys[k]) >= tuple(keys[k + 1]):
                    raise AssertionError(
                        f"Causal completion order violated at index {k}: {tuple(keys[k])} vs {tuple(keys[k+1])}."
                    )
        # Verify per-slice boundedness theorem: 0 <= frame_lengths[tau] <= nh * nw
        if len(self.frame_lengths) != t or sum(self.frame_lengths) != self.output_length:
            raise AssertionError("frame_lengths mismatch.")
        for tau, fl in enumerate(self.frame_lengths):
            if not (0 <= fl <= nh * nw):
                raise AssertionError(f"Per-slice bound violated at tau={tau}: {fl} > {nh * nw}.")


def hard_tree_route_layout(
    grid: Sequence[int],
    p_s1: np.ndarray | torch.Tensor,
    p_s2: np.ndarray | torch.Tensor,
    p_t1: np.ndarray | torch.Tensor,
    p_t2: np.ndarray | torch.Tensor,
    *,
    theta_s1: float = 0.5,
    theta_s2: float = 0.5,
    theta_t1: float = 0.5,
    theta_t2: float = 0.5,
    max_spatial_level: int = 2,
    max_temporal_level: int = 2,
    allow_fine_temporal_merge: bool = True,
) -> TreeRouteLayout:
    """Two-step consistent tree routing (Spatial Quadtree -> Isomorphic Temporal Binary Tree).

    Fast vectorized numpy implementation over quadrant/macroblock grids (`<= 1024` blocks/frame),
    taking ~0.05 ms per image/video frame on CPU.
    """
    t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
    if t <= 0 or h <= 0 or w <= 0 or h % 4 != 0 or w % 4 != 0:
        raise ValueError(f"Grid must be positive with H, W divisible by 4, got {(t, h, w)}.")

    nh, nw = h // 2, w // 2
    bh, bw = h // 4, w // 4
    mh, mw = bh // 2, bw // 2

    def _np(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
        return np.asarray(x, dtype=np.float32)

    s1_arr = _np(p_s1).reshape(t, bh, bw)
    s2_arr = _np(p_s2).reshape(t, mh, mw) if mh > 0 and mw > 0 else np.zeros((t, 0, 0), dtype=np.float32)
    t1_arr = _np(p_t1).reshape(t // 2, bh, bw) if t >= 2 else np.zeros((0, bh, bw), dtype=np.float32)
    t2_arr = _np(p_t2).reshape(t // 4, bh, bw) if t >= 4 else np.zeros((0, bh, bw), dtype=np.float32)

    # ---- Step A: Frame-wise Spatial Quadtree (`s_quad[tau, I_1, J_1] in {0, 1, 2}`) ----
    m_s1 = (s1_arr >= theta_s1) if max_spatial_level >= 1 else np.zeros((t, bh, bw), dtype=bool)
    s_quad = m_s1.astype(np.int8)  # 0 or 1

    if max_spatial_level >= 2 and mh > 0 and mw > 0:
        sub_s1 = m_s1[:, : 2 * mh, : 2 * mw].reshape(t, mh, 2, mw, 2)
        all_four_s1 = sub_s1.all(axis=(2, 4))
        m_s2 = all_four_s1 & (s2_arr >= theta_s2)
        m_s2_expanded = np.repeat(np.repeat(m_s2, 2, axis=1), 2, axis=2)
        s_quad[:, : 2 * mh, : 2 * mw] = np.where(
            m_s2_expanded, np.int8(2), s_quad[:, : 2 * mh, : 2 * mw]
        )

    # ---- Step B: Isomorphic Cross-Frame Binary Tree (`t_quad[tau, I_1, J_1] in {0, 1, 2}`) ----
    # `active_quad[tau, I_1, J_1]` marks whether slice `tau` is the COMPLETION slice (`tau_end == tau`)
    # of a temporal node, and `t_quad[tau, I_1, J_1]` stores that node's temporal level `t in {0, 1, 2}`.
    active_quad = np.ones((t, bh, bw), dtype=bool)
    t_quad = np.zeros((t, bh, bw), dtype=np.int8)

    if max_temporal_level >= 1 and t >= 2:
        t1_len = t // 2
        s_even = s_quad[0 : 2 * t1_len : 2]
        s_odd = s_quad[1 : 2 * t1_len : 2]
        same_s_pair = s_even == s_odd

        if not allow_fine_temporal_merge:
            # Rigid Octree ablation (Q2): forbid s=0 blocks from folding across time
            same_s_pair = same_s_pair & (s_even > 0)

        pass_t1 = t1_arr >= theta_t1
        # For s=2 macroblocks, require all 4 quadrants of the macroblock to agree on t1 merge
        if mh > 0 and mw > 0:
            is_s2_pair = same_s_pair[:, : 2 * mh, : 2 * mw] & (s_even[:, : 2 * mh, : 2 * mw] == 2)
            macro_t1_all = (
                pass_t1[:, : 2 * mh, : 2 * mw]
                .reshape(t1_len, mh, 2, mw, 2)
                .all(axis=(2, 4))
            )
            macro_t1_exp = np.repeat(np.repeat(macro_t1_all, 2, axis=1), 2, axis=2)
            pass_t1 = pass_t1.copy()
            pass_t1[:, : 2 * mh, : 2 * mw] = np.where(
                is_s2_pair, macro_t1_exp, pass_t1[:, : 2 * mh, : 2 * mw]
            )

        merge_t1 = same_s_pair & pass_t1  # `(t1_len, bh, bw)`
        # When `merge_t1[p]` holds, slice `2*p` folds into completion slice `2*p + 1`!
        active_quad[0 : 2 * t1_len : 2] = ~merge_t1
        t_quad[1 : 2 * t1_len : 2] = np.where(merge_t1, np.int8(1), np.int8(0))

        if max_temporal_level >= 2 and t >= 4:
            t2_len = t // 4
            both_pairs_t1 = merge_t1[0 : 2 * t2_len : 2] & merge_t1[1 : 2 * t2_len : 2]
            s_left = s_quad[1 : 4 * t2_len : 4]
            s_right = s_quad[3 : 4 * t2_len : 4]
            same_s_quad = both_pairs_t1 & (s_left == s_right)

            pass_t2 = t2_arr >= theta_t2
            if mh > 0 and mw > 0:
                is_s2_quad = same_s_quad[:, : 2 * mh, : 2 * mw] & (
                    s_right[:, : 2 * mh, : 2 * mw] == 2
                )
                macro_t2_all = (
                    pass_t2[:, : 2 * mh, : 2 * mw]
                    .reshape(t2_len, mh, 2, mw, 2)
                    .all(axis=(2, 4))
                )
                macro_t2_exp = np.repeat(np.repeat(macro_t2_all, 2, axis=1), 2, axis=2)
                pass_t2 = pass_t2.copy()
                pass_t2[:, : 2 * mh, : 2 * mw] = np.where(
                    is_s2_quad, macro_t2_exp, pass_t2[:, : 2 * mh, : 2 * mw]
                )

            merge_t2 = same_s_quad & pass_t2
            # When `merge_t2[q]` holds, completion slice `4*q + 1` folds into `4*q + 3`!
            active_quad[1 : 4 * t2_len : 4] = np.where(
                merge_t2, False, active_quad[1 : 4 * t2_len : 4]
            )
            t_quad[3 : 4 * t2_len : 4] = np.where(
                merge_t2, np.int8(2), t_quad[3 : 4 * t2_len : 4]
            )

    # ---- Step C: Emit tokens per completion slice `tau` in `(cell_row_start, cell_col_start)` raster order ----
    cell_rows = np.arange(nh, dtype=np.int32)[:, None]
    cell_cols = np.arange(nw, dtype=np.int32)[None, :]
    quad_r = cell_rows >> 1
    quad_c = cell_cols >> 1
    top_left_s1 = ((cell_rows & 1) == 0) & ((cell_cols & 1) == 0)
    top_left_s2 = ((cell_rows & 3) == 0) & ((cell_cols & 3) == 0)

    log_vol_lut = np.asarray(GRANULARITY_LOG_VOLUMES, dtype=np.float32)

    scale_parts: List[np.ndarray] = []
    t_start_parts: List[np.ndarray] = []
    t_end_parts: List[np.ndarray] = []
    r_start_parts: List[np.ndarray] = []
    c_start_parts: List[np.ndarray] = []
    pos_hw_parts: List[np.ndarray] = []
    frame_lengths: List[int] = []

    for tau in range(t):
        act_cell = active_quad[tau][quad_r, quad_c]
        s_cell = s_quad[tau][quad_r, quad_c]
        t_cell = t_quad[tau][quad_r, quad_c]

        emit_mask = act_cell & (
            (s_cell == 0)
            | ((s_cell == 1) & top_left_s1)
            | ((s_cell == 2) & top_left_s2)
        )
        flat_emit = np.flatnonzero(emit_mask.reshape(-1))
        n_emit = int(flat_emit.size)
        frame_lengths.append(n_emit)
        if n_emit == 0:
            continue

        rows = (flat_emit // nw).astype(np.int32)
        cols = (flat_emit % nw).astype(np.int32)
        s_vals = s_cell.reshape(-1)[flat_emit]
        t_vals = t_cell.reshape(-1)[flat_emit]
        g_ids = (s_vals * 3 + t_vals).astype(np.int8)

        # Dyadic center in x2 coordinates: `2 * row + (2^s - 1)`!
        # Check: s=0 -> 2*r + 0; s=1 -> 2*(2*I1) + 1 = 4*I1 + 1; s=2 -> 2*(4*I2) + 3 = 8*I2 + 3!
        offset_x2 = ((1 << s_vals.astype(np.int32)) - 1).astype(np.int32)
        pos_r_x2 = POSITION_SCALE * rows + offset_x2
        pos_c_x2 = POSITION_SCALE * cols + offset_x2

        span_t = 1 << t_vals.astype(np.int32)
        tau_end_arr = np.full(n_emit, tau, dtype=np.int32)
        tau_start_arr = tau_end_arr - span_t + 1

        scale_parts.append(g_ids)
        t_start_parts.append(tau_start_arr)
        t_end_parts.append(tau_end_arr)
        r_start_parts.append(rows)
        c_start_parts.append(cols)
        pos_hw_parts.append(np.stack((pos_r_x2, pos_c_x2), axis=1))

    if scale_parts:
        scale_ids = np.concatenate(scale_parts)
        tau_start = np.concatenate(t_start_parts)
        tau_end = np.concatenate(t_end_parts)
        cell_row_start = np.concatenate(r_start_parts)
        cell_col_start = np.concatenate(c_start_parts)
        spatial_pos_x2 = np.concatenate(pos_hw_parts)
        log_volumes = log_vol_lut[scale_ids]
    else:
        scale_ids = np.zeros((0,), dtype=np.int8)
        tau_start = np.zeros((0,), dtype=np.int32)
        tau_end = np.zeros((0,), dtype=np.int32)
        cell_row_start = np.zeros((0,), dtype=np.int32)
        cell_col_start = np.zeros((0,), dtype=np.int32)
        spatial_pos_x2 = np.zeros((0, 2), dtype=np.int32)
        log_volumes = np.zeros((0,), dtype=np.float32)

    return TreeRouteLayout(
        grid=(t, h, w),
        scale_ids=scale_ids,
        tau_start=tau_start,
        tau_end=tau_end,
        cell_row_start=cell_row_start,
        cell_col_start=cell_col_start,
        spatial_pos_x2=spatial_pos_x2,
        log_volumes=log_volumes,
        frame_lengths=tuple(frame_lengths),
    )


def sample_stochastic_tree_layout(
    grid: Sequence[int],
    rng: np.random.Generator,
    *,
    prob_s1: float = 0.65,
    prob_s2_given_s1: float = 0.45,
    prob_t1: float = 0.60,
    prob_t2_given_t1: float = 0.45,
    max_spatial_level: int = 2,
    max_temporal_level: int = 2,
    allow_fine_temporal_merge: bool = True,
    uniform_single_scale: Optional[Tuple[int, int]] = None,
) -> TreeRouteLayout:
    """Stage 1 Stochastic Pyramid Weaving: sample valid mixed-granularity `(s, t)` trees without the router.

    In `uniform_single_scale=(s, t)` mode (for Ablation Q5), forces a uniform deterministic scale across the grid.
    To ensure temporal isomorphism hits all `(s, t)` scales richly during Stage 1 warmup, spatial scale decisions
    within each 4-slice temporal window are shared with probability 0.75 and varied independently with 0.25.
    """
    t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
    bh, bw = h // 4, w // 4
    mh, mw = bh // 2, bw // 2

    if uniform_single_scale is not None:
        s_target, t_target = uniform_single_scale
        p_s1 = np.full((t, bh, bw), 1.0 if s_target >= 1 else 0.0, dtype=np.float32)
        p_s2 = np.full((t, mh, mw), 1.0 if s_target >= 2 else 0.0, dtype=np.float32)
        p_t1 = np.full((t // 2, bh, bw), 1.0 if t_target >= 1 else 0.0, dtype=np.float32)
        p_t2 = np.full((t // 4, bh, bw), 1.0 if t_target >= 2 else 0.0, dtype=np.float32)
        return hard_tree_route_layout(
            grid,
            p_s1,
            p_s2,
            p_t1,
            p_t2,
            max_spatial_level=max_spatial_level,
            max_temporal_level=max_temporal_level,
            allow_fine_temporal_merge=allow_fine_temporal_merge,
        )

    # Sample macroblock & quadrant spatial choices with temporal persistence inside 4-slice windows
    num_win = (t + 3) // 4
    win_s2 = rng.random((num_win, mh, mw)) < prob_s2_given_s1 if mh > 0 and mw > 0 else np.zeros((num_win, 0, 0), dtype=bool)
    win_s1 = rng.random((num_win, bh, bw)) < prob_s1
    if mh > 0 and mw > 0:
        win_s2_exp = np.repeat(np.repeat(win_s2, 2, axis=1), 2, axis=2)
        win_s1[:, : 2 * mh, : 2 * mw] |= win_s2_exp

    p_s1 = np.zeros((t, bh, bw), dtype=np.float32)
    p_s2 = np.zeros((t, mh, mw), dtype=np.float32)
    for tau in range(t):
        w_idx = tau // 4
        if t == 1 or rng.random() < 0.75:
            s1_frame = win_s1[w_idx]
            s2_frame = win_s2[w_idx]
        else:
            s2_frame = rng.random((mh, mw)) < prob_s2_given_s1 if mh > 0 and mw > 0 else np.zeros((0, 0), dtype=bool)
            s1_frame = rng.random((bh, bw)) < prob_s1
            if mh > 0 and mw > 0:
                s1_frame[: 2 * mh, : 2 * mw] |= np.repeat(np.repeat(s2_frame, 2, axis=0), 2, axis=1)
        p_s1[tau] = s1_frame.astype(np.float32)
        if mh > 0 and mw > 0:
            p_s2[tau] = s2_frame.astype(np.float32)

    t1_len, t2_len = t // 2, t // 4
    m_t1 = rng.random((t1_len, bh, bw)) < prob_t1
    m_t2 = rng.random((t2_len, bh, bw)) < prob_t2_given_t1
    if mh > 0 and mw > 0 and t1_len > 0:
        # Synchronize macroblock temporal decisions so (s=2, t=1) and (s=2, t=2) activate frequently
        macro_t1 = rng.random((t1_len, mh, mw)) < prob_t1
        m_t1[:, : 2 * mh, : 2 * mw] = np.where(
            p_s1[1 : 2 * t1_len : 2, : 2 * mh, : 2 * mw] > 0.5,
            np.repeat(np.repeat(macro_t1, 2, axis=1), 2, axis=2),
            m_t1[:, : 2 * mh, : 2 * mw],
        )
        if t2_len > 0:
            macro_t2 = rng.random((t2_len, mh, mw)) < prob_t2_given_t1
            m_t2[:, : 2 * mh, : 2 * mw] = np.repeat(np.repeat(macro_t2, 2, axis=1), 2, axis=2)

    return hard_tree_route_layout(
        grid,
        p_s1,
        p_s2,
        m_t1.astype(np.float32),
        m_t2.astype(np.float32),
        max_spatial_level=max_spatial_level,
        max_temporal_level=max_temporal_level,
        allow_fine_temporal_merge=allow_fine_temporal_merge,
    )


def shuffle_router_outputs(outputs: RouterOutputs, seed: int = 0) -> RouterOutputs:
    """Ablation Q3(a): Shuffle routing scores within each media item while preserving exact score multisets."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def _shuf(t_list: List[torch.Tensor]) -> List[torch.Tensor]:
        res = []
        for x in t_list:
            if x.numel() <= 1:
                res.append(x)
                continue
            perm = torch.randperm(x.numel(), generator=gen).to(x.device)
            res.append(x.reshape(-1)[perm].view_as(x))
        return res

    return RouterOutputs(
        p_s1=_shuf(outputs.p_s1),
        p_s2=_shuf(outputs.p_s2),
        p_t1=_shuf(outputs.p_t1),
        p_t2=_shuf(outputs.p_t2),
    )


def compute_heuristic_router_outputs(
    raster_list: Sequence[torch.Tensor],
    grid_thw_list: Sequence[Sequence[int]],
) -> RouterOutputs:
    """Ablation Q3(b): Low-level feature-variance / frame-difference heuristic router (normalized to `[0, 1]`)."""
    p_s1_list, p_s2_list, p_t1_list, p_t2_list = [], [], [], []
    for r, grid in zip(raster_list, grid_thw_list):
        t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
        bh, bw = h // 4, w // 4
        mh, mw = bh // 2, bw // 2
        rf = r.detach().float()

        # Spatial homogeneity score = 1 - rank_normalized(patch variance)
        quads = rf.view(t, bh, 4, bw, 4, -1).permute(0, 1, 3, 2, 4, 5).reshape(t, bh, bw, 16, -1)
        var_s1 = quads.var(dim=3).mean(dim=-1)  # (t, bh, bw)
        p_s1 = torch.exp(-var_s1 / (var_s1.median().clamp_min(1e-6)))
        p_s1_list.append(p_s1)

        if mh > 0 and mw > 0:
            macros = (
                rf[:, : 8 * mh, : 8 * mw, :]
                .view(t, mh, 8, mw, 8, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(t, mh, mw, 64, -1)
            )
            var_s2 = macros.var(dim=3).mean(dim=-1)
            p_s2 = torch.exp(-var_s2 / (var_s2.median().clamp_min(1e-6)))
        else:
            p_s2 = torch.zeros((t, mh, mw), dtype=torch.float32, device=r.device)
        p_s2_list.append(p_s2)

        t1_len = t // 2
        if t1_len > 0:
            q_mean = quads.mean(dim=3)  # (t, bh, bw, d)
            diff1 = (q_mean[1 : 2 * t1_len : 2] - q_mean[0 : 2 * t1_len : 2]).norm(dim=-1)
            p_t1 = torch.exp(-diff1 / (diff1.median().clamp_min(1e-6)))
        else:
            p_t1 = torch.zeros((0, bh, bw), dtype=torch.float32, device=r.device)
        p_t1_list.append(p_t1)

        t2_len = t // 4
        if t2_len > 0:
            q_mean = quads.mean(dim=3)
            w4 = q_mean[: 4 * t2_len].view(t2_len, 4, bh, bw, -1)
            diff2 = w4.var(dim=1).mean(dim=-1)
            p_t2 = torch.exp(-diff2 / (diff2.median().clamp_min(1e-6)))
        else:
            p_t2 = torch.zeros((0, bh, bw), dtype=torch.float32, device=r.device)
        p_t2_list.append(p_t2)

    return RouterOutputs(p_s1=p_s1_list, p_s2=p_s2_list, p_t1=p_t1_list, p_t2=p_t2_list)


def norm_direction_interpolate(
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Norm-direction decoupled interpolation `I(u, v; p)` (VisionWeave Eq. 9).

    `u, v`: `(..., D)`; `p`: `(...)` or `(..., 1)` in `[0, 1]`.
    """
    orig_dtype = u.dtype
    uf, vf = u.float(), v.float()
    pf = p.float().unsqueeze(-1) if p.ndim == u.ndim - 1 else p.float()
    norm_u = uf.norm(dim=-1, keepdim=True)
    norm_v = vf.norm(dim=-1, keepdim=True)
    u_hat = uf / norm_u.clamp_min(eps)
    v_hat = vf / norm_v.clamp_min(eps)
    mag = (1.0 - pf) * norm_u + pf * norm_v
    mixed_dir = (1.0 - pf) * u_hat + pf * v_hat
    mixed_dir = mixed_dir / mixed_dir.norm(dim=-1, keepdim=True).clamp_min(eps)
    return (mag * mixed_dir).to(orig_dtype)


def moe_binary_balance_loss(p: torch.Tensor, target_rho: float) -> torch.Tensor:
    """Per-item MoE load-balancing penalty (VisionWeave Eq. 10):
        `L_bal = (F / rho) * G + ((1 - F) / (1 - rho)) * (1 - G)`
    Minimized (`L_bal == 1.0`) when hard fraction `F == rho` and mean gate `G == rho`.
    """
    if p.numel() == 0 or not (0.0 < target_rho < 1.0):
        return p.new_zeros(())
    pf = p.float()
    hard_frac = (pf.detach() >= 0.5).float().mean()
    mean_gate = pf.mean()
    return (hard_frac / target_rho) * mean_gate + ((1.0 - hard_frac) / (1.0 - target_rho)) * (1.0 - mean_gate)


@dataclass
class SoftTreeResult:
    """Outputs of Stage 2 bottom-up differentiable tree soft-mixing."""

    mixed_tokens: List[torch.Tensor]       # each `(T * (H//2) * (W//2), D)` matching native sequence length
    expected_retention: torch.Tensor       # scalar mean expected token retention `r_total in [1/64, 1]`
    balance_loss: torch.Tensor             # scalar mean 4-gate MoE balance loss
    metrics: Dict[str, float]              # detached diagnostics (entropy, per-gate means, retention)


def _upsample_spatial(v_st: torch.Tensor, s: int, target_nh: int, target_nw: int) -> torch.Tensor:
    """Broadcast `(T_t, H_s, W_s, D)` by `2^s` spatially into `(T_t, target_nh, target_nw, D)`."""
    if s == 0:
        return v_st
    rep = 1 << s
    up = v_st.repeat_interleave(rep, dim=1).repeat_interleave(rep, dim=2)
    if up.shape[1] == target_nh and up.shape[2] == target_nw:
        return up
    out = v_st.new_zeros((v_st.shape[0], target_nh, target_nw, v_st.shape[-1]))
    out[:, : up.shape[1], : up.shape[2], :] = up
    return out


def _smooth_geom2(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    """Smooth, finite-gradient geometric mean `sqrt(a * b)` in FP32 (`== 0` when `a*b == 0`, MPS & CUDA safe)."""
    d = max(min(float(eps), 1e-5), 1e-8)
    prod = a.float().clamp_min(0.0) * b.float().clamp_min(0.0)
    out = torch.sqrt(prod + d * d) - d
    return out.clamp(0.0, 1.0).to(a.dtype)


def _smooth_geom4_list(
    t0: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    t3: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Smooth, finite-gradient 4-way geometric mean `(t0 * t1 * t2 * t3)^(1/4)` in FP32 (MPS & CUDA safe)."""
    d = max(min(float(eps), 1e-5), 1e-8)
    prod = (
        t0.float().clamp_min(0.0)
        * t1.float().clamp_min(0.0)
        * t2.float().clamp_min(0.0)
        * t3.float().clamp_min(0.0)
    )
    out = torch.pow(prod + (d**4), 0.25) - d
    return out.clamp(0.0, 1.0).to(t0.dtype)


def soft_tree_mix(
    projected_scales: Sequence[Dict[Tuple[int, int], torch.Tensor]],
    router_outputs: RouterOutputs,
    grid_thw_list: Sequence[Sequence[int]],
    *,
    rho_s1: float = 0.75,
    rho_s2: float = 0.50,
    rho_t1: float = 0.65,
    rho_t2: float = 0.50,
    max_spatial_level: int = 2,
    max_temporal_level: int = 2,
    allow_fine_temporal_merge: bool = True,
    eps_gate: float = 1e-6,
) -> SoftTreeResult:
    """Bottom-up Spatial (`0 -> 1 -> 2`) then Isomorphic Temporal (`0 -> 1 -> 2`) Soft-Tree Mixing + Exact Rate Loss.

    Enforces:
    1. Tree-consistent spatial simplex probabilities `(pi_0, pi_1, pi_2)` on every quadrant `(tau, I_1, J_1)`.
    2. Cross-frame Bhattacharyya spatial isomorphism (`sum_s (pi_s(tau_a) * pi_s(tau_b))^{1/k} <= 1`) AND
       4-quadrant macroblock temporal agreement on `s = 2`, guaranteeing that at EVERY binary assignment
       (including spatially heterogeneous & non-isomorphic cross-frame states!), both `mixed_tokens` and
       `expected_retention` match `hard_tree_route_layout` identically.
    """
    mixed_list: List[torch.Tensor] = []
    retentions: List[torch.Tensor] = []
    bal_terms: List[torch.Tensor] = []

    entropies: List[float] = []
    gate_means = {"s1": [], "s2": [], "t1": [], "t2": []}

    for item_scales, p_s1, p_s2, p_t1, p_t2, grid in zip(
        projected_scales,
        router_outputs.p_s1,
        router_outputs.p_s2,
        router_outputs.p_t1,
        router_outputs.p_t2,
        grid_thw_list,
    ):
        t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
        nh, nw = h // 2, w // 2
        bh, bw = h // 4, w // 4
        mh, mw = bh // 2, bw // 2

        p_s1_eff = p_s1 if max_spatial_level >= 1 else torch.zeros_like(p_s1)
        p_s2_eff_quad = torch.zeros_like(p_s1)  # (T, bh, bw)
        if max_spatial_level >= 2 and mh > 0 and mw > 0 and (2, 0) in item_scales:
            sub_s1 = p_s1_eff[:, : 2 * mh, : 2 * mw].view(t, mh, 2, mw, 2)
            geom_s1 = _smooth_geom4_list(
                sub_s1[:, :, 0, :, 0],
                sub_s1[:, :, 0, :, 1],
                sub_s1[:, :, 1, :, 0],
                sub_s1[:, :, 1, :, 1],
                eps_gate,
            )  # (T, mh, mw)
            p_s2_eff_macro = p_s2 * geom_s1
            p_s2_eff_quad[:, : 2 * mh, : 2 * mw] = p_s2_eff_macro.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

        # Mutually-exclusive spatial simplex `(pi_0, pi_1, pi_2)` per quadrant `(T, bh, bw)`
        pi_2 = p_s2_eff_quad
        pi_1 = (1.0 - pi_2) * p_s1_eff
        pi_0 = (1.0 - pi_2) * (1.0 - p_s1_eff)

        p_s1_cell = p_s1_eff.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)  # (T, nh, nw)
        p_s2_cell = p_s2_eff_quad.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)  # (T, nh, nw)

        # Bottom-up spatial mixing at each active temporal scale `tm in {0, 1, 2}`
        v_sp_by_t: Dict[int, torch.Tensor] = {}
        active_t_levels = [0]
        if max_temporal_level >= 1 and t >= 2 and (0, 1) in item_scales:
            active_t_levels.append(1)
        if max_temporal_level >= 2 and t >= 4 and (0, 2) in item_scales:
            active_t_levels.append(2)

        for tm in active_t_levels:
            if tm == 0:
                ps1_tm = p_s1_cell
                ps2_tm = p_s2_cell
            elif tm == 1:
                t1_len = t // 2
                ps1_tm = _smooth_geom2(p_s1_cell[0 : 2 * t1_len : 2], p_s1_cell[1 : 2 * t1_len : 2], eps_gate)
                ps2_tm = _smooth_geom2(p_s2_cell[0 : 2 * t1_len : 2], p_s2_cell[1 : 2 * t1_len : 2], eps_gate)
            else:
                t2_len = t // 4
                ps1_tm = _smooth_geom4_list(
                    p_s1_cell[0 : 4 * t2_len : 4],
                    p_s1_cell[1 : 4 * t2_len : 4],
                    p_s1_cell[2 : 4 * t2_len : 4],
                    p_s1_cell[3 : 4 * t2_len : 4],
                    eps_gate,
                )
                ps2_tm = _smooth_geom4_list(
                    p_s2_cell[0 : 4 * t2_len : 4],
                    p_s2_cell[1 : 4 * t2_len : 4],
                    p_s2_cell[2 : 4 * t2_len : 4],
                    p_s2_cell[3 : 4 * t2_len : 4],
                    eps_gate,
                )

            v_cur = item_scales[(0, tm)]  # (t_tm, nh, nw, D)
            if max_spatial_level >= 1 and (1, tm) in item_scales:
                v1_up = _upsample_spatial(item_scales[(1, tm)], 1, nh, nw)
                v_cur = norm_direction_interpolate(v_cur, v1_up, ps1_tm)
            if max_spatial_level >= 2 and (2, tm) in item_scales and mh > 0 and mw > 0:
                v2_up = _upsample_spatial(item_scales[(2, tm)], 2, nh, nw)
                v_cur = norm_direction_interpolate(v_cur, v2_up, ps2_tm)
            v_sp_by_t[tm] = v_cur

        # Per-quadrant spatial retention `r_sp in [1/16, 1]`
        r_sp = pi_0 + 0.25 * pi_1 + (1.0 / 16.0) * pi_2
        delta_r_t1_full = torch.zeros_like(r_sp)
        delta_r_t2_full = torch.zeros_like(r_sp)

        # Bottom-up isomorphic temporal mixing (`0 -> 1 -> 2`) back onto `(T, nh, nw, D)`
        v_mix = v_sp_by_t[0]
        if 1 in v_sp_by_t:
            t1_len = t // 2
            # For `s=2` macroblocks, require all 4 quadrants to agree on `t=1` merge (matching Step B of hard routing)
            p_t1_s2 = p_t1.clone()
            if mh > 0 and mw > 0:
                sub_t1 = p_t1[:, : 2 * mh, : 2 * mw].view(t1_len, mh, 2, mw, 2)
                macro_t1 = _smooth_geom4_list(
                    sub_t1[:, :, 0, :, 0],
                    sub_t1[:, :, 0, :, 1],
                    sub_t1[:, :, 1, :, 0],
                    sub_t1[:, :, 1, :, 1],
                    eps_gate,
                )
                p_t1_s2[:, : 2 * mh, : 2 * mw] = macro_t1.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

            iso1_0 = (
                _smooth_geom2(pi_0[0 : 2 * t1_len : 2], pi_0[1 : 2 * t1_len : 2], eps_gate)
                if allow_fine_temporal_merge
                else torch.zeros_like(p_t1)
            )
            iso1_1 = _smooth_geom2(pi_1[0 : 2 * t1_len : 2], pi_1[1 : 2 * t1_len : 2], eps_gate)
            iso1_2 = _smooth_geom2(pi_2[0 : 2 * t1_len : 2], pi_2[1 : 2 * t1_len : 2], eps_gate)

            m1_0 = iso1_0 * p_t1
            m1_1 = iso1_1 * p_t1
            m1_2 = iso1_2 * p_t1_s2
            p_t1_eff = (m1_0 + m1_1 + m1_2).clamp(0.0, 1.0)

            p_t1_full = torch.zeros_like(p_s1)
            p_t1_full[: 2 * t1_len] = p_t1_eff.repeat_interleave(2, dim=0)
            p_t1_cell = p_t1_full.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

            v1_t_up = v_sp_by_t[0].clone()
            v1_t_up[: 2 * t1_len] = v_sp_by_t[1].repeat_interleave(2, dim=0)
            v_mix = norm_direction_interpolate(v_mix, v1_t_up, p_t1_cell)

            dr_pair = 0.5 * (m1_0 + 0.25 * m1_1 + (1.0 / 16.0) * m1_2)
            delta_r_t1_full[: 2 * t1_len] = dr_pair.repeat_interleave(2, dim=0)

            if 2 in v_sp_by_t:
                t2_len = t // 4
                p_t2_s2 = p_t2.clone()
                if mh > 0 and mw > 0:
                    sub_t2 = p_t2[:, : 2 * mh, : 2 * mw].view(t2_len, mh, 2, mw, 2)
                    macro_t2 = _smooth_geom4_list(
                        sub_t2[:, :, 0, :, 0],
                        sub_t2[:, :, 0, :, 1],
                        sub_t2[:, :, 1, :, 0],
                        sub_t2[:, :, 1, :, 1],
                        eps_gate,
                    )
                    p_t2_s2[:, : 2 * mh, : 2 * mw] = macro_t2.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

                iso2_0 = (
                    _smooth_geom4_list(
                        pi_0[0 : 4 * t2_len : 4],
                        pi_0[1 : 4 * t2_len : 4],
                        pi_0[2 : 4 * t2_len : 4],
                        pi_0[3 : 4 * t2_len : 4],
                        eps_gate,
                    )
                    if allow_fine_temporal_merge
                    else torch.zeros_like(p_t2)
                )
                iso2_1 = _smooth_geom4_list(
                    pi_1[0 : 4 * t2_len : 4],
                    pi_1[1 : 4 * t2_len : 4],
                    pi_1[2 : 4 * t2_len : 4],
                    pi_1[3 : 4 * t2_len : 4],
                    eps_gate,
                )
                iso2_2 = _smooth_geom4_list(
                    pi_2[0 : 4 * t2_len : 4],
                    pi_2[1 : 4 * t2_len : 4],
                    pi_2[2 : 4 * t2_len : 4],
                    pi_2[3 : 4 * t2_len : 4],
                    eps_gate,
                )

                g_t1_quad = _smooth_geom2(p_t1[0 : 2 * t2_len : 2], p_t1[1 : 2 * t2_len : 2], eps_gate)
                g_t1_s2 = _smooth_geom2(p_t1_s2[0 : 2 * t2_len : 2], p_t1_s2[1 : 2 * t2_len : 2], eps_gate)

                m2_0 = iso2_0 * p_t2 * g_t1_quad
                m2_1 = iso2_1 * p_t2 * g_t1_quad
                m2_2 = iso2_2 * p_t2_s2 * g_t1_s2
                p_t2_eff = (m2_0 + m2_1 + m2_2).clamp(0.0, 1.0)

                p_t2_full = torch.zeros_like(p_s1)
                p_t2_full[: 4 * t2_len] = p_t2_eff.repeat_interleave(4, dim=0)
                p_t2_cell = p_t2_full.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

                v2_t_up = v_mix.clone()
                v2_t_up[: 4 * t2_len] = v_sp_by_t[2].repeat_interleave(4, dim=0)
                v_mix = norm_direction_interpolate(v_mix, v2_t_up, p_t2_cell)

                dr_quad = 0.25 * (m2_0 + 0.25 * m2_1 + (1.0 / 16.0) * m2_2)
                delta_r_t2_full[: 4 * t2_len] = dr_quad.repeat_interleave(4, dim=0)

        mixed_list.append(v_mix.reshape(t * nh * nw, -1))

        r_quad = (r_sp - delta_r_t1_full - delta_r_t2_full).clamp_min(1.0 / 64.0)
        retentions.append(r_quad.mean())

        # Per-item MoE load-balance penalties across active gates
        item_bals: List[torch.Tensor] = []
        if max_spatial_level >= 1 and p_s1.numel() > 0:
            item_bals.append(moe_binary_balance_loss(p_s1, rho_s1))
            gate_means["s1"].append(float(p_s1.detach().mean().item()))
        if max_spatial_level >= 2 and p_s2.numel() > 0:
            item_bals.append(moe_binary_balance_loss(p_s2, rho_s2))
            gate_means["s2"].append(float(p_s2.detach().mean().item()))
        if max_temporal_level >= 1 and p_t1.numel() > 0:
            item_bals.append(moe_binary_balance_loss(p_t1, rho_t1))
            gate_means["t1"].append(float(p_t1.detach().mean().item()))
        if max_temporal_level >= 2 and p_t2.numel() > 0:
            item_bals.append(moe_binary_balance_loss(p_t2, rho_t2))
            gate_means["t2"].append(float(p_t2.detach().mean().item()))

        if item_bals:
            bal_terms.append(torch.stack(item_bals).mean())

        # Binary entropy diagnostic
        for g_tensor in (p_s1, p_s2, p_t1, p_t2):
            if g_tensor.numel() > 0:
                gf = g_tensor.detach().float().clamp(1e-6, 1.0 - 1e-6)
                ent = -(gf * gf.log() + (1.0 - gf) * (1.0 - gf).log()).mean().item()
                entropies.append(ent)

    dev = mixed_list[0].device if mixed_list else torch.device("cpu")
    expected_ret = torch.stack(retentions).mean() if retentions else torch.zeros((), device=dev)
    bal_loss = torch.stack(bal_terms).mean() if bal_terms else torch.zeros((), device=dev)

    metrics = {
        "expected_retention": float(expected_ret.detach().item()),
        "expected_savings": float(1.0 - expected_ret.detach().item()),
        "router_entropy": float(sum(entropies) / max(len(entropies), 1)),
    }
    for k, vals in gate_means.items():
        if vals:
            metrics[f"p_{k}_mean"] = float(sum(vals) / len(vals))

    return SoftTreeResult(
        mixed_tokens=mixed_list,
        expected_retention=expected_ret,
        balance_loss=bal_loss,
        metrics=metrics,
    )


def _find_modality_runs_1d(mm_type_row: np.ndarray, modality_id: int) -> List[Tuple[int, int]]:
    """Return `(start, length)` of every contiguous run where `mm_type_row == modality_id`."""
    is_mod = mm_type_row == modality_id
    if not is_mod.any():
        return []
    padded = np.empty(is_mod.size + 2, dtype=bool)
    padded[0] = False
    padded[-1] = False
    padded[1:-1] = is_mod
    diffs = np.flatnonzero(padded[1:] != padded[:-1])
    starts = diffs[0::2]
    ends = diffs[1::2]
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]


@dataclass
class RoutedBatchPack:
    """Compacted student inputs & exact 3D half-integer MRoPE after variable-length hard tree routing."""

    inputs_embeds: torch.Tensor                 # `(B, L_routed, D)`
    labels: Optional[torch.Tensor]              # `(B, L_routed)`
    attention_mask: Optional[torch.Tensor]      # `(B, L_routed)` or None when packed
    position_ids_4d: torch.Tensor               # `(4, B, L_routed)` float32 (row 0 = integer text_pos, rows 1..3 = 3D MRoPE)
    text_position_ids: torch.Tensor             # `(B, L_routed)` int64
    token_log_volumes: torch.Tensor             # `(B, L_routed)` float32 (`0.0` for text & fine `(0,0)` tokens)
    rope_deltas: torch.Tensor                   # `(B, 1)` float32
    cu_seq_lens_q: Optional[torch.Tensor] = None
    cu_seq_lens_k: Optional[torch.Tensor] = None
    max_length_q: Optional[int] = None
    max_length_k: Optional[int] = None


def build_routed_sequence_batch(
    input_ids: torch.Tensor,
    base_inputs_embeds: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    *,
    image_layouts: Sequence[TreeRouteLayout] = (),
    image_routed_embeds: Sequence[torch.Tensor] = (),
    video_layouts: Sequence[TreeRouteLayout] = (),
    video_routed_embeds: Sequence[torch.Tensor] = (),
    labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    text_position_ids: Optional[torch.Tensor] = None,
    is_packed: bool = False,
    pad_embedding: Optional[torch.Tensor] = None,
    force_left_pad: bool = False,
) -> RoutedBatchPack:
    """Compact native placeholder runs into routed mixed-granularity sequences with exact 3D MRoPE.

    Supports:
    - Single/multi-image + single/multi-video interleaved prompts
    - Packed multi-sample sequences (`B == 1`, `is_packed=True`, resetting positions at sample boundaries `text_position_ids == 0`)
    - Batched padded sequences (`B >= 1`, `attention_mask` left- or right-padded)
    - Zero-token slices (`frame_lengths[tau] == 0` when an entire video slice folded into a later slice `tau_end > tau`)
      while still advancing `current_pos` by `native_span` so subsequent timestamps & text keep identical MRoPE positions!
    """
    bsz, seq_len, hidden_dim = base_inputs_embeds.shape
    device = base_inputs_embeds.device
    dtype = base_inputs_embeds.dtype

    mm_np = mm_token_type_ids.detach().cpu().numpy()
    attn_np = attention_mask.detach().cpu().numpy().astype(bool) if attention_mask is not None else None
    tpos_np = text_position_ids.detach().cpu().numpy() if text_position_ids is not None else None

    img_cursor = 0
    vid_cursor = 0

    per_batch_embeds: List[torch.Tensor] = []
    per_batch_labels: List[torch.Tensor] = []
    per_batch_mrope: List[np.ndarray] = []      # each `(3, L_b)` float32
    per_batch_tpos: List[np.ndarray] = []       # each `(L_b,)` int64
    per_batch_logvol: List[torch.Tensor] = []   # each `(L_b,)` float32
    per_batch_deltas: List[float] = []
    packed_subseq_lens: List[int] = []

    for b in range(bsz):
        valid_mask = attn_np[b] if (attn_np is not None and not is_packed) else np.ones(seq_len, dtype=bool)
        valid_indices = np.flatnonzero(valid_mask)
        v_len = int(valid_indices.size)
        mm_valid = mm_np[b, valid_indices]

        # Determine packed sub-sample segments within row `b`
        if is_packed and tpos_np is not None:
            tpos_valid = tpos_np[b, valid_indices]
            seg_starts = np.flatnonzero(tpos_valid == 0).tolist()
            if not seg_starts or seg_starts[0] != 0:
                seg_starts = [0] + [s for s in seg_starts if s > 0]
            seg_ends = seg_starts[1:] + [v_len]
        else:
            seg_starts = [0]
            seg_ends = [v_len]

        keep_valid = np.ones(v_len, dtype=bool)
        scatter_positions_valid: List[np.ndarray] = []
        scatter_embeds_list: List[torch.Tensor] = []
        scatter_logvols_list: List[np.ndarray] = []

        mrope_valid_parts: List[np.ndarray] = []
        tpos_routed_parts: List[np.ndarray] = []
        last_delta = 0.0

        for seg_s, seg_e in zip(seg_starts, seg_ends):
            seg_mm = mm_valid[seg_s:seg_e]
            # Parse contiguous modality groups `(mod_type, rel_start, rel_end)` in `[seg_s, seg_e)`
            groups: List[Tuple[int, int, int]] = []
            if seg_e > seg_s:
                changes = np.flatnonzero(seg_mm[1:] != seg_mm[:-1]) + 1
                g_starts = [0] + changes.tolist()
                g_ends = changes.tolist() + [seg_e - seg_s]
                for gs, ge in zip(g_starts, g_ends):
                    groups.append((int(seg_mm[gs]), seg_s + gs, seg_s + ge))

            cur_pos = 0.0
            seg_mrope_chunks: List[np.ndarray] = []
            seg_routed_len = 0

            g_idx = 0
            while g_idx < len(groups):
                mod_type, gs, ge = groups[g_idx]
                run_len = ge - gs
                if mod_type == 0:
                    # Text run: keep 100% of tokens
                    txt_pos = np.arange(run_len, dtype=np.float32)[None, :] + cur_pos
                    seg_mrope_chunks.append(np.repeat(txt_pos, 3, axis=0))
                    cur_pos += float(run_len)
                    seg_routed_len += run_len
                    g_idx += 1
                elif mod_type == 1:
                    # Single image run
                    if img_cursor >= len(image_layouts):
                        raise ValueError("More image placeholder runs in input_ids than image_layouts.")
                    layout = image_layouts[img_cursor]
                    r_emb = image_routed_embeds[img_cursor]
                    img_cursor += 1

                    if run_len != layout.native_tokens:
                        raise ValueError(
                            f"Image run has {run_len} placeholders, expected {layout.native_tokens} for grid {layout.grid}."
                        )
                    k_out = layout.output_length
                    keep_valid[gs + k_out : ge] = False
                    if k_out > 0:
                        scatter_positions_valid.append(np.arange(gs, gs + k_out, dtype=np.int64))
                        scatter_embeds_list.append(r_emb)
                        scatter_logvols_list.append(layout.log_volumes)

                        hw_half = layout.spatial_pos_x2.astype(np.float32) / float(POSITION_SCALE)
                        vis_mrope = np.stack(
                            (
                                np.full(k_out, cur_pos, dtype=np.float32),
                                cur_pos + hw_half[:, 0],
                                cur_pos + hw_half[:, 1],
                            ),
                            axis=0,
                        )
                        seg_mrope_chunks.append(vis_mrope)

                    native_span = max(layout.grid[1], layout.grid[2]) // 2
                    cur_pos += float(native_span)
                    seg_routed_len += k_out
                    g_idx += 1
                elif mod_type == 2:
                    # Video item! In Qwen3.5, one video of `T` slices spans `T` consecutive `mod_type == 2` groups
                    # separated by timestamp text groups (`mod_type == 0`).
                    if vid_cursor >= len(video_layouts):
                        raise ValueError("More video placeholder runs in input_ids than video_layouts.")
                    layout = video_layouts[vid_cursor]
                    r_emb = video_routed_embeds[vid_cursor]
                    vid_cursor += 1

                    t_slices, vh, vw = layout.grid
                    per_slice_native = (vh // 2) * (vw // 2)
                    native_span = max(vh, vw) // 2

                    # Consume `t_slices` video groups and the `t_slices - 1` inter-frame timestamp text groups between them!
                    slice_base_positions = np.zeros(t_slices, dtype=np.float32)
                    emb_offset = 0
                    slices_seen = 0
                    while slices_seen < t_slices:
                        if g_idx >= len(groups):
                            raise ValueError(
                                f"Video {vid_cursor - 1} expected {t_slices} slice runs, ran out at {slices_seen}."
                            )
                        sub_mod, s_gs, s_ge = groups[g_idx]
                        s_len = s_ge - s_gs
                        if sub_mod == 0:
                            # Inter-slice timestamp text (`<1.5 seconds><|vision_start|>`)
                            txt_pos = np.arange(s_len, dtype=np.float32)[None, :] + cur_pos
                            seg_mrope_chunks.append(np.repeat(txt_pos, 3, axis=0))
                            cur_pos += float(s_len)
                            seg_routed_len += s_len
                            g_idx += 1
                            continue
                        if sub_mod != 2:
                            raise ValueError(
                                f"Unexpected modality {sub_mod} interleaved inside video {vid_cursor - 1} slices."
                            )
                        if s_len != per_slice_native:
                            raise ValueError(
                                f"Video slice {slices_seen} has {s_len} placeholders, expected {per_slice_native}."
                            )

                        tau = slices_seen
                        slice_base_positions[tau] = cur_pos
                        k_tau = layout.frame_lengths[tau]
                        keep_valid[s_gs + k_tau : s_ge] = False

                        if k_tau > 0:
                            sl = slice(emb_offset, emb_offset + k_tau)
                            scatter_positions_valid.append(np.arange(s_gs, s_gs + k_tau, dtype=np.int64))
                            scatter_embeds_list.append(r_emb[sl])
                            scatter_logvols_list.append(layout.log_volumes[sl])

                            # Exact dyadic midpoint base `0.5 * (c[tau_start] + c[tau_end])` (always in `0.5 * Z`!)
                            t0_arr = layout.tau_start[sl]
                            t1_arr = layout.tau_end[sl]
                            c_mid = 0.5 * (slice_base_positions[t0_arr] + slice_base_positions[t1_arr])
                            hw_half = layout.spatial_pos_x2[sl].astype(np.float32) / float(POSITION_SCALE)
                            vis_mrope = np.stack(
                                (
                                    c_mid,
                                    c_mid + hw_half[:, 0],
                                    c_mid + hw_half[:, 1],
                                ),
                                axis=0,
                            )
                            seg_mrope_chunks.append(vis_mrope)
                            emb_offset += k_tau

                        cur_pos += float(native_span)
                        seg_routed_len += k_tau
                        slices_seen += 1
                        g_idx += 1
                else:
                    raise ValueError(f"Unsupported mm_token_type_id: {mod_type}.")

            if seg_mrope_chunks:
                seg_mrope = np.concatenate(seg_mrope_chunks, axis=1)
                last_delta = float(seg_mrope.max()) + 1.0 - float(seg_routed_len)
                mrope_valid_parts.append(seg_mrope)
                tpos_routed_parts.append(np.arange(seg_routed_len, dtype=np.int64))
            else:
                last_delta = 0.0
            if is_packed and seg_routed_len > 0:
                packed_subseq_lens.append(seg_routed_len)

        # Materialize compacted row `b` on GPU
        row_embeds = base_inputs_embeds[b, valid_indices].clone()
        row_logvol = np.zeros(v_len, dtype=np.float32)
        if scatter_positions_valid:
            cat_pos = torch.from_numpy(np.concatenate(scatter_positions_valid)).to(device=device, dtype=torch.long)
            cat_emb = torch.cat(scatter_embeds_list, dim=0).to(device=device, dtype=dtype)
            row_embeds.index_copy_(0, cat_pos, cat_emb)
            row_logvol[np.concatenate(scatter_positions_valid)] = np.concatenate(scatter_logvols_list)

        keep_idx_t = torch.from_numpy(np.flatnonzero(keep_valid)).to(device=device, dtype=torch.long)
        per_batch_embeds.append(row_embeds.index_select(0, keep_idx_t))
        per_batch_logvol.append(torch.from_numpy(row_logvol[keep_valid]).to(device=device, dtype=torch.float32))

        if labels is not None:
            row_labels = labels[b, valid_indices]
            per_batch_labels.append(row_labels.index_select(0, keep_idx_t))

        per_batch_mrope.append(
            np.concatenate(mrope_valid_parts, axis=1)
            if mrope_valid_parts
            else np.zeros((3, 0), dtype=np.float32)
        )
        per_batch_tpos.append(
            np.concatenate(tpos_routed_parts, axis=0)
            if tpos_routed_parts
            else np.zeros((0,), dtype=np.int64)
        )
        per_batch_deltas.append(last_delta)

    if img_cursor != len(image_layouts) or vid_cursor != len(video_layouts):
        raise ValueError(
            f"Unconsumed routed layouts: images {img_cursor}/{len(image_layouts)}, videos {vid_cursor}/{len(video_layouts)}."
        )

    # Collate across `B` rows (left-padding on inference/generation so column `-1` is every row's final token)
    max_routed_len = max((e.shape[0] for e in per_batch_embeds), default=0)
    left_padded = bool(force_left_pad) or bool(
        attn_np is not None
        and not is_packed
        and bsz > 0
        and attn_np[:, 0].sum() < attn_np[:, -1].sum()
    )

    if bsz == 1 and per_batch_embeds[0].shape[0] == max_routed_len:
        out_embeds = per_batch_embeds[0].unsqueeze(0)
        out_labels = per_batch_labels[0].unsqueeze(0) if labels is not None else None
        out_attn = (
            torch.ones((1, max_routed_len), dtype=attention_mask.dtype, device=device)
            if (attention_mask is not None and not is_packed)
            else None
        )
        out_tpos = torch.from_numpy(per_batch_tpos[0]).to(device=device, dtype=torch.long).unsqueeze(0)
        out_mrope = torch.from_numpy(per_batch_mrope[0]).to(device=device, dtype=torch.float32).unsqueeze(1)
        out_logvol = per_batch_logvol[0].unsqueeze(0)
    else:
        pad_vec = (
            pad_embedding.to(device=device, dtype=dtype).view(1, hidden_dim)
            if pad_embedding is not None
            else torch.zeros((1, hidden_dim), device=device, dtype=dtype)
        )
        out_embeds = pad_vec.expand(bsz, max_routed_len, hidden_dim).clone()
        out_labels = (
            torch.full((bsz, max_routed_len), -100, dtype=labels.dtype, device=device)
            if labels is not None
            else None
        )
        out_attn = torch.zeros(
            (bsz, max_routed_len),
            dtype=attention_mask.dtype if attention_mask is not None else torch.long,
            device=device,
        )
        out_tpos = torch.zeros((bsz, max_routed_len), dtype=torch.long, device=device)
        out_mrope = torch.zeros((3, bsz, max_routed_len), dtype=torch.float32, device=device)
        out_logvol = torch.zeros((bsz, max_routed_len), dtype=torch.float32, device=device)

        for b in range(bsz):
            lb = per_batch_embeds[b].shape[0]
            if lb == 0:
                continue
            sl = slice(max_routed_len - lb, max_routed_len) if left_padded else slice(0, lb)
            out_embeds[b, sl] = per_batch_embeds[b]
            out_attn[b, sl] = 1
            out_tpos[b, sl] = torch.from_numpy(per_batch_tpos[b]).to(device=device, dtype=torch.long)
            out_mrope[:, b, sl] = torch.from_numpy(per_batch_mrope[b]).to(device=device, dtype=torch.float32)
            out_logvol[b, sl] = per_batch_logvol[b]
            if out_labels is not None:
                out_labels[b, sl] = per_batch_labels[b]

    pos_4d = torch.cat((out_tpos.float().unsqueeze(0), out_mrope), dim=0)
    rope_deltas_t = torch.tensor(per_batch_deltas, dtype=torch.float32, device=device).unsqueeze(1)

    cu_q = cu_k = None
    max_q = max_k = None
    if is_packed and packed_subseq_lens:
        lens_t = torch.tensor([0] + packed_subseq_lens, dtype=torch.int32, device=device)
        cu_q = lens_t.cumsum(dim=0, dtype=torch.int32)
        cu_k = cu_q
        max_q = int(max(packed_subseq_lens))
        max_k = max_q

    return RoutedBatchPack(
        inputs_embeds=out_embeds,
        labels=out_labels,
        attention_mask=out_attn,
        position_ids_4d=pos_4d,
        text_position_ids=out_tpos,
        token_log_volumes=out_logvol,
        rope_deltas=rope_deltas_t,
        cu_seq_lens_q=cu_q,
        cu_seq_lens_k=cu_k,
        max_length_q=max_q,
        max_length_k=max_k,
    )

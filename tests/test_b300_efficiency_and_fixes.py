"""Regression & Parity Tests for B300 Efficiency (P0) and Runtime Fixes (P1--P4, E5)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from tests.test_modeling_qwen3_5_treeweave import make_mini_qwen3_5_config, make_synthetic_image_batch
from treeweave import TreeWeaveConfig
from treeweave.losses import chunked_topk_kl_distill_loss
from treeweave.modeling_treeweave import (
    Qwen3_5TreeWeaveForConditionalGeneration,
    _has_fused_qwen3_5_varlen_kernels,
)
from treeweave.pooler import TreeWeavePyramid
from treeweave.rollout.cache_topk_logits import _verify_and_log_gpu_pin, cache_dataset_topk
from treeweave.rollout.offline_rollout import _verify_and_log_gpu_pin as _verify_rollout_pin
from treeweave.router import SpatiotemporalTreeRouter
from treeweave.swift_plugin.template import (
    Qwen3_5TreeWeaveTemplate,
    get_topk_alignment_stats,
    reset_topk_alignment_stats,
)
from treeweave.swift_plugin.trainer import rebalance_packing_dataset, repack_balanced_indices
from treeweave.layout import hard_tree_route_layout
from treeweave.volume_bias import (
    install_volume_bias_on_language_model,
    varlen_bounds_context,
    volume_bias_context,
)


# =============================================================================
# P0a: Single-Pass Inner Varlen SDPA Parity & Zero Cross-Segment Leakage
# =============================================================================


@pytest.mark.parametrize("mode", ["key_direction", "exact_logit_bias", "none"])
def test_p0_inner_varlen_sdpa_matches_separate_segments_forward_and_backward(mode: str):
    torch.manual_seed(123)
    cfg = make_mini_qwen3_5_config()
    cfg.text_config._attn_implementation = "sdpa"
    cfg.treeweave = TreeWeaveConfig(
        router_num_heads=4,
        volume_bias_mode=mode,
        use_volume_attn_bias=(mode != "none"),
    ).to_dict()
    model = Qwen3_5TreeWeaveForConditionalGeneration(cfg).train()
    lm = model.model.language_model
    assert getattr(lm, "_treeweave_varlen_sdpa_patched", False) is True

    # Locate the first full-attention layer (`Qwen3_5Attention`)
    attn_layer = next(layer.self_attn for layer in lm.layers if hasattr(layer, "self_attn"))
    rotary = lm.rotary_emb

    # Give non-zero volume bias weights so volume compensation actively affects outputs & grads
    if hasattr(attn_layer, "treeweave_volume_bias"):
        with torch.no_grad():
            attn_layer.treeweave_volume_bias.logit_gamma.fill_(0.7)
            attn_layer.treeweave_volume_bias.key_volume_vec.normal_(std=0.25)

    seg_lens = [7, 11, 5]
    cum_bounds = (0, 7, 18, 23)
    seg_ranges = list(zip(cum_bounds[:-1], cum_bounds[1:]))
    cu_q = torch.tensor(list(cum_bounds), dtype=torch.int32)
    total_len = sum(seg_lens)
    hidden_dim = cfg.text_config.hidden_size

    h_packed = torch.randn(1, total_len, hidden_dim, requires_grad=True)
    lv_packed = torch.rand(1, total_len) * 2.5
    pos_segs = [torch.arange(L, dtype=torch.long).view(1, 1, L).expand(3, 1, L) for L in seg_lens]
    pos_packed = torch.cat(pos_segs, dim=-1)
    cos_p, sin_p = rotary(h_packed, pos_packed)

    # 1) Single-pass packed varlen SDPA forward + backward
    with varlen_bounds_context(cum_bounds), volume_bias_context(lv_packed if mode != "none" else None):
        out_packed, _ = attn_layer(
            hidden_states=h_packed,
            position_embeddings=(cos_p, sin_p),
            attention_mask=None,
            past_key_values=None,
            cu_seq_lens_q=cu_q,
            cu_seq_lens_k=cu_q,
            max_length_q=max(seg_lens),
            max_length_k=max(seg_lens),
        )
    grad_target = torch.randn_like(out_packed)
    (out_packed * grad_target).sum().backward()
    packed_h_grad = h_packed.grad.detach().clone()
    packed_param_grads = {
        n: p.grad.detach().clone()
        for n, p in attn_layer.named_parameters()
        if p.grad is not None
    }

    # 2) Separate per-segment reference forward + backward
    attn_layer.zero_grad()
    ref_outs = []
    ref_h_grads = []
    for (s, e), pos_s in zip(seg_ranges, pos_segs):
        h_s = h_packed.detach()[:, s:e, :].clone().requires_grad_(True)
        lv_s = lv_packed[:, s:e] if mode != "none" else None
        cos_s, sin_s = rotary(h_s, pos_s)
        with varlen_bounds_context(None), volume_bias_context(lv_s):
            o_s, _ = attn_layer(
                hidden_states=h_s,
                position_embeddings=(cos_s, sin_s),
                attention_mask=None,
                past_key_values=None,
            )
        (o_s * grad_target[:, s:e, :]).sum().backward()
        ref_outs.append(o_s.detach())
        ref_h_grads.append(h_s.grad.detach())

    out_ref = torch.cat(ref_outs, dim=1)
    h_grad_ref = torch.cat(ref_h_grads, dim=1)

    torch.testing.assert_close(out_packed.detach(), out_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(packed_h_grad, h_grad_ref, atol=1e-5, rtol=1e-5)
    for n, g_p in packed_param_grads.items():
        g_r = dict(attn_layer.named_parameters())[n].grad
        assert g_r is not None, f"Missing reference grad for {n}"
        torch.testing.assert_close(g_p, g_r, atol=1e-5, rtol=1e-5)

    # 3) Zero cross-segment leakage check: wildly corrupt segment 0, verify segments 1 & 2 stay identical
    with torch.no_grad():
        with varlen_bounds_context(cum_bounds), volume_bias_context(lv_packed if mode != "none" else None):
            out_clean, _ = attn_layer(
                hidden_states=h_packed.detach(),
                position_embeddings=(cos_p, sin_p),
                attention_mask=None,
                past_key_values=None,
                cu_seq_lens_q=cu_q,
                cu_seq_lens_k=cu_q,
                max_length_q=max(seg_lens),
                max_length_k=max(seg_lens),
            )
        h_corrupt = h_packed.detach().clone()
        h_corrupt[:, :7, :] += 100.0
        with varlen_bounds_context(cum_bounds), volume_bias_context(lv_packed if mode != "none" else None):
            out_corrupt, _ = attn_layer(
                hidden_states=h_corrupt,
                position_embeddings=(cos_p, sin_p),
                attention_mask=None,
                past_key_values=None,
                cu_seq_lens_q=cu_q,
                cu_seq_lens_k=cu_q,
                max_length_q=max(seg_lens),
                max_length_k=max(seg_lens),
            )
    torch.testing.assert_close(out_corrupt[:, 7:, :], out_clean[:, 7:, :], atol=1e-6, rtol=1e-6)


def test_p0_linear_attn_strips_volume_kwarg_and_fused_detector_sdpa(monkeypatch):
    cfg = make_mini_qwen3_5_config()
    cfg.text_config._attn_implementation = "sdpa"
    native = Qwen3_5ForConditionalGeneration(cfg)
    lm = native.model.language_model
    install_volume_bias_on_language_model(lm, mode="none")
    assert getattr(lm, "_treeweave_varlen_sdpa_patched", False) is True

    # Ensure linear_attn layers never fail when `treeweave_token_log_volumes` is in kwargs
    lin_layer = next(layer for layer in lm.layers if hasattr(layer, "linear_attn"))
    h = torch.randn(1, 8, cfg.text_config.hidden_size)
    out = lin_layer.linear_attn(h, treeweave_token_log_volumes=torch.zeros(1, 8))
    assert out.shape == h.shape

    # Simulate CUDA + fla + causal_conv1d availability -> `_has_fused_qwen3_5_varlen_kernels` must accept SDPA!
    import importlib.util

    # On CPU model, must return False even if CUDA/fla exist!
    assert _has_fused_qwen3_5_varlen_kernels(lm) is False

    # Simulate a CUDA model + installed fla & causal_conv1d -> SDPA patched LM must return True!
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name in ("fla", "causal_conv1d") else None)
    monkeypatch.setattr(lm, "parameters", lambda recurse=True: iter([]))
    assert _has_fused_qwen3_5_varlen_kernels(lm) is True
    lm._treeweave_varlen_sdpa_patched = False
    assert _has_fused_qwen3_5_varlen_kernels(lm) is False


# =============================================================================
# P0b: Vision- & Segment-Balanced Re-Packing (Eliminates Multi-Rank Stragglers)
# =============================================================================


def test_p0_balanced_repacking_preserves_bijection_and_reduces_cost_skew():
    # 12 heavy videos + 36 medium images + 72 short images
    lengths = [4000] * 12 + [1200] * 36 + [400] * 72
    vis_list = [3600] * 12 + [1024] * 36 + [256] * 72
    sample_indices = list(range(len(lengths)))
    tok_lens = {i: lengths[i] for i in sample_indices}
    vis_costs = {i: vis_list[i] for i in sample_indices}
    packing_length = 8192

    bins, bin_lens, bin_work_costs = repack_balanced_indices(
        sample_indices, tok_lens, vis_costs, packing_length
    )
    flat = [i for b in bins for i in b]
    assert sorted(flat) == sample_indices, "Re-packing must be an exact bijection over all samples"
    assert len(bins) == len(bin_lens) == len(bin_work_costs)

    for b, bl in zip(bins, bin_lens):
        assert bl == sum(lengths[i] for i in b)
        assert bl <= packing_length

    # Balanced bin visual-cost spread must be dramatically tighter than clustering all 12 videos into 6 bins!
    bin_vis = [sum(vis_costs[i] for i in b) for b in bins]
    assert max(bin_vis) / max(min(bin_vis), 1) < 1.8

    # Test `rebalance_packing_dataset` in-place mutation on a mock PackingDataset
    class _MockBase:
        def __init__(self, lens):
            self._lens = lens

        def __len__(self):
            return len(self._lens)

        def __getitem__(self, idx):
            return {"length": self._lens[idx]}

    class _MockPackingDataset:
        def __init__(self, lens, pack_len):
            self.dataset = _MockBase(lens)
            self.packing_length = pack_len
            self.packed_idx = [[i] for i in range(len(lens))]
            self.packed_length = list(lens)

        def __len__(self):
            return len(self.packed_idx)

    mock_ds = _MockPackingDataset(lengths, packing_length)
    out_costs = rebalance_packing_dataset(mock_ds)
    assert out_costs is not None
    assert len(out_costs) == len(mock_ds.packed_idx) == len(mock_ds.packed_length)
    assert sorted(i for b in mock_ds.packed_idx for i in b) == sample_indices


# =============================================================================
# P1: Subprocess GPU Pin Verification
# =============================================================================


def test_p1_verify_and_log_gpu_pin_cpu_and_subprocess_multiprocess_cache():
    # On CPU / non-CUDA machines, `_verify_and_log_gpu_pin` must cleanly report CPU without crashing
    _verify_and_log_gpu_pin(0, "cpu")
    _verify_rollout_pin(0, "cpu")

    # End-to-end test of `subprocess.Popen` 2-worker sharding in `cache_dataset_topk(..., gpus=["cpu", "cpu"])`!
    from PIL import Image
    from safetensors import safe_open
    from tests.test_ms_swift_integration import _build_local_mini_qwen3_5_checkpoint_with_processor

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        native_dir = root / "native-qwen35"
        _build_local_mini_qwen3_5_checkpoint_with_processor(native_dir)

        img = root / "img.png"
        Image.new("RGB", (128, 128), color=(80, 140, 220)).save(img)

        rollout_jsonl = root / "rollouts.jsonl"
        rows = [
            {
                "messages": [
                    {"role": "user", "content": f"<image> Sample {i}"},
                    {"role": "assistant", "content": f"Answer {i} ok"},
                ],
                "images": [str(img)],
            }
            for i in range(4)
        ]
        rollout_jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        out_jsonl = root / "topk_out.jsonl"
        cache_dir = root / "topk_shards"
        cache_dataset_topk(
            teacher_model_path=str(native_dir),
            rollout_jsonl=rollout_jsonl,
            output_jsonl=out_jsonl,
            cache_dir=cache_dir,
            topk=8,
            pixel_factor=128,
            device="cpu",
            gpus=["cpu", "cpu"],
            batch_size=2,
            pack_sequences=True,
        )

        out_lines = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(out_lines) == 4
        for rec in out_lines:
            st_path = Path(rec["topk_cache_path"])
            assert st_path.is_file()
            tensors = load_file(str(st_path))
            assert {"topk_indices", "topk_logprobs", "target_labels", "source_labels"} <= set(tensors.keys())
            src_lbl = tensors["source_labels"]
            assert src_lbl.ndim == 1 and int((src_lbl[1:] != -100).sum().item()) == int(tensors["target_labels"].shape[0])
            with safe_open(str(st_path), framework="pt", device="cpu") as sf:
                meta = sf.metadata() or {}
                assert meta.get("loss_scale") == "default+ignore_empty_think"


# =============================================================================
# P2: Offline Top-K Valid Mask & Legacy Pos-0 BPE Boundary Alignment
# =============================================================================


@pytest.mark.parametrize("reduction", ["token_mean", "sample_sqrt_len"])
def test_p2_chunked_topk_kl_valid_mask_ignores_masked_tokens_and_zeroes_grad(reduction: str):
    torch.manual_seed(7)
    n_tok, hidden_dim, vocab, k = 8, 16, 64, 8
    lm_head = torch.nn.Linear(hidden_dim, vocab, bias=False)
    s_hid = torch.randn(n_tok, hidden_dim, requires_grad=True)
    sample_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    t_idx = torch.randint(0, vocab, (n_tok, k))
    t_lp = torch.randn(n_tok, k)
    valid_mask = torch.tensor([False, True, True, True, False, True, True, True], dtype=torch.bool)

    # Inject NaN at masked positions to prove `torch.where` never leaks NaN into loss or grads!
    t_lp_poisoned = t_lp.clone()
    t_lp_poisoned[0] = float("nan")
    t_lp_poisoned[4] = float("nan")

    loss_masked = chunked_topk_kl_distill_loss(
        s_hid,
        lm_head,
        sample_ids=sample_ids,
        teacher_topk_indices=t_idx,
        teacher_topk_logprobs=t_lp_poisoned,
        teacher_topk_valid_mask=valid_mask,
        topk=k,
        reduction=reduction,
        chunk_size=4,
    )
    assert torch.isfinite(loss_masked)
    loss_masked.backward()
    grad_masked = s_hid.grad.detach().clone()

    # Masked positions MUST receive strictly zero gradient
    assert torch.all(grad_masked[0] == 0.0)
    assert torch.all(grad_masked[4] == 0.0)
    assert torch.any(grad_masked[1] != 0.0)

    # Compare against computing KL directly on ONLY the valid subset
    s_hid_valid = s_hid.detach()[valid_mask].clone().requires_grad_(True)
    loss_ref = chunked_topk_kl_distill_loss(
        s_hid_valid,
        lm_head,
        sample_ids=sample_ids[valid_mask],
        teacher_topk_indices=t_idx[valid_mask],
        teacher_topk_logprobs=t_lp[valid_mask],
        topk=k,
        reduction=reduction,
        chunk_size=4,
    )
    loss_ref.backward()
    torch.testing.assert_close(loss_masked.detach(), loss_ref.detach(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad_masked[valid_mask], s_hid_valid.grad.detach(), atol=1e-5, rtol=1e-5)

    # All-False valid mask must return exact 0.0 with finite zero gradient
    s_hid_zero = s_hid.detach().clone().requires_grad_(True)
    loss_empty = chunked_topk_kl_distill_loss(
        s_hid_zero,
        lm_head,
        sample_ids=sample_ids,
        teacher_topk_indices=t_idx,
        teacher_topk_logprobs=t_lp,
        teacher_topk_valid_mask=torch.zeros(n_tok, dtype=torch.bool),
        topk=k,
        reduction=reduction,
    )
    loss_empty.backward()
    assert float(loss_empty.item()) == 0.0
    assert torch.all(s_hid_zero.grad == 0.0)


def test_p2_template_alignment_handles_exact_and_legacy_pos0_bpe_and_rejects_corrupt():
    reset_topk_alignment_stats()
    with tempfile.TemporaryDirectory() as tmp:
        shard_path = Path(tmp) / "legacy_shard.safetensors"
        # Legacy `loss_scale='default'` cache: [<think>, \n\n, </think>, \n\n, 561, 100, 101, 102, <|im_end|>]
        cached_targets = torch.tensor([151667, 271, 151668, 271, 561, 100, 101, 102, 151645], dtype=torch.int32)
        n = cached_targets.shape[0]
        save_file(
            {
                "topk_indices": torch.arange(n * 4, dtype=torch.int32).view(n, 4),
                "topk_logprobs": torch.zeros((n, 4), dtype=torch.bfloat16),
                "target_labels": cached_targets,
            },
            str(shard_path),
        )

        # Case 1: Exact match (e.g. new `default+ignore_empty_think` shard or non-merged first token)
        exact_res = Qwen3_5TreeWeaveTemplate._load_offline_topk_list(
            [str(shard_path)],
            expected_counts=[5],
            truncation_sides=["right"],
            expected_targets=[[561, 100, 101, 102, 151645]],
        )
        assert exact_res is not None
        assert "teacher_topk_valid_mask" not in exact_res
        assert exact_res["_teacher_target_labels"].tolist() == [561, 100, 101, 102, 151645]

        # Case 2: Legacy pos-0 BPE boundary split (`561` vs `760`, tokens `[1:]` identical)
        pos0_res = Qwen3_5TreeWeaveTemplate._load_offline_topk_list(
            [str(shard_path)],
            expected_counts=[5],
            truncation_sides=["right"],
            expected_targets=[[760, 100, 101, 102, 151645]],
        )
        assert pos0_res is not None
        assert "teacher_topk_valid_mask" in pos0_res
        assert pos0_res["teacher_topk_valid_mask"].tolist() == [False, True, True, True, True]
        assert pos0_res["_teacher_target_labels"][1:].tolist() == [100, 101, 102, 151645]

        # Case 3: Right-truncated legacy pos-0 BPE boundary split
        trunc_res = Qwen3_5TreeWeaveTemplate._load_offline_topk_list(
            [str(shard_path)],
            expected_counts=[3],
            truncation_sides=["right"],
            expected_targets=[[760, 100, 101]],
        )
        assert trunc_res is not None
        assert trunc_res["teacher_topk_valid_mask"].tolist() == [False, True, True]

        # Case 4: Genuine misalignment (2 tokens differ or pos-1 differs) MUST still raise ValueError
        with pytest.raises(ValueError, match="do not match Student supervised"):
            Qwen3_5TreeWeaveTemplate._load_offline_topk_list(
                [str(shard_path)],
                expected_counts=[5],
                truncation_sides=["right"],
                expected_targets=[[760, 999, 101, 102, 151645]],
            )

        stats = get_topk_alignment_stats()
        assert stats["exact"] == 1
        assert stats["pos0_bpe_boundary"] == 2
        assert stats["mismatched"] == 1


def test_p2_forward_distill_accepts_teacher_topk_valid_mask_end_to_end():
    torch.manual_seed(0)
    cfg = make_mini_qwen3_5_config()
    model = Qwen3_5TreeWeaveForConditionalGeneration(cfg)
    batch = make_synthetic_image_batch(cfg)
    n_act = 3  # labels has 3 supervised tokens `[30, 31, 32]`
    k = 8
    t_idx = torch.randint(0, cfg.text_config.vocab_size, (n_act, k))
    t_lp = torch.randn(n_act, k)
    valid_mask = torch.tensor([False, True, True], dtype=torch.bool)

    for stage in (1, 2, 3):
        model.configure_stage_trainability(stage)
        out = model(
            distill_stage=stage,
            teacher_topk_indices=t_idx,
            teacher_topk_logprobs=t_lp,
            teacher_topk_valid_mask=valid_mask,
            distill_topk=k,
            **batch,
        )
        assert torch.isfinite(out.loss)
        out.loss.backward()
        model.zero_grad()


# =============================================================================
# P3: Pooler BF16 / Autocast `index_copy_` Dtype Safety
# =============================================================================


def test_p3_pooler_bf16_and_autocast_index_copy_no_dtype_mismatch():
    torch.manual_seed(11)
    pyramid_mod = TreeWeavePyramid(
        vision_hidden_size=32,
        llm_hidden_size=64,
        fine=2,
    ).to(dtype=torch.bfloat16)

    # Video raster (T=4, H=8, W=8, d=32) -> post-merger native grid (4, 4, 4), bh=2, bw=2
    grid = (4, 8, 8)
    x_raster = torch.randn(4, 8, 8, 32, dtype=torch.bfloat16, requires_grad=True)

    def _dummy_merger(z: torch.Tensor) -> torch.Tensor:
        return z.view(-1, 4, 32).mean(dim=1).repeat(1, 2)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        pyr = pyramid_mod.build_item_pyramid(x_raster, max_spatial_level=2, max_temporal_level=2)
        assert len(pyr) == 9
        for key, tensor in pyr.items():
            assert tensor.dtype == torch.bfloat16, f"Pyramid scale {key} had dtype {tensor.dtype}, expected bfloat16"

        # Build a mixed-scale hard layout exercising multiple spatial & temporal levels via `index_copy_`
        import numpy as np

        p_s1 = np.array([[[0.8, 0.2], [0.9, 0.1]]] * 4, dtype=np.float32)
        p_s2 = np.full((4, 1, 1), 0.3, dtype=np.float32)
        p_t1 = np.full((2, 2, 2), 0.7, dtype=np.float32)
        p_t2 = np.full((1, 2, 2), 0.3, dtype=np.float32)
        layout = hard_tree_route_layout(
            grid, p_s1, p_s2, p_t1, p_t2, theta_s1=0.5, theta_s2=0.5, theta_t1=0.5, theta_t2=0.5
        )
        proj_list = pyramid_mod.project_selected_nodes([pyr], [layout], merger=_dummy_merger)
        proj = proj_list[0]
        assert proj.dtype == torch.bfloat16
        assert proj.shape == (layout.output_length, 64)
        proj.float().sum().backward()

    assert x_raster.grad is not None and torch.isfinite(x_raster.grad).all()


# =============================================================================
# P4: Router `_binary_compress_prob` BF16 Weight Cast Safety
# =============================================================================


def test_p4_router_bf16_cast_heads_forward_and_backward():
    torch.manual_seed(19)
    router = SpatiotemporalTreeRouter(
        hidden_size=32,
        num_heads=4,  # head_dim = 8 (divisible by 4 for 2D axial RoPE)
        mlp_ratio=2,
        use_temporal_attn=True,
    )
    # Simulate DeepSpeed/Accelerate casting ALL router modules (including head_s1..t2) to BF16!
    router = router.to(dtype=torch.bfloat16)
    assert router.head_s1[1].weight.dtype == torch.bfloat16

    grid = (4, 8, 8)  # bh=2, bw=2 -> n_q = 4 * 2 * 2 = 16
    queries = torch.randn(16, 32, dtype=torch.bfloat16, requires_grad=True)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = router.score_queries(
            queries,
            [grid],
            max_spatial_level=2,
            max_temporal_level=2,
        )
        for name, p in [("s1", out.p_s1[0]), ("s2", out.p_s2[0]), ("t1", out.p_t1[0]), ("t2", out.p_t2[0])]:
            assert p.dtype == torch.float32, f"{name} must stay FP32"
            assert torch.all((p > 0.0) & (p < 1.0))

        loss = out.p_s1[0].mean() + out.p_s2[0].mean() + out.p_t1[0].mean() + out.p_t2[0].mean()
        loss.backward()

    assert queries.grad is not None and torch.isfinite(queries.grad).all()
    assert router.head_s1[1].weight.grad is not None and torch.isfinite(router.head_s1[1].weight.grad).all()

    # Also test full `Qwen3_5TreeWeaveForConditionalGeneration` Stage 2 forward+backward after `.to(bfloat16)`!
    cfg = make_mini_qwen3_5_config()
    model = Qwen3_5TreeWeaveForConditionalGeneration(cfg).to(dtype=torch.bfloat16)
    model.configure_stage_trainability(2)
    batch = make_synthetic_image_batch(cfg)
    batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        res = model(distill_stage=2, distill_topk=8, **batch)
        assert torch.isfinite(res.loss)
        res.loss.backward()

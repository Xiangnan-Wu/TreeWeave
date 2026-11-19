"""Deep Integration Tests on Real `Qwen3_5ForConditionalGeneration` Hybrid (GatedDeltaNet + FullAttn + MRoPE) Models."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from treeweave.modeling_treeweave import Qwen3_5TreeWeaveForConditionalGeneration


def make_mini_qwen3_5_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        vision_config={
            "depth": 4,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,       # head_dim = 8 (divisible by 4 for 2D axial RoPE!)
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 64,
            "initializer_range": 0.02,
            "rope_parameters": {"rope_type": "axial", "rope_theta": 10000.0},
        },
        text_config={
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,  # 3 GatedDeltaNet linear_attention + 1 full_attention!
            "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "tie_word_embeddings": True,
            "pad_token_id": 0,
            "eos_token_id": 2,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.5,  # rotary_dim = 16, non_rotary_dim = 16
                "mrope_section": [3, 3, 2],
            },
        },
        image_token_id=200,
        video_token_id=201,
        vision_start_token_id=202,
        vision_end_token_id=203,
        tie_word_embeddings=True,
    )


def make_synthetic_image_batch(cfg: Qwen3_5Config, grid=(1, 8, 8)):
    t, h, w = grid
    n_vis = t * (h // 2) * (w // 2)  # 16 native tokens
    ids = [10, 11, cfg.vision_start_token_id] + [cfg.image_token_id] * n_vis + [cfg.vision_end_token_id, 20, 21, 30, 31, 32]
    mm = [0, 0, 0] + [1] * n_vis + [0, 0, 0, 0, 0, 0]
    labels = [-100] * (len(ids) - 3) + [30, 31, 32]
    patch_dim = 3 * 2 * 16 * 16
    pixel_values = torch.randn(t * h * w, patch_dim)
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "mm_token_type_ids": torch.tensor([mm], dtype=torch.long),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "pixel_values": pixel_values,
        "image_grid_thw": torch.tensor([list(grid)], dtype=torch.long),
    }


def make_synthetic_video_batch(cfg: Qwen3_5Config, grid=(4, 8, 8)):
    t, h, w = grid
    per_slice = (h // 2) * (w // 2)  # 16 native tokens/slice
    ids = [10, 11]
    mm = [0, 0]
    for tau in range(t):
        # <timestamp> <|vision_start|> [<|video_pad|> * 16] <|vision_end|>
        ids.extend([40 + tau, cfg.vision_start_token_id] + [cfg.video_token_id] * per_slice + [cfg.vision_end_token_id])
        mm.extend([0, 0] + [2] * per_slice + [0])
    ids.extend([25, 26, 51, 52, 53, 54])
    mm.extend([0, 0, 0, 0, 0, 0])
    labels = [-100] * (len(ids) - 4) + [51, 52, 53, 54]
    patch_dim = 3 * 2 * 16 * 16
    pixel_values_videos = torch.randn(t * h * w, patch_dim)
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "mm_token_type_ids": torch.tensor([mm], dtype=torch.long),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "pixel_values_videos": pixel_values_videos,
        "video_grid_thw": torch.tensor([list(grid)], dtype=torch.long),
    }


def test_untrained_init_matches_native_qwen3_5_bit_exact_on_image_and_video():
    torch.manual_seed(0)
    cfg = make_mini_qwen3_5_config()
    native = Qwen3_5ForConditionalGeneration(cfg).eval()
    tw = Qwen3_5TreeWeaveForConditionalGeneration(cfg).eval()
    # Copy native weights into TreeWeave (missing keys are only pyramid/router/volume_bias!)
    missing, unexpected = tw.load_state_dict(native.state_dict(), strict=False)
    assert not unexpected
    assert all(("pyramid" in k or "router" in k or "treeweave_volume_bias" in k) for k in missing)

    for batch in (make_synthetic_image_batch(cfg), make_synthetic_video_batch(cfg)):
        b_nat = {k: v for k, v in batch.items() if k != "labels"}
        b_tw = {k: v for k, v in batch.items() if k != "labels"}
        with torch.no_grad():
            out_nat = native(**b_nat)
            out_tw = tw(**b_tw)
        torch.testing.assert_close(out_tw.logits, out_nat.logits, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_three_stage_distillation_forward_backward_and_ddp_grad_completeness(stage):
    """Test Stage 1, 2, 3 on BOTH image-only (`T=1`) AND packed image+video (`T=1` + `T=4`) sequences."""
    torch.manual_seed(10 + stage)
    cfg = make_mini_qwen3_5_config()
    model = Qwen3_5TreeWeaveForConditionalGeneration(cfg)
    # Set aggressive threshold in Stage 3 so compression actually exercises all scales!
    if stage == 3:
        model.tw_config.theta_s1 = 0.05
        model.tw_config.theta_s2 = 0.05
        model.tw_config.theta_t1 = 0.05
        model.tw_config.theta_t2 = 0.05
        model.sync_config_to_hf()

    model.configure_stage_trainability(stage)
    model.enforce_frozen_eval_mode(stage)

    # 1) Image-only batch (`T=1`): verify EVERY trainable param (even temporal pooler/router!) gets a non-None grad!
    img_batch = make_synthetic_image_batch(cfg, grid=(1, 8, 8))
    model.zero_grad(set_to_none=True)
    out_img = model(distill_stage=stage, distill_topk=32, **img_batch)
    assert torch.isfinite(out_img.loss)
    out_img.loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Stage {stage} image-only batch left trainable param {name} with None grad (DDP deadlock risk)!"
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"
        else:
            assert p.grad is None, f"Frozen param {name} unexpectedly received gradient in Stage {stage}!"

    # 2) Packed multi-sample sequence combining 1 image sample (`T=1`) + 1 4-slice video sample (`T=4`)!
    vid_batch = make_synthetic_video_batch(cfg, grid=(4, 8, 8))
    l1 = img_batch["input_ids"].shape[1]
    l2 = vid_batch["input_ids"].shape[1]
    packed_batch = {
        "input_ids": torch.cat((img_batch["input_ids"], vid_batch["input_ids"]), dim=1),
        "labels": torch.cat((img_batch["labels"], vid_batch["labels"]), dim=1),
        "mm_token_type_ids": torch.cat((img_batch["mm_token_type_ids"], vid_batch["mm_token_type_ids"]), dim=1),
        "text_position_ids": torch.cat((torch.arange(l1), torch.arange(l2))).unsqueeze(0),
        "cu_seq_lens_q": torch.tensor([0, l1, l1 + l2], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, l1, l1 + l2], dtype=torch.int32),
        "max_length_q": max(l1, l2),
        "max_length_k": max(l1, l2),
        "pixel_values": img_batch["pixel_values"],
        "image_grid_thw": img_batch["image_grid_thw"],
        "pixel_values_videos": vid_batch["pixel_values_videos"],
        "video_grid_thw": vid_batch["video_grid_thw"],
    }
    model.zero_grad(set_to_none=True)
    out_packed = model(distill_stage=stage, distill_topk=32, **packed_batch)
    assert torch.isfinite(out_packed.loss)
    out_packed.loss.backward()
    if stage in (1, 3):
        assert out_packed.treeweave_metrics["actual_token_savings"] > 0.0
    else:
        assert 0.0 < out_packed.treeweave_metrics["expected_retention"] < 1.0


@pytest.mark.parametrize("vol_mode", ["key_direction", "exact_logit_bias"])
def test_volume_attention_bias_modes_and_kv_cached_generate(vol_mode):
    torch.manual_seed(99)
    cfg = make_mini_qwen3_5_config()
    model = Qwen3_5TreeWeaveForConditionalGeneration(cfg).eval()
    # Force compression (theta = 0.0 -> all blocks merge to s=2, t=2 => 64->1, ln(M) = ln(64) > 0!)
    model.tw_config.theta_s1 = 0.0
    model.tw_config.theta_s2 = 0.0
    model.tw_config.theta_t1 = 0.0
    model.tw_config.theta_t2 = 0.0
    model.tw_config.use_volume_attn_bias = True
    model.tw_config.volume_bias_mode = vol_mode
    model.sync_config_to_hf()

    # Give non-zero key_volume_vec so `key_direction` visibly shifts logits relative to `none`
    for m in model.model.language_model.modules():
        if hasattr(m, "treeweave_volume_bias"):
            nn.init.normal_(m.treeweave_volume_bias.key_volume_vec, std=0.2)

    vid_batch = make_synthetic_video_batch(cfg, grid=(4, 8, 8))
    vid_inputs = {k: v for k, v in vid_batch.items() if k != "labels"}

    with torch.no_grad():
        logits_with_bias = model(**vid_inputs).logits
        model.tw_config.use_volume_attn_bias = False
        model.sync_config_to_hf()
        logits_no_bias = model(**vid_inputs).logits
        model.tw_config.use_volume_attn_bias = True
        model.tw_config.volume_bias_mode = vol_mode
        model.sync_config_to_hf()

    # Compressed sequence length: 64 visual tokens became 1 visual token (savings = 63 tokens)!
    assert logits_with_bias.shape[1] == vid_inputs["input_ids"].shape[1] - 63
    assert (logits_with_bias - logits_no_bias).abs().max().item() > 1e-4

    # Test multi-step KV-cached `.generate()` on BOTH compressed video (with 0-token slices!) and compressed image,
    # explicitly asserting that `.generate()` ACTUALLY invoked TreeWeave routing & compressed the KV cache!
    model.model.last_route_layouts = {"image": [], "video": []}
    gen_vid = model.generate(**vid_inputs, max_new_tokens=5, do_sample=False, return_dict_in_generate=True)
    assert gen_vid.sequences.shape == (1, vid_inputs["input_ids"].shape[1] + 5)
    assert len(model.model.last_route_layouts["video"]) == 1
    assert model.model.last_route_layouts["video"][0].output_length == 1
    assert gen_vid.past_key_values.get_seq_length() == vid_inputs["input_ids"].shape[1] - 63 + 4

    img_batch = make_synthetic_image_batch(cfg, grid=(1, 8, 8))
    img_inputs = {k: v for k, v in img_batch.items() if k != "labels"}
    model.model.last_route_layouts = {"image": [], "video": []}
    gen_img = model.generate(**img_inputs, max_new_tokens=5, do_sample=False, return_dict_in_generate=True)
    assert gen_img.sequences.shape == (1, img_inputs["input_ids"].shape[1] + 5)
    assert len(model.model.last_route_layouts["image"]) == 1
    assert model.model.last_route_layouts["image"][0].output_length == 1
    assert gen_img.past_key_values.get_seq_length() == img_inputs["input_ids"].shape[1] - 15 + 4

    # Also verify Q3 ablation routing modes ("shuffled" and "heuristic") execute cleanly
    with torch.no_grad():
        out_shuf = model(**img_inputs, routing_mode="shuffled", shuffle_seed=7)
        out_heur = model(**img_inputs, routing_mode="heuristic")
    assert torch.isfinite(out_shuf.logits).all()
    assert torch.isfinite(out_heur.logits).all()


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_gradient_checkpointing_parity_and_no_warnings_across_stages(stage):
    """Verify LLM activation checkpointing stays active in Stages 1--3 with exact gradient parity & zero warnings."""
    import warnings
    import numpy as np

    torch.manual_seed(2026 + stage)
    cfg = make_mini_qwen3_5_config()
    model = Qwen3_5TreeWeaveForConditionalGeneration(cfg)
    if stage == 3:
        model.tw_config.theta_s1 = 0.05
        model.tw_config.theta_t1 = 0.05
        model.sync_config_to_hf()

    vid_batch = make_synthetic_video_batch(cfg, grid=(4, 8, 8))

    # 1. Baseline uncheckpointed forward + backward
    model.gradient_checkpointing_disable()
    model.configure_stage_trainability(stage)
    model.enforce_frozen_eval_mode(stage)
    model.zero_grad(set_to_none=True)
    out_ref = model(distill_stage=stage, distill_topk=32, stochastic_rng=np.random.default_rng(123), **vid_batch)
    out_ref.loss.backward()
    ref_grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    # 2. Checkpointed forward + backward (ensure LLM layers are actively checkpointing even in Stage 1 & Stage 2!)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enforce_frozen_eval_mode(stage)
    assert not model.model.visual.blocks.training
    assert all(layer.gradient_checkpointing and layer.training for layer in model.model.language_model.layers)

    model.zero_grad(set_to_none=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_ckpt = model(distill_stage=stage, distill_topk=32, stochastic_rng=np.random.default_rng(123), **vid_batch)
        out_ckpt.loss.backward()

    ckpt_warnings = [
        str(w.message) for w in caught if "None of the inputs have requires_grad=True" in str(w.message)
    ]
    assert ckpt_warnings == [], f"Unexpected checkpoint warnings in stage {stage}: {ckpt_warnings}"
    torch.testing.assert_close(out_ckpt.loss, out_ref.loss, atol=1e-6, rtol=1e-6)
    for name, p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.grad, ref_grads[name], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("vol_mode", ["key_direction", "exact_logit_bias"])
def test_batched_unequal_compression_and_left_padded_generate_matches_single_item(vol_mode):
    """Verify B=2 batched `.generate()` with left-padding + unequal visual compression matches B=1 per-item bit-for-bit."""
    torch.manual_seed(7)
    cfg = make_mini_qwen3_5_config()
    model = Qwen3_5TreeWeaveForConditionalGeneration(cfg).eval()
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.4)
    model.visual.router.bias_s1.data.zero_()
    model.visual.router.bias_s2.data.zero_()

    pv0 = torch.randn(64, 3 * 2 * 16 * 16)
    pv1 = torch.randn(64, 3 * 2 * 16 * 16)
    enc = model.visual.forward_vit_with_taps(torch.cat([pv0, pv1], 0), torch.tensor([[1, 8, 8], [1, 8, 8]]))
    r = model.visual.router.score_queries(enc.router_query, enc.grid_thw_list)
    model.tw_config.theta_s1 = 0.5 * (r.p_s1[0].max().item() + r.p_s1[1].max().item())
    model.tw_config.theta_s2 = 0.99
    model.tw_config.use_volume_attn_bias = True
    model.tw_config.volume_bias_mode = vol_mode
    model.sync_config_to_hf()

    for m in model.model.language_model.modules():
        if hasattr(m, "treeweave_volume_bias"):
            nn.init.normal_(m.treeweave_volume_bias.key_volume_vec, std=0.2)
            m.treeweave_volume_bias.logit_gamma.data.fill_(0.8)

    n_vis = 16
    ids0_unpad = [10, 11, cfg.vision_start_token_id] + [cfg.image_token_id] * n_vis + [cfg.vision_end_token_id, 20, 21]
    mm0_unpad = [0, 0, 0] + [1] * n_vis + [0, 0, 0]
    ids1_unpad = [14, 15, 16, 12, 13, cfg.vision_start_token_id] + [cfg.image_token_id] * n_vis + [cfg.vision_end_token_id, 22, 23]
    mm1_unpad = [0] * 6 + [1] * n_vis + [0, 0, 0]

    ids0_pad = [0, 0, 0] + ids0_unpad
    attn0_pad = [0, 0, 0] + [1] * len(ids0_unpad)
    mm0_pad = [0, 0, 0] + mm0_unpad
    attn1_pad = [1] * len(ids1_unpad)

    with torch.no_grad():
        g0 = model.generate(
            input_ids=torch.tensor([ids0_unpad]),
            attention_mask=torch.ones((1, len(ids0_unpad)), dtype=torch.long),
            mm_token_type_ids=torch.tensor([mm0_unpad]),
            pixel_values=pv0,
            image_grid_thw=torch.tensor([[1, 8, 8]]),
            max_new_tokens=5,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        g1 = model.generate(
            input_ids=torch.tensor([ids1_unpad]),
            attention_mask=torch.ones((1, len(ids1_unpad)), dtype=torch.long),
            mm_token_type_ids=torch.tensor([mm1_unpad]),
            pixel_values=pv1,
            image_grid_thw=torch.tensor([[1, 8, 8]]),
            max_new_tokens=5,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        gb2 = model.generate(
            input_ids=torch.tensor([ids0_pad, ids1_unpad]),
            attention_mask=torch.tensor([attn0_pad, attn1_pad], dtype=torch.long),
            mm_token_type_ids=torch.tensor([mm0_pad, mm1_unpad]),
            pixel_values=torch.cat([pv0, pv1], 0),
            image_grid_thw=torch.tensor([[1, 8, 8], [1, 8, 8]]),
            max_new_tokens=5,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    for step in range(5):
        torch.testing.assert_close(gb2.scores[step][0:1], g0.scores[step], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(gb2.scores[step][1:2], g1.scores[step], atol=1e-4, rtol=1e-4)
    assert torch.equal(gb2.sequences[0:1, -5:], g0.sequences[:, -5:])
    assert torch.equal(gb2.sequences[1:2, -5:], g1.sequences[:, -5:])


def test_interleaved_multi_image_and_video_conversation_and_no_dead_volume_params():
    """Stress-test `<image> ... <video> ... <image>` interleaved prompts & active-only volume bias grad flags."""
    torch.manual_seed(19)
    cfg = make_mini_qwen3_5_config()
    native = Qwen3_5ForConditionalGeneration(cfg).eval()
    tw = Qwen3_5TreeWeaveForConditionalGeneration(cfg).eval()
    tw.load_state_dict(native.state_dict(), strict=False)

    # Check zero dead trainable params across all volume bias modes in Stages 1 & 3
    for v_enable, v_mode in [(True, "key_direction"), (True, "exact_logit_bias"), (False, "key_direction")]:
        tw.tw_config.use_volume_attn_bias = v_enable
        tw.tw_config.volume_bias_mode = v_mode
        tw.sync_config_to_hf()
        for stage in (1, 3):
            tw.configure_stage_trainability(stage)
            for m in tw.model.language_model.modules():
                if hasattr(m, "treeweave_volume_bias"):
                    vb = m.treeweave_volume_bias
                    assert vb.key_volume_vec.requires_grad == bool(v_enable and v_mode == "key_direction")
                    assert vb.logit_gamma.requires_grad == bool(v_enable and v_mode == "exact_logit_bias")

    # Build `<image 0> text <video 0 (4 slices)> text <image 1>` interleaved prompt!
    tw.eval()
    ids = [10, cfg.vision_start_token_id] + [cfg.image_token_id] * 16 + [cfg.vision_end_token_id, 11]
    mm = [0, 0] + [1] * 16 + [0, 0]
    for tau in range(4):
        ids.extend([40 + tau, cfg.vision_start_token_id] + [cfg.video_token_id] * 16 + [cfg.vision_end_token_id])
        mm.extend([0, 0] + [2] * 16 + [0])
    ids.extend([12, cfg.vision_start_token_id] + [cfg.image_token_id] * 16 + [cfg.vision_end_token_id, 30, 31])
    mm.extend([0, 0] + [1] * 16 + [0, 0, 0])

    patch_dim = 3 * 2 * 16 * 16
    batch = {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "mm_token_type_ids": torch.tensor([mm], dtype=torch.long),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        "pixel_values": torch.randn(128, patch_dim),
        "image_grid_thw": torch.tensor([[1, 8, 8], [1, 8, 8]], dtype=torch.long),
        "pixel_values_videos": torch.randn(256, patch_dim),
        "video_grid_thw": torch.tensor([[4, 8, 8]], dtype=torch.long),
    }

    # 1) At `theta = 0.5` with untrained `p_init = 0.1`, matches native Qwen3.5 bit-exact!
    with torch.no_grad():
        torch.testing.assert_close(tw(**batch).logits, native(**batch).logits, atol=1e-5, rtol=1e-5)

    # 2) At `theta = 0.0` (maximum compression: 16->1 + 64->1 + 16->1 = 93 tokens saved!), `.generate()` runs cleanly!
    tw.tw_config.theta_s1 = 0.0
    tw.tw_config.theta_s2 = 0.0
    tw.tw_config.theta_t1 = 0.0
    tw.tw_config.theta_t2 = 0.0
    tw.sync_config_to_hf()
    gen = tw.generate(**batch, max_new_tokens=4, do_sample=False, return_dict_in_generate=True)
    assert gen.sequences.shape == (1, len(ids) + 4)
    assert gen.past_key_values.get_seq_length() == len(ids) - 93 + 3

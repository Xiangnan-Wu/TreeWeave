"""Unit & Parity Tests for TreeWeave Disaggregated SGLang Serving (`treeweave/serving/`).

Covers:
1. `python -m treeweave.serving.serve --dry-run` & `prepare_serving_checkpoint` across `128px` (`s<=2`)
   and `64px` (`s<=1`), multi-encoder + TP1/TP2 language roles, port checks, and process-group cleanup.
2. Placeholder shrinking, run splitting, and `0`-token folded video slices (`frame_lengths[tau] == 0`).
3. Exact 3D MRoPE (`POSITION_SCALE = 2`, `inv_freq / 2`), `rope_deltas`, rotary `(cos, sin)`, and
   per-token `log_volumes` parity against `Qwen3_5TreeWeaveForConditionalGeneration` across all 9 scales.
4. Language-side `TreeWeaveAttentionVolumeAdapter` SGLang `RadixAttention` wrapper (`key_direction`
   post-`k_norm` non-rotary key offset) across full prefill, chunked prefill, decode, and `TP=2`.
5. Live SGLang package entrypoints (`pytest.importorskip("sglang")` when `sglang` is not installed).
"""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from tests.test_checkpoint_and_rollout import _write_synthetic_native_ckpt
from tests.test_modeling_qwen3_5_treeweave import make_mini_qwen3_5_config
from treeweave import ARCHITECTURE, POSITION_SCALE, ROUTE_THRESHOLD_ENV
from treeweave.layout import build_routed_sequence_batch, hard_tree_route_layout, sample_stochastic_tree_layout
from treeweave.modeling_treeweave import Qwen3_5TreeWeaveForConditionalGeneration
from treeweave.serving import (
    BUDGET_SHIFT_ENV,
    PIXEL_FACTOR_ENV,
    TreeRouteItem,
    TreeRoutePayload,
    build_routed_input_ids_and_runs,
    build_sglang_extend_log_volumes,
    config_fingerprint,
    initialize_treeweave_checkpoint,
    prepare_serving_checkpoint,
    tree_routed_mrope_positions,
)
from treeweave.serving.serve import Role, build_roles, parser, run, stop_children
from treeweave.serving.sglang_volume_patch import (
    finalize_sglang_volume_bias_loaded_names,
    install_sglang_volume_bias_on_causal_lm,
    load_sglang_volume_bias_tensor,
    sglang_volume_bias_context,
)


def test_prepare_and_dry_run_launch_plans_for_128px_and_64px_checkpoints():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        native_dir = root / "native"
        _write_synthetic_native_ckpt(native_dir, sharded=True)

        # 1) Full 3-level quadtree (`max_spatial_level=2` -> `128px`, `min_image_tokens=16`, `shortest_edge=16384`)
        ckpt_128 = initialize_treeweave_checkpoint(
            native_dir, root / "tw-128", max_spatial_level=2, max_temporal_level=2
        )
        serve_128 = prepare_serving_checkpoint(
            ckpt_128, root / "serve-128", route_threshold=0.55, budget_logit_shift=0.25
        )
        assert (serve_128 / "config.json").is_file() and not (serve_128 / "config.json").is_symlink()
        assert (serve_128 / "model.safetensors.index.json").is_symlink()

        args_128 = parser().parse_args(
            [
                "--model-path",
                str(serve_128),
                "--gpus",
                "0,1",
                "--encoder-gpus",
                "2,3,4",
                "--port",
                "30100",
                "--encoder-port",
                "31100",
            ]
        )
        roles_128 = build_roles(args_128)
        assert [r.name for r in roles_128] == ["encoder-0", "encoder-1", "encoder-2", "language"]
        enc0, lang = roles_128[0], roles_128[-1]
        for role in roles_128:
            assert role.environment["SGLANG_EXTERNAL_MODEL_PACKAGE"] == "treeweave.serving.models"
            assert role.environment["SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"] == "treeweave.serving.processors"
            assert role.environment["SGLANG_EXTERNAL_MM_MODEL_ARCH"] == ARCHITECTURE
            assert role.environment[ROUTE_THRESHOLD_ENV] == "0.55"
            assert role.environment[BUDGET_SHIFT_ENV] == "0.25"
            assert role.environment[PIXEL_FACTOR_ENV] == "128"
            assert role.command[role.command.index("--encoder-transfer-backend") + 1] == "zmq_to_tokenizer"

        assert enc0.environment["CUDA_VISIBLE_DEVICES"] == "2"
        assert lang.environment["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert lang.command[lang.command.index("--tp-size") + 1] == "2"
        lang_budget = json.loads(lang.command[lang.command.index("--mm-process-config") + 1])
        assert lang_budget["image"]["size"]["shortest_edge"] == 128 * 128

        # `image_tokens < 16` must be rejected for 128px (`s<=2`)!
        bad_budget_args = parser().parse_args(["--model-path", str(serve_128), "--image-tokens", "8"])
        with pytest.raises(ValueError, match="image-tokens >= 16"):
            build_roles(bad_budget_args)

        # Port collision between encoder and language must be rejected!
        args_128.encoder_port = args_128.port
        with pytest.raises(ValueError, match="ports"):
            build_roles(args_128)

        # 2) 2D binary ablation (`max_spatial_level=1` -> `64px`, `min_image_tokens=4`, `shortest_edge=4096`)
        ckpt_64 = initialize_treeweave_checkpoint(
            native_dir, root / "tw-64", max_spatial_level=1, max_temporal_level=0
        )
        args_64 = parser().parse_args(
            ["--model-path", str(ckpt_64), "--gpus", "0", "--image-tokens", "8", "--route-threshold", "0.4"]
        )
        roles_64 = build_roles(args_64)
        assert len(roles_64) == 2
        assert roles_64[0].environment[PIXEL_FACTOR_ENV] == "64"
        lang_budget_64 = json.loads(
            roles_64[-1].command[roles_64[-1].command.index("--mm-process-config") + 1]
        )
        assert lang_budget_64["image"]["size"]["shortest_edge"] == 64 * 64

        # Single-file native checkpoint without TreeWeave weights must be rejected by `prepare_serving_checkpoint`!
        native_single = root / "native-single"
        _write_synthetic_native_ckpt(native_single, sharded=False)
        bad_cfg = json.loads((native_single / "config.json").read_text(encoding="utf-8"))
        bad_cfg["architectures"] = [ARCHITECTURE]
        (native_single / "config.json").write_text(json.dumps(bad_cfg), encoding="utf-8")
        with pytest.raises(ValueError, match="missing required TreeWeave weights"):
            prepare_serving_checkpoint(native_single, root / "serve-bad-single")

        # `volume_bias_mode='exact_logit_bias'` must fail fast in SGLang serving validation!
        bad_vol_cfg = json.loads((serve_128 / "config.json").read_text(encoding="utf-8"))
        bad_vol_cfg["treeweave"]["use_volume_attn_bias"] = True
        bad_vol_cfg["treeweave"]["volume_bias_mode"] = "exact_logit_bias"
        from treeweave.serving.sglang_route import validate_serving_geometry

        with pytest.raises(ValueError, match="volume_bias_mode"):
            validate_serving_geometry(bad_vol_cfg)

        # 3) CLI subprocess `python -m treeweave.serving.serve --dry-run` executes cleanly!
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "treeweave.serving.serve",
                "--model-path",
                str(serve_128),
                "--gpus",
                "1",
                "--encoder-gpus",
                "0",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        assert len(lines) == 2
        assert lines[0].startswith("encoder-0:") and "--encoder-only" in lines[0]
        assert lines[1].startswith("language:") and "--language-only" in lines[1]


def test_occupied_port_probe_and_owned_process_group_cleanup():
    with socket.socket() as listener, tempfile.TemporaryDirectory() as tmp:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        role = Role("probe", [], {}, "127.0.0.1", listener.getsockname()[1])
        with patch("subprocess.Popen") as spawn_mock:
            with pytest.raises(OSError):
                run([role], Path(tmp), 1)
            spawn_mock.assert_not_called()

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        stop_children([child])
        assert child.poll() is not None
        with pytest.raises(ProcessLookupError):
            os.killpg(child.pid, 0)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_placeholder_shrinking_run_splitting_zero_token_video_slices_and_wire_validation():
    """Verify `build_routed_input_ids_and_runs` handles `frame_lengths[tau] == 0` without empty/inverted
    SGLang `MultimodalDataItem` offsets while keeping 0-length `TreeVisualRun`s for exact MRoPE spans!
    """
    VS, VE, IMG, VID = 202, 203, 200, 201
    # Full 64 -> 1 spatiotemporal fold on (4, 8, 8): `frame_lengths == (0, 0, 0, 1)`!
    ones = lambda sh: np.ones(sh, dtype=np.float32)
    vid_layout = hard_tree_route_layout(
        (4, 8, 8), ones((4, 2, 2)), ones((4, 1, 1)), ones((2, 2, 2)), ones((1, 2, 2))
    )
    assert vid_layout.frame_lengths == (0, 0, 0, 1) and vid_layout.output_length == 1

    # Partial quadtree on (1, 8, 8) image: top-left 2x2 compressed to s=1 (`1` tok), other 3 blocks native (`12` toks) -> 13 toks
    p_s1_img = np.array([[[1.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    img_layout = hard_tree_route_layout((1, 8, 8), p_s1_img, np.zeros((1, 1, 1)), np.zeros((0, 2, 2)), np.zeros((0, 2, 2)))
    assert img_layout.output_length == 13

    img_item = TreeRouteItem.from_layout("image", img_layout)
    vid_item = TreeRouteItem.from_layout("video", vid_layout)

    # Validate payload roundtrip & skew guards
    payload = TreeRoutePayload(
        fingerprint="fp1234",
        items=(img_item, vid_item),
        threshold=0.5,
        budget_logit_shift=0.15,
    )
    restored = TreeRoutePayload.from_dict(payload.to_dict())
    restored.validate(fingerprint="fp1234", threshold=0.5, embedding_rows=14)
    with pytest.raises(ValueError, match="fingerprint"):
        restored.validate(fingerprint="wrong", threshold=0.5, embedding_rows=14)
    with pytest.raises(ValueError, match="threshold"):
        restored.validate(fingerprint="fp1234", threshold=0.7, embedding_rows=14)
    with pytest.raises(ValueError, match="embedding rows"):
        restored.validate(fingerprint="fp1234", threshold=0.5, embedding_rows=15)

    # SGLang raw prompt: [10, VS, IMG, VE, 11, VID, 12]
    sgl_prompt = [10, VS, IMG, VE, 11, VID, 12]
    ts_map = {"<0.0 seconds>": [40], "<0.5 seconds>": [41], "<1.0 seconds>": [42], "<1.5 seconds>": [43]}
    built = build_routed_input_ids_and_runs(
        sgl_prompt,
        image_items=[img_item],
        video_items=[vid_item],
        video_timestamps=[[0.0, 0.5, 1.0, 1.5]],
        encode_timestamp_fn=lambda s: ts_map[s],
        img_token_id=IMG,
        video_token_id=VID,
        vision_start_token_id=VS,
        vision_end_token_id=VE,
    )

    # Slices 0, 1, 2 must emit `[40, VS, VE]`, `[41, VS, VE]`, `[42, VS, VE]`, and slice 3 `[43, VS, VID, VE]`!
    expected_ids = (
        [10, VS] + [IMG] * 13 + [VE, 11]
        + [40, VS, VE]
        + [41, VS, VE]
        + [42, VS, VE]
        + [43, VS, VID, VE]
        + [12]
    )
    assert built.input_ids == expected_ids

    # Non-empty chunks ONLY (1 image chunk of 13 + 1 video chunk of 1; NO inverted `(start, start-1)` chunks!)
    assert len(built.chunks) == 2
    assert built.chunks[0].modality == "image" and built.chunks[0].embedding_slice == (0, 13)
    assert built.chunks[1].modality == "video" and built.chunks[1].embedding_slice == (0, 1)
    for ch in built.chunks:
        s_off, e_off = ch.offset
        assert s_off <= e_off
        expected_tok = IMG if ch.modality == "image" else VID
        assert built.input_ids[s_off : e_off + 1] == [expected_tok] * (e_off - s_off + 1)

    # Runs MUST still contain all 5 visual runs (1 image + 4 video slices, 3 of which have length 0)!
    assert len(built.runs) == 5
    assert [int(r.spatial_pos_x2.shape[0]) for r in built.runs] == [13, 0, 0, 0, 1]
    assert math.isclose(float(built.chunks[1].log_volumes[0]), math.log(64.0), rel_tol=1e-6)

    # Also verify SGLang chat-template `[VS, VID, VE]` outer wrapper matches `Qwen3VLTemplate` / HF processor!
    sgl_wrapped_prompt = [10, VS, IMG, VE, 11, VS, VID, VE, 12]
    built_wrapped = build_routed_input_ids_and_runs(
        sgl_wrapped_prompt,
        image_items=[img_item],
        video_items=[vid_item],
        video_timestamps=[[0.0, 0.5, 1.0, 1.5]],
        encode_timestamp_fn=lambda s: ts_map[s],
        img_token_id=IMG,
        video_token_id=VID,
        vision_start_token_id=VS,
        vision_end_token_id=VE,
    )
    assert built_wrapped.input_ids == (
        [10, VS] + [IMG] * 13 + [VE, 11, VS]
        + [40, VS, VE] + [41, VS, VE] + [42, VS, VE] + [43, VS, VID, VE]
        + [VE, 12]
    )


def test_sglang_mrope_rotary_angles_and_log_volumes_match_hf_treeweave_exact():
    """Bit-exact parity between SGLang `build_routed_input_ids_and_runs` + `tree_routed_mrope_positions`
    (`POSITION_SCALE = 2`, `inv_freq / 2`) and `Qwen3_5TreeWeaveForConditionalGeneration` across
    random 9-scale interleaved `<image> + <video> + <image>` sequences and decode continuations!
    """
    cfg = make_mini_qwen3_5_config()
    tw_model = Qwen3_5TreeWeaveForConditionalGeneration(cfg).eval()
    rotary_emb = tw_model.model.language_model.rotary_emb
    VS, VE, IMG, VID = cfg.vision_start_token_id, cfg.vision_end_token_id, cfg.image_token_id, cfg.video_token_id

    rng = np.random.default_rng(2026)
    for trial in range(12):
        if trial == 0:
            # Force maximum 64->1 video compression (3 zero-token slices!) + 16->1 image compression
            z = lambda sh, v: np.full(sh, v, dtype=np.float32)
            img0_l = hard_tree_route_layout((1, 8, 8), z((1, 2, 2), 1), z((1, 1, 1), 1), z((0, 2, 2), 0), z((0, 2, 2), 0))
            vid0_l = hard_tree_route_layout((4, 8, 8), z((4, 2, 2), 1), z((4, 1, 1), 1), z((2, 2, 2), 1), z((1, 2, 2), 1))
            img1_l = hard_tree_route_layout((1, 8, 16), z((1, 2, 4), 1), z((1, 1, 2), 0), z((0, 2, 4), 0), z((0, 2, 4), 0))
        else:
            img0_l = sample_stochastic_tree_layout((1, 8, 8), rng)
            vid0_l = sample_stochastic_tree_layout((4, 8, 8), rng, allow_fine_temporal_merge=(trial % 2 == 0))
            img1_l = sample_stochastic_tree_layout((1, 8, 16), rng)

        # 1) Build native HF input_ids & mm_token_type_ids
        nat_ids = [10, 11, VS] + [IMG] * img0_l.native_tokens + [VE, 12]
        nat_mm = [0, 0, 0] + [1] * img0_l.native_tokens + [0, 0]
        per_slice = (vid0_l.grid[1] // 2) * (vid0_l.grid[2] // 2)
        for tau in range(vid0_l.grid[0]):
            nat_ids.extend([40 + tau, 50 + tau, VS] + [VID] * per_slice + [VE])
            nat_mm.extend([0, 0, 0] + [2] * per_slice + [0])
        nat_ids.extend([13, VS] + [IMG] * img1_l.native_tokens + [VE, 30, 31, 32])
        nat_mm.extend([0, 0] + [1] * img1_l.native_tokens + [0, 0, 0, 0])

        nat_ids_t = torch.tensor([nat_ids], dtype=torch.long)
        base_emb = tw_model.model.get_input_embeddings()(nat_ids_t)
        d = base_emb.shape[-1]
        packed_hf = build_routed_sequence_batch(
            input_ids=nat_ids_t,
            base_inputs_embeds=base_emb,
            mm_token_type_ids=torch.tensor([nat_mm], dtype=torch.long),
            image_layouts=[img0_l, img1_l],
            image_routed_embeds=[torch.randn(img0_l.output_length, d), torch.randn(img1_l.output_length, d)],
            video_layouts=[vid0_l],
            video_routed_embeds=[torch.randn(vid0_l.output_length, d)],
            attention_mask=torch.ones_like(nat_ids_t),
        )

        # 2) Build SGLang unexpanded prompt & run `build_routed_input_ids_and_runs` + `tree_routed_mrope_positions`
        sgl_prompt = [10, 11, VS, IMG, VE, 12, VID, 13, VS, IMG, VE, 30, 31, 32]
        ts_toks = {f"<{0.5 * tau:.1f} seconds>": [40 + tau, 50 + tau] for tau in range(4)}
        built_sgl = build_routed_input_ids_and_runs(
            sgl_prompt,
            image_items=[TreeRouteItem.from_layout("image", img0_l), TreeRouteItem.from_layout("image", img1_l)],
            video_items=[TreeRouteItem.from_layout("video", vid0_l)],
            video_timestamps=[[0.0, 0.5, 1.0, 1.5]],
            encode_timestamp_fn=lambda txt: ts_toks[txt],
            img_token_id=IMG,
            video_token_id=VID,
            vision_start_token_id=VS,
            vision_end_token_id=VE,
        )
        pos_x2, delta_x2 = tree_routed_mrope_positions(len(built_sgl.input_ids), built_sgl.runs)

        # Exact parity checks!
        assert len(built_sgl.input_ids) == packed_hf.inputs_embeds.shape[1]
        hf_mrope = packed_hf.position_ids_4d[1:, 0, :].detach().cpu().numpy()
        np.testing.assert_allclose(pos_x2.astype(np.float32) / float(POSITION_SCALE), hf_mrope, atol=0.0, rtol=0.0)
        assert delta_x2 / float(POSITION_SCALE) == float(packed_hf.rope_deltas.item())
        np.testing.assert_allclose(
            built_sgl.token_log_volumes,
            packed_hf.token_log_volumes[0].detach().cpu().numpy(),
            atol=1e-7,
            rtol=0.0,
        )

        # Verify scaled integer `pos_x2 * (inv_freq / 2)` gives identical `(cos, sin)` to HF float MRoPE!
        dummy_x = packed_hf.inputs_embeds
        cos_hf, sin_hf = rotary_emb(dummy_x, packed_hf.position_ids_4d[1:])
        with torch.no_grad():
            orig_inv = rotary_emb.inv_freq.clone()
            try:
                rotary_emb.inv_freq.div_(float(POSITION_SCALE))
                pos_sgl_t = torch.from_numpy(pos_x2).unsqueeze(1).float()
                cos_sgl, sin_sgl = rotary_emb(dummy_x, pos_sgl_t)
            finally:
                rotary_emb.inv_freq.copy_(orig_inv)
        torch.testing.assert_close(cos_sgl, cos_hf, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(sin_sgl, sin_hf, atol=1e-6, rtol=1e-6)

        # Verify decode step `k` continuation matches HF `_prepare_position_ids_for_generation`!
        prefill_len = len(built_sgl.input_ids)
        for decode_step in range(1, 5):
            seq_len_now = prefill_len + decode_step
            sgl_decode_x2 = delta_x2 - POSITION_SCALE + POSITION_SCALE * seq_len_now
            hf_decode = (prefill_len + decode_step - 1) + float(packed_hf.rope_deltas.item())
            assert sgl_decode_x2 / float(POSITION_SCALE) == hf_decode


class _MockRadixAttention(nn.Module):
    def __init__(self, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.captured_keys: list[torch.Tensor] = []

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        self.captured_keys.append(k.detach().clone())
        return q


class _MockSGLangDecoderLayer(nn.Module):
    def __init__(self, layer_id: int, is_full_attn: bool):
        super().__init__()
        if is_full_attn:
            self.self_attn = nn.Module()
            self.self_attn.attn = _MockRadixAttention(layer_id)
        else:
            self.linear_attn = nn.Identity()


def test_sglang_volume_bias_patch_prefill_chunked_prefill_decode_and_tp2():
    """Verify `install_sglang_volume_bias_on_causal_lm` + `_SGLangVolumeBiasedRadixAttention`
    matches `TreeWeaveAttentionVolumeAdapter.apply_to_keys` across full prefill, chunked prefill,
    decode (`is_decode=True`), and `TP=2` head sharding!
    """
    cfg = make_mini_qwen3_5_config()
    cfg_dict = cfg.to_dict()
    cfg_dict["architectures"] = [ARCHITECTURE]
    cfg_dict["treeweave"] = {"use_volume_attn_bias": True, "volume_bias_mode": "key_direction"}

    # Build synthetic SGLang-style 4-layer CausalLM (`["linear_attention", "full_attention", "linear_attention", "full_attention"]`)
    def _make_mock_lm():
        lm = nn.Module()
        lm.model = nn.Module()
        lm.model.layers = nn.ModuleList(
            [_MockSGLangDecoderLayer(i, ltype == "full_attention") for i, ltype in enumerate(cfg.text_config.layer_types)]
        )
        return lm

    lm_tp1 = _make_mock_lm()
    adapters_tp1 = install_sglang_volume_bias_on_causal_lm(lm_tp1, cfg_dict, tp_rank=0, tp_size=1)
    assert sorted(adapters_tp1.keys()) == [1, 3]  # Only full-attention layers patched!
    assert lm_tp1.model.layers[1].self_attn.attn.layer_id == 1  # Transparent attribute forwarding!

    torch.manual_seed(42)
    num_kv = cfg.text_config.num_key_value_heads  # 2
    head_dim = cfg.text_config.head_dim           # 32
    rot_dim = 16
    w_layer1 = torch.randn(num_kv, head_dim - rot_dim)
    w_layer3 = torch.randn(num_kv, head_dim - rot_dim)

    loaded_keys: set[str] = set()
    load_sglang_volume_bias_tensor(
        adapters_tp1,
        "model.language_model.layers.1.self_attn.treeweave_volume_bias.key_volume_vec",
        w_layer1,
        tp_rank=0,
        tp_size=1,
        loaded_record=loaded_keys,
    )
    load_sglang_volume_bias_tensor(
        adapters_tp1,
        "model.language_model.layers.3.self_attn.treeweave_volume_bias.key_volume_vec",
        w_layer3,
        tp_rank=0,
        tp_size=1,
        loaded_record=loaded_keys,
    )
    assert loaded_keys == {"1:key_volume_vec", "3:key_volume_vec"}

    # Construct a 12-token request where visual tokens at offsets `[3, 7]` have non-zero log-volumes!
    vis_lv = np.array([math.log(4.0), 0.0, math.log(16.0), math.log(64.0), math.log(8.0)], dtype=np.float32)
    mm_item = SimpleNamespace(
        offsets=[(3, 7)],
        model_specific_data={"treeweave_log_volumes": vis_lv},
    )
    mm_req = SimpleNamespace(mm_items=[mm_item])

    q_full = torch.randn(12, cfg.text_config.num_attention_heads * head_dim)
    k_full = torch.randn(12, num_kv * head_dim)
    v_full = torch.randn(12, num_kv * head_dim)

    # 1) Full 12-token prefill
    fb_full = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: False),
        mm_inputs=[mm_req],
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[12],
    )
    lv_full = build_sglang_extend_log_volumes(fb_full, 12, "cpu")
    assert lv_full is not None and lv_full.shape == (12,)
    with sglang_volume_bias_context(lv_full):
        lm_tp1.model.layers[1].self_attn.attn(q_full, k_full, v_full)
    k_out_full = lm_tp1.model.layers[1].self_attn.attn._orig_attn.captured_keys[-1]

    # Must match HF `TreeWeaveAttentionVolumeAdapter.apply_to_keys` bit-for-bit!
    k_hf_in = k_full.view(1, 12, num_kv, head_dim).transpose(1, 2)
    k_hf_ref = adapters_tp1[1].apply_to_keys(k_hf_in, lv_full.unsqueeze(0)).transpose(1, 2).reshape(12, -1)
    torch.testing.assert_close(k_out_full, k_hf_ref, atol=0.0, rtol=0.0)
    # Rotary channels `[..., :16]` MUST remain 100% untouched!
    torch.testing.assert_close(
        k_out_full.view(12, num_kv, head_dim)[:, :, :rot_dim],
        k_full.view(12, num_kv, head_dim)[:, :, :rot_dim],
        atol=0.0,
        rtol=0.0,
    )

    # 2) Chunked prefill splitting right across the visual span (`[0:5]` then `[5:12]`)!
    chunk_parts = []
    for p_len, e_len in ((0, 5), (5, 7)):
        fb_chunk = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: False),
            mm_inputs=[mm_req],
            extend_prefix_lens_cpu=[p_len],
            extend_seq_lens_cpu=[e_len],
        )
        lv_chunk = build_sglang_extend_log_volumes(fb_chunk, e_len, "cpu")
        with sglang_volume_bias_context(lv_chunk):
            lm_tp1.model.layers[1].self_attn.attn(
                q_full[p_len : p_len + e_len],
                k_full[p_len : p_len + e_len],
                v_full[p_len : p_len + e_len],
            )
        chunk_parts.append(lm_tp1.model.layers[1].self_attn.attn._orig_attn.captured_keys[-1])
    torch.testing.assert_close(torch.cat(chunk_parts, dim=0), k_out_full, atol=0.0, rtol=0.0)

    # 3) Decode step (`is_decode() == True`) -> `lv_dec is None` -> decode text keys untouched!
    fb_dec = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        mm_inputs=[mm_req],
        extend_prefix_lens_cpu=[12],
        extend_seq_lens_cpu=[1],
    )
    assert build_sglang_extend_log_volumes(fb_dec, 1, "cpu") is None
    k_dec = torch.randn(1, num_kv * head_dim)
    with sglang_volume_bias_context(None):
        lm_tp1.model.layers[1].self_attn.attn(q_full[:1], k_dec, v_full[:1])
    torch.testing.assert_close(
        lm_tp1.model.layers[1].self_attn.attn._orig_attn.captured_keys[-1], k_dec, atol=0.0, rtol=0.0
    )

    # 4) `TP=2` head sharding (`tp_rank=0` and `tp_rank=1`) concatenated equals `TP=1` bit-for-bit!
    tp_keys = []
    for rank in (0, 1):
        lm_r = _make_mock_lm()
        ad_r = install_sglang_volume_bias_on_causal_lm(lm_r, cfg_dict, tp_rank=rank, tp_size=2)
        load_sglang_volume_bias_tensor(
            ad_r,
            "model.language_model.layers.1.self_attn.treeweave_volume_bias.key_volume_vec",
            w_layer1,
            tp_rank=rank,
            tp_size=2,
        )
        k_shard = k_full.view(12, num_kv, head_dim)[:, rank : rank + 1, :].reshape(12, head_dim)
        with sglang_volume_bias_context(lv_full):
            lm_r.model.layers[1].self_attn.attn(q_full[:, : 2 * head_dim], k_shard, k_shard)
        tp_keys.append(lm_r.model.layers[1].self_attn.attn._orig_attn.captured_keys[-1].view(12, 1, head_dim))
    torch.testing.assert_close(torch.cat(tp_keys, dim=1).reshape(12, -1), k_out_full, atol=0.0, rtol=0.0)

    # 5) Verify `finalize_sglang_volume_bias_loaded_names` covers 100% of `named_parameters()` for SGLang DefaultModelLoader
    vol_names = finalize_sglang_volume_bias_loaded_names(
        lm_tp1, adapters_tp1, loaded_keys, use_volume_attn_bias=True, volume_bias_mode="key_direction"
    )
    all_lm_params = {n for n, _ in lm_tp1.named_parameters()}
    assert all_lm_params - vol_names == set()
    with pytest.raises(RuntimeError, match="key_volume_vec"):
        finalize_sglang_volume_bias_loaded_names(
            lm_tp1, adapters_tp1, {"1:key_volume_vec"}, use_volume_attn_bias=True, volume_bias_mode="key_direction"
        )

    # 6) Verify GPU-tensor fallback (`extend_prefix_lens` / `extend_seq_lens`) AND tail-padded `total_tokens > sum_extend`!
    fb_padded_gpu = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: False),
        mm_inputs=[mm_req],
        extend_prefix_lens_cpu=None,
        extend_seq_lens_cpu=None,
        extend_prefix_lens=torch.tensor([0]),
        extend_seq_lens=torch.tensor([12]),
    )
    lv_padded = build_sglang_extend_log_volumes(fb_padded_gpu, 16, "cpu")
    assert lv_padded is not None and lv_padded.shape == (16,)
    torch.testing.assert_close(lv_padded[:12], lv_full, atol=0.0, rtol=0.0)
    assert float(lv_padded[12:].abs().max().item()) == 0.0


def test_live_sglang_entrypoints_when_sglang_installed():
    pytest.importorskip("sglang")
    from treeweave.serving.models import EntryClass
    from treeweave.serving.processors import TreeWeaveSGLangImageProcessor

    assert EntryClass[0].__name__ == ARCHITECTURE
    assert TreeWeaveSGLangImageProcessor.models[0].__name__ == ARCHITECTURE
    assert config_fingerprint(make_mini_qwen3_5_config())

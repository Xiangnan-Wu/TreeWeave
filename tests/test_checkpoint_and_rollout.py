"""Tests for Streaming Safetensors Checkpoint Init, Router Calibration, and 128px Resize Patch Edge Cases."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from tests.test_modeling_qwen3_5_treeweave import make_mini_qwen3_5_config
from treeweave import ARCHITECTURE
from treeweave.modeling_treeweave import Qwen3_5TreeWeaveForConditionalGeneration
from treeweave.serving.calibrate_router import calibrate_router_bias
from treeweave.serving.init_checkpoint import initialize_treeweave_checkpoint
from treeweave.swift_plugin.template import install_treeweave_resize_patch


def _write_synthetic_native_ckpt(root: Path, *, sharded: bool = True, dtype: torch.dtype = torch.bfloat16) -> Qwen3_5ForConditionalGeneration:
    root.mkdir(parents=True, exist_ok=True)
    cfg = make_mini_qwen3_5_config()
    native = Qwen3_5ForConditionalGeneration(cfg).to(dtype=dtype)

    cfg_dict = cfg.to_dict()
    cfg_dict["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    cfg_dict["model_type"] = "qwen3_5"
    (root / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    # Exclude tied `lm_head.weight` just like HF native safetensors checkpoints do!
    sd = {k: v.contiguous() for k, v in native.state_dict().items() if k != "lm_head.weight"}
    if sharded:
        items = list(sd.items())
        mid = len(items) // 2
        s1, s2 = dict(items[:mid]), dict(items[mid:])
        save_file(s1, str(root / "part-1.safetensors"), metadata={"format": "pt"})
        save_file(s2, str(root / "part-2.safetensors"), metadata={"format": "pt"})
        wmap = {k: "part-1.safetensors" for k in s1}
        wmap.update({k: "part-2.safetensors" for k in s2})
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wmap}), encoding="utf-8")
    else:
        save_file(sd, str(root / "model.safetensors"), metadata={"format": "pt"})

    for f in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "video_preprocessor_config.json"):
        (root / f).write_text('{"patch_size": 16, "merge_size": 2}', encoding="utf-8")
    return native


@pytest.mark.parametrize("sharded,link_base", [(False, False), (True, True)])
def test_streaming_checkpoint_initializer_and_calibration(sharded, link_base):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "native"
        dst = root / "treeweave-init"
        _write_synthetic_native_ckpt(src, sharded=sharded, dtype=torch.bfloat16)

        out = initialize_treeweave_checkpoint(src, dst, seed=42, link_base_weights=link_base)
        assert out == dst.resolve()
        cfg_json = json.loads((dst / "config.json").read_text(encoding="utf-8"))
        assert cfg_json["architectures"] == [ARCHITECTURE]

        # Verify HF `from_pretrained` loads every single tensor cleanly with FP32 router heads intact!
        hf_cfg = AutoConfig.from_pretrained(dst)
        loaded = Qwen3_5TreeWeaveForConditionalGeneration.from_pretrained(dst, config=hf_cfg, dtype=torch.bfloat16)
        assert loaded.model.visual.router.head_s1[1].weight.dtype == torch.float32
        assert loaded.model.visual.router.bias_s1.dtype == torch.float32

        # Test offline router bias calibration targeting ~75% savings on synthetic query states
        torch.manual_seed(5)
        synth_queries = [
            ((1, 8, 8), torch.randn(4, 32)),
            ((4, 8, 8), torch.randn(16, 32)),
        ]
        shift, achieved = calibrate_router_bias(
            checkpoint_dir=dst,
            synthetic_grids_and_queries=synth_queries,
            target_savings=0.75,
            write_in_place=True,
            device="cpu",
        )
        assert achieved >= 0.70

        # Reload checkpoint and confirm `router_calib_bias` persisted in safetensors!
        reloaded = Qwen3_5TreeWeaveForConditionalGeneration.from_pretrained(dst, config=AutoConfig.from_pretrained(dst))
        torch.testing.assert_close(
            reloaded.model.visual.router.router_calib_bias,
            torch.full((4,), float(shift), dtype=torch.float32),
            atol=1e-5,
            rtol=1e-5,
        )


def test_checkpoint_initializer_rejects_invalid_inputs_cleanly():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "native"
        _write_synthetic_native_ckpt(src, sharded=False)

        with pytest.raises(ValueError):
            initialize_treeweave_checkpoint(src, src / "inside")
        with pytest.raises(ValueError):
            initialize_treeweave_checkpoint(src, root / "bad-p", initial_compression_prob=0.0)
        with pytest.raises(ValueError):
            initialize_treeweave_checkpoint(src, root / "bad-t", route_threshold=-1.0)


def test_resize_patch_enforces_128px_multiples_even_under_tiny_pixel_budgets():
    install_treeweave_resize_patch(128)
    import qwen_vl_utils.vision_process as vp
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as hf_smart_resize

    for h_in, w_in, max_px in [
        (37, 51, 4096),          # tiny image + tiny max_pixels (< 128*128!) -> must clamp cleanly to 128x128!
        (1080, 1920, 512 * 1024),
        (2400, 160, 256 * 1024), # extreme 15:1 aspect ratio!
    ]:
        h1, w1 = vp.smart_resize(h_in, w_in, factor=32, min_pixels=4096, max_pixels=max_px)
        h2, w2 = hf_smart_resize(h_in, w_in, factor=32, min_pixels=4096, max_pixels=max_px)
        assert h1 >= 128 and w1 >= 128 and h1 % 128 == 0 and w1 % 128 == 0
        assert h2 >= 128 and w2 >= 128 and h2 % 128 == 0 and w2 % 128 == 0


def _spawned_worker_check_resize(expected_factor: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Runs inside a `multiprocessing` `spawn` child WITHOUT importing `treeweave` explicitly."""
    import qwen_vl_utils.vision_process as vp
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as hf_sr

    r1 = vp.smart_resize(37, 51, factor=28, min_pixels=3136, max_pixels=4096)
    r2 = hf_sr(515, 777, factor=32, min_pixels=3136, max_pixels=expected_factor * expected_factor * 4)
    return r1, r2


def test_subprocess_sitecustomize_bootstrap_enforces_128px_and_64px_and_chains_downstream():
    """Verify spawned OS child processes (`vLLM` EngineCore / `SGLang` TokenizerManager) inherit exact
    `128px` or `64px` `smart_resize` via `sitecustomize.py` without clobbering user `sitecustomize.py`.
    """
    import multiprocessing as mp
    import os
    import subprocess
    import sys

    from treeweave.rollout.offline_rollout import enable_subprocess_resize_bootstrap

    orig_pypath = os.environ.get("PYTHONPATH")
    orig_factor = os.environ.get("TREEWEAVE_PIXEL_FACTOR")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            downstream_dir = Path(tmp) / "user_site"
            downstream_dir.mkdir()
            downstream_sc = downstream_dir / "sitecustomize.py"
            downstream_sc.write_text(
                "import os\n"
                "os.environ['USER_SITECUSTOMIZE_RAN'] = 'yes'\n"
                "os.environ['USER_SITECUSTOMIZE_FILE'] = str(__file__)\n",
                encoding="utf-8",
            )

            for factor in (128, 64):
                os.environ["PYTHONPATH"] = str(downstream_dir)
                py_path = enable_subprocess_resize_bootstrap(factor)
                env = dict(os.environ, PYTHONPATH=py_path, TREEWEAVE_PIXEL_FACTOR=str(factor))

                # 1) Fresh OS subprocess (`python -c ...`) importing qwen_vl_utils & transformers cold
                probe_code = (
                    "import asyncio, json, os\n"
                    "import qwen_vl_utils.vision_process as vp\n"
                    "from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as hf_sr\n"
                    "from sitecustomize import _wrap_sglang_preprocess_video\n"
                    "async def _fake_pv(vr, image_factor=28):\n"
                    "    return image_factor\n"
                    "wrapped_pv = _wrap_sglang_preprocess_video(_fake_pv, 128)\n"
                    "pv_kw = asyncio.run(wrapped_pv('v'))\n"
                    "pv_pos = asyncio.run(wrapped_pv('v', 28))\n"
                    "a = vp.smart_resize(37, 51, factor=28, min_pixels=3136, max_pixels=4096)\n"
                    "b = hf_sr(300, 500, factor=32, min_pixels=3136, max_pixels=262144)\n"
                    "print(json.dumps({'a': list(a), 'b': list(b), 'pv_kw': pv_kw, 'pv_pos': pv_pos, "
                    "'chained': os.environ.get('USER_SITECUSTOMIZE_RAN'), "
                    "'chained_file': os.environ.get('USER_SITECUSTOMIZE_FILE')}))\n"
                )
                proc = subprocess.run(
                    [sys.executable, "-c", probe_code],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                res = json.loads(proc.stdout.strip().splitlines()[-1])
                assert res["chained"] == "yes"
                assert Path(res["chained_file"]).resolve() == downstream_sc.resolve()
                assert res["a"] == [factor, factor]
                assert res["b"][0] % factor == 0 and res["b"][1] % factor == 0
                assert res["pv_kw"] == factor and res["pv_pos"] == factor

                # 2) `multiprocessing.get_context("spawn")` Pool worker (exact mechanism used by vLLM/SGLang!)
                ctx = mp.get_context("spawn")
                with ctx.Pool(1) as pool:
                    r1, r2 = pool.apply(_spawned_worker_check_resize, (factor,))
                assert r1 == (factor, factor)
                assert r2[0] % factor == 0 and r2[1] % factor == 0
    finally:
        if orig_pypath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = orig_pypath
        if orig_factor is None:
            os.environ.pop("TREEWEAVE_PIXEL_FACTOR", None)
        else:
            os.environ["TREEWEAVE_PIXEL_FACTOR"] = orig_factor
        install_treeweave_resize_patch(128)


def test_cache_topk_logits_dp_sharding_batching_packing_and_atomic_resume():
    """Verify `cache_topk_logits.py` produces consistent Top-K shards across `batch_size=1`,
    multi-sample batching (`batch_size=3`), sequence packing (`pack_sequences=True`), and multi-worker
    `DP=2` spawn sharding, while skipping valid cached shards and recovering corrupted/temp files!
    """
    from PIL import Image
    from tests.test_ms_swift_integration import _build_local_mini_qwen3_5_checkpoint_with_processor
    from treeweave.rollout.cache_topk_logits import cache_dataset_topk
    from treeweave.rollout.offline_rollout import compute_topk_cache_uuid

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        native_dir = root / "native-qwen35"
        _build_local_mini_qwen3_5_checkpoint_with_processor(native_dir)

        img1 = root / "c1.png"
        img2 = root / "c2.png"
        Image.new("RGB", (128, 128), color=(10, 200, 80)).save(img1)
        Image.new("RGB", (256, 128), color=(200, 50, 120)).save(img2)

        rows = [
            {
                "messages": [
                    {"role": "user", "content": "<image> First image question?"},
                    {"role": "assistant", "content": "Green square answer."},
                ],
                "images": [str(img1)],
            },
            {
                "messages": [
                    {"role": "user", "content": "<image> Second wider image question?"},
                    {"role": "assistant", "content": "Wide pink rectangle detailed description."},
                ],
                "images": [str(img2)],
            },
            {
                "messages": [
                    {"role": "user", "content": "Pure text prompt for hybrid batch."},
                    {"role": "assistant", "content": "Concise text reply."},
                ],
            },
            # Same prompt + image as row 0, but DIFFERENT assistant response -> MUST NOT collide!
            {
                "messages": [
                    {"role": "user", "content": "<image> First image question?"},
                    {"role": "assistant", "content": "Alternative much longer green square explanation."},
                ],
                "images": [str(img1)],
            },
            # Intentional exact duplicate of row 0 to test deduplication before DP sharding!
            {
                "messages": [
                    {"role": "user", "content": "<image> First image question?"},
                    {"role": "assistant", "content": "Green square answer."},
                ],
                "images": [str(img1)],
            },
        ]
        rollout_jsonl = root / "in_rollouts.jsonl"
        rollout_jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        # 1. Baseline single-sample sequential reference (`batch_size=1`)
        dir_ref = root / "cache_ref"
        cache_dataset_topk(
            teacher_model_path=str(native_dir),
            rollout_jsonl=rollout_jsonl,
            output_jsonl=root / "out_ref.jsonl",
            cache_dir=dir_ref,
            topk=24,
            device="cpu",
            batch_size=1,
        )
        ref_files = {p.name: load_file(str(p)) for p in dir_ref.glob("*.safetensors")}
        assert len(ref_files) == 4  # 4 unique (prompt + response) trajectories across 5 rows!
        assert all("target_labels" in v for v in ref_files.values())

        # 2. Multi-sample padded micro-batching (`batch_size=3`)
        dir_batched = root / "cache_batched"
        cache_dataset_topk(
            teacher_model_path=str(native_dir),
            rollout_jsonl=rollout_jsonl,
            output_jsonl=root / "out_batched.jsonl",
            cache_dir=dir_batched,
            topk=24,
            device="cpu",
            batch_size=3,
            pack_sequences=False,
        )
        for fname, ref_t in ref_files.items():
            got_t = load_file(str(dir_batched / fname))
            torch.testing.assert_close(got_t["topk_logprobs"], ref_t["topk_logprobs"], atol=1e-3, rtol=1e-3)
            assert got_t["topk_indices"].shape == ref_t["topk_indices"].shape
            assert torch.equal(got_t["target_labels"], ref_t["target_labels"])

        # 3. Padding-free packed forward extraction (`pack_sequences=True`)
        dir_packed = root / "cache_packed"
        cache_dataset_topk(
            teacher_model_path=str(native_dir),
            rollout_jsonl=rollout_jsonl,
            output_jsonl=root / "out_packed.jsonl",
            cache_dir=dir_packed,
            topk=24,
            device="cpu",
            batch_size=3,
            pack_sequences=True,
        )
        for fname, ref_t in ref_files.items():
            got_t = load_file(str(dir_packed / fname))
            torch.testing.assert_close(got_t["topk_logprobs"], ref_t["topk_logprobs"], atol=1e-3, rtol=1e-3)
            assert torch.equal(got_t["topk_indices"][:, 0], ref_t["topk_indices"][:, 0])
            assert torch.equal(got_t["target_labels"], ref_t["target_labels"])

        # 4. Multi-worker `DP=2` spawn sharding + resume skip + corrupted/stale-temp recovery!
        dir_dp = root / "cache_dp"
        dir_dp.mkdir()
        uid0 = compute_topk_cache_uuid(rows[0])
        uid1 = compute_topk_cache_uuid(rows[1])
        # Pre-seed uid0 as a valid shard (should be skipped on resume), uid1 as 0-byte corrupt file, plus a stale .tmp file!
        valid_pre = dir_dp / f"{uid0}.safetensors"
        save_file(ref_files[f"{uid0}.safetensors"], str(valid_pre), metadata={"format": "pt"})
        pre_bytes = valid_pre.read_bytes()
        corrupt_pre = dir_dp / f"{uid1}.safetensors"
        corrupt_pre.write_bytes(b"")
        stale_tmp = dir_dp / ".stale.99999.tmp.safetensors"
        stale_tmp.write_bytes(b"partial")

        out_dp = root / "out_dp.jsonl"
        cache_dataset_topk(
            teacher_model_path=str(native_dir),
            rollout_jsonl=rollout_jsonl,
            output_jsonl=out_dp,
            cache_dir=dir_dp,
            topk=24,
            gpus=["cpu", "cpu"],
            batch_size=2,
            pack_sequences=True,
        )
        assert not stale_tmp.exists()
        assert valid_pre.read_bytes() == pre_bytes  # valid pre-existing shard untouched!
        repaired_t = load_file(str(corrupt_pre))
        torch.testing.assert_close(
            repaired_t["topk_logprobs"], ref_files[f"{uid1}.safetensors"]["topk_logprobs"], atol=1e-3, rtol=1e-3
        )
        out_lines = [json.loads(line) for line in out_dp.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(out_lines) == 5 and all(Path(r["topk_cache_path"]).is_file() for r in out_lines)

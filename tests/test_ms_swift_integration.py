"""End-to-End Real `ms-swift` Pipeline Test: Template Encoding + Sequence Packing + Offline TopK + Stage 1/2/3 Training."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import AutoConfig, PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

from tests.test_modeling_qwen3_5_treeweave import make_mini_qwen3_5_config
from treeweave.modeling_treeweave import Qwen3_5TreeWeaveForConditionalGeneration
from treeweave.rollout.cache_topk_logits import cache_dataset_topk
from treeweave.rollout.offline_rollout import compute_sample_uuid, compute_topk_cache_uuid, run_sharded_rollout
from treeweave.serving.calibrate_router import calibrate_router_bias
from treeweave.serving.init_checkpoint import initialize_treeweave_checkpoint
from treeweave.swift_plugin.cli import TreeWeaveSft, TreeWeaveSftArguments
from treeweave.swift_plugin.register import register_treeweave_with_swift


def _build_local_mini_qwen3_5_checkpoint_with_processor(native_dir: Path) -> None:
    """Create a complete miniature Qwen3.5 checkpoint + ByteLevel BPE tokenizer + image/video processors on disk."""
    native_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_mini_qwen3_5_config()

    # ByteLevel BPE preserves `\n` so real `ms-swift` Qwen3VL/Qwen3.5 video timestamp splitting works natively!
    special_tokens = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<think>",
        "</think>",
    ]
    vocab = {ch: idx for idx, ch in enumerate(ByteLevel.alphabet())}
    for st in ["<pad>", "<unk>"] + special_tokens:
        if st not in vocab:
            vocab[st] = len(vocab)
    while len(vocab) < 320:
        vocab[f"tok_{len(vocab)}"] = len(vocab)

    cfg.text_config.vocab_size = len(vocab)
    cfg.image_token_id = vocab["<|image_pad|>"]
    cfg.video_token_id = vocab["<|video_pad|>"]
    cfg.vision_start_token_id = vocab["<|vision_start|>"]
    cfg.vision_end_token_id = vocab["<|vision_end|>"]
    cfg.text_config.pad_token_id = vocab["<pad>"]
    cfg.text_config.eos_token_id = vocab["<|im_end|>"]

    tok_backend = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tok_backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok_backend.decoder = ByteLevelDecoder()
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|im_end|>",
        additional_special_tokens=special_tokens,
    )
    fast_tok.chat_template = (
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )

    img_proc = Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2)
    vid_proc = Qwen3VLVideoProcessor(patch_size=16, temporal_patch_size=2, merge_size=2)
    proc = Qwen3VLProcessor(image_processor=img_proc, video_processor=vid_proc, tokenizer=fast_tok)
    proc.save_pretrained(native_dir)

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration(cfg)
    with torch.no_grad():
        model.model.language_model.embed_tokens.weight[40:60].fill_(0.5)
    model.save_pretrained(native_dir, safe_serialization=True)
    # Ensure config.json explicitly records architectures & model_type expected by initializer
    c_json = json.loads((native_dir / "config.json").read_text(encoding="utf-8"))
    c_json["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    c_json["model_type"] = "qwen3_5"
    (native_dir / "config.json").write_text(json.dumps(c_json, indent=2), encoding="utf-8")


def test_real_ms_swift_stage1_packed_stage2_soft_and_stage3_offline_topk_pipeline():
    register_treeweave_with_swift(activate_distill_trainer=True)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        native_dir = root / "native-qwen35"
        tw_init_dir = root / "treeweave-init"
        _build_local_mini_qwen3_5_checkpoint_with_processor(native_dir)
        initialize_treeweave_checkpoint(native_dir, tw_init_dir, seed=0)

        # Create 2 synthetic RGB images and a rollout JSONL dataset
        img1 = root / "blue.png"
        img2 = root / "red.png"
        Image.new("RGB", (200, 150), color=(30, 80, 220)).save(img1)
        Image.new("RGB", (140, 260), color=(220, 40, 40)).save(img2)

        rollout_jsonl = root / "rollouts.jsonl"
        samples = [
            {
                "messages": [
                    {"role": "user", "content": "<image> Describe the image and color."},
                    {"role": "assistant", "content": "It shows a bright blue square."},
                ],
                "images": [str(img1)],
            },
            {
                "messages": [
                    {"role": "user", "content": "<image> Describe the image and color."},
                    {"role": "assistant", "content": "It shows a red circle."},
                ],
                "images": [str(img2)],
            },
        ] * 2  # 4 samples so PackingDataset packs multiple samples per row!
        rollout_jsonl.write_text(
            "\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8"
        )

        # ---- 1. Run REAL ms-swift Stage 1 (with Sequence Packing enabled!) ----
        s1_out = root / "out-stage1"
        args_s1 = TreeWeaveSftArguments(
            treeweave_stage=1,
            model=str(tw_init_dir),
            dataset=[str(rollout_jsonl)],
            output_dir=str(s1_out),
            torch_dtype="float32",
            attn_impl="sdpa",
            packing=True,
            packing_length=1024,
            max_length=1024,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            max_steps=2,
            save_steps=2,
            logging_steps=1,
            distill_topk=32,
            dataloader_num_workers=0,
            dataset_num_proc=2,
            report_to=["none"],
        )
        res_s1 = TreeWeaveSft(args_s1).main()
        s1_ckpt = s1_out / "last"
        assert s1_ckpt.is_symlink() and s1_ckpt.resolve() == Path(res_s1["last_model_checkpoint"]).resolve()

        # ---- 2. Run REAL ms-swift Stage 2 (Soft-Tree Routing with Rate + Balance Losses!) ----
        s2_out = root / "out-stage2"
        args_s2 = TreeWeaveSftArguments(
            treeweave_stage=2,
            model=str(s1_ckpt),
            dataset=[str(rollout_jsonl)],
            output_dir=str(s2_out),
            torch_dtype="float32",
            attn_impl="sdpa",
            packing=False,
            max_length=512,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=5e-4,
            max_steps=2,
            save_steps=2,
            logging_steps=1,
            distill_topk=32,
            lambda_rate=0.05,
            lambda_bal=0.01,
            dataloader_num_workers=0,
            dataset_num_proc=1,
            report_to=["none"],
        )
        res_s2 = TreeWeaveSft(args_s2).main()
        s2_ckpt = Path(res_s2["last_model_checkpoint"])
        assert s2_ckpt.is_dir()

        # ---- 2.5. Run REAL JSONL-based Offline Router Calibration on Stage 2 Checkpoint! ----
        shift, cal_sav = calibrate_router_bias(
            checkpoint_dir=s2_ckpt,
            calibration_jsonl=rollout_jsonl,
            target_savings=0.50,
            max_samples=2,
            write_in_place=True,
            device="cpu",
        )
        assert cal_sav >= 0.40

        # ---- 3. Precompute Offline Top-K Cache & Run REAL ms-swift Stage 3 (Zero-Teacher-VRAM Mode!) ----
        topk_jsonl = root / "rollouts_topk.jsonl"
        topk_cache_dir = root / "topk_shards"
        cache_dataset_topk(
            teacher_model_path=str(native_dir),
            rollout_jsonl=rollout_jsonl,
            output_jsonl=topk_jsonl,
            cache_dir=topk_cache_dir,
            topk=32,
            pixel_factor=128,
            device="cpu",
        )
        assert len(list(topk_cache_dir.glob("*.safetensors"))) == 2  # deduplicated by deterministic UUID!

        s3_out = root / "out-stage3"
        args_s3 = TreeWeaveSftArguments(
            treeweave_stage=3,
            distill_teacher_mode="offline_topk_cache",
            topk_cache_dir=str(topk_cache_dir),
            model=str(s2_ckpt),
            dataset=[str(rollout_jsonl)],
            output_dir=str(s3_out),
            torch_dtype="float32",
            attn_impl="sdpa",
            packing=False,
            max_length=512,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            max_steps=2,
            save_steps=2,
            logging_steps=1,
            distill_topk=32,
            dataloader_num_workers=0,
            dataset_num_proc=2,
            report_to=["none"],
        )
        res_s3 = TreeWeaveSft(args_s3).main()
        s3_ckpt = Path(res_s3["last_model_checkpoint"])
        assert s3_ckpt.is_dir()

        # Confirm final Stage 3 checkpoint reloads cleanly as Qwen3_5TreeWeaveForConditionalGeneration!
        final_model = Qwen3_5TreeWeaveForConditionalGeneration.from_pretrained(
            s3_ckpt, config=AutoConfig.from_pretrained(s3_ckpt)
        )
        assert final_model.config.architectures == ["Qwen3_5TreeWeaveForConditionalGeneration"]
        assert "treeweave" in final_model.config.to_dict()

        # ---- 4. Also verify Stage 3 with `external_teacher` mode (Lightweight Merger + LLM Teacher)! ----
        s3_ext_out = root / "out-stage3-ext"
        args_s3_ext = TreeWeaveSftArguments(
            treeweave_stage=3,
            distill_teacher_mode="external_teacher",
            teacher_model_path=str(native_dir),
            model=str(s2_ckpt),
            dataset=[str(rollout_jsonl)],
            output_dir=str(s3_ext_out),
            torch_dtype="float32",
            attn_impl="sdpa",
            packing=False,
            max_length=512,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            max_steps=1,
            save_steps=1,
            logging_steps=1,
            distill_topk=32,
            dataloader_num_workers=0,
            dataset_num_proc=1,
            report_to=["none"],
        )
        res_s3_ext = TreeWeaveSft(args_s3_ext).main()
        assert Path(res_s3_ext["last_model_checkpoint"]).is_dir()

        # ---- 5. Verify Offline Rollout UUID Deduplication, End-to-End Rollout Execution, Video Packing & `swift sft` CLI ----
        uid1 = compute_sample_uuid(samples[0])
        uid1_stripped = compute_sample_uuid({"messages": samples[0]["messages"][:1], "images": samples[0]["images"]})
        assert uid1 == uid1_stripped  # trailing assistant message ignored when hashing prompt!

        vf1 = root / "vf1.png"
        vf2 = root / "vf2.png"
        Image.new("RGB", (160, 160), color=(20, 90, 210)).save(vf1)
        Image.new("RGB", (160, 160), color=(220, 140, 30)).save(vf2)
        frames = [str(vf1), str(vf2), str(vf1), str(vf2)]
        raw_rollout_in = root / "raw_prompts.jsonl"
        gen_rollout_out = root / "generated_rollouts.jsonl"
        raw_rollout_in.write_text(
            "\n".join(
                json.dumps(r)
                for r in [
                    {"messages": [{"role": "user", "content": "<image> color?"}], "images": [str(img1)]},
                    {"messages": [{"role": "user", "content": "<video> motion?"}], "videos": [frames]},
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        run_sharded_rollout(
            model_path=str(native_dir),
            input_jsonl=raw_rollout_in,
            output_jsonl=gen_rollout_out,
            gpus=None,
            backend="transformers",
            max_new_tokens=6,
            temperature=0.0,
        )
        gen_rows = [json.loads(line) for line in gen_rollout_out.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(gen_rows) == 2 and all(r["messages"][-1]["role"] == "assistant" for r in gen_rows)

        # ---- 6. Verify `swift sft --custom_register_path` subprocess on tri-modal (text + image + video) packed mixture! ----
        tri_jsonl = root / "tri_modal.jsonl"
        tri_jsonl.write_text(
            "\n".join(
                json.dumps(r)
                for r in [
                    {
                        "messages": [
                            {"role": "user", "content": "Pure text prompt."},
                            {"role": "assistant", "content": "Pure text reply."},
                        ]
                    },
                    gen_rows[0],
                    gen_rows[1],
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        cli_out = root / "out-swift-cli"
        env = dict(os.environ, TREEWEAVE_STAGE="2")
        env.pop("PYTHONPATH", None)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "swift.cli.sft",
                "--custom_register_path",
                "treeweave/swift_plugin/register.py",
                "--model",
                str(s1_ckpt),
                "--dataset",
                str(tri_jsonl),
                "--output_dir",
                str(cli_out),
                "--add_version",
                "false",
                "--tuner_type",
                "full",
                "--torch_dtype",
                "float32",
                "--attn_impl",
                "sdpa",
                "--packing",
                "true",
                "--packing_length",
                "1024",
                "--max_length",
                "1024",
                "--max_steps",
                "1",
                "--save_steps",
                "1",
                "--dataloader_num_workers",
                "0",
                "--dataset_num_proc",
                "1",
                "--report_to",
                "none",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"swift.cli.sft failed:\n{proc.stderr[-2000:]}"
        assert (cli_out / "last" / "config.json").is_file()


def test_offline_topk_cache_max_length_truncation_alignment_in_padded_and_packed_modes():
    """Regression test: when `--max_length` truncates Sample 0's response in a collated or packed batch,
    Sample 0's cached Top-K tensor MUST be sliced to its exact post-truncation active count so Sample 1's
    Teacher Top-K targets are never shifted or misaligned!
    """
    import pytest
    from safetensors.torch import load_file, save_file
    from swift.model import get_model_processor
    from swift.template import get_template
    from treeweave.swift_plugin.template import QWEN3_5_TREEWEAVE_TEMPLATE_TYPE

    register_treeweave_with_swift(activate_distill_trainer=True)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        native_dir = root / "native-qwen35"
        tw_init_dir = root / "treeweave-init"
        _build_local_mini_qwen3_5_checkpoint_with_processor(native_dir)
        initialize_treeweave_checkpoint(native_dir, tw_init_dir, seed=0)

        img = root / "sample.png"
        Image.new("RGB", (128, 128), color=(40, 120, 200)).save(img)

        # Sample 0 & Sample 1 intentionally share the EXACT same prompt + image (plus a custom `_rollout_uuid`
        # that `ms-swift` AutoPreprocessor strips) while having DIFFERENT assistant responses!
        long_sample = {
            "_rollout_uuid": "custom_upstream_id_001",
            "messages": [
                {"role": "user", "content": "<image> Go."},
                {"role": "assistant", "content": "A" * 70 + "TAIL_PART"},
            ],
            "images": [str(img)],
        }
        short_sample = {
            "messages": [
                {"role": "user", "content": "<image> Go."},
                {"role": "assistant", "content": "SHORT_OK"},
            ],
            "images": [str(img)],
        }

        rollout_jsonl = root / "trunc_rollouts.jsonl"
        rollout_jsonl.write_text(
            json.dumps(long_sample) + "\n" + json.dumps(short_sample) + "\n", encoding="utf-8"
        )
        topk_jsonl = root / "trunc_topk.jsonl"
        cache_dir = root / "trunc_cache"

        # 1. Cache full UNTRUNCATED Top-K shards
        cache_dataset_topk(
            teacher_model_path=str(native_dir),
            rollout_jsonl=rollout_jsonl,
            output_jsonl=topk_jsonl,
            cache_dir=cache_dir,
            topk=16,
            pixel_factor=128,
            device="cpu",
            batch_size=2,
        )

        uid_long = compute_topk_cache_uuid(long_sample)
        uid_short = compute_topk_cache_uuid(short_sample)
        assert uid_long != uid_short  # Same prompt + different response MUST NOT collide!
        st_long = cache_dir / f"{uid_long}.safetensors"
        st_short = cache_dir / f"{uid_short}.safetensors"

        raw_long = load_file(str(st_long))
        raw_short = load_file(str(st_short))
        orig_n_long = int(raw_long["topk_indices"].shape[0])
        orig_n_short = int(raw_short["topk_indices"].shape[0])
        assert orig_n_long > 40 and orig_n_short > 4

        # Stamp unmistakable distinct sentinel patterns into Sample 0 vs Sample 1 shards
        # while preserving `target_labels` so token-ID verification also runs!
        sentinel_long_idx = torch.arange(orig_n_long, dtype=torch.int32)[:, None].expand(orig_n_long, 16).contiguous()
        sentinel_short_idx = (
            (100 + torch.arange(orig_n_short, dtype=torch.int32))[:, None].expand(orig_n_short, 16).contiguous()
        )
        save_file(
            {
                "topk_indices": sentinel_long_idx,
                "topk_logprobs": raw_long["topk_logprobs"],
                "target_labels": raw_long["target_labels"],
            },
            str(st_long),
        )
        save_file(
            {
                "topk_indices": sentinel_short_idx,
                "topk_logprobs": raw_short["topk_logprobs"],
                "target_labels": raw_short["target_labels"],
            },
            str(st_short),
        )

        _, processor = get_model_processor(str(tw_init_dir), load_model=False)
        tmpl = get_template(
            processor,
            template_type=QWEN3_5_TREEWEAVE_TEMPLATE_TYPE,
            max_length=72,  # Tight enough to truncate ~half of Sample 0, loose enough to keep 100% of Sample 1!
            truncation_strategy="right",
        )
        tmpl.set_mode("train")
        tmpl.topk_cache_dir = str(cache_dir)

        enc_long = tmpl.encode(long_sample)
        enc_short = tmpl.encode(short_sample)
        trunc_n_long = enc_long["topk_active_tokens"]
        trunc_n_short = enc_short["topk_active_tokens"]
        assert 0 < trunc_n_long < orig_n_long, f"Expected Sample 0 truncation, got {trunc_n_long}/{orig_n_long}"
        assert trunc_n_short == orig_n_short, f"Expected Sample 1 intact, got {trunc_n_short}/{orig_n_short}"

        # 2a. Verify PADDED collation (`padding_free = False`) across MULTIPLE EPOCHS WITHOUT copying dicts!
        tmpl.padding_free = False
        for _epoch in range(2):
            collated_pad = tmpl.data_collator([enc_long, enc_short])
            got_pad_idx = collated_pad["teacher_topk_indices"]
            assert got_pad_idx.shape == (trunc_n_long + trunc_n_short, 16)
            assert torch.equal(got_pad_idx[:trunc_n_long], sentinel_long_idx[:trunc_n_long])
            assert torch.equal(got_pad_idx[trunc_n_long:], sentinel_short_idx)

        # 2b. Verify PACKED collation (`padding_free = True` / `--packing true`) across MULTIPLE EPOCHS!
        tmpl.padding_free = True
        for _epoch in range(2):
            collated_pack = tmpl.data_collator([enc_long, enc_short])
            got_pack_idx = collated_pack["teacher_topk_indices"]
            assert got_pack_idx.shape == (trunc_n_long + trunc_n_short, 16)
            assert torch.equal(got_pack_idx[:trunc_n_long], sentinel_long_idx[:trunc_n_long])
            assert torch.equal(got_pack_idx[trunc_n_long:], sentinel_short_idx)

        # 2b-ii. Verify `loss_scale='default+ignore_empty_think'` (ms-swift SftArguments default for Qwen3.5!)
        # skips the leading `<think>\n\n</think>\n\n` prefix (`s_off > 0`) rather than shifting `<think>` onto Student tokens!
        tmpl_ignore_think = get_template(
            processor,
            template_type=QWEN3_5_TREEWEAVE_TEMPLATE_TYPE,
            max_length=72,
            truncation_strategy="right",
            loss_scale="default+ignore_empty_think",
        )
        tmpl_ignore_think.set_mode("train")
        tmpl_ignore_think.topk_cache_dir = str(cache_dir)
        tmpl_ignore_think.padding_free = True
        enc_long_it = tmpl_ignore_think.encode(long_sample)
        enc_short_it = tmpl_ignore_think.encode(short_sample)
        n_long_it = enc_long_it["topk_active_tokens"]
        n_short_it = enc_short_it["topk_active_tokens"]
        think_skip = orig_n_short - n_short_it
        assert think_skip > 0  # `<think>\n\n</think>\n\n` tokens were masked to -100 by ignore_empty_think!
        collated_it = tmpl_ignore_think.data_collator([enc_long_it, enc_short_it])
        got_it_idx = collated_it["teacher_topk_indices"]
        assert torch.equal(got_it_idx[:n_long_it], sentinel_long_idx[think_skip : think_skip + n_long_it])
        assert torch.equal(got_it_idx[n_long_it:], sentinel_short_idx[think_skip : think_skip + n_short_it])

        # 2c. Verify extreme truncation where Sample 0 loses ALL response tokens (`topk_active_tokens == 0`)!
        tmpl_extreme = get_template(
            processor,
            template_type=QWEN3_5_TREEWEAVE_TEMPLATE_TYPE,
            max_length=28,
            truncation_strategy="right",
        )
        tmpl_extreme.set_mode("train")
        tmpl_extreme.topk_cache_dir = str(cache_dir)
        tmpl_extreme.padding_free = True
        enc_zero = tmpl_extreme.encode(long_sample)
        assert enc_zero["topk_active_tokens"] == 0
        collated_zero_pack = tmpl_extreme.data_collator([enc_zero, enc_short])
        assert torch.equal(collated_zero_pack["teacher_topk_indices"], sentinel_short_idx)

        # 2d. Verify corrupted/under-sized cache AND mismatched `target_labels` raise ValueError immediately!
        bad_st = root / "too_short.safetensors"
        save_file(
            {"topk_indices": sentinel_short_idx[:2], "topk_logprobs": raw_short["topk_logprobs"][:2]},
            str(bad_st),
        )
        tampered = dict(enc_short, topk_cache_path=str(bad_st))
        with pytest.raises(ValueError, match="fewer than"):
            tmpl.data_collator([tampered])

        wrong_tok_st = root / "wrong_targets.safetensors"
        bad_targets = raw_short["target_labels"].clone()
        bad_targets[-1] = (bad_targets[-1] + 1) % 200
        save_file(
            {
                "topk_indices": sentinel_short_idx,
                "topk_logprobs": raw_short["topk_logprobs"],
                "target_labels": bad_targets,
            },
            str(wrong_tok_st),
        )
        tampered_tok = dict(enc_short, topk_cache_path=str(wrong_tok_st))
        with pytest.raises(ValueError, match="target_labels"):
            tmpl.data_collator([tampered_tok])

        # 3. End-to-end `TreeWeaveSft` Stage 1 & Stage 3 training with `--packing true` + `max_length` truncation + `offline_topk_cache`!
        out_s3_trunc = root / "out-s3-trunc-packed"
        args_trunc = TreeWeaveSftArguments(
            treeweave_stage=3,
            distill_teacher_mode="offline_topk_cache",
            topk_cache_dir=str(cache_dir),
            model=str(tw_init_dir),
            dataset=[str(rollout_jsonl)],
            output_dir=str(out_s3_trunc),
            torch_dtype="float32",
            attn_impl="sdpa",
            packing=True,
            packing_length=256,
            max_length=72,
            truncation_strategy="right",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            max_steps=1,
            save_steps=1,
            logging_steps=1,
            distill_topk=16,
            dataloader_num_workers=0,
            dataset_num_proc=1,
            report_to=["none"],
        )
        res_trunc = TreeWeaveSft(args_trunc).main()
        assert Path(res_trunc["last_model_checkpoint"]).is_dir()


def test_tensorboard_offline_logging_writes_treeweave_scalars_and_honors_logging_dir():
    """Verify `--report_to tensorboard` writes real `events.out.tfevents.*` with TreeWeave custom scalars
    (`train/loss`, `train/distill_kl`, `train/rate_retention`, `train/expected_savings`, `train/p_s1_mean`)
    and honors explicit `--logging_dir` even when a stale `TENSORBOARD_LOGGING_DIR` env var exists.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    register_treeweave_with_swift(activate_distill_trainer=True)
    prev_env = os.environ.get("TENSORBOARD_LOGGING_DIR")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native_dir = root / "native-qwen35"
            tw_init_dir = root / "treeweave-init"
            _build_local_mini_qwen3_5_checkpoint_with_processor(native_dir)
            initialize_treeweave_checkpoint(native_dir, tw_init_dir, seed=0)

            img = root / "tb_img.png"
            Image.new("RGB", (128, 128), color=(90, 140, 210)).save(img)
            ds_jsonl = root / "tb_ds.jsonl"
            ds_jsonl.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "<image> Describe briefly."},
                            {"role": "assistant", "content": "Blue square."},
                        ],
                        "images": [str(img)],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            stale_dir = root / "stale_tb"
            target_tb_dir = root / "tb_logs" / "stage2_exp"
            os.environ["TENSORBOARD_LOGGING_DIR"] = str(stale_dir)

            args_tb = TreeWeaveSftArguments(
                treeweave_stage=2,
                model=str(tw_init_dir),
                dataset=[str(ds_jsonl)],
                output_dir=str(root / "out-s2-tb"),
                torch_dtype="float32",
                attn_impl="sdpa",
                packing=False,
                max_length=256,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=1e-4,
                max_steps=1,
                save_steps=1,
                logging_steps=1,
                logging_first_step=True,
                report_to=["tensorboard"],
                logging_dir=str(target_tb_dir),
                distill_topk=16,
                dataloader_num_workers=0,
                dataset_num_proc=1,
            )
            TreeWeaveSft(args_tb).main()

            assert not stale_dir.exists(), "Stale TENSORBOARD_LOGGING_DIR must not receive new logs!"
            event_files = list(target_tb_dir.glob("events.out.tfevents.*"))
            assert len(event_files) >= 1, f"No TensorBoard event file found under {target_tb_dir}"

            ea = EventAccumulator(str(target_tb_dir))
            ea.Reload()
            scalar_tags = set(ea.Tags().get("scalars", []))
            for expected_tag in (
                "train/loss",
                "train/distill_kl",
                "train/rate_retention",
                "train/moe_balance",
                "train/expected_savings",
                "train/img_expected_savings",
                "train/router_entropy",
                "train/p_s1_mean",
                "train/p_s2_mean",
            ):
                assert expected_tag in scalar_tags, f"Missing {expected_tag} in TensorBoard tags: {sorted(scalar_tags)}"
    finally:
        if prev_env is None:
            os.environ.pop("TENSORBOARD_LOGGING_DIR", None)
        else:
            os.environ["TENSORBOARD_LOGGING_DIR"] = prev_env

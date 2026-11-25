"""First-Class CLI Entrypoint (`treeweave-train` / `python -m treeweave.swift_plugin.cli`) for ms-swift Training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

from swift.arguments import SftArguments
from swift.pipelines.train.sft import SwiftSft

from .register import (
    QWEN3_5_TREEWEAVE_MODEL_TYPE,
    QWEN3_5_TREEWEAVE_TEMPLATE_TYPE,
    register_treeweave_with_swift,
)
from .template import get_treeweave_pixel_factor, install_treeweave_resize_patch
from .trainer import TreeWeaveDistillConfig, set_active_distill_config, update_last_checkpoint_symlink


@dataclass
class TreeWeaveSftArguments(SftArguments):
    """Extends `ms-swift`'s `SftArguments` with explicit TreeWeave 3-stage self-distillation & ablation flags."""

    treeweave_stage: int = field(default=1, metadata={"help": "Self-distillation stage: 1, 2, or 3."})
    distill_topk: int = field(default=512, metadata={"help": "Vocabulary Top-K for forward KL distillation."})
    distill_temperature: float = field(default=1.0, metadata={"help": "Distillation softmax temperature."})
    distill_chunk_size: int = field(default=1024, metadata={"help": "Active-token chunk size for LM head & KL."})
    distill_teacher_mode: str = field(
        default="auto",
        metadata={"help": "Teacher mode: auto | shared_frozen | external_teacher | offline_topk_cache."},
    )
    teacher_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to native Qwen3.5 checkpoint for Stage 3 `external_teacher` mode."},
    )
    topk_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing precomputed `<uuid>.safetensors` teacher Top-K shards."},
    )

    # Stage 2 Rate-Distortion & MoE Load-Balancing Hyperparameters
    lambda_rate: float = field(default=0.05, metadata={"help": "Weight on continuous expected retention rate r_total."})
    lambda_bal: float = field(default=0.01, metadata={"help": "Weight on 4-level MoE load-balancing loss."})
    rho_s1: float = field(default=0.75, metadata={"help": "Target activation anchor for spatial s=1 gate."})
    rho_s2: float = field(default=0.50, metadata={"help": "Target activation anchor for spatial s=2 gate."})
    rho_t1: float = field(default=0.65, metadata={"help": "Target activation anchor for temporal t=1 gate."})
    rho_t2: float = field(default=0.50, metadata={"help": "Target activation anchor for temporal t=2 gate."})

    # Architecture & Ablation Overrides (Q1--Q5)
    max_spatial_level: Optional[int] = field(default=None, metadata={"help": "Override max spatial depth (1 or 2)."})
    max_temporal_level: Optional[int] = field(default=None, metadata={"help": "Override max temporal depth (0, 1, or 2)."})
    allow_fine_temporal_merge: Optional[bool] = field(
        default=None, metadata={"help": "Set False for Q2 Rigid Octree ablation."}
    )
    use_volume_attn_bias: Optional[bool] = field(
        default=None, metadata={"help": "Set False for Q4 w/o Volume Bias ablation."}
    )
    volume_bias_mode: Optional[str] = field(
        default=None, metadata={"help": "key_direction | exact_logit_bias | none."}
    )
    stage1_uniform_scale: Optional[str] = field(
        default=None, metadata={"help": "Comma-separated 's,t' (e.g. '1,0') for Q5 single-scale Stage 1 ablation."}
    )
    train_volume_bias_in_stage1: bool = field(default=True)
    freeze_merger_in_stage3: bool = field(default=False)
    add_version: bool = field(
        default=False,
        metadata={"help": "Keep deterministic output_dir without timestamped v0-* nesting by default."},
    )

    def __post_init__(self):
        self._user_root_output_dir = self.output_dir
        register_treeweave_with_swift(activate_distill_trainer=True)
        if self.model_type is None:
            self.model_type = QWEN3_5_TREEWEAVE_MODEL_TYPE
        if self.template is None:
            self.template = QWEN3_5_TREEWEAVE_TEMPLATE_TYPE
        # TreeWeave always uses selective full-parameter subsets per stage (not LoRA)
        self.tuner_type = "full"
        # ViT blocks are frozen across all 3 stages; use non-reentrant checkpointing on the 32-layer LLM!
        if self.vit_gradient_checkpointing is None:
            self.vit_gradient_checkpointing = False
        if self.gradient_checkpointing_kwargs is None:
            self.gradient_checkpointing_kwargs = {"use_reentrant": False}

        u_scale = None
        if self.stage1_uniform_scale:
            parts = [int(x) for x in self.stage1_uniform_scale.split(",")]
            if len(parts) != 2:
                raise ValueError(f"--stage1_uniform_scale expects 's,t', got {self.stage1_uniform_scale!r}.")
            u_scale = (parts[0], parts[1])

        set_active_distill_config(
            TreeWeaveDistillConfig(
                stage=self.treeweave_stage,
                distill_topk=self.distill_topk,
                distill_temperature=self.distill_temperature,
                distill_chunk_size=self.distill_chunk_size,
                teacher_mode=self.distill_teacher_mode,
                teacher_model_path=self.teacher_model_path,
                topk_cache_dir=self.topk_cache_dir,
                lambda_rate=self.lambda_rate,
                lambda_bal=self.lambda_bal,
                rho_s1=self.rho_s1,
                rho_s2=self.rho_s2,
                rho_t1=self.rho_t1,
                rho_t2=self.rho_t2,
                train_volume_bias_in_stage1=self.train_volume_bias_in_stage1,
                freeze_merger_in_stage3=self.freeze_merger_in_stage3,
                stage1_uniform_scale=u_scale,
            )
        )
        super().__post_init__()
        # `transformers>=5.15` reads `TENSORBOARD_LOGGING_DIR` directly inside `TensorBoardCallback`.
        # Force-sync it to `self.logging_dir` so explicit `--logging_dir` overrides never collide with stale env vars.
        if self.logging_dir:
            os.environ["TENSORBOARD_LOGGING_DIR"] = str(self.logging_dir)

    def _check_padding_free(self):
        if (self.padding_free or self.packing) and self.attn_impl in ("sdpa", "eager"):
            if self.packing:
                self.padding_free = True
            from swift.utils import get_logger

            get_logger().info_once(
                f"[TreeWeave] Using single-pass fused GatedDeltaNet + inner per-segment varlen SDPA for `--packing true` "
                f"with attn_impl={self.attn_impl!r} (zero outer segment loop, zero O(L^2) mask)."
            )
            return
        super()._check_padding_free()


class TreeWeaveSft(SwiftSft):
    """`SwiftSft` pipeline applying runtime TreeWeave ablation overrides onto loaded models."""

    args_class = TreeWeaveSftArguments
    args: TreeWeaveSftArguments

    def _prepare_model_tokenizer(self, **kwargs):
        super()._prepare_model_tokenizer(**kwargs)
        if self.model is None:
            return
        tw = self.model.tw_config
        args = self.args
        if args.max_spatial_level is not None:
            tw.max_spatial_level = int(args.max_spatial_level)
        if args.max_temporal_level is not None:
            tw.max_temporal_level = int(args.max_temporal_level)
        if args.allow_fine_temporal_merge is not None:
            tw.allow_fine_temporal_merge = bool(args.allow_fine_temporal_merge)
        if args.use_volume_attn_bias is not None:
            tw.use_volume_attn_bias = bool(args.use_volume_attn_bias)
        if args.volume_bias_mode is not None:
            tw.volume_bias_mode = str(args.volume_bias_mode)
        self.model.sync_config_to_hf()
        forced_pf_raw = os.environ.get("TREEWEAVE_FORCE_PIXEL_FACTOR", "").strip()
        if forced_pf_raw:
            eff_pf = int(forced_pf_raw)
            if eff_pf <= 0 or eff_pf % tw.image_resize_factor != 0:
                raise ValueError(
                    f"TREEWEAVE_FORCE_PIXEL_FACTOR={eff_pf} must be a positive multiple of "
                    f"model image_resize_factor={tw.image_resize_factor}."
                )
        else:
            eff_pf = tw.image_resize_factor
        install_treeweave_resize_patch(eff_pf)

    def _prepare_template(self) -> None:
        super()._prepare_template()
        if self.template is not None:
            self.template.pixel_factor = get_treeweave_pixel_factor()
            self.template.topk_cache_dir = self.args.topk_cache_dir

    def main(self):
        res = super().main()
        if isinstance(res, dict) and res.get("last_model_checkpoint"):
            root_out = getattr(self.args, "_user_root_output_dir", None) or self.args.output_dir
            if root_out:
                update_last_checkpoint_symlink(root_out, res["last_model_checkpoint"])
        return res


def main(argv: Optional[Union[List[str], TreeWeaveSftArguments]] = None):
    register_treeweave_with_swift(activate_distill_trainer=True)
    return TreeWeaveSft(argv).main()


if __name__ == "__main__":
    main()

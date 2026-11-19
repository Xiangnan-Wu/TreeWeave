"""ms-swift Custom Registration Plugin (`--custom_register_path treeweave/swift_plugin/register.py`)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

# Support standalone loading via `swift sft --custom_register_path treeweave/swift_plugin/register.py`
# where `ms-swift`'s `import_external_file` executes this file with `__package__ = None`.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import PretrainedConfig, PreTrainedModel

from swift.model import Model, ModelGroup, ModelMeta, register_model
from swift.model.model_arch import ModelArch
from swift.model.models.qwen import (
    Qwen2VLLoader,
    Qwen3_5Loader,
    _get_qwen3_5_keep_in_fp32_modules,
    _patch_qwen3_5_keep_in_fp32_modules,
    _patch_qwen3_5_linear_attention_sequence_parallel,
    _register_qwen3_5_vision_kwargs_hook,
)
from swift.trainers import TrainerFactory
from swift.utils import get_logger

from treeweave import ARCHITECTURE
from treeweave.modeling_treeweave import Qwen3_5TreeWeaveForConditionalGeneration
from treeweave.swift_plugin.template import (
    QWEN3_5_TREEWEAVE_TEMPLATE_TYPE,
    install_treeweave_resize_patch,
    register_treeweave_template,
)
from treeweave.swift_plugin.trainer import (
    TreeWeaveDistillConfig,
    get_active_distill_config,
    set_active_distill_config,
)

logger = get_logger()

QWEN3_5_TREEWEAVE_MODEL_TYPE = "qwen3_5_treeweave"
_REGISTERED = False


class Qwen3_5TreeWeaveLoader(Qwen3_5Loader):
    """ms-swift ModelLoader for `Qwen3_5TreeWeaveForConditionalGeneration`."""

    def get_model(self, model_dir: str, config: PretrainedConfig, processor, model_kwargs) -> PreTrainedModel:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel

        self.auto_model_cls = Qwen3_5TreeWeaveForConditionalGeneration
        _patch_qwen3_5_linear_attention_sequence_parallel()

        base_fp32: Tuple[str, ...] = _get_qwen3_5_keep_in_fp32_modules(model_dir, config, model_kwargs)
        extra_fp32 = tuple(Qwen3_5TreeWeaveForConditionalGeneration._keep_in_fp32_modules)
        keep_in_fp32 = tuple(dict.fromkeys(base_fp32 + extra_fp32))

        with _patch_qwen3_5_keep_in_fp32_modules(Qwen3_5PreTrainedModel, keep_in_fp32):
            model: Qwen3_5TreeWeaveForConditionalGeneration = Qwen2VLLoader.get_model(  # type: ignore[assignment]
                self, model_dir, config, processor, model_kwargs
            )

        _register_qwen3_5_vision_kwargs_hook(model)
        # Apply optional `TREEWEAVE_*` ablation overrides when invoked via plain `swift sft` CLI
        tw = model.tw_config
        if os.environ.get("TREEWEAVE_MAX_SPATIAL_LEVEL"):
            tw.max_spatial_level = int(os.environ["TREEWEAVE_MAX_SPATIAL_LEVEL"])
        if os.environ.get("TREEWEAVE_MAX_TEMPORAL_LEVEL"):
            tw.max_temporal_level = int(os.environ["TREEWEAVE_MAX_TEMPORAL_LEVEL"])
        if os.environ.get("TREEWEAVE_ALLOW_FINE_TEMPORAL_MERGE"):
            tw.allow_fine_temporal_merge = os.environ["TREEWEAVE_ALLOW_FINE_TEMPORAL_MERGE"].strip().lower() in (
                "1", "true", "yes", "on"
            )
        if os.environ.get("TREEWEAVE_USE_VOLUME_ATTN_BIAS"):
            tw.use_volume_attn_bias = os.environ["TREEWEAVE_USE_VOLUME_ATTN_BIAS"].strip().lower() in (
                "1", "true", "yes", "on"
            )
        if os.environ.get("TREEWEAVE_VOLUME_BIAS_MODE"):
            tw.volume_bias_mode = os.environ["TREEWEAVE_VOLUME_BIAS_MODE"].strip()
        model.sync_config_to_hf()
        install_treeweave_resize_patch(model.tw_config.image_resize_factor)
        return model


def register_treeweave_with_swift(activate_distill_trainer: bool = True) -> None:
    """Idempotently register `qwen3_5_treeweave` model, template, and distillation trainer into `ms-swift`."""
    global _REGISTERED
    register_treeweave_template()

    from swift.model.register import MODEL_MAPPING

    if QWEN3_5_TREEWEAVE_MODEL_TYPE not in MODEL_MAPPING:
        register_model(
            ModelMeta(
                QWEN3_5_TREEWEAVE_MODEL_TYPE,
                [
                    ModelGroup(
                        [
                            Model("TreeWeave/Qwen3.5-4B-TreeWeave", "TreeWeave/Qwen3.5-4B-TreeWeave"),
                        ],
                        QWEN3_5_TREEWEAVE_TEMPLATE_TYPE,
                    )
                ],
                Qwen3_5TreeWeaveLoader,
                template=QWEN3_5_TREEWEAVE_TEMPLATE_TYPE,
                model_arch=ModelArch.qwen3_5,
                architectures=[ARCHITECTURE],
                requires=["transformers>=5.2.0", "qwen_vl_utils>=0.0.14"],
                tags=["vision", "video"],
            )
        )

    if activate_distill_trainer:
        TrainerFactory.TRAINER_MAPPING["causal_lm"] = "treeweave.swift_plugin.trainer.TreeWeaveDistillTrainer"

        # Hook `TunerMixin.prepare_model` & `SftArguments._check_padding_free` once so `swift sft` CLI works natively!
        if not _REGISTERED:
            from swift.arguments import SftArguments
            from swift.pipelines.train.tuner import TunerMixin

            orig_check_pf = SftArguments._check_padding_free

            def _treeweave_check_padding_free(self):
                is_tw = getattr(self, "model_type", None) == QWEN3_5_TREEWEAVE_MODEL_TYPE or (
                    getattr(self, "template", None) == QWEN3_5_TREEWEAVE_TEMPLATE_TYPE
                )
                if is_tw and (self.padding_free or self.packing) and self.attn_impl in ("sdpa", "eager"):
                    if self.packing:
                        self.padding_free = True
                    return
                return orig_check_pf(self)

            SftArguments._check_padding_free = _treeweave_check_padding_free

            orig_prepare = TunerMixin.prepare_model

            @classmethod  # type: ignore[misc]
            def _treeweave_prepare_model(cls, args, model, **kwargs):
                prepared = orig_prepare.__func__(cls, args, model, **kwargs)
                if isinstance(prepared, Qwen3_5TreeWeaveForConditionalGeneration):
                    d_cfg = get_active_distill_config()
                    stats = prepared.configure_stage_trainability(
                        d_cfg.stage,
                        train_volume_bias_in_stage1=d_cfg.train_volume_bias_in_stage1,
                        freeze_merger_in_stage3=d_cfg.freeze_merger_in_stage3,
                    )
                    prepared.enforce_frozen_eval_mode(d_cfg.stage)
                    logger.info(
                        f"[TreeWeave] Stage {stats['stage']} parameter freeze applied: "
                        f"{stats['trainable_params']:,} / {stats['total_params']:,} "
                        f"({100.0 * stats['trainable_params'] / max(stats['total_params'], 1):.2f}%) trainable."
                    )
                return prepared

            TunerMixin.prepare_model = _treeweave_prepare_model
            _REGISTERED = True


# When imported by `swift sft --custom_register_path .../treeweave/swift_plugin/register.py`,
# initialize config from environment and register automatically!
set_active_distill_config(TreeWeaveDistillConfig.from_env())
register_treeweave_with_swift(activate_distill_trainer=True)

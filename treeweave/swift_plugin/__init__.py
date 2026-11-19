"""Native ms-swift plugin for Qwen3.5-TreeWeave 3-Stage Self-Distillation."""

from .register import register_treeweave_with_swift
from .template import Qwen3_5TreeWeaveTemplate, install_treeweave_resize_patch
from .trainer import TreeWeaveDistillConfig, TreeWeaveDistillTrainer

__all__ = [
    "Qwen3_5TreeWeaveTemplate",
    "TreeWeaveDistillConfig",
    "TreeWeaveDistillTrainer",
    "install_treeweave_resize_patch",
    "register_treeweave_with_swift",
]

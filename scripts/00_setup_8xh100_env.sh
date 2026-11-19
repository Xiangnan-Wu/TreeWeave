#!/usr/bin/env bash
# Step 0: Complete 8xH100 Server Environment Setup for Qwen3.5-4B TreeWeave
# Installs ms-swift, transformers>=5.9, FlashAttention, Flash-Linear-Attention (required by Qwen3.5 GatedDeltaNet packing!),
# causal-conv1d, DeepSpeed, and SGLang / vLLM offline rollout engine.
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -U "ms-swift>=4.6.0" "transformers>=5.9.0" "qwen-vl-utils>=0.0.14" \
  deepspeed accelerate peft safetensors pillow einops binpacking decord tensorboard

# Fused CUDA varlen kernels for Qwen3.5 Full-Attention + GatedDeltaNet Linear-Attention sequence packing!
python -m pip install "flash-attn>=2.7.0" "causal-conv1d>=1.5.0" "flash-linear-attention>=0.2.0" --no-build-isolation

# Install offline Teacher rollout engine (vLLM shares exact Qwen3.5 128px + video-timestamp parity with ms-swift;
# SGLang `fa3` is also supported if preferred):
python -m pip install "vllm>=0.14.0"
# python -m pip install "sglang[all]"

python -m pip install --no-deps -e .
chmod +x scripts/*.sh
echo "[TreeWeave] 8xH100 environment setup complete."

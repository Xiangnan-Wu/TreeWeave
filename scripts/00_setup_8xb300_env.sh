#!/usr/bin/env bash
# Step 0 (NVIDIA B300 / Blackwell Ultra 288GB HBM3e): Environment Setup for Qwen3.5-4B TreeWeave
# Requirements on B300 (sm103 / Blackwell Ultra):
#   - PyTorch compiled with CUDA 12.8+ / 13.0 (`cu128` or `cu130`, e.g., torch>=2.7/2.8)
#   - Do NOT force Hopper-only `fa3` on B300; TreeWeave automatically selects `flashinfer` on sm100/sm103!
set -euo pipefail

export TORCH_CUDA_ARCH_LIST="10.0;10.3"

python -c '
import torch
assert torch.cuda.is_available(), "CUDA is not available in this Python environment!"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"[TreeWeave-B300] Detected GPU 0: {name} | sm{cap[0]}{cap[1]} | {vram_gb:.1f} GB VRAM | PyTorch {torch.__version__} (CUDA {torch.version.cuda})")
'

python -m pip install --upgrade pip
python -m pip install -U "ms-swift>=4.6.0" "transformers>=5.9.0" "qwen-vl-utils>=0.0.14" \
  deepspeed accelerate peft safetensors pillow einops binpacking decord tensorboard

# Qwen3.5 GatedDeltaNet linear-attention Triton kernels (`flash-linear-attention` works natively on Blackwell Triton!)
python -m pip install -U "flash-linear-attention>=0.2.0"

# Install `flash-attn` and `causal-conv1d` targeting Blackwell (`sm100;sm103`).
# If prebuilt wheels for your exact CUDA/Torch ABI are not available on PyPI, install offline/release wheels
# OR run training with `ATTN_IMPL=sdpa` (PyTorch cuDNN FlashAttention on B300)!
python -m pip install "causal-conv1d>=1.5.0" --no-build-isolation || \
  echo "[WARN] causal-conv1d failed to build; TreeWeave will fall back to PyTorch conv1d (still exact, slightly slower)."
python -m pip install "flash-attn>=2.7.4" --no-build-isolation || \
  echo "[WARN] flash-attn wheel/build unavailable for sm103; pass ATTN_IMPL=sdpa (uses native B300 cuDNN SDPA) or install a sm103 flash-attn wheel."

# Offline DP=8 rollout engine (vLLM or SGLang with Blackwell flashinfer backend; note fa3 is Hopper-only!)
python -m pip install "vllm>=0.14.0"
# python -m pip install "sglang[all]"

python -m pip install --no-deps -e .
chmod +x scripts/*.sh scripts/prepare_minimal_data.py
echo "[TreeWeave-B300] Environment setup complete."

#!/usr/bin/env bash
# Step 2b (Optional): 8xH100 Data-Parallel (DP=8) Offline Teacher Top-512 Logprob Extraction
# Pre-extracts crash-safe `<uuid>.safetensors` Top-512 shards so Stage 1 / Stage 2 / Stage 3 and
# all Q1--Q5 ablations can run with ZERO Teacher LLM forward FLOPs and ZERO Teacher VRAM (`TOPK_CACHE_DIR=...`).
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

NATIVE_MODEL_DIR="${NATIVE_MODEL_DIR:-/path/to/Qwen3.5-4B}"
ROLLOUT_JSONL="${ROLLOUT_JSONL:-./rollout_data/teacher_rollouts.jsonl}"
TOPK_JSONL="${TOPK_JSONL:-./rollout_data/teacher_rollouts_topk.jsonl}"
TOPK_CACHE_DIR="${TOPK_CACHE_DIR:-./rollout_data/topk_cache}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TOPK_BATCH_SIZE="${TOPK_BATCH_SIZE:-4}"
TOPK_MAX_TOKENS="${TOPK_MAX_TOKENS:-16384}"
TOPK_ATTN_IMPL="${TOPK_ATTN_IMPL:-auto}"

python -m treeweave.rollout.cache_topk_logits \
  --teacher-model "${NATIVE_MODEL_DIR}" \
  --rollout-jsonl "${ROLLOUT_JSONL}" \
  --output-jsonl "${TOPK_JSONL}" \
  --cache-dir "${TOPK_CACHE_DIR}" \
  --gpus "${GPUS}" \
  --batch-size "${TOPK_BATCH_SIZE}" \
  --max-batch-tokens "${TOPK_MAX_TOKENS}" \
  --pack-sequences \
  --attn-impl "${TOPK_ATTN_IMPL}" \
  --topk 512 \
  --temperature 1.0 \
  --image-max-tokens 4096 \
  --video-max-tokens 512 \
  --max-frames 64 \
  --pixel-factor 128 \
  "$@"

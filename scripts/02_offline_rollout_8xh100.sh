#!/usr/bin/env bash
# Step 2: 8xH100 Data-Parallel (DP=8, TP=1) Offline Self-Distillation Rollout Generation
# Reuses generated trajectories across Stage 1, Stage 2, Stage 3, and all Q1--Q5 ablations!
# --backend auto selects vLLM first (exact ms-swift Qwen3.5 video-timestamp & 128px parity), or SGLang (`fa3`).
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

NATIVE_MODEL_DIR="${NATIVE_MODEL_DIR:-/path/to/Qwen3.5-4B}"
RAW_DATASET_JSONL="${RAW_DATASET_JSONL:-./data/raw_multimodal_mix.jsonl}"
ROLLOUT_JSONL="${ROLLOUT_JSONL:-./rollout_data/teacher_rollouts.jsonl}"
BACKEND="${BACKEND:-auto}"   # auto | sglang | vllm
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
ROLLOUT_MAX_BATCH_SIZE="${ROLLOUT_MAX_BATCH_SIZE:-64}"
ROLLOUT_CHUNK_SIZE="${ROLLOUT_CHUNK_SIZE:-256}"

python -m treeweave.rollout.offline_rollout \
  --model "${NATIVE_MODEL_DIR}" \
  --input-jsonl "${RAW_DATASET_JSONL}" \
  --output-jsonl "${ROLLOUT_JSONL}" \
  --gpus "${GPUS}" \
  --backend "${BACKEND}" \
  --temperature 0.0 \
  --max-new-tokens 1024 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.88 \
  --max-batch-size "${ROLLOUT_MAX_BATCH_SIZE}" \
  --chunk-size "${ROLLOUT_CHUNK_SIZE}" \
  --image-max-tokens 4096 \
  --video-max-tokens 512 \
  --max-frames 64 \
  --pixel-factor 128 \
  "$@"

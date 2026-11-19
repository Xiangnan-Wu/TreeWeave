#!/usr/bin/env bash
# Step 1: Initialize Qwen3.5-4B-TreeWeave checkpoint on CPU (0 GPU required, < 1 second with --link-base-weights)
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

NATIVE_MODEL_DIR="${NATIVE_MODEL_DIR:-/path/to/Qwen3.5-4B}"
OUTPUT_INIT_DIR="${OUTPUT_INIT_DIR:-./checkpoints/qwen3_5-4b-treeweave-init}"
MAX_SPATIAL_LEVEL="${MAX_SPATIAL_LEVEL:-2}"
MAX_TEMPORAL_LEVEL="${MAX_TEMPORAL_LEVEL:-2}"

python -m treeweave.serving.init_checkpoint \
  --source "${NATIVE_MODEL_DIR}" \
  --destination "${OUTPUT_INIT_DIR}" \
  --seed 0 \
  --initial-compression-prob 0.1 \
  --route-threshold 0.5 \
  --max-spatial-level "${MAX_SPATIAL_LEVEL}" \
  --max-temporal-level "${MAX_TEMPORAL_LEVEL}" \
  --link-base-weights \
  "$@"

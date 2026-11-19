#!/usr/bin/env bash
# Step 4: Stage 2 — Bottom-Up Soft-Tree Rate-Distortion & MoE Load-Balanced Router Distillation (8xH100)
# - Trains ONLY Hierarchical Spatiotemporal Router (Spatial Local/Global Cross-Attn + Windowed Temporal Attn + 4 FP32 Heads)
# - 24,576 Packed Sequence Length, Global Batch = 64 packed sequences (1 * 8 * 8 GPUs)
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

STAGE1_CKPT="${STAGE1_CKPT:-./output/qwen3_5-4b-treeweave-stage1/last}"
ROLLOUT_JSONL="${ROLLOUT_JSONL:-./rollout_data/teacher_rollouts.jsonl}"
STAGE2_OUT="${STAGE2_OUT:-./output/qwen3_5-4b-treeweave-stage2}"
TOPK_CACHE_DIR="${TOPK_CACHE_DIR:-}"
TB_LOG_ROOT="${TB_LOG_ROOT:-./tb_logs}"
RUN_TAG="${RUN_TAG:-default}"
TB_LOG_DIR="${TB_LOG_DIR:-${TB_LOG_ROOT}/stage2_${RUN_TAG}}"
export TENSORBOARD_LOGGING_DIR="${TB_LOG_DIR}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29502}"
MAX_STEPS="${MAX_STEPS:-2500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
PACKING_LENGTH="${PACKING_LENGTH:-24576}"
MAX_LENGTH="${MAX_LENGTH:-24576}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

EXTRA_DISTILL_ARGS=()
if [[ -n "${TOPK_CACHE_DIR}" ]]; then
  EXTRA_DISTILL_ARGS+=(--distill_teacher_mode offline_topk_cache --topk_cache_dir "${TOPK_CACHE_DIR}")
fi

export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-4096}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-512}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-64}"

NPROC_PER_NODE="${NPROC_PER_NODE}" torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" -m treeweave.swift_plugin.cli \
  --treeweave_stage 2 \
  --model "${STAGE1_CKPT}" \
  --dataset "${ROLLOUT_JSONL}" \
  --output_dir "${STAGE2_OUT}" \
  ${EXTRA_DISTILL_ARGS[@]+"${EXTRA_DISTILL_ARGS[@]}"} \
  --add_version false \
  --torch_dtype bfloat16 \
  --attn_impl "${ATTN_IMPL}" \
  --deepspeed scripts/ds_zero2_h100.json \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --packing true \
  --packing_length "${PACKING_LENGTH}" \
  --max_length "${MAX_LENGTH}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --max_steps "${MAX_STEPS}" \
  --distill_topk 512 \
  --lambda_rate 0.05 \
  --lambda_bal 0.01 \
  --rho_s1 0.75 \
  --rho_s2 0.50 \
  --rho_t1 0.65 \
  --rho_t2 0.50 \
  --dataloader_num_workers 8 \
  --dataset_num_proc 16 \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 2 \
  --report_to tensorboard \
  --logging_dir "${TB_LOG_DIR}" \
  --logging_first_step true \
  --logging_steps "${LOGGING_STEPS}" \
  "$@"

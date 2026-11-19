#!/usr/bin/env bash
# Step 3: Stage 1 — Stochastic 9-Scale Pyramid Self-Distillation Warmup (ms-swift on 8xH100)
# - Trains ONLY Spatial Poolers (P_sp^1,2), Diff-Aware Temporal Poolers (P_tm^1,2), Scale Embeddings, & Volume Bias
# - Single-ViT-Pass `shared_frozen` Teacher (0 extra ViT FLOPs, 0 extra model VRAM!)
# - 16,384 Packed Sequence Length, Global Batch = 32 packed sequences (1 * 4 * 8 GPUs)
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

INIT_CKPT="${INIT_CKPT:-./checkpoints/qwen3_5-4b-treeweave-init}"
ROLLOUT_JSONL="${ROLLOUT_JSONL:-./rollout_data/teacher_rollouts.jsonl}"
STAGE1_OUT="${STAGE1_OUT:-./output/qwen3_5-4b-treeweave-stage1}"
TOPK_CACHE_DIR="${TOPK_CACHE_DIR:-}"
TB_LOG_ROOT="${TB_LOG_ROOT:-./tb_logs}"
RUN_TAG="${RUN_TAG:-default}"
TB_LOG_DIR="${TB_LOG_DIR:-${TB_LOG_ROOT}/stage1_${RUN_TAG}}"
export TENSORBOARD_LOGGING_DIR="${TB_LOG_DIR}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"
MAX_STEPS="${MAX_STEPS:-5257}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
PACKING_LENGTH="${PACKING_LENGTH:-16384}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
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
  --treeweave_stage 1 \
  --model "${INIT_CKPT}" \
  --dataset "${ROLLOUT_JSONL}" \
  --output_dir "${STAGE1_OUT}" \
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
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr": 5e-6}' \
  --warmup_ratio 0.05 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --max_steps "${MAX_STEPS}" \
  --distill_topk 512 \
  --distill_chunk_size 1024 \
  --dataloader_num_workers 8 \
  --dataset_num_proc 16 \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 2 \
  --report_to tensorboard \
  --logging_dir "${TB_LOG_DIR}" \
  --logging_first_step true \
  --logging_steps "${LOGGING_STEPS}" \
  "$@"

#!/usr/bin/env bash
# Step 5: Offline Router Calibration + Stage 3 Hard-Routed Mixed-Granularity Alignment (8xH100)
# - First calibrates scalar `router_calib_bias` on a 1,024-sample subset targeting e.g. 55% savings
# - Then fine-tunes Poolers + Merger + LLM + Volume Bias under discrete hard routing (Global Batch = 512)
# - Reuses frozen ViT features across Student & Lightweight Teacher (or pass --distill_teacher_mode offline_topk_cache)
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

NATIVE_MODEL_DIR="${NATIVE_MODEL_DIR:-/path/to/Qwen3.5-4B}"
STAGE2_CKPT="${STAGE2_CKPT:-./output/qwen3_5-4b-treeweave-stage2/last}"
ROLLOUT_JSONL="${ROLLOUT_JSONL:-./rollout_data/teacher_rollouts.jsonl}"
STAGE3_OUT="${STAGE3_OUT:-./output/qwen3_5-4b-treeweave-stage3}"
TARGET_SAVINGS="${TARGET_SAVINGS:-0.55}"
TOPK_CACHE_DIR="${TOPK_CACHE_DIR:-}"
TB_LOG_ROOT="${TB_LOG_ROOT:-./tb_logs}"
RUN_TAG="${RUN_TAG:-default}"
TB_LOG_DIR="${TB_LOG_DIR:-${TB_LOG_ROOT}/stage3_${RUN_TAG}}"
export TENSORBOARD_LOGGING_DIR="${TB_LOG_DIR}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29503}"
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-250}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_LENGTH="${MAX_LENGTH:-24576}"
CALIB_MAX_SAMPLES="${CALIB_MAX_SAMPLES:-1024}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

if [[ -n "${TOPK_CACHE_DIR}" ]]; then
  TEACHER_ARGS=(--distill_teacher_mode offline_topk_cache --topk_cache_dir "${TOPK_CACHE_DIR}")
else
  TEACHER_ARGS=(--teacher_model_path "${NATIVE_MODEL_DIR}")
fi

export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-4096}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-512}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-64}"

# 5a. Calibrate fixed scalar router bias (takes ~20s on first assigned GPU, Vision-only)
_FIRST_GPU="${GPUS:-0}"
CALIB_GPU="${CALIB_GPU:-${_FIRST_GPU%%,*}}"
CUDA_VISIBLE_DEVICES="${CALIB_GPU:-0}" python -m treeweave.serving.calibrate_router \
  --checkpoint "${STAGE2_CKPT}" \
  --calibration-jsonl "${ROLLOUT_JSONL}" \
  --target-savings "${TARGET_SAVINGS}" \
  --max-samples "${CALIB_MAX_SAMPLES}"

# 5b. Stage 3 Hard-Routed Self-Distillation (Global Batch = 4 * 16 * 8 = 512, Packing Disabled per VisionWeave recipe)
NPROC_PER_NODE="${NPROC_PER_NODE}" torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" -m treeweave.swift_plugin.cli \
  --treeweave_stage 3 \
  --model "${STAGE2_CKPT}" \
  "${TEACHER_ARGS[@]}" \
  --dataset "${ROLLOUT_JSONL}" \
  --output_dir "${STAGE3_OUT}" \
  --add_version false \
  --torch_dtype bfloat16 \
  --attn_impl "${ATTN_IMPL}" \
  --deepspeed scripts/ds_zero2_h100.json \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --packing false \
  --max_length "${MAX_LENGTH}" \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate 3e-6 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.02 \
  --weight_decay 0.0 \
  --max_grad_norm 0.5 \
  --max_steps "${MAX_STEPS}" \
  --distill_topk 512 \
  --dataloader_num_workers 8 \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 2 \
  --report_to tensorboard \
  --logging_dir "${TB_LOG_DIR}" \
  --logging_first_step true \
  --logging_steps "${LOGGING_STEPS}" \
  "$@"

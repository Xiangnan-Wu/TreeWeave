#!/usr/bin/env bash
# ==============================================================================
# TreeWeave All-in-One Single-Job Launcher (Tuned for NVIDIA 8xB300 288GB HBM3e / 8xH100)
# Merges Shared Offline Rollout + Shared Offline Top-512 Cache + 3 Complete
# Comparative Experiments (Each: Init -> Stage 1 -> Stage 2 -> Calibrate -> Stage 3)
# into ONE resumable job submission!
#
# Hardware Default: `GPU_ARCH=b300` (NVIDIA B300 Blackwell Ultra, 288 GB HBM3e per GPU, sm103)
#   - Disables activation recomputation (`GRADIENT_CHECKPOINTING=false`, ~1.5x faster on 288 GB VRAM!)
#   - Scales rollout (`max_batch_size=256`) & Top-K caching (`batch_size=16, max_tokens=32768`)
#   - Uses larger Stage 3 per-device microbatch (`bs=8`, 4x fewer grad-accum steps)
#   - Auto-detects `flash_attn` vs native B300 cuDNN `sdpa` (`ATTN_IMPL=auto`)
#   - Never forces Hopper-only `fa3` on Blackwell (`offline_rollout.py` & `serve.py` auto-select `flashinfer`)
#   - Set `GPU_ARCH=h100` to switch back to 80 GB H100 memory budgets.
#
# The 3 Built-In Minimal Comparative Experiments:
#   - exp1 (exp1_visionweave_2d): max_spatial_level=1, max_temporal_level=0 (VisionWeave 2D baseline)
#   - exp2 (exp2_quadtree_3lvl):  max_spatial_level=2, max_temporal_level=0 (3-level Spatial Quadtree)
#   - exp3 (exp3_treeweave_full): max_spatial_level=2, max_temporal_level=2 (Full Orthogonal TreeWeave)
#
# Usage Examples:
#   # Dry-run plan inspection (0 GPU required):
#   NATIVE_MODEL_DIR=/data/models/Qwen3.5-4B RAW_DATASET_JSONL=./data/minimal_train_prompts.jsonl \
#     GPU_ARCH=b300 PROFILE=minimal bash scripts/run_three_experiments_one_job.sh --dry-run
#
#   # 3-5 min end-to-end smoke verification on 8xB300 (auto-generates synthetic smoke data):
#   NATIVE_MODEL_DIR=/data/models/Qwen3.5-4B GPU_ARCH=b300 PROFILE=smoke bash scripts/run_three_experiments_one_job.sh
#
#   # Full 3-experiment minimal validation on 8xB300 (~24K samples, 1 single job, ~1.5--2.5 hours total):
#   NATIVE_MODEL_DIR=/data/models/Qwen3.5-4B RAW_DATASET_JSONL=./data/minimal_train_prompts.jsonl \
#     GPU_ARCH=b300 PROFILE=minimal bash scripts/run_three_experiments_one_job.sh
# ==============================================================================
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Force identical 128px native visual grid across exp1 (s<=1), exp2 (s<=2,t=0), and exp3 (s<=2,t<=2)
# so all 3 experiments have 100% identical native token counts AND share ONE `topk_cache`!
export TREEWEAVE_FORCE_PIXEL_FACTOR="${TREEWEAVE_FORCE_PIXEL_FACTOR:-128}"

# -------- User Configurable Environment Variables --------
NATIVE_MODEL_DIR="${NATIVE_MODEL_DIR:-/path/to/Qwen3.5-4B}"
RAW_DATASET_JSONL="${RAW_DATASET_JSONL:-}"
WORK_ROOT="${WORK_ROOT:-./work_minimal}"
GPU_ARCH="${GPU_ARCH:-b300}"                  # b300 (default, 288GB HBM3e) | h100 (80GB)
PROFILE="${PROFILE:-minimal}"                 # smoke | minimal | full
EXPERIMENTS="${EXPERIMENTS:-exp1 exp2 exp3}"  # Space-separated subset of: exp1 exp2 exp3
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"    # vllm | sglang | auto
ENABLE_TOPK_CACHE="${ENABLE_TOPK_CACHE:-1}"   # 1: pre-extract shared DP=8 Top-512 cache (0 GB Teacher VRAM across all 3 exps!)
SKIP_IF_DONE="${SKIP_IF_DONE:-1}"             # 1: skip completed stages automatically on job resubmission
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if [[ -z "${GPUS:-}" ]]; then
  GPUS="0"
  for ((g=1; g<NPROC_PER_NODE; g++)); do
    GPUS="${GPUS},${g}"
  done
fi
BASE_MASTER_PORT="${BASE_MASTER_PORT:-29510}"
TB_LOG_ROOT="${TB_LOG_ROOT:-./tb_logs}"

# Auto-resolve `ATTN_IMPL`: use `flash_attn` if installed & importable, otherwise seamlessly use PyTorch cuDNN `sdpa`!
if [[ -z "${ATTN_IMPL:-}" || "${ATTN_IMPL}" == "auto" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    ATTN_IMPL="flash_attn"
  elif python -c "import flash_attn" >/dev/null 2>&1; then
    ATTN_IMPL="flash_attn"
  else
    echo "[TreeWeave-Job] [INFO] flash_attn not found in Python env; using PyTorch native cuDNN sdpa (exact segment-isolated packing)."
    ATTN_IMPL="sdpa"
  fi
fi

# -------- Hardware-Aware Defaults (B300 288GB vs H100 80GB) --------
case "${GPU_ARCH}" in
  b300|B300|blackwell)
    GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
    ROLLOUT_MAX_BATCH_SIZE="${ROLLOUT_MAX_BATCH_SIZE:-256}"
    ROLLOUT_CHUNK_SIZE="${ROLLOUT_CHUNK_SIZE:-1024}"
    TOPK_BATCH_SIZE="${TOPK_BATCH_SIZE:-16}"
    TOPK_MAX_TOKENS="${TOPK_MAX_TOKENS:-32768}"
    DEFAULT_S1_PACK=24576
    DEFAULT_S2_PACK=32768
    DEFAULT_S3_BS=8
    DEFAULT_S3_GA=4
    DEFAULT_S3_BS_FULL=8
    DEFAULT_S3_GA_FULL=8
    ;;
  h100|H100|hopper)
    GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
    ROLLOUT_MAX_BATCH_SIZE="${ROLLOUT_MAX_BATCH_SIZE:-64}"
    ROLLOUT_CHUNK_SIZE="${ROLLOUT_CHUNK_SIZE:-256}"
    TOPK_BATCH_SIZE="${TOPK_BATCH_SIZE:-4}"
    TOPK_MAX_TOKENS="${TOPK_MAX_TOKENS:-16384}"
    DEFAULT_S1_PACK=16384
    DEFAULT_S2_PACK=24576
    DEFAULT_S3_BS=2
    DEFAULT_S3_GA=16
    DEFAULT_S3_BS_FULL=4
    DEFAULT_S3_GA_FULL=16
    ;;
  *)
    echo "[ERROR] Unknown GPU_ARCH='${GPU_ARCH}'. Use: b300 | h100" >&2
    exit 1
    ;;
esac

# -------- Profile Presets (Step Budgets & Batch Sizes) --------
case "${PROFILE}" in
  smoke)
    S1_STEPS="${S1_STEPS:-20}"      ; S1_SAVE="${S1_SAVE:-20}"    ; S1_GA="${S1_GA:-1}" ; S1_PACK="${S1_PACK:-4096}"
    S2_STEPS="${S2_STEPS:-20}"      ; S2_SAVE="${S2_SAVE:-20}"    ; S2_GA="${S2_GA:-1}" ; S2_PACK="${S2_PACK:-4096}"
    S3_STEPS="${S3_STEPS:-10}"      ; S3_SAVE="${S3_SAVE:-10}"    ; S3_BS="${S3_BS:-1}" ; S3_GA="${S3_GA:-2}" ; S3_MAXLEN="${S3_MAXLEN:-4096}"
    CALIB_SAMPLES="${CALIB_SAMPLES:-64}"
    LOG_STEPS="${LOG_STEPS:-2}"
    ;;
  minimal)
    # Designed for ~24K multimodal mix (~1.5--2.5 hours total on 8xB300, ~4--6 hours on 8xH100!)
    S1_STEPS="${S1_STEPS:-800}"     ; S1_SAVE="${S1_SAVE:-400}"   ; S1_GA="${S1_GA:-4}" ; S1_PACK="${S1_PACK:-${DEFAULT_S1_PACK}}"
    S2_STEPS="${S2_STEPS:-500}"     ; S2_SAVE="${S2_SAVE:-250}"   ; S2_GA="${S2_GA:-4}" ; S2_PACK="${S2_PACK:-${DEFAULT_S2_PACK}}"
    S3_STEPS="${S3_STEPS:-250}"     ; S3_SAVE="${S3_SAVE:-250}"   ; S3_BS="${S3_BS:-${DEFAULT_S3_BS}}" ; S3_GA="${S3_GA:-${DEFAULT_S3_GA}}" ; S3_MAXLEN="${S3_MAXLEN:-24576}"
    CALIB_SAMPLES="${CALIB_SAMPLES:-512}"
    LOG_STEPS="${LOG_STEPS:-10}"
    ;;
  full)
    S1_STEPS="${S1_STEPS:-5257}"    ; S1_SAVE="${S1_SAVE:-1000}"  ; S1_GA="${S1_GA:-4}" ; S1_PACK="${S1_PACK:-${DEFAULT_S1_PACK}}"
    S2_STEPS="${S2_STEPS:-2500}"    ; S2_SAVE="${S2_SAVE:-500}"   ; S2_GA="${S2_GA:-8}" ; S2_PACK="${S2_PACK:-${DEFAULT_S2_PACK}}"
    S3_STEPS="${S3_STEPS:-1000}"    ; S3_SAVE="${S3_SAVE:-250}"   ; S3_BS="${S3_BS:-${DEFAULT_S3_BS_FULL}}" ; S3_GA="${S3_GA:-${DEFAULT_S3_GA_FULL}}" ; S3_MAXLEN="${S3_MAXLEN:-24576}"
    CALIB_SAMPLES="${CALIB_SAMPLES:-1024}"
    LOG_STEPS="${LOG_STEPS:-10}"
    ;;
  *)
    echo "[ERROR] Unknown PROFILE='${PROFILE}'. Use: smoke | minimal | full" >&2
    exit 1
    ;;
esac

if [[ -z "${RAW_DATASET_JSONL}" ]]; then
  if [[ "${PROFILE}" == "smoke" ]]; then
    RAW_DATASET_JSONL="${WORK_ROOT}/smoke_data/smoke_prompts.jsonl"
  else
    echo "[ERROR] Please set RAW_DATASET_JSONL=/path/to/prompts.jsonl (or use PROFILE=smoke)." >&2
    exit 1
  fi
fi

SHARED_DIR="${WORK_ROOT}/shared"
ROLLOUT_JSONL="${SHARED_DIR}/teacher_rollouts.jsonl"
TOPK_JSONL="${SHARED_DIR}/teacher_rollouts_topk.jsonl"
TOPK_CACHE_DIR="${SHARED_DIR}/topk_cache"
MARKER_DIR="${WORK_ROOT}/.markers"
PORT_COUNTER="${BASE_MASTER_PORT}"

next_port() {
  PORT_COUNTER=$((PORT_COUNTER + 1))
}

run_step() {
  local step_name="$1"
  shift
  local marker="${MARKER_DIR}/${step_name}.done"

  if [[ "${SKIP_IF_DONE}" == "1" && -f "${marker}" ]]; then
    echo "[TreeWeave-Job] [SKIP] ${step_name} already completed (${marker})"
    return 0
  fi

  echo ""
  echo "================================================================================"
  echo "[TreeWeave-Job] [START] ${step_name}  ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "================================================================================"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] $*"
    return 0
  fi

  mkdir -p "${MARKER_DIR}" "${SHARED_DIR}" "${TB_LOG_ROOT}"
  "$@"
  touch "${marker}"
  echo "[TreeWeave-Job] [DONE] ${step_name}  ($(date '+%Y-%m-%d %H:%M:%S'))"
  # Short pause so NCCL/CUDA sockets and GPU memory settle cleanly before next phase
  sleep 3
}

echo "[TreeWeave-Job] Configuration:"
echo "  GPU_ARCH=${GPU_ARCH} | PROFILE=${PROFILE} | EXPERIMENTS=${EXPERIMENTS}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE} | GPUS=${GPUS} | ATTN_IMPL=${ATTN_IMPL} | GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}"
echo "  NATIVE_MODEL_DIR=${NATIVE_MODEL_DIR}"
echo "  RAW_DATASET_JSONL=${RAW_DATASET_JSONL}"
echo "  WORK_ROOT=${WORK_ROOT} | TB_LOG_ROOT=${TB_LOG_ROOT}"
echo "  ENABLE_TOPK_CACHE=${ENABLE_TOPK_CACHE} | TREEWEAVE_FORCE_PIXEL_FACTOR=${TREEWEAVE_FORCE_PIXEL_FACTOR}"

if [[ "${DRY_RUN}" != "1" && ! -d "${NATIVE_MODEL_DIR}" ]]; then
  echo "[ERROR] NATIVE_MODEL_DIR does not exist: ${NATIVE_MODEL_DIR}" >&2
  exit 1
fi

# ==============================================================================
# PHASE 0 (SHARED ONCE): Optional Smoke Data + Offline DP=8 Rollout + Top-512 Cache
# ==============================================================================
if [[ "${PROFILE}" == "smoke" && ! -f "${RAW_DATASET_JSONL}" ]]; then
  run_step "phase0_generate_smoke_data" \
    python scripts/prepare_minimal_data.py \
      --mode smoke \
      --output-dir "${WORK_ROOT}/smoke_data" \
      --num-images 64 \
      --num-videos 32
fi

run_step "phase0_offline_rollout" \
  env NATIVE_MODEL_DIR="${NATIVE_MODEL_DIR}" \
      RAW_DATASET_JSONL="${RAW_DATASET_JSONL}" \
      ROLLOUT_JSONL="${ROLLOUT_JSONL}" \
      BACKEND="${ROLLOUT_BACKEND}" \
      GPUS="${GPUS}" \
      ROLLOUT_MAX_BATCH_SIZE="${ROLLOUT_MAX_BATCH_SIZE}" \
      ROLLOUT_CHUNK_SIZE="${ROLLOUT_CHUNK_SIZE}" \
      bash scripts/02_offline_rollout_8xh100.sh

ACTIVE_TOPK_DIR=""
if [[ "${ENABLE_TOPK_CACHE}" == "1" ]]; then
  run_step "phase0_cache_topk_512" \
    env NATIVE_MODEL_DIR="${NATIVE_MODEL_DIR}" \
        ROLLOUT_JSONL="${ROLLOUT_JSONL}" \
        TOPK_JSONL="${TOPK_JSONL}" \
        TOPK_CACHE_DIR="${TOPK_CACHE_DIR}" \
        GPUS="${GPUS}" \
        TOPK_BATCH_SIZE="${TOPK_BATCH_SIZE}" \
        TOPK_MAX_TOKENS="${TOPK_MAX_TOKENS}" \
        bash scripts/02b_cache_topk_8xh100.sh
  ACTIVE_TOPK_DIR="${TOPK_CACHE_DIR}"
fi

# ==============================================================================
# PHASE 1--3: Run Each Selected Comparative Experiment End-to-End in Sequence
# ==============================================================================
for exp_id in ${EXPERIMENTS}; do
  case "${exp_id}" in
    exp1|exp1_visionweave_2d)
      EXP_NAME="exp1_visionweave_2d"
      SP_LVL=1
      TM_LVL=0
      TARGET_SAV="0.38"
      ;;
    exp2|exp2_quadtree_3lvl)
      EXP_NAME="exp2_quadtree_3lvl"
      SP_LVL=2
      TM_LVL=0
      TARGET_SAV="0.50"
      ;;
    exp3|exp3_treeweave_full)
      EXP_NAME="exp3_treeweave_full"
      SP_LVL=2
      TM_LVL=2
      TARGET_SAV="0.58"
      ;;
    *)
      echo "[ERROR] Unknown experiment id '${exp_id}'. Supported: exp1 exp2 exp3" >&2
      exit 1
      ;;
  esac

  EXP_DIR="${WORK_ROOT}/${EXP_NAME}"
  INIT_DIR="${EXP_DIR}/ckpt_init"
  S1_DIR="${EXP_DIR}/stage1"
  S2_DIR="${EXP_DIR}/stage2"
  S3_DIR="${EXP_DIR}/stage3"
  RUN_TAG="${PROFILE}_${EXP_NAME}"

  # Step A: Zero-init checkpoint with target tree depth
  run_step "${EXP_NAME}_init" \
    env NATIVE_MODEL_DIR="${NATIVE_MODEL_DIR}" \
        OUTPUT_INIT_DIR="${INIT_DIR}" \
        MAX_SPATIAL_LEVEL="${SP_LVL}" \
        MAX_TEMPORAL_LEVEL="${TM_LVL}" \
        bash scripts/01_init_qwen3_5_4b.sh

  # Step B: Stage 1 — Stochastic Multi-Scale Pyramid Warmup
  next_port; P1="${PORT_COUNTER}"
  run_step "${EXP_NAME}_stage1" \
    env INIT_CKPT="${INIT_DIR}" \
        ROLLOUT_JSONL="${ROLLOUT_JSONL}" \
        STAGE1_OUT="${S1_DIR}" \
        TOPK_CACHE_DIR="${ACTIVE_TOPK_DIR}" \
        TB_LOG_ROOT="${TB_LOG_ROOT}" \
        RUN_TAG="${RUN_TAG}" \
        GPUS="${GPUS}" \
        NPROC_PER_NODE="${NPROC_PER_NODE}" \
        MASTER_PORT="${P1}" \
        ATTN_IMPL="${ATTN_IMPL}" \
        GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING}" \
        MAX_STEPS="${S1_STEPS}" \
        SAVE_STEPS="${S1_SAVE}" \
        LOGGING_STEPS="${LOG_STEPS}" \
        GRAD_ACCUM="${S1_GA}" \
        PACKING_LENGTH="${S1_PACK}" \
        MAX_LENGTH="${S1_PACK}" \
        bash scripts/03_train_stage1_8xh100.sh \
          --max_spatial_level "${SP_LVL}" \
          --max_temporal_level "${TM_LVL}"

  # Step C: Stage 2 — Soft-Tree Rate-Distortion & MoE Load-Balanced Router Training
  next_port; P2="${PORT_COUNTER}"
  run_step "${EXP_NAME}_stage2" \
    env STAGE1_CKPT="${S1_DIR}/last" \
        ROLLOUT_JSONL="${ROLLOUT_JSONL}" \
        STAGE2_OUT="${S2_DIR}" \
        TOPK_CACHE_DIR="${ACTIVE_TOPK_DIR}" \
        TB_LOG_ROOT="${TB_LOG_ROOT}" \
        RUN_TAG="${RUN_TAG}" \
        GPUS="${GPUS}" \
        NPROC_PER_NODE="${NPROC_PER_NODE}" \
        MASTER_PORT="${P2}" \
        ATTN_IMPL="${ATTN_IMPL}" \
        GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING}" \
        MAX_STEPS="${S2_STEPS}" \
        SAVE_STEPS="${S2_SAVE}" \
        LOGGING_STEPS="${LOG_STEPS}" \
        GRAD_ACCUM="${S2_GA}" \
        PACKING_LENGTH="${S2_PACK}" \
        MAX_LENGTH="${S2_PACK}" \
        bash scripts/04_train_stage2_8xh100.sh \
          --max_spatial_level "${SP_LVL}" \
          --max_temporal_level "${TM_LVL}"

  # Step D: Router Calibration + Stage 3 — Hard-Routed Full-Pipeline Alignment
  next_port; P3="${PORT_COUNTER}"
  run_step "${EXP_NAME}_stage3" \
    env NATIVE_MODEL_DIR="${NATIVE_MODEL_DIR}" \
        STAGE2_CKPT="${S2_DIR}/last" \
        ROLLOUT_JSONL="${ROLLOUT_JSONL}" \
        STAGE3_OUT="${S3_DIR}" \
        TOPK_CACHE_DIR="${ACTIVE_TOPK_DIR}" \
        TARGET_SAVINGS="${TARGET_SAV}" \
        CALIB_MAX_SAMPLES="${CALIB_SAMPLES}" \
        TB_LOG_ROOT="${TB_LOG_ROOT}" \
        RUN_TAG="${RUN_TAG}" \
        GPUS="${GPUS}" \
        NPROC_PER_NODE="${NPROC_PER_NODE}" \
        MASTER_PORT="${P3}" \
        ATTN_IMPL="${ATTN_IMPL}" \
        GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING}" \
        MAX_STEPS="${S3_STEPS}" \
        SAVE_STEPS="${S3_SAVE}" \
        LOGGING_STEPS="${LOG_STEPS}" \
        PER_DEVICE_BS="${S3_BS}" \
        GRAD_ACCUM="${S3_GA}" \
        MAX_LENGTH="${S3_MAXLEN}" \
        bash scripts/05_calibrate_and_stage3_8xh100.sh \
          --max_spatial_level "${SP_LVL}" \
          --max_temporal_level "${TM_LVL}"
done

echo ""
echo "================================================================================"
echo "[TreeWeave-Job] ALL REQUESTED EXPERIMENTS FINISHED SUCCESSFULLY! (GPU_ARCH=${GPU_ARCH})"
echo "  Checkpoints: ${WORK_ROOT}/exp*/stage3/last"
echo "  TensorBoard: TB_LOG_ROOT=${TB_LOG_ROOT} bash scripts/06_view_tensorboard.sh"
echo "================================================================================"

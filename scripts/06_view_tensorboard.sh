#!/usr/bin/env bash
# Offline TensorBoard Viewer for Stage 1 / Stage 2 / Stage 3 & Q1--Q5 Ablation Runs
# Works completely offline on intranet / air-gapped GPU servers (no internet required).
set -euo pipefail

TB_LOG_ROOT="${TB_LOG_ROOT:-./tb_logs}"
TB_PORT="${TB_PORT:-6006}"
TB_HOST="${TB_HOST:-0.0.0.0}"

mkdir -p "${TB_LOG_ROOT}"
echo "[TreeWeave] Starting offline TensorBoard on http://${TB_HOST}:${TB_PORT} (logdir: ${TB_LOG_ROOT})"
echo "[TreeWeave] Local browser SSH tunnel: ssh -N -L ${TB_PORT}:127.0.0.1:${TB_PORT} <user>@<server_ip>"

exec tensorboard \
  --logdir "${TB_LOG_ROOT}" \
  --host "${TB_HOST}" \
  --port "${TB_PORT}" \
  --reload_interval 15

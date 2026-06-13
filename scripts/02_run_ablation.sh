#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
DATA=${1:-./data/latest_history.csv}
DEVICE=${DEVICE:-cuda:0}
python run_ablation.py --data "$DATA" --out-dir ./output/industrial/ablation --epochs 120 --device "$DEVICE" --strict-experts

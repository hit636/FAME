#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
M5_DIR=${1:?Usage: scripts/04_run_public_m5.sh /path/to/m5_dir}
DEVICE=${DEVICE:-cuda:0}
python public_benchmarks/prepare_m5.py --m5-dir "$M5_DIR" --out ./data/m5_fame.csv --max-series ${MAX_SERIES:-5000}
python public_benchmarks/run_public_retail.py --data ./data/m5_fame.csv --name m5 --out-root ./output/public --model-root ./fame_model_public --device "$DEVICE" --epochs 120

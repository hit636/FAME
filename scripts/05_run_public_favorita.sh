#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
FAV_DIR=${1:?Usage: scripts/05_run_public_favorita.sh /path/to/favorita_dir}
DEVICE=${DEVICE:-cuda:1}
python public_benchmarks/prepare_favorita.py --favorita-dir "$FAV_DIR" --out ./data/favorita_fame.csv --max-series ${MAX_SERIES:-5000}
python public_benchmarks/run_public_retail.py --data ./data/favorita_fame.csv --name favorita --out-root ./output/public --model-root ./fame_model_public --device "$DEVICE" --epochs 120

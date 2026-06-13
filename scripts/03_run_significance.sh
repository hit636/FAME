#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
DEVICE=${DEVICE:-cuda:0}
python run_significance_tests.py --model ./fame_model_industrial --history ./output/industrial/history_for_table_v.csv --test ./output/industrial/experiment/split_test.csv --out-dir ./output/industrial/significance --device "$DEVICE"

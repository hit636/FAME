#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
python -m pytest -q
if [ "${RUN_SYNTHETIC_EXAMPLE:-0}" = "1" ]; then
  python example_synthetic.py
fi
python run_fame_experiment.py --data ./data/latest_history.csv --out-dir ./output/smoke/experiment --model-out ./output/smoke/model --fast-artifact --epochs 5 --device cpu
python predict_fame.py --model ./output/smoke/model --history ./data/latest_history.csv --future ./data/future_weather.csv --out ./output/smoke/prediction --device cpu

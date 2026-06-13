#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
DATA=${1:-./data/latest_history.csv}
DEVICE=${DEVICE:-cuda:0}
python run_fame_experiment.py --data "$DATA" --out-dir ./output/industrial/experiment --model-out ./fame_model_industrial --epochs 200 --device "$DEVICE" --strict-experts
python - <<'PY'
import pandas as pd
train=pd.read_csv('./output/industrial/experiment/split_train.csv')
val=pd.read_csv('./output/industrial/experiment/split_validation.csv')
pd.concat([train,val],ignore_index=True).to_csv('./output/industrial/history_for_table_v.csv',index=False)
PY
python run_table_v.py --model ./fame_model_industrial --history ./output/industrial/history_for_table_v.csv --test ./output/industrial/experiment/split_test.csv --out-dir ./output/industrial/table_v --device "$DEVICE" --train-costaware --costaware-model-out ./fame_model_industrial_costaware

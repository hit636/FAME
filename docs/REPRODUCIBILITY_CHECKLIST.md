# FAME Reproducibility Checklist

## 0. Smoke test
```bash
bash scripts/00_smoke_test.sh
```

## 1. Industrial-style Table V on your confidential CSV
```bash
DEVICE=cuda:0 bash scripts/01_run_industrial_table_v.sh /path/to/latest_history.csv
```
Outputs:
- `output/industrial/experiment/experiment_metrics.csv`
- `output/industrial/table_v/table_v_metrics.csv`
- `fame_model_industrial/training_manifest.json`
- `fame_model_industrial/model_metadata.json`

## 2. Ablation / Table X
```bash
DEVICE=cuda:0 bash scripts/02_run_ablation.sh /path/to/latest_history.csv
```
Output: `output/industrial/ablation/ablation_metrics.csv`

## 3. Significance / Table XV
```bash
DEVICE=cuda:0 bash scripts/03_run_significance.sh
```
Output: `output/industrial/significance/significance_tests.csv`

## 4. Public M5
Download the M5 Accuracy files from Kaggle and run:
```bash
MAX_SERIES=5000 DEVICE=cuda:0 bash scripts/04_run_public_m5.sh /path/to/m5
```

## 5. Public Favorita
Download the Corporacion Favorita files from Kaggle and run:
```bash
MAX_SERIES=5000 DEVICE=cuda:1 bash scripts/05_run_public_favorita.sh /path/to/favorita
```

## Strict expert mode
Use `--strict-experts` for paper-style reproduction. In this mode missing Prophet,
LightGBM, XGBoost, statsmodels, or failed expert fitting raises an error instead
of silently falling back.

## Backend audit
Every trained model writes expert backend information to:
- `training_manifest.json`
- `model_metadata.json`

Check that backends are `external_lightgbm`, `external_xgboost`, `prophet`,
`statsmodels`, or intentionally documented portable proxies for TimeMixer/TimesNet.

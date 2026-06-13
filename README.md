# FAME: Forecastability-Aware Mixture of Experts for Heterogeneous Time Series Forecasting

FAME is a sparse mixture-of-experts framework for heterogeneous retail and industrial time-series forecasting. It extracts forecastability fingerprints, mines expert suitability from validation losses, and routes each series to a small budgeted set of forecasting experts.

```text
series + metadata + context
→ forecastability fingerprint
→ expert validation loss matrix
→ oracle suitability targets
→ sparse router
→ Top-r expert execution and forecast fusion
```

This repository provides the official implementation of the FAME workflow.

---
<img width="1716" height="930" alt="image" src="https://github.com/user-attachments/assets/8bba13c3-91bd-4e58-bc08-69fe16670e32" />

## Highlights

- Forecastability fingerprints covering lifecycle, sparsity, intermittency, volatility, trend, seasonality, spectral, metadata and context features.
- Fixed heterogeneous expert pool: SARIMA, ETS, Prophet, Croston/TSB, Linear Regression, XGBoost, LightGBM, DLinear, TimeMixer-style and TimesNet-style experts.
- Oracle expert mining from validation-window loss matrices.
- Cost-aware sparse router with Top-r expert activation.
- Reproducibility scripts for main comparison, ablation, significance testing and public retail benchmarks.

---

## Installation

```bash
pip install -r requirements.txt
```

For GPU experiments, install PyTorch with CUDA support and run with `--device cuda:0` or `--device cuda:1`.

For paper-style reproduction, install the full dependency set and use `--strict-experts` to prevent silent fallbacks.

---

## Quick start

Run a fast CPU smoke test on the included demo data:

```bash
bash scripts/00_smoke_test.sh
```

This checks unit tests, fast training, model saving, prediction and routing explanation generation.

---

## Basic usage

Train FAME:

```bash
python train_fame.py \
  --data ./data/latest_history.csv \
  --out ./fame_model \
  --horizon 14 \
  --top-r 2 \
  --device auto
```

Run deployment-style prediction:

```bash
python predict_fame.py \
  --model ./fame_model \
  --history ./data/latest_history.csv \
  --future ./data/future_weather.csv \
  --out ./output/prediction \
  --device auto
```

Validate predictions when future actual demand is available:

```bash
python validate_fame.py \
  --model ./fame_model \
  --prediction ./output/prediction/prediction.csv \
  --explain ./output/prediction/prediction_explain.csv \
  --actual ./data/future_actual.csv
```

---

## Data format

Minimum required columns:

```text
vem_id, merc_id, date, daily_quantity
```

Recommended optional columns:

```text
merc_brand_code, merc_type_code, machine_type, capacity, scene_code, city_name,
merc_sale_price, max_temperature, min_temperature, weather, wind_level,
event_name, is_offday, coupon_amount, discount_quantity,
is_available, stockout_flag, outage_flag, suspension_flag
```

The default forecasting unit is `(vem_id, merc_id, date)`.

---

## Reproducing paper-style experiments

Main chronological experiment and Table-V-style baselines:

```bash
DEVICE=cuda:0 bash scripts/01_run_industrial_table_v.sh ./data/latest_history.csv
```

Ablation study:

```bash
DEVICE=cuda:0 bash scripts/02_run_ablation.sh ./data/latest_history.csv
```

Paired significance tests:

```bash
DEVICE=cuda:0 bash scripts/03_run_significance.sh
```

Public benchmarks after downloading Kaggle data:

```bash
MAX_SERIES=5000 DEVICE=cuda:0 bash scripts/04_run_public_m5.sh /path/to/m5
MAX_SERIES=5000 DEVICE=cuda:1 bash scripts/05_run_public_favorita.sh /path/to/favorita
```

Detailed reproducibility notes are provided in:

```text
docs/REPRODUCIBILITY_CHECKLIST.md
```

---

## Expert backends and strict mode

The default full configuration follows the paper Table IV settings for SARIMA, ETS, Prophet, LightGBM, XGBoost, DLinear and the neural experts.

Use strict mode for paper-style runs:

```bash
python run_fame_experiment.py \
  --data ./data/latest_history.csv \
  --out-dir ./output/icdm_full \
  --model-out ./fame_model_full \
  --device cuda:0 \
  --epochs 200 \
  --strict-experts
```

Strict mode raises an error if a required expert backend is missing or fails during training or prediction.

Backend audit information is saved in:

```text
training_manifest.json
model_metadata.json
deployment_manifest.json
```

---

## Notes

- The included demo data are for smoke testing only and are not expected to reproduce the paper's industrial numbers.
- Exact reproduction of confidential SNBC results requires the industrial dataset and the same production expert implementations.
- TimeMixer and TimesNet are provided as portable proxy experts in this public artifact. If official implementations are used for final paper numbers, connect them through the expert interface and run with `--strict-experts`.
- Inventory replay is an offline fixed-policy simulation, not an online A/B-test result.

---

## Citation

```bibtex
@inproceedings{fame2026,
  title     = {FAME: Forecastability-Aware Mixture of Experts for Heterogeneous Time Series Forecasting},
  author    = {Li, Qianyang and Zhang, Xingjun and Wang, Shaoxun and Peng, Tao and Wei, Jia},
  year      = {2026}
}
```

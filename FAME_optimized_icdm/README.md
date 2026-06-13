# FAME: Forecastability-Aware Mixture of Experts

This repository is an ICDM-style, production-oriented reference implementation of **FAME** for heterogeneous retail / industrial time-series forecasting.

The code follows the paper pipeline:

1. **Forecastability fingerprint extraction**: lifecycle, sparsity, intermittency, volatility, trend, seasonality, spectral, metadata and context-sensitivity features.
2. **Heterogeneous expert pool**: statistical experts, global machine-learning experts and lightweight deep experts.
3. **Oracle expert mining**: validation-window expert loss matrix, hard oracle labels and soft suitability targets.
4. **Sparse router training**: cost-aware MLP router with prediction loss, KL supervision, balance regularization and cost regularization.
5. **Top-r deployment inference**: select at most `r` active experts, execute selected experts by batch and fuse forecasts with router weights.
6. **Experiment closure**: forecast metrics, routing explanation, expert usage, oracle-hit analysis and optional inventory replay.

The implementation intentionally keeps all default paths relative to the **current directory (`.`)** so it can be copied directly to a Linux server or container.

---

## Directory layout

```text
.
├── train_fame.py              # Offline training: experts + oracle mining + router
├── predict_fame.py            # Deployment inference: batch Top-r routing + fusion
├── validate_fame.py           # Prediction/routing validation and optional replay
├── run_icdm_experiment.py     # Closed-loop chronological experiment for paper evidence
├── generate_fame_csv.py       # Demo CSV generator
├── example_synthetic.py       # Minimal synthetic smoke test
├── requirements.txt
├── data/
│   ├── latest_history.csv     # Historical product-terminal daily demand
│   └── future_weather.csv     # Future date/city context, e.g. weather/holiday/promotion
├── fame/
│   ├── config.py              # FAMEConfig and expert registry
│   ├── features.py            # Forecastability fingerprint and supervised TS features
│   ├── experts.py             # Expert pool implementations
│   ├── oracle.py              # Validation loss matrix and oracle target mining
│   ├── router.py              # Sparse MLP router
│   ├── pipeline.py            # End-to-end FAMEModel
│   ├── data_checks.py         # Industrial data audit and schema validation
│   ├── metrics.py             # Forecast metrics and stock-aware loss
│   ├── monitoring.py          # Router/expert monitoring metrics
│   ├── inventory_replay.py    # Fixed-policy inventory replay simulator
│   └── logging_utils.py       # File + console logging
├── fame_model/                # Saved model and oracle artifacts
├── output/                    # Predictions, explanations and validation reports
└── logs/                      # Runtime logs
```

---

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

`lightgbm` and `xgboost` are optional. If unavailable, the corresponding expert implementation falls back to a scikit-learn model where possible.

---

## 2. Generate demo data

```bash
python generate_fame_csv.py
```

This creates:

```text
data/latest_history.csv
/data/future_weather.csv
```

Historical data requires at least:

```text
vem_id, merc_id, date, daily_quantity
```

The implementation also uses optional fields when present:

```text
merc_brand_code, merc_type_code, machine_type, capacity, scene_code, city_name,
merc_sale_price, max_temperature, min_temperature, weather, wind_level,
event_name, is_offday, coupon_amount, discount_quantity,
is_available, stockout_flag, outage_flag, suspension_flag
```

---

## 3. Offline training

```bash
python train_fame.py \
  --data ./data/latest_history.csv \
  --out ./fame_model \
  --horizon 14 \
  --top-r 2 \
  --delta 0.05 \
  --tau 0.30 \
  --device auto
```

Training outputs:

```text
fame_model/fame_model.joblib
fame_model/oracle_loss_matrix.csv
fame_model/oracle_soft_targets.csv
fame_model/oracle_hard_targets.csv
fame_model/training_manifest.json
fame_model/training_data_audit.json
logs/train_fame.log
```

These files correspond to the paper's offline workflow: fixed expert pool, validation loss matrix, oracle suitability targets and sparse router training.

---

## 4. Deployment inference

```bash
python predict_fame.py \
  --model ./fame_model \
  --history ./data/latest_history.csv \
  --future ./data/future_weather.csv \
  --out ./output/prediction.csv \
  --explain-out ./output/prediction_explain.csv
```

Outputs:

```text
output/prediction.csv
output/prediction_explain.csv
output/deployment_manifest.json
output/deployment_data_audit.json
logs/predict_fame.log
```

`prediction_explain.csv` records the selected expert, normalized routing weight and raw router probability for each series. This is the deployable explanation layer used to inspect expert specialization.

If your future file already contains full `vem_id, merc_id, date` rows, use:

```bash
python predict_fame.py --future-is-full-frame
```

Otherwise, `future_weather.csv` is treated as future context merged by `date + city_name` when possible.

---

## 5. Validation and monitoring

If true future demand is available, save it as:

```text
data/future_actual.csv
```

It must contain:

```text
vem_id, merc_id, date, daily_quantity
```

Then run:

```bash
python validate_fame.py \
  --prediction ./output/prediction.csv \
  --explain ./output/prediction_explain.csv \
  --actual ./data/future_actual.csv \
  --run-replay
```

Outputs:

```text
output/validation/validation_metrics.json
output/validation/validation_metrics_flat.csv
output/validation/validation_joined_predictions.csv
output/validation/inventory_replay.csv        # only when --run-replay is used
logs/validate_fame.log
```

If no actual future file exists, validation still performs deployment QA: empty-output checks, NaN/negative prediction checks, expert usage entropy and average active experts.

---

## 6. Closed-loop ICDM experiment

For a paper-style chronological experiment:

```bash
python run_icdm_experiment.py \
  --data ./data/latest_history.csv \
  --out-dir ./output/icdm_experiment \
  --model-out ./fame_model
```

This script performs:

```text
chronological split
→ expert training
→ oracle mining
→ router training
→ held-out test prediction
→ metric calculation
```

Outputs include split CSVs, test predictions, explanations and a compact metrics table for reporting.

---

## 7. Notes for industrial deployment

- `predict_fame.py` uses **group-by-expert batch inference**: the router first determines selected experts for all series, then each expert is invoked once on its assigned batch.
- `data_checks.py` provides schema validation, duplicate id-date checks, negative demand checks, short-series warnings, date-gap diagnostics and future-context mode detection.
- `deployment_manifest.json` and `training_manifest.json` preserve configuration, data audit and runtime metadata for reproducibility.
- Use `--device cpu`, `--device cuda:0` or `--device cuda:1` to control placement in Kubernetes / Yarn / server deployments.
- The offline replay in `inventory_replay.py` estimates replenishment impact under a fixed policy; it is not an online A/B-test result.

---

## Typical one-command workflow

```bash
python train_fame.py && \
python predict_fame.py && \
python validate_fame.py
```

For paper experiments:

```bash
python run_icdm_experiment.py
```


## ICDM Artifact Alignment Notes

This optimized release is designed to match the FAME paper workflow more closely:

1. **Paper-aligned expert pool.** The default pool exposes SARIMA, ETS,
   Prophet, Croston/TSB, Linear Regression, XGBoost, LightGBM, DLinear,
   TimeMixer and TimesNet. Prophet is optional and falls back to ETS when the
   package is unavailable. TimeMixer/TimesNet are compatible wrappers in the
   portable release; users can replace them with official implementations while
   keeping the same expert names and router interface.
2. **Validation split protocol.** The validation window is separated into an
   oracle-mining section and a router-calibration section. The calibration
   section now selects the deployment-time Top-r pruning threshold `delta`, and
   the selected value is saved in `fame_model/router_calibration.json`.
3. **Recursive ML multi-step prediction.** Tabular experts now predict future
   horizons recursively so lag and rolling features for horizon t+1 are based on
   previous predictions rather than placeholder zeros.
4. **Censored demand masking.** Oracle loss construction and evaluation exclude
   rows censored by stockout, outage, suspension or unavailability flags.
5. **Leakage-safe grid completion.** Chronological experiment scripts split the
   data before grid completion; static metadata may be backfilled within a split,
   while dynamic context is forward-filled only by default.
6. **Experiment closure.** In addition to `train_fame.py`, `predict_fame.py` and
   `validate_fame.py`, the package includes `run_icdm_experiment.py` for the
   paper-style closed loop and `run_baselines.py` for expert-pool baseline tables.

Recommended closed-loop commands:

```bash
python train_fame.py --data ./data/latest_history.csv --out ./fame_model --device cpu
python predict_fame.py --history ./data/latest_history.csv --future ./data/future_weather.csv --model ./fame_model --out ./output --device cpu
python validate_fame.py --model ./fame_model --prediction ./output/prediction.csv --explain ./output/prediction_explain.csv --actual ./data/future_weather.csv
python run_icdm_experiment.py --data ./data/latest_history.csv --out-dir ./output/icdm_experiment --model-out ./fame_model --device cpu
```

## v4 small artifact optimizations

This release adds several small artifact-oriented improvements requested during
code/paper alignment checks:

- `run_icdm_experiment.py --fast-artifact`: a CPU-friendly closed-loop mode that
  uses a reduced expert pool and shorter router/deep training while preserving
  the full FAME workflow. This is intended for reviewers to verify the pipeline
  quickly; production experiments should run the default full expert pool.
- `run_baselines.py`: now reports single experts, best single expert, uniform
  ensemble, dense soft MoE, FAME Top-1, FAME Top-r, a cost-aware deployment proxy,
  and a diagnostic oracle reference.
- Additional smoke tests cover leakage-safe grid completion and the Top-r budget
  guarantee.

Example quick artifact run:

```bash
python run_icdm_experiment.py \
  --data ./data/latest_history.csv \
  --out-dir ./output/icdm_fast \
  --model-out ./fame_model_fast \
  --device cpu \
  --fast-artifact

python run_baselines.py \
  --model ./fame_model_fast \
  --history ./output/icdm_fast/split_train.csv \
  --test ./output/icdm_fast/split_test.csv \
  --out-dir ./output/icdm_fast/baselines \
  --device cpu
```

Artifact scope: the portable release follows the complete FAME workflow, but it
does not exactly reproduce the confidential SNBC production Table V results.
TimeMixer and TimesNet are paper-aligned expert names with DLinear-compatible
fallback wrappers unless users plug in the official implementations through the
`BaseExpert` interface.


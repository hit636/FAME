#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run component ablations for FAME and save a Table-X-style CSV.

The script supports quick reviewer runs and paper-style GPU runs. It retrains a
model for each ablation, evaluates on the same chronological test window, and
reports accuracy, active experts, backend report, and runtime.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import pandas as pd

from fame import FAMEConfig, FAMEModel, ExpertSpec
from fame.metrics import regression_metrics
from fame.utils import set_seed, valid_demand_mask, json_dump


def parse_args():
    ap = argparse.ArgumentParser(description="Run FAME ablation study.")
    ap.add_argument("--data", default="./data/latest_history.csv")
    ap.add_argument("--out-dir", default="./output/ablation")
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--top-r", type=int, default=2)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast-artifact", action="store_true")
    ap.add_argument("--strict-experts", action="store_true")
    ap.add_argument("--variant", required=True)
    return ap.parse_args()


def make_config(args, variant: str) -> FAMEConfig:
    cfg = FAMEConfig(horizon=args.horizon, top_r=args.top_r, delta=args.delta, tau=args.tau,
                     router_epochs=args.epochs, device=args.device, seed=args.seed,
                     strict_experts=args.strict_experts)
    if args.fast_artifact:
        cfg.router_epochs = min(args.epochs, 15)
        cfg.patience = 5
        cfg.hidden_size = 64
        cfg.batch_size = 128
        cfg.lookback = 28
        cfg.max_dlinear_windows = 2_000
        cfg.recursive_ml_prediction = False
        cfg.expert_specs = [
            ExpertSpec("ets", "ets", cost=0.8, params={"seasonal_period": cfg.seasonal_period, "use_statsmodels": False}),
            ExpertSpec("croston_tsb", "tsb", cost=0.6, params={"alpha": 0.1, "beta": 0.1}),
            ExpertSpec("linear", "linear", cost=0.6),
            ExpertSpec("lightgbm", "lightgbm", cost=1.0, params={"n_estimators": 10, "learning_rate": 0.05, "num_leaves": 31, "use_external_lightgbm": False}),
            ExpertSpec("dlinear", "dlinear", cost=3.0, params={"seq_len": 28, "epochs": 1, "batch_size": 64}),
        ]
    if variant == "wo_sparsity_features":
        cfg.disabled_fingerprint_keywords = ("zero_ratio", "adi", "nonzero", "cv2")
    elif variant == "wo_seasonality_features":
        cfg.disabled_fingerprint_keywords = ("seasonal", "acf_peak")
    elif variant == "wo_spectral_features":
        cfg.disabled_fingerprint_keywords = ("spectral", "dominant_frequency", "band_energy")
    elif variant == "wo_metadata_context":
        cfg.metadata_cols = ()
        cfg.context_cols = ()
    elif variant == "wo_balance_loss":
        cfg.beta_balance = 0.0
    elif variant == "fame_acc_top2":
        cfg.gamma_cost = 0.0
        cfg.eta_oracle_cost = 0.0
    elif variant == "fame_costaware":
        cfg.gamma_cost = 0.05
        cfg.eta_oracle_cost = 0.01
        cfg.delta = max(cfg.delta, 0.10)
    return cfg


def eval_model(model: FAMEModel, history: pd.DataFrame, test: pd.DataFrame) -> dict:
    cfg = model.config
    pred, explain = model.predict(history, future_df=test, return_explanations=True)
    merged = test[list(cfg.id_cols) + [cfg.date_col, cfg.target_col]].merge(
        pred[list(cfg.id_cols) + [cfg.date_col, "predicted_sales"]], on=list(cfg.id_cols) + [cfg.date_col], how="inner")
    valid = valid_demand_mask(merged, cfg)
    m = regression_metrics(merged[cfg.target_col], merged["predicted_sales"], valid_mask=valid)
    if not explain.empty:
        per_series = explain.groupby(list(cfg.id_cols)).size()
        m["exec"] = float(per_series.mean())
        cost_map = dict(zip(model.expert_pool.expert_names, model.expert_pool.costs))
        tmp = explain.copy(); tmp["cost"] = tmp["expert"].map(cost_map).fillna(1.0)
        m["norm_cost"] = float(tmp.groupby(list(cfg.id_cols))["cost"].sum().mean())
    else:
        m["exec"] = 0.0; m["norm_cost"] = 0.0
    m["valid_rows"] = int(valid.sum())
    return m


def main() -> int:
    args = parse_args(); set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    v = args.variant
    start = time.time()
    cfg = make_config(args, v)
    df = pd.read_csv(args.data, encoding="utf-8")
    splitter = FAMEModel(cfg)
    raw = splitter._prepare(df, complete_grid=False)
    train_raw, val_raw, test_raw = splitter.chronological_split(raw)
    train_df = splitter._prepare(train_raw, complete_grid=True)
    val_df = splitter._prepare(val_raw, complete_grid=True)
    test_df = splitter._prepare(test_raw, complete_grid=True)
    prepared = pd.concat([train_df, val_df, test_df], ignore_index=True, sort=False)
    model = FAMEModel(cfg).fit(prepared, train_df=train_df, validation_df=val_df, complete_grid=False)
    history_for_test = prepared[prepared[cfg.date_col] < test_df[cfg.date_col].min()].copy()
    metrics = eval_model(model, history_for_test, test_df)
    metrics.update({"variant": v, "runtime_seconds": round(time.time() - start, 3),
                    "expert_backends": str(model.expert_pool.backend_report())})
    result = pd.DataFrame([metrics])
    cols = ["variant", "mse", "rmse", "mae", "wape", "smape", "exec", "norm_cost", "valid_rows", "runtime_seconds", "expert_backends"]
    result = result[[c for c in cols if c in result.columns] + [c for c in result.columns if c not in cols]]
    out_file = out_dir / f"ablation_{v}.csv"
    result.to_csv(out_file, index=False, encoding="utf-8")
    print(result.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-style end-to-end experiment runner for FAME.

This is a reproducible offline protocol that mirrors the FAME paper narrative:
chronological split -> expert training -> oracle mining -> router training ->
held-out test prediction -> accuracy/cost/routing validation.

It is intentionally separate from deployment inference. Deployment uses
``predict_fame.py`` with a saved model and latest history; this file is for
experimental evidence and ablation-style reporting.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from fame import FAMEConfig, FAMEModel, ExpertSpec
from fame.data_checks import audit_history, write_audit_report
from fame.logging_utils import setup_logger
from fame.metrics import regression_metrics, stock_aware_metric
from fame.monitoring import expert_usage_entropy
from fame.utils import json_dump, set_seed, ensure_datetime, chronological_split_dates, valid_demand_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run closed-loop FAME chronological experiment.")
    parser.add_argument("--data", default="./data/latest_history.csv", help="Full historical CSV with target.")
    parser.add_argument("--out-dir", default="./output/fame_experiment", help="Experiment output directory.")
    parser.add_argument("--model-out", default="./fame_model", help="Directory for saved trained model.")
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--top-r", type=int, default=2)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round-output", action="store_true")
    parser.add_argument("--fast-artifact", action="store_true", help="Run a lightweight artifact mode that finishes quickly on CPU by using a reduced expert pool and shorter router/deep training.")
    parser.add_argument("--strict-experts", action="store_true", help="Disable fallback and fail if a paper expert backend is unavailable.")
    parser.add_argument("--keep-censored-training", action="store_true", help="Keep censored stockout/outage rows in expert training/fingerprinting.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger("FAME_EXPERIMENT", log_dir=args.log_dir, log_file="run_fame_experiment.log")
    started = time.time()
    set_seed(args.seed)

    try:
        data_path = Path(args.data)
        out_dir = Path(args.out_dir)
        model_out = Path(args.model_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_out.mkdir(parents=True, exist_ok=True)
        if not data_path.exists():
            raise FileNotFoundError(f"Data not found: {data_path.resolve()}")

        df = pd.read_csv(data_path, encoding="utf-8")
        cfg = FAMEConfig(
            horizon=args.horizon,
            top_r=args.top_r,
            delta=args.delta,
            tau=args.tau,
            router_epochs=args.epochs,
            device=args.device,
            seed=args.seed,
            round_output=args.round_output,
            strict_experts=args.strict_experts,
            exclude_censored_in_training=not args.keep_censored_training,
            exclude_censored_in_fingerprint=not args.keep_censored_training,
        )
        if args.fast_artifact:
            # Fast artifact mode is designed for reviewers who need to verify the
            # full closed loop on a CPU-only machine within a short time budget.
            # It preserves the FAME workflow while using a smaller expert pool and
            # much shorter neural/router training. Production/paper experiments
            # should run without this flag.
            cfg.router_epochs = min(int(args.epochs), 20)
            cfg.patience = min(cfg.patience, 5)
            cfg.hidden_size = min(cfg.hidden_size, 64)
            cfg.batch_size = min(cfg.batch_size, 128)
            cfg.lookback = min(cfg.lookback, 28)
            cfg.max_dlinear_windows = 2_000
            # Disable recursive tabular prediction only in this quick smoke mode
            # to keep artifact verification fast; the default production path
            # keeps recursive_ml_prediction=True to avoid horizon lag pollution.
            cfg.recursive_ml_prediction = False
            cfg.expert_specs = [
                ExpertSpec("ets", "ets", cost=0.8, params={"seasonal_period": cfg.seasonal_period, "use_statsmodels": False}),
                ExpertSpec("croston_tsb", "tsb", cost=0.6, params={"alpha": 0.1, "beta": 0.1}),
                ExpertSpec("linear", "linear", cost=0.6),
                ExpertSpec("lightgbm", "lightgbm", cost=1.0, params={"n_estimators": 5, "learning_rate": 0.05, "num_leaves": 31, "use_external_lightgbm": False}),
                ExpertSpec("dlinear", "dlinear", cost=3.0, params={"seq_len": 28, "epochs": 1, "batch_size": 64}),
            ]
            logger.info("Fast artifact mode enabled: experts=%s router_epochs=%d", [e.name for e in cfg.expert_specs], cfg.router_epochs)
        audit = audit_history(df, cfg, name="experiment_full_data")
        write_audit_report([audit], out_dir / "experiment_data_audit.json")
        for w in audit.warnings:
            logger.warning(w)
        if not audit.ok:
            raise ValueError(f"Experiment data audit failed: {audit.errors}")

        # Split before grid completion to avoid future-context leakage. Each split
        # is then completed independently by FAMEModel._prepare using static-only
        # backfill and dynamic-context forward fill.
        splitter = FAMEModel(cfg)
        raw = splitter._prepare(df, complete_grid=False)
        train_raw, val_raw, test_raw = splitter.chronological_split(raw)
        train_df = splitter._prepare(train_raw, complete_grid=True)
        val_df = splitter._prepare(val_raw, complete_grid=True)
        test_df = splitter._prepare(test_raw, complete_grid=True)
        prepared = pd.concat([train_df, val_df, test_df], ignore_index=True, sort=False)
        if test_df.empty:
            raise ValueError("Test window is empty; provide longer chronological data.")
        train_df.to_csv(out_dir / "split_train.csv", index=False, encoding="utf-8")
        val_df.to_csv(out_dir / "split_validation.csv", index=False, encoding="utf-8")
        test_df.to_csv(out_dir / "split_test.csv", index=False, encoding="utf-8")
        logger.info("Split rows: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

        model = FAMEModel(cfg)
        model.fit(prepared, train_df=train_df, validation_df=val_df, complete_grid=False)
        model.save(str(model_out))
        if model.oracle_ is not None:
            model.oracle_.loss_matrix.to_csv(model_out / "oracle_loss_matrix.csv", index=False, encoding="utf-8")
            model.oracle_.soft_targets.to_csv(model_out / "oracle_soft_targets.csv", index=False, encoding="utf-8")
            model.oracle_.hard_targets.to_csv(model_out / "oracle_hard_targets.csv", index=False, encoding="utf-8")

        history_for_test = prepared[prepared[cfg.date_col] < test_df[cfg.date_col].min()].copy()
        pred, explain = model.predict(history_for_test, future_df=test_df, return_explanations=True)
        pred.to_csv(out_dir / "test_prediction.csv", index=False, encoding="utf-8")
        explain.to_csv(out_dir / "test_prediction_explain.csv", index=False, encoding="utf-8")
        merged = test_df[list(cfg.id_cols) + [cfg.date_col, cfg.target_col]].merge(
            pred[list(cfg.id_cols) + [cfg.date_col, "predicted_sales"]],
            on=list(cfg.id_cols) + [cfg.date_col],
            how="inner",
        )
        valid = valid_demand_mask(merged, cfg)
        metrics = regression_metrics(merged[cfg.target_col], merged["predicted_sales"], valid_mask=valid)
        metrics["stock_aware_loss"] = stock_aware_metric(merged[cfg.target_col], merged["predicted_sales"], valid_mask=valid)
        metrics["valid_demand_rows"] = int(valid.sum())
        metrics["censored_rows_excluded"] = int((~valid).sum())
        metrics["matched_rows"] = int(len(merged))
        metrics["matched_series"] = int(merged[list(cfg.id_cols)].drop_duplicates().shape[0]) if not merged.empty else 0
        metrics["expert_usage_entropy"] = expert_usage_entropy(explain)
        if not explain.empty:
            per_series = explain.groupby(list(cfg.id_cols)).size()
            metrics["avg_active_experts"] = float(per_series.mean())
            metrics["max_active_experts"] = int(per_series.max())
        metrics["runtime_seconds"] = round(time.time() - started, 3)

        json_dump({
            "stage": "closed_loop_fame_experiment",
            "data_path": str(data_path),
            "model_out": str(model_out),
            "split_rows": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
            "metrics": metrics,
            "top_r": cfg.top_r,
            "delta": cfg.delta,
            "tau": cfg.tau,
            "fast_artifact": bool(args.fast_artifact),
            "expert_names": [e.name for e in cfg.enabled_experts()],
            "expert_backends": model.expert_pool.backend_report() if model.expert_pool is not None else [],
        }, out_dir / "experiment_metrics.json")
        pd.DataFrame([metrics]).to_csv(out_dir / "experiment_metrics.csv", index=False, encoding="utf-8")
        logger.info("Experiment metrics: %s", metrics)
        logger.info("Experiment artifacts saved to: %s", out_dir.resolve())
        return 0
    except Exception as exc:
        logger.exception("Experiment failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

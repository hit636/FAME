#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation and experiment-closure script for FAME.

This script closes the experimental loop required by applied data-mining work:
- prediction quality checks for deployment outputs;
- point-forecast metrics when true future demand is available;
- routing/expert usage analysis;
- optional oracle-hit analysis when saved oracle hard targets exist;
- optional fixed-policy inventory replay when actual demand is available.

Default paths are current-directory relative:
    python validate_fame.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from fame import FAMEModel
from fame.inventory_replay import ReplayConfig, fixed_policy_replay
from fame.logging_utils import setup_logger
from fame.metrics import regression_metrics, stock_aware_metric
from fame.utils import valid_demand_mask
from fame.monitoring import expert_usage_entropy
from fame.utils import json_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FAME predictions and routing explanations.")
    parser.add_argument("--model", default="./fame_model", help="Saved model directory. Default: ./fame_model")
    parser.add_argument("--prediction", default="./output/prediction.csv", help="Prediction CSV. Default: ./output/prediction.csv")
    parser.add_argument("--explain", default="./output/prediction_explain.csv", help="Routing explanation CSV.")
    parser.add_argument("--actual", default="./data/future_actual.csv", help="Optional actual future demand CSV. If absent, QA-only validation is run.")
    parser.add_argument("--history", default="./data/latest_history.csv", help="Optional history CSV for context in replay or QA.")
    parser.add_argument("--out-dir", default="./output/validation", help="Validation artifact directory.")
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--under-weight", type=float, default=3.0, help="Under-forecast penalty for stock-aware loss.")
    parser.add_argument("--over-weight", type=float, default=1.0, help="Over-forecast penalty for stock-aware loss.")
    parser.add_argument("--run-replay", action="store_true", help="Run fixed-policy inventory replay when actual demand is available.")
    return parser.parse_args()


def _load_optional(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path, encoding="utf-8")
    return None


def _qa_checks(pred: pd.DataFrame, explain: pd.DataFrame, model: FAMEModel) -> Dict:
    cfg = model.config
    id_cols = list(cfg.id_cols)
    pred_series = pred[id_cols].drop_duplicates().shape[0] if not pred.empty and all(c in pred.columns for c in id_cols) else 0
    exp_series = explain[id_cols].drop_duplicates().shape[0] if not explain.empty and all(c in explain.columns for c in id_cols) else 0
    qa = {
        "prediction_rows": int(len(pred)),
        "prediction_series": int(pred_series),
        "explanation_rows": int(len(explain)),
        "explanation_series": int(exp_series),
        "nan_prediction_rows": int(pred["predicted_sales"].isna().sum()) if "predicted_sales" in pred.columns else None,
        "negative_prediction_rows": int((pred["predicted_sales"] < 0).sum()) if "predicted_sales" in pred.columns else None,
        "expert_usage_entropy": expert_usage_entropy(explain) if not explain.empty else 0.0,
    }
    if not explain.empty and "expert" in explain.columns:
        usage = explain.groupby("expert")["weight"].agg(active_count="count", weight_sum="sum").reset_index()
        qa["expert_usage"] = usage.to_dict("records")
        per_series = explain.groupby(id_cols).size()
        qa["avg_active_experts"] = float(per_series.mean()) if len(per_series) else 0.0
        qa["max_active_experts"] = int(per_series.max()) if len(per_series) else 0
    return qa


def _oracle_hit_analysis(explain: pd.DataFrame, model_dir: Path, model: FAMEModel) -> Dict:
    hard_path = model_dir / "oracle_hard_targets.csv"
    if not hard_path.exists() or explain.empty:
        return {"available": False}
    id_cols = list(model.config.id_cols)
    hard = pd.read_csv(hard_path, encoding="utf-8")
    if "oracle_expert" not in hard.columns:
        return {"available": False}
    routed = explain.groupby(id_cols)["expert"].apply(lambda x: set(map(str, x))).reset_index(name="routed_experts")
    joined = routed.merge(hard[id_cols + ["oracle_expert"]], on=id_cols, how="inner")
    if joined.empty:
        return {"available": False, "reason": "no overlapping series"}
    hits = joined.apply(lambda r: str(r["oracle_expert"]) in r["routed_experts"], axis=1)
    return {
        "available": True,
        "overlap_series": int(len(joined)),
        "selected_oracle_hit_rate": float(hits.mean()),
    }


def _evaluate_with_actual(pred: pd.DataFrame, actual: pd.DataFrame, model: FAMEModel, under_weight: float, over_weight: float) -> tuple[Dict, pd.DataFrame]:
    cfg = model.config
    id_cols = list(cfg.id_cols)
    key_cols = id_cols + [cfg.date_col]
    actual = actual.copy()
    pred = pred.copy()
    actual[cfg.date_col] = pd.to_datetime(actual[cfg.date_col], errors="coerce")
    pred[cfg.date_col] = pd.to_datetime(pred[cfg.date_col], errors="coerce")
    if cfg.target_col not in actual.columns:
        raise ValueError(f"Actual file does not contain target column: {cfg.target_col}")
    merged = actual[key_cols + [cfg.target_col] + [c for c in ["capacity", "initial_inventory"] if c in actual.columns]].merge(
        pred[key_cols + ["predicted_sales"]], on=key_cols, how="inner"
    )
    if merged.empty:
        raise ValueError("No overlapping id-date rows between prediction and actual files.")
    y_true = pd.to_numeric(merged[cfg.target_col], errors="coerce").fillna(0.0)
    y_pred = pd.to_numeric(merged["predicted_sales"], errors="coerce").fillna(0.0)
    valid = valid_demand_mask(merged, cfg)
    metrics = regression_metrics(y_true, y_pred, valid_mask=valid)
    metrics["stock_aware_loss"] = stock_aware_metric(y_true, y_pred, under_weight=under_weight, over_weight=over_weight, valid_mask=valid)
    metrics["valid_demand_rows"] = int(valid.sum())
    metrics["censored_rows_excluded"] = int((~valid).sum())
    metrics["matched_rows"] = int(len(merged))
    metrics["matched_series"] = int(merged[id_cols].drop_duplicates().shape[0])
    return metrics, merged


def main() -> int:
    args = parse_args()
    logger = setup_logger("FAME_VALIDATE", log_dir=args.log_dir, log_file="validate_fame.log")
    started = time.time()

    try:
        model_dir = Path(args.model)
        pred_path = Path(args.prediction)
        explain_path = Path(args.explain)
        actual_path = Path(args.actual)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir.resolve()}")
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_path.resolve()}")
        if not explain_path.exists():
            raise FileNotFoundError(f"Explanation file not found: {explain_path.resolve()}")

        model = FAMEModel.load(str(model_dir))
        pred = pd.read_csv(pred_path, encoding="utf-8")
        explain = pd.read_csv(explain_path, encoding="utf-8")
        qa = _qa_checks(pred, explain, model)
        logger.info("QA prediction rows=%s series=%s", qa.get("prediction_rows"), qa.get("prediction_series"))

        result: Dict = {
            "stage": "validation_and_monitoring",
            "paper_component": "metrics + router quality + deployment QA + optional replay",
            "prediction_path": str(pred_path),
            "explanation_path": str(explain_path),
            "actual_path": str(actual_path) if actual_path.exists() else None,
            "runtime_seconds": None,
            "qa": qa,
            "oracle_hit_analysis": _oracle_hit_analysis(explain, model_dir, model),
            "forecast_metrics": None,
            "replay_metrics": None,
        }

        actual = _load_optional(actual_path)
        if actual is None:
            logger.warning("Actual future demand file not found: %s. Running QA-only validation.", actual_path)
        elif model.config.target_col not in actual.columns:
            logger.warning("Actual file exists but target column %s is missing. Running QA-only validation.", model.config.target_col)
        else:
            metrics, joined = _evaluate_with_actual(pred, actual, model, args.under_weight, args.over_weight)
            result["forecast_metrics"] = metrics
            joined.to_csv(out_dir / "validation_joined_predictions.csv", index=False, encoding="utf-8")
            logger.info("Forecast metrics: %s", metrics)

            if args.run_replay:
                replay_input = joined.copy()
                # Replay expects both target and forecast columns.
                replay_cfg = ReplayConfig(
                    id_cols=model.config.id_cols,
                    date_col=model.config.date_col,
                    target_col=model.config.target_col,
                    forecast_col="predicted_sales",
                    horizon=model.config.horizon,
                    understock_weight=args.under_weight,
                    overstock_weight=args.over_weight,
                )
                replay = fixed_policy_replay(replay_input, replay_cfg)
                replay.to_csv(out_dir / "inventory_replay.csv", index=False, encoding="utf-8")
                result["replay_metrics"] = {
                    "stockout_events": int(replay["stockout_event"].sum()) if not replay.empty else 0,
                    "overstock_exposure_sum": float(replay["overstock_exposure"].sum()) if not replay.empty else 0.0,
                    "stock_aware_loss_mean": float(replay["stock_aware_loss"].mean()) if not replay.empty else None,
                }

        result["runtime_seconds"] = round(time.time() - started, 3)
        json_dump(result, out_dir / "validation_metrics.json")
        # Flat CSV for quick table insertion in paper appendix.
        flat = {"metric": [], "value": []}
        for section, payload in result.items():
            if isinstance(payload, dict):
                for k, v in payload.items():
                    if not isinstance(v, (dict, list)):
                        flat["metric"].append(f"{section}.{k}")
                        flat["value"].append(v)
        pd.DataFrame(flat).to_csv(out_dir / "validation_metrics_flat.csv", index=False, encoding="utf-8")
        logger.info("Validation artifacts saved to: %s", out_dir.resolve())
        return 0
    except Exception as exc:
        logger.exception("Validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

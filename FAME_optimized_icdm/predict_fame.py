#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch deployment inference entrypoint for FAME.

This script corresponds to the online inference workflow in the paper:
1. load the saved expert pool + sparse router;
2. audit latest history and future context;
3. extract current forecastability fingerprints;
4. route each series to Top-r experts;
5. execute selected experts by expert-wise batches;
6. save predictions, routing explanations and deployment manifest.

Default paths use the current project directory:
    python predict_fame.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from fame import FAMEModel
from fame.data_checks import audit_future_context, audit_history, write_audit_report
from fame.logging_utils import setup_logger
from fame.monitoring import expert_usage_entropy
from fame.utils import json_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FAME sparse Top-r batch inference.")
    parser.add_argument("--model", default="./fame_model", help="Saved model directory. Default: ./fame_model")
    parser.add_argument("--history", default="./data/latest_history.csv", help="Latest history CSV. Default: ./data/latest_history.csv")
    parser.add_argument("--future", default="./data/future_weather.csv", help="Future frame/context CSV. Default: ./data/future_weather.csv")
    parser.add_argument("--future-is-full-frame", action="store_true", help="Treat --future as full id-date future frame; otherwise context by date/city.")
    parser.add_argument(
        "--out",
        default="./output/prediction.csv",
        help=(
            "Prediction output path. It can be either a CSV file path, such as "
            "./output/prediction.csv, or a directory path, such as ./output. "
            "When a directory is given, prediction.csv will be created inside it."
        ),
    )
    parser.add_argument("--explain-out", default="./output/prediction_explain.csv", help="Routing explanation output CSV.")
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--device", default=None, help="Override saved device: auto, cpu, cuda:0, cuda:1, ...")
    parser.add_argument("--fail-on-empty", action="store_true", help="Return non-zero when prediction is empty.")
    return parser.parse_args()


def _prediction_quality(pred: pd.DataFrame, explain: pd.DataFrame, model: FAMEModel) -> Dict:
    cfg = model.config
    id_cols = list(cfg.id_cols)
    quality = {
        "prediction_rows": int(len(pred)),
        "prediction_series": int(pred[id_cols].drop_duplicates().shape[0]) if not pred.empty else 0,
        "negative_prediction_rows": int((pred["predicted_sales"] < 0).sum()) if "predicted_sales" in pred.columns else 0,
        "nan_prediction_rows": int(pred["predicted_sales"].isna().sum()) if "predicted_sales" in pred.columns else 0,
        "explanation_rows": int(len(explain)),
        "expert_usage_entropy": expert_usage_entropy(explain) if not explain.empty else 0.0,
    }
    if not explain.empty and "expert" in explain.columns:
        usage = explain.groupby("expert")["weight"].agg(["count", "sum"]).reset_index()
        quality["expert_usage"] = usage.to_dict("records")
        per_series = explain.groupby(id_cols).size()
        quality["avg_active_experts"] = float(per_series.mean()) if len(per_series) else 0.0
        quality["max_active_experts"] = int(per_series.max()) if len(per_series) else 0
    return quality


def main() -> int:
    args = parse_args()
    logger = setup_logger("FAME_DEPLOY", log_dir=args.log_dir, log_file="predict_fame.log")
    started = time.time()

    try:
        model_dir = Path(args.model)
        history_path = Path(args.history)
        future_path = Path(args.future) if args.future else None
        # --out accepts both a concrete CSV file and an output directory.
        # This makes deployment scripts friendlier:
        #   python predict_fame.py --out ./output
        # is equivalent to:
        #   python predict_fame.py --out ./output/prediction.csv
        out_arg = Path(args.out)
        default_explain_arg = "./output/prediction_explain.csv"
        if (out_arg.exists() and out_arg.is_dir()) or out_arg.suffix.lower() != ".csv":
            output_dir = out_arg
            out_path = output_dir / "prediction.csv"
            # If the user did not explicitly override --explain-out, keep the
            # explanation file in the same output directory as prediction.csv.
            if args.explain_out == default_explain_arg:
                explain_path = output_dir / "prediction_explain.csv"
            else:
                explain_path = Path(args.explain_out)
        else:
            out_path = out_arg
            explain_path = Path(args.explain_out)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        explain_path.parent.mkdir(parents=True, exist_ok=True)

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir.resolve()}")
        if not history_path.exists():
            raise FileNotFoundError(f"History CSV not found: {history_path.resolve()}")

        logger.info("Loading model from %s", model_dir)
        model = FAMEModel.load(str(model_dir))
        if args.device:
            model.set_device(args.device)
            logger.info("Runtime device overridden to: %s", args.device)

        history = pd.read_csv(history_path, encoding="utf-8")
        hist_audit = audit_history(history, model.config, name="deployment_history")
        audits = [hist_audit]
        for w in hist_audit.warnings:
            logger.warning(w)
        if not hist_audit.ok:
            for e in hist_audit.errors:
                logger.error(e)
            raise ValueError("Deployment history audit failed.")

        future_df = None
        future_context_df = None
        if future_path is not None and future_path.exists():
            future = pd.read_csv(future_path, encoding="utf-8")
            fut_audit = audit_future_context(future, model.config, name="future_input")
            audits.append(fut_audit)
            for w in fut_audit.warnings:
                logger.warning(w)
            if not fut_audit.ok:
                for e in fut_audit.errors:
                    logger.error(e)
                raise ValueError("Future input audit failed.")
            if args.future_is_full_frame:
                future_df = future
            else:
                future_context_df = future
        elif future_path is not None:
            logger.warning("Future CSV not found: %s. FAME will generate a future frame from latest history.", future_path)

        write_audit_report(audits, out_path.parent / "deployment_data_audit.json")

        logger.info("Running FAME Top-r sparse inference")
        pred, explain = model.predict(
            history_df=history,
            future_df=future_df,
            future_context_df=future_context_df,
            return_explanations=True,
        )
        if pred.empty and args.fail_on_empty:
            raise RuntimeError("Prediction result is empty.")

        pred.to_csv(out_path, index=False, encoding="utf-8")
        explain.to_csv(explain_path, index=False, encoding="utf-8")

        quality = _prediction_quality(pred, explain, model)
        manifest = {
            "stage": "batch_deployment_inference",
            "paper_component": "Top-r budgeted sparse routing and forecast fusion",
            "model_dir": str(model_dir),
            "history_path": str(history_path),
            "future_path": str(future_path) if future_path else None,
            "future_is_full_frame": bool(args.future_is_full_frame),
            "prediction_path": str(out_path),
            "explanation_path": str(explain_path),
            "runtime_seconds": round(time.time() - started, 3),
            "top_r": model.config.top_r,
            "delta": model.config.delta,
            "quality": quality,
        }
        json_dump(manifest, out_path.parent / "deployment_manifest.json")

        logger.info("Predictions saved to: %s", out_path.resolve())
        logger.info("Explanations saved to: %s", explain_path.resolve())
        logger.info("Deployment manifest saved to: %s", out_path.parent / "deployment_manifest.json")
        logger.info("Prediction rows=%d series=%d avg_active_experts=%.3f", quality["prediction_rows"], quality["prediction_series"], quality.get("avg_active_experts", 0.0))
        return 0
    except Exception as exc:
        logger.exception("Batch inference failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

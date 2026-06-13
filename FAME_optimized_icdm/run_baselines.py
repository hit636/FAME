#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-style baseline runner for the FAME artifact.

This script reuses a trained FAME model and a known future/test window to
produce a compact Table-V-style comparison. It is intentionally portable: the
industrial confidential baselines are approximated with the released expert
interfaces, while the method names and evaluation logic match the paper's
accuracy/cost narrative.

Reported rows include:
- every single expert in the pool;
- best single expert;
- uniform ensemble;
- dense soft MoE using router probabilities over all experts;
- FAME Top-1;
- FAME Top-2 or the configured Top-r;
- FAME-CostAware proxy using stricter pruning delta;
- diagnostic oracle reference selected from test-window expert losses.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from fame import FAMEModel
from fame.metrics import regression_metrics
from fame.utils import ensure_datetime, valid_demand_mask, json_dump


def parse_args():
    ap = argparse.ArgumentParser(description="Run paper-style baselines from a saved FAME model.")
    ap.add_argument("--model", default="./fame_model")
    ap.add_argument("--history", default="./data/latest_history.csv")
    ap.add_argument("--test", default="./data/future_weather.csv", help="Known future/test CSV containing daily_quantity.")
    ap.add_argument("--out-dir", default="./output/baselines")
    ap.add_argument("--device", default=None)
    ap.add_argument("--costaware-delta", type=float, default=0.10, help="Proxy deployment threshold for the cost-aware sparse row.")
    return ap.parse_args()


def _metric_row(method: str, actual: pd.DataFrame, pred: pd.DataFrame, cfg, exec_count: float, norm_cost: float) -> Dict:
    merged = actual.merge(
        pred[list(cfg.id_cols) + [cfg.date_col, "prediction"]],
        on=list(cfg.id_cols) + [cfg.date_col],
        how="inner",
    )
    valid = valid_demand_mask(merged, cfg)
    metrics = regression_metrics(merged[cfg.target_col], merged["prediction"], valid_mask=valid)
    metrics.update({
        "method": method,
        "exec": float(exec_count),
        "norm_cost": float(norm_cost),
        "valid_rows": int(valid.sum()),
        "matched_rows": int(len(merged)),
    })
    return metrics


def _fame_variant(model: FAMEModel, name: str, hist: pd.DataFrame, test: pd.DataFrame, top_r: int, delta: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = model.config
    old_top_r, old_delta = cfg.top_r, cfg.delta
    try:
        cfg.top_r = int(top_r)
        cfg.delta = float(delta)
        pred, explain = model.predict(hist, future_df=test, return_explanations=True)
        pred = pred.rename(columns={"predicted_sales": "prediction"})
        pred["method"] = name
        return pred, explain
    finally:
        cfg.top_r, cfg.delta = old_top_r, old_delta


def _route_cost(explain: pd.DataFrame, cost_map: Dict[str, float], id_cols: List[str]) -> Tuple[float, float]:
    if explain.empty:
        return 0.0, 0.0
    per_series_exec = explain.groupby(id_cols).size()
    tmp = explain.copy()
    tmp["expert_cost"] = tmp["expert"].map(cost_map).fillna(1.0)
    per_series_cost = tmp.groupby(id_cols)["expert_cost"].sum()
    return float(per_series_exec.mean()), float(per_series_cost.mean())


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = FAMEModel.load(args.model, device=args.device)
        cfg = model.config
        hist = pd.read_csv(args.history, encoding="utf-8")
        test = pd.read_csv(args.test, encoding="utf-8")
        hist = ensure_datetime(hist, cfg.date_col)
        test = ensure_datetime(test, cfg.date_col)
        if cfg.target_col not in test.columns:
            raise ValueError(f"Test file must contain target column {cfg.target_col}. For deployment-only future_weather.csv, run predict_fame.py instead.")

        actual = test[list(cfg.id_cols) + [cfg.date_col, cfg.target_col]].copy()
        cost_map = dict(zip(model.expert_pool.expert_names, model.expert_pool.costs))
        rows: List[Dict] = []
        pred_frames: List[pd.DataFrame] = []

        # 1) Single experts and prediction panel.
        preds_by_expert = model.expert_pool.predict_all(hist, test)
        for expert_name, pred in preds_by_expert.items():
            pf = pred.copy()
            pf["expert"] = expert_name
            pred_frames.append(pf)
            rows.append(_metric_row(expert_name, actual, pf, cfg, exec_count=1.0, norm_cost=cost_map.get(expert_name, 1.0)))

        if not pred_frames:
            raise RuntimeError("No expert predictions were produced.")

        panel = pd.concat(pred_frames, ignore_index=True)
        id_date = list(cfg.id_cols) + [cfg.date_col]
        expert_names = list(model.expert_pool.expert_names)

        # 2) Best single expert row.
        single_df = pd.DataFrame(rows)
        best_single = single_df.sort_values("mse").iloc[0].to_dict()
        best_single["method"] = "best_single_expert"
        rows.append(best_single)

        # 3) Uniform ensemble.
        uniform = panel.groupby(id_date, as_index=False)["prediction"].mean()
        rows.append(_metric_row(
            "uniform_ensemble", actual, uniform, cfg,
            exec_count=float(len(expert_names)), norm_cost=float(np.sum(model.expert_pool.costs))
        ))

        # 4) Dense soft MoE: router probabilities over all experts, all experts executed.
        hist_prepared = model._prepare(hist, complete_grid=True)
        fp = model.fingerprint_extractor.transform(hist_prepared, reference_date=hist_prepared[cfg.date_col].max())
        probs = model.router.predict_proba(fp)
        dense = panel.merge(probs[list(cfg.id_cols) + expert_names], on=list(cfg.id_cols), how="left")
        dense["router_weight"] = dense.apply(lambda r: float(r.get(r["expert"], 0.0)), axis=1)
        # Normalize only over experts that successfully predicted this row.
        denom = dense.groupby(id_date)["router_weight"].transform("sum").replace(0, np.nan)
        dense["router_weight"] = (dense["router_weight"] / denom).fillna(1.0 / max(1, len(expert_names)))
        dense["weighted_prediction"] = dense["prediction"] * dense["router_weight"]
        dense_pred = dense.groupby(id_date, as_index=False)["weighted_prediction"].sum().rename(columns={"weighted_prediction": "prediction"})
        rows.append(_metric_row(
            "dense_soft_moe", actual, dense_pred, cfg,
            exec_count=float(len(expert_names)), norm_cost=float(np.sum(model.expert_pool.costs))
        ))

        # 5) FAME sparse variants using the trained router.
        for method, top_r, delta in [
            ("fame_top1", 1, cfg.delta),
            (f"fame_top{cfg.top_r}", cfg.top_r, cfg.delta),
            ("fame_costaware_proxy", cfg.top_r, args.costaware_delta),
        ]:
            sparse_pred, explain = _fame_variant(model, method, hist, test, top_r=top_r, delta=delta)
            exec_count, norm_cost = _route_cost(explain, cost_map, list(cfg.id_cols))
            rows.append(_metric_row(method, actual, sparse_pred, cfg, exec_count=exec_count, norm_cost=norm_cost))

        # 6) Diagnostic oracle reference from the test expert-loss panel.
        merged_panel = actual.merge(panel[id_date + ["expert", "prediction"]], on=id_date, how="inner")
        valid = valid_demand_mask(merged_panel, cfg)
        merged_panel["sq_error"] = np.where(valid, (merged_panel[cfg.target_col] - merged_panel["prediction"]) ** 2, np.nan)
        series_expert_loss = merged_panel.groupby(list(cfg.id_cols) + ["expert"], as_index=False)["sq_error"].mean()
        best_expert = series_expert_loss.sort_values("sq_error").groupby(list(cfg.id_cols), as_index=False).first()[list(cfg.id_cols) + ["expert"]]
        oracle_pred = panel.merge(best_expert, on=list(cfg.id_cols) + ["expert"], how="inner")
        oracle_cost = float(best_expert["expert"].map(cost_map).fillna(1.0).mean()) if not best_expert.empty else 0.0
        rows.append(_metric_row("oracle_reference_diagnostic", actual, oracle_pred, cfg, exec_count=1.0, norm_cost=oracle_cost))

        df = pd.DataFrame(rows)
        order = ["method", "mse", "rmse", "mae", "wape", "smape", "exec", "norm_cost", "valid_rows", "matched_rows"]
        df = df[[c for c in order if c in df.columns] + [c for c in df.columns if c not in order]]
        df = df.sort_values("mse")
        df.to_csv(out_dir / "baseline_metrics.csv", index=False, encoding="utf-8")
        json_dump({
            "methods": df.to_dict("records"),
            "expert_names": expert_names,
            "note": "CostAware row is a deployment proxy using stricter delta; full gamma retraining is available in production experiments.",
        }, out_dir / "baseline_metrics.json")
        print(df.to_string(index=False))
        return 0
    except Exception as exc:
        print(f"[ERROR] Baseline runner failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

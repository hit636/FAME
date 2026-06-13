#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce a Table-V-style comparison for the  FAME paper.

This script is intentionally more complete than run_baselines.py. It evaluates:
  - all single experts and best single expert;
  - Uniform ensemble;
  - FFORMA-style learned dense weighting from fingerprints;
  - Stacking ensemble trained on the oracle-mining window;
  - Rule-based USFF-style selector;
  - Cluster-then-forecast selector;
  - AutoML-style selector over forecastability fingerprints;
  - Dense soft MoE;
  - FAME Top-1 / Top-r / independently retrained CostAware FAME;
  - diagnostic validation/test oracle reference.

Industrial confidential data cannot be distributed, but this script runs on any
CSV with the paper schema and on the generated demo CSVs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder

from fame import FAMEModel
from fame.metrics import regression_metrics
from fame.utils import ensure_datetime, valid_demand_mask, json_dump


def parse_args():
    ap = argparse.ArgumentParser(description="Run complete Table-V-style baseline comparison.")
    ap.add_argument("--model", default="./fame_model")
    ap.add_argument("--history", default="./data/latest_history.csv")
    ap.add_argument("--test", required=True, help="Known future/test CSV containing daily_quantity.")
    ap.add_argument("--out-dir", default="./output/table_v")
    ap.add_argument("--device", default=None)
    ap.add_argument("--costaware-delta", type=float, default=0.10)
    ap.add_argument("--train-costaware", action="store_true", help="Retrain a true cost-aware FAME model instead of reporting a delta-only proxy.")
    ap.add_argument("--costaware-model-out", default=None, help="Directory to save the retrained cost-aware model.")
    ap.add_argument("--n-clusters", type=int, default=8)
    return ap.parse_args()


def _id_date(cfg):
    return list(cfg.id_cols) + [cfg.date_col]


def _metric_row(method: str, actual: pd.DataFrame, pred: pd.DataFrame, cfg, exec_count: float, norm_cost: float) -> Dict:
    pred_col = "prediction" if "prediction" in pred.columns else "predicted_sales"
    merged = actual.merge(pred[list(cfg.id_cols) + [cfg.date_col, pred_col]], on=_id_date(cfg), how="inner")
    valid = valid_demand_mask(merged, cfg)
    metrics = regression_metrics(merged[cfg.target_col], merged[pred_col], valid_mask=valid)
    metrics.update({
        "method": method,
        "exec": float(exec_count),
        "norm_cost": float(norm_cost),
        "valid_rows": int(valid.sum()),
        "matched_rows": int(len(merged)),
    })
    return metrics


def _panel_to_wide(panel: pd.DataFrame, cfg, expert_names: List[str]) -> pd.DataFrame:
    wide = panel.pivot_table(index=_id_date(cfg), columns="expert", values="prediction", aggfunc="mean").reset_index()
    for e in expert_names:
        if e not in wide.columns:
            wide[e] = np.nan
    wide[expert_names] = wide[expert_names].replace([np.inf, -np.inf], np.nan)
    wide[expert_names] = wide[expert_names].fillna(wide[expert_names].median()).fillna(0.0)
    return wide[list(cfg.id_cols) + [cfg.date_col] + expert_names]


def _weighted_from_weights(panel: pd.DataFrame, weights: pd.DataFrame, cfg, expert_names: List[str], method: str) -> pd.DataFrame:
    dense = panel.merge(weights[list(cfg.id_cols) + expert_names], on=list(cfg.id_cols), how="left")
    dense["router_weight"] = dense.apply(lambda r: float(r.get(r["expert"], 0.0)), axis=1)
    denom = dense.groupby(_id_date(cfg))["router_weight"].transform("sum").replace(0, np.nan)
    dense["router_weight"] = (dense["router_weight"] / denom).fillna(1.0 / max(1, len(expert_names)))
    dense["weighted_prediction"] = dense["prediction"] * dense["router_weight"]
    out = dense.groupby(_id_date(cfg), as_index=False)["weighted_prediction"].sum().rename(columns={"weighted_prediction": "prediction"})
    out["method"] = method
    return out


def _selector_prediction(panel: pd.DataFrame, selected: pd.DataFrame, cfg, method: str) -> pd.DataFrame:
    sel = selected.copy().rename(columns={"selected_expert": "expert"})
    out = panel.merge(sel[list(cfg.id_cols) + ["expert"]], on=list(cfg.id_cols) + ["expert"], how="inner")
    out = out[list(cfg.id_cols) + [cfg.date_col, "prediction"]].copy()
    out["method"] = method
    return out


def _fame_variant(model: FAMEModel, name: str, hist: pd.DataFrame, test: pd.DataFrame, top_r: int, delta: float):
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


def _usff_rules(fp_raw: pd.DataFrame, available_experts: List[str]) -> pd.DataFrame:
    def pick(r):
        if "croston_tsb" in available_experts and (r.get("zero_ratio", 0) >= 0.5 or r.get("adi", 0) >= 1.32):
            return "croston_tsb"
        if r.get("seasonal_strength", 0) >= 0.45:
            for e in ["timemixer", "sarima", "ets"]:
                if e in available_experts:
                    return e
        if r.get("cv", 0) >= 0.8:
            for e in ["xgboost", "lightgbm", "linear"]:
                if e in available_experts:
                    return e
        for e in ["lightgbm", "xgboost", "linear", available_experts[0]]:
            if e in available_experts:
                return e
        return available_experts[0]
    out = fp_raw[list(fp_raw.columns[:2])].copy()
    # The first two columns are id cols because raw_transform keeps config order.
    out["selected_expert"] = fp_raw.apply(pick, axis=1)
    return out


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = FAMEModel.load(args.model, device=args.device)
        cfg = model.config
        hist = ensure_datetime(pd.read_csv(args.history, encoding="utf-8"), cfg.date_col)
        test = ensure_datetime(pd.read_csv(args.test, encoding="utf-8"), cfg.date_col)
        if cfg.target_col not in test.columns:
            raise ValueError(f"Test file must contain target column {cfg.target_col}.")
        actual = test[list(cfg.id_cols) + [cfg.date_col, cfg.target_col]].copy()
        expert_names = list(model.expert_pool.expert_names)
        cost_map = dict(zip(expert_names, model.expert_pool.costs))
        total_cost = float(np.sum(model.expert_pool.costs))
        rows: List[Dict] = []
        pred_frames: List[pd.DataFrame] = []

        # Shared expert prediction panel.
        preds_by_expert = model.expert_pool.predict_all(hist, test)
        for expert_name, pred in preds_by_expert.items():
            pf = pred.copy(); pf["expert"] = expert_name
            pred_frames.append(pf)
            rows.append(_metric_row(expert_name, actual, pf, cfg, 1.0, cost_map.get(expert_name, 1.0)))
        if not pred_frames:
            raise RuntimeError("No expert predictions were produced.")
        panel = pd.concat(pred_frames, ignore_index=True)
        panel.to_csv(out_dir / "expert_prediction_panel.csv", index=False, encoding="utf-8")

        single_df = pd.DataFrame(rows)
        best_single = single_df.sort_values("mse").iloc[0].to_dict(); best_single["method"] = "best_single_expert"
        rows.append(best_single)

        # Uniform ensemble.
        uniform = panel.groupby(_id_date(cfg), as_index=False)["prediction"].mean()
        rows.append(_metric_row("uniform_ensemble", actual, uniform, cfg, len(expert_names), total_cost))

        # Dense router MoE.
        hist_prep = model._prepare(hist, complete_grid=True)
        fp = model.fingerprint_extractor.transform(hist_prep, reference_date=hist_prep[cfg.date_col].max())
        probs = model.router.predict_proba(fp)
        dense_pred = _weighted_from_weights(panel, probs, cfg, expert_names, "dense_soft_moe")
        rows.append(_metric_row("dense_soft_moe", actual, dense_pred, cfg, len(expert_names), total_cost))

        # FFORMA-style learned dense weighting: train a fingerprint -> soft-oracle
        # multi-output regressor on the oracle-mining window, then use predicted
        # dense weights at test time. This is no longer an oracle-weight proxy.
        if model.oracle_ is not None and model.oracle_.soft_targets is not None:
            train_w = fp.merge(model.oracle_.soft_targets[list(cfg.id_cols) + expert_names], on=list(cfg.id_cols), how="inner")
            Xw = train_w.drop(columns=list(cfg.id_cols) + expert_names).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            Yw = train_w[expert_names].replace([np.inf, -np.inf], np.nan).fillna(1.0 / len(expert_names)).to_numpy(dtype=float)
            if len(Xw) >= 3:
                base = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=cfg.seed, n_jobs=1, verbose=0)
                fforma_model = MultiOutputRegressor(base).fit(Xw, Yw)
                Xp = fp.drop(columns=list(cfg.id_cols)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                W = np.asarray(fforma_model.predict(Xp), dtype=float)
                W = np.clip(W, 0.0, None)
                denom = W.sum(axis=1, keepdims=True)
                W = np.divide(W, np.where(denom > 0, denom, 1.0))
                W[denom.reshape(-1) <= 0] = 1.0 / len(expert_names)
                fforma_weights = fp[list(cfg.id_cols)].copy()
                for j, e in enumerate(expert_names):
                    fforma_weights[e] = W[:, j]
                fforma_pred = _weighted_from_weights(panel, fforma_weights, cfg, expert_names, "fforma_style_weighting")
                rows.append(_metric_row("fforma_style_weighting", actual, fforma_pred, cfg, len(expert_names), total_cost))

        # Stacking ensemble trained on oracle-window tensor.
        if model.oracle_ is not None and model.oracle_.prediction_tensor is not None:
            P = np.asarray(model.oracle_.prediction_tensor, dtype=float)  # [N,M,H]
            Y = np.asarray(model.oracle_.target_tensor, dtype=float)      # [N,H]
            Xs = P.transpose(0, 2, 1).reshape(-1, len(expert_names))
            ys = Y.reshape(-1)
            mask = np.isfinite(ys) & np.all(np.isfinite(Xs), axis=1)
            if mask.sum() >= max(5, len(expert_names)):
                stack = Ridge(alpha=1.0).fit(np.nan_to_num(Xs[mask], nan=0.0), ys[mask])
                wide = _panel_to_wide(panel, cfg, expert_names)
                stack_pred = wide[list(cfg.id_cols) + [cfg.date_col]].copy()
                stack_pred["prediction"] = stack.predict(wide[expert_names].to_numpy(dtype=float))
                rows.append(_metric_row("stacking_ensemble", actual, stack_pred, cfg, len(expert_names), total_cost))

        # Rule-USFF.
        fp_raw = model.fingerprint_extractor.raw_transform(hist_prep, reference_date=hist_prep[cfg.date_col].max())
        usff_sel = _usff_rules(fp_raw, expert_names)
        usff_sel.columns = list(cfg.id_cols) + ["selected_expert"]
        usff_pred = _selector_prediction(panel, usff_sel, cfg, "rule_based_usff")
        rows.append(_metric_row("rule_based_usff", actual, usff_pred, cfg, 1.0, float(usff_sel["selected_expert"].map(cost_map).fillna(1.0).mean())))

        # Cluster-then-forecast.
        if model.oracle_ is not None:
            fp_num = fp.drop(columns=list(cfg.id_cols)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            k = min(args.n_clusters, max(1, len(fp_num)))
            labels = KMeans(n_clusters=k, random_state=cfg.seed, n_init=10).fit_predict(fp_num)
            cluster_df = fp[list(cfg.id_cols)].copy(); cluster_df["cluster"] = labels
            loss = model.oracle_.loss_matrix.merge(cluster_df, on=list(cfg.id_cols), how="inner")
            cluster_best = loss.groupby("cluster")[expert_names].mean().idxmin(axis=1).to_dict()
            cluster_df["selected_expert"] = cluster_df["cluster"].map(cluster_best)
            cluster_pred = _selector_prediction(panel, cluster_df, cfg, "cluster_then_forecast")
            rows.append(_metric_row("cluster_then_forecast", actual, cluster_pred, cfg, 1.0, float(cluster_df["selected_expert"].map(cost_map).fillna(1.0).mean())))

        # AutoML-style selector. Prefer LightGBM classifier to match the paper;
        # fall back to RandomForest only in non-strict artifact environments.
        if model.oracle_ is not None and not model.oracle_.hard_targets.empty:
            train_sel = fp.merge(model.oracle_.hard_targets[list(cfg.id_cols) + ["oracle_expert"]], on=list(cfg.id_cols), how="inner")
            if len(train_sel) >= 3:
                le = LabelEncoder().fit(expert_names)
                X = train_sel.drop(columns=list(cfg.id_cols) + ["oracle_expert"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                y = le.transform(train_sel["oracle_expert"].astype(str))
                try:
                    from lightgbm import LGBMClassifier
                    clf = LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=64,
                                         subsample=0.8, colsample_bytree=0.8,
                                         random_state=cfg.seed, n_jobs=1, verbosity=-1)
                    clf_name = "lightgbm_classifier"
                except Exception as exc:
                    if getattr(cfg, "strict_experts", False):
                        raise RuntimeError(f"AutoML LightGBM classifier unavailable in strict mode: {exc}") from exc
                    clf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=cfg.seed, n_jobs=1, verbose=0)
                    clf_name = "random_forest_classifier_fallback"
                clf.fit(X, y)
                pred_sel = fp[list(cfg.id_cols)].copy()
                Xp = fp.drop(columns=list(cfg.id_cols)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                pred_sel["selected_expert"] = le.inverse_transform(clf.predict(Xp))
                automl_pred = _selector_prediction(panel, pred_sel, cfg, "automl_selector")
                row = _metric_row("automl_selector", actual, automl_pred, cfg, 1.0, float(pred_sel["selected_expert"].map(cost_map).fillna(1.0).mean()))
                row["selector_backend"] = clf_name
                rows.append(row)

        # Sparse FAME variants.
        for method, top_r, delta in [
            ("fame_top1", 1, cfg.delta),
            (f"fame_top{cfg.top_r}", cfg.top_r, cfg.delta),
        ]:
            sparse_pred, explain = _fame_variant(model, method, hist, test, top_r=top_r, delta=delta)
            exec_count, norm_cost = _route_cost(explain, cost_map, list(cfg.id_cols))
            rows.append(_metric_row(method, actual, sparse_pred, cfg, exec_count, norm_cost))
            explain.to_csv(out_dir / f"{method}_explain.csv", index=False, encoding="utf-8")

        if args.train_costaware:
            import copy
            ca_cfg = copy.deepcopy(cfg)
            ca_cfg.gamma_cost = 0.05
            ca_cfg.eta_oracle_cost = 0.01
            ca_cfg.delta = max(float(args.costaware_delta), 0.10)
            ca_cfg.device = args.device or ca_cfg.device
            ca_model = FAMEModel(ca_cfg).fit(hist, complete_grid=True)
            ca_pred, ca_explain = ca_model.predict(hist, future_df=test, return_explanations=True)
            ca_pred = ca_pred.rename(columns={"predicted_sales": "prediction"})
            ca_cost_map = dict(zip(ca_model.expert_pool.expert_names, ca_model.expert_pool.costs))
            exec_count, norm_cost = _route_cost(ca_explain, ca_cost_map, list(cfg.id_cols))
            rows.append(_metric_row("fame_costaware", actual, ca_pred, cfg, exec_count, norm_cost))
            ca_explain.to_csv(out_dir / "fame_costaware_explain.csv", index=False, encoding="utf-8")
            if args.costaware_model_out:
                ca_model.save(args.costaware_model_out)
        else:
            # Keep an explicitly named delta-only diagnostic row for fast reviewer runs.
            sparse_pred, explain = _fame_variant(model, "fame_costaware_delta_proxy", hist, test, top_r=cfg.top_r, delta=args.costaware_delta)
            exec_count, norm_cost = _route_cost(explain, cost_map, list(cfg.id_cols))
            rows.append(_metric_row("fame_costaware_delta_proxy", actual, sparse_pred, cfg, exec_count, norm_cost))
            explain.to_csv(out_dir / "fame_costaware_delta_proxy_explain.csv", index=False, encoding="utf-8")

        # Diagnostic oracle reference from test expert losses. Not a deployable method.
        merged_panel = actual.merge(panel[_id_date(cfg) + ["expert", "prediction"]], on=_id_date(cfg), how="inner")
        valid = valid_demand_mask(merged_panel, cfg)
        merged_panel["sq_error"] = np.where(valid, (merged_panel[cfg.target_col] - merged_panel["prediction"]) ** 2, np.nan)
        series_expert_loss = merged_panel.groupby(list(cfg.id_cols) + ["expert"], as_index=False)["sq_error"].mean()
        best_expert = series_expert_loss.sort_values("sq_error").groupby(list(cfg.id_cols), as_index=False).first()[list(cfg.id_cols) + ["expert"]]
        oracle_pred = panel.merge(best_expert, on=list(cfg.id_cols) + ["expert"], how="inner")
        oracle_cost = float(best_expert["expert"].map(cost_map).fillna(1.0).mean()) if not best_expert.empty else 0.0
        rows.append(_metric_row("oracle_reference_diagnostic", actual, oracle_pred, cfg, 1.0, oracle_cost))

        df = pd.DataFrame(rows)
        order = ["method", "mse", "rmse", "mae", "wape", "smape", "exec", "norm_cost", "valid_rows", "matched_rows"]
        df = df[[c for c in order if c in df.columns] + [c for c in df.columns if c not in order]].sort_values("mse")
        df.to_csv(out_dir / "table_v_metrics.csv", index=False, encoding="utf-8")
        json_dump({
            "methods": df.to_dict("records"),
            "expert_names": expert_names,
            "expert_backends": model.expert_pool.backend_report(),
            "note": "oracle_reference_diagnostic uses test outcomes and is not deployable.",
        }, out_dir / "table_v_metrics.json")
        print(df.to_string(index=False))
        return 0
    except Exception as exc:
        print(f"[ERROR] run_table_v failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

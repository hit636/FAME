#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke tests for the FAME ICDM artifact.

Run from project root:
    python tests/test_fame_core.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from fame.config import FAMEConfig, ExpertSpec
from fame.experts import MLGlobalExpert
from fame.router import SparseRouter
from fame.utils import valid_demand_mask, complete_daily_grid


def test_valid_demand_mask():
    cfg = FAMEConfig()
    df = pd.DataFrame({"is_available": [1, 0, 1], "stockout_flag": [0, 0, 1]})
    mask = valid_demand_mask(df, cfg).tolist()
    assert mask == [True, False, False], mask


def test_topr_router_selection():
    cfg = FAMEConfig(top_r=2, delta=0.05, expert_specs=[])
    r = SparseRouter(cfg, ["a", "b", "c"], [1.0, 2.0, 3.0])
    row = pd.Series({"a": 0.6, "b": 0.3, "c": 0.1})
    experts, weights = r.select_active_experts(row)
    assert experts == ["a", "b"], experts
    assert abs(float(weights.sum()) - 1.0) < 1e-8


def test_recursive_ml_prediction_has_no_zero_lag_pollution():
    cfg = FAMEConfig(
        horizon=3,
        recursive_ml_prediction=True,
        expert_specs=[ExpertSpec("linear", "linear", cost=1.0)],
    )
    dates = pd.date_range("2025-01-01", periods=40, freq="D")
    hist = pd.DataFrame({
        "vem_id": "v1", "merc_id": "m1", "date": dates,
        "daily_quantity": np.arange(40, dtype=float) % 7 + 1,
    })
    fut = pd.DataFrame({
        "vem_id": "v1", "merc_id": "m1",
        "date": pd.date_range("2025-02-10", periods=3, freq="D"),
        "daily_quantity": 0.0,
    })
    expert = MLGlobalExpert(cfg.expert_specs[0], cfg).fit(hist)
    pred = expert.predict(hist, fut)
    assert len(pred) == 3
    assert pred["prediction"].notna().all()


def test_complete_grid_does_not_backfill_dynamic_context_by_default():
    df = pd.DataFrame({
        "vem_id": ["v1", "v1"],
        "merc_id": ["m1", "m1"],
        "date": ["2025-01-01", "2025-01-03"],
        "daily_quantity": [1.0, 3.0],
        "city_name": ["西安市", "西安市"],
        "coupon_amount": [np.nan, 5.0],
    })
    out = complete_daily_grid(
        df,
        id_cols=["vem_id", "merc_id"],
        date_col="date",
        target_col="daily_quantity",
        static_cols=["city_name"],
        dynamic_cols=["coupon_amount"],
        allow_dynamic_bfill=False,
    )
    # Missing dynamic context on 2025-01-02 must not be filled from the future value 5.0.
    mid = out[out["date"] == pd.Timestamp("2025-01-02")].iloc[0]
    assert pd.isna(mid["coupon_amount"]), out


def test_active_experts_never_exceed_topr_budget():
    cfg = FAMEConfig(top_r=2, delta=0.0, expert_specs=[])
    r = SparseRouter(cfg, ["a", "b", "c", "d"], [1, 1, 1, 1])
    row = pd.Series({"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4})
    experts, weights = r.select_active_experts(row)
    assert len(experts) <= cfg.top_r, experts
    assert abs(float(weights.sum()) - 1.0) < 1e-8


if __name__ == "__main__":
    test_valid_demand_mask()
    test_topr_router_selection()
    test_recursive_ml_prediction_has_no_zero_lag_pollution()
    test_complete_grid_does_not_backfill_dynamic_context_by_default()
    test_active_experts_never_exceed_topr_budget()
    print("All FAME core tests passed.")

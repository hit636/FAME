#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation metrics used by FAME experiments.

The metrics mirror the paper-style reporting: MSE, MAE, RMSE, WAPE, sMAPE,
and an asymmetric stock-aware loss used by offline replenishment replay.
"""
from __future__ import annotations

from typing import Dict
import numpy as np

EPS = 1e-8


def regression_metrics(y_true, y_pred, valid_mask=None) -> Dict[str, float]:
    """Return standard point-forecast metrics.

    Inputs are converted to finite numpy arrays. Missing or infinite values are
    replaced by zero to prevent metric jobs from crashing during deployment
    validation; data-quality reports should still be inspected separately.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if valid_mask is not None:
        mask &= np.asarray(valid_mask, dtype=bool)
    yt = yt[mask]
    yp = yp[mask]
    if yt.size == 0:
        return {"mse": np.nan, "rmse": np.nan, "mae": np.nan, "wape": np.nan, "smape": np.nan}
    err = yp - yt
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "wape": float(np.sum(np.abs(err)) / (np.sum(np.abs(yt)) + EPS)),
        "smape": float(np.mean(2 * np.abs(err) / (np.abs(yt) + np.abs(yp) + EPS))),
    }


def stock_aware_metric(y_true, y_pred, under_weight: float = 3.0, over_weight: float = 1.0, valid_mask=None) -> float:
    """Asymmetric inventory-oriented loss used in the paper's replay setting."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if valid_mask is not None:
        mask &= np.asarray(valid_mask, dtype=bool)
    if not mask.any():
        return float("nan")
    yt = yt[mask]
    yp = yp[mask]
    return float(np.mean(under_weight * np.maximum(yt - yp, 0.0) + over_weight * np.maximum(yp - yt, 0.0)))

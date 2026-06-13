# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import FAMEConfig
from .experts import ExpertPool
from .utils import mae, mse, wape, valid_demand_mask


@dataclass
class OracleMiningResult:
    loss_matrix: pd.DataFrame
    soft_targets: pd.DataFrame
    hard_targets: pd.DataFrame
    prediction_tensor: np.ndarray
    target_tensor: np.ndarray
    series_ids: List[str]
    expert_names: List[str]
    horizon: int


def _series_id_from_cols(df: pd.DataFrame, id_cols: Sequence[str]) -> pd.Series:
    return df[list(id_cols)].astype(str).agg("__".join, axis=1)


def build_future_from_window(window_df: pd.DataFrame, cfg: FAMEConfig) -> pd.DataFrame:
    """Return window rows with target masked to 0 so experts only use covariates."""
    future = window_df.copy()
    future[cfg.target_col] = 0.0
    return future


def compute_expert_loss_matrix(
    expert_pool: ExpertPool,
    history_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    cfg: FAMEConfig,
    metric: str = "mse",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """Evaluate every fitted expert on the oracle-mining window.

    Returns
    -------
    loss_matrix:
        [n_series, n_experts] per-series losses.
    pred_tensor:
        [n_series, n_experts, horizon] predictions aligned by series and date.
    target_tensor:
        [n_series, horizon] true targets.
    series_ids:
        aligned series IDs.
    """
    id_cols = list(cfg.id_cols)
    oracle = oracle_df.sort_values(id_cols + [cfg.date_col]).copy()
    oracle["__valid_demand__"] = valid_demand_mask(oracle, cfg).astype(bool)
    oracle["series_id"] = _series_id_from_cols(oracle, id_cols)
    future = build_future_from_window(oracle, cfg)
    preds_by_expert = expert_pool.predict_all(history_df, future)

    # series with complete horizon in oracle window
    groups = []
    for sid, g in oracle.groupby("series_id", sort=False):
        if len(g) >= 1:
            groups.append((sid, g.head(cfg.horizon).copy()))
    series_ids = [sid for sid, _ in groups]
    n = len(series_ids)
    m = len(expert_pool.expert_names)
    horizon = max([len(g) for _, g in groups], default=cfg.horizon)
    pred_tensor = np.full((n, m, horizon), np.nan, dtype=float)
    target_tensor = np.full((n, horizon), np.nan, dtype=float)

    for i, (sid, g) in enumerate(groups):
        target = pd.to_numeric(g[cfg.target_col], errors="coerce").to_numpy(dtype=float)
        valid_mask = g["__valid_demand__"].to_numpy(dtype=bool)
        target = np.where(valid_mask, target, np.nan)
        target_tensor[i, : len(target)] = target

    for j, expert_name in enumerate(expert_pool.expert_names):
        p = preds_by_expert.get(expert_name)
        if p is None or p.empty:
            continue
        p = p.copy()
        p["series_id"] = _series_id_from_cols(p, id_cols)
        for i, (sid, g) in enumerate(groups):
            dates = list(pd.to_datetime(g[cfg.date_col]))
            pp = p[(p["series_id"] == sid) & (pd.to_datetime(p[cfg.date_col]).isin(dates))].sort_values(cfg.date_col)
            vals = pd.to_numeric(pp["prediction"], errors="coerce").to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            pred_tensor[i, j, : min(len(vals), horizon)] = vals[:horizon]

    loss = np.full((n, m), np.inf, dtype=float)
    for i in range(n):
        y = target_tensor[i]
        valid = ~np.isnan(y)
        for j in range(m):
            yhat = pred_tensor[i, j]
            valid2 = valid & ~np.isnan(yhat)
            if valid2.sum() == 0:
                continue
            if metric == "mae":
                loss[i, j] = mae(y[valid2], yhat[valid2])
            elif metric == "wape":
                loss[i, j] = wape(y[valid2], yhat[valid2])
            else:
                loss[i, j] = mse(y[valid2], yhat[valid2])

    loss_df = pd.DataFrame(loss, columns=expert_pool.expert_names)
    # recover ID columns from concatenated series IDs
    for k, col in enumerate(id_cols):
        loss_df.insert(k, col, [sid.split("__")[k] if "__" in sid else sid for sid in series_ids])
    return loss_df, pred_tensor, target_tensor, series_ids


def mine_oracle_targets(
    loss_matrix: pd.DataFrame,
    expert_names: Sequence[str],
    costs: Sequence[float],
    cfg: FAMEConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Mine hard/soft expert suitability from validation loss matrix."""
    id_cols = list(cfg.id_cols)
    losses = loss_matrix[list(expert_names)].to_numpy(dtype=float)
    costs_arr = np.asarray(costs, dtype=float).reshape(1, -1)
    adjusted = losses + float(cfg.eta_oracle_cost) * costs_arr
    # avoid all-inf rows
    max_finite = np.nanmax(np.where(np.isfinite(adjusted), adjusted, np.nan))
    if not np.isfinite(max_finite):
        max_finite = 1.0
    adjusted = np.where(np.isfinite(adjusted), adjusted, max_finite * 10.0)
    hard_idx = np.argmin(adjusted, axis=1)
    logits = -adjusted / max(float(cfg.tau), 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    expv = np.exp(logits)
    q = expv / np.maximum(expv.sum(axis=1, keepdims=True), 1e-12)

    hard = loss_matrix[id_cols].copy()
    hard["oracle_expert"] = [expert_names[i] for i in hard_idx]
    hard["oracle_index"] = hard_idx
    soft = loss_matrix[id_cols].copy()
    for j, name in enumerate(expert_names):
        soft[name] = q[:, j]
    return hard, soft


def mine_oracle(
    expert_pool: ExpertPool,
    history_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    cfg: FAMEConfig,
    metric: str = "mse",
) -> OracleMiningResult:
    loss_df, pred_tensor, target_tensor, series_ids = compute_expert_loss_matrix(
        expert_pool, history_df, oracle_df, cfg, metric=metric
    )
    hard, soft = mine_oracle_targets(loss_df, expert_pool.expert_names, expert_pool.costs, cfg)
    return OracleMiningResult(
        loss_matrix=loss_df,
        soft_targets=soft,
        hard_targets=hard,
        prediction_tensor=pred_tensor,
        target_tensor=target_tensor,
        series_ids=series_ids,
        expert_names=expert_pool.expert_names,
        horizon=pred_tensor.shape[-1] if pred_tensor.ndim == 3 else cfg.horizon,
    )

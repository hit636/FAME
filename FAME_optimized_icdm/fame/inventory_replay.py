# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class ReplayConfig:
    id_cols: Sequence[str] = ("vem_id", "merc_id")
    date_col: str = "date"
    target_col: str = "daily_quantity"
    forecast_col: str = "predicted_sales"
    horizon: int = 14
    lead_time: int = 1
    z_service: float = 1.645  # approx 95% service target
    understock_weight: float = 3.0
    overstock_weight: float = 1.0
    capacity_col: str = "capacity"
    initial_inventory_col: str = "initial_inventory"


def stock_aware_loss(y_true, y_pred, under_weight: float = 3.0, over_weight: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(under_weight * np.maximum(y_true - y_pred, 0) + over_weight * np.maximum(y_pred - y_true, 0)))


def fixed_policy_replay(df: pd.DataFrame, cfg: ReplayConfig = ReplayConfig()) -> pd.DataFrame:
    """Simple fixed-policy offline replenishment replay.

    This implements the paper-style simulator idea: replace only the demand
    forecast while keeping an order-up-to policy fixed.
    """
    rows = []
    data = df.sort_values(list(cfg.id_cols) + [cfg.date_col]).copy()
    for key, g in data.groupby(list(cfg.id_cols), sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        g = g.reset_index(drop=True)
        demand = pd.to_numeric(g[cfg.target_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        forecast = pd.to_numeric(g[cfg.forecast_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        capacity = float(g[cfg.capacity_col].dropna().iloc[0]) if cfg.capacity_col in g.columns and g[cfg.capacity_col].notna().any() else np.inf
        inv = float(g[cfg.initial_inventory_col].dropna().iloc[0]) if cfg.initial_inventory_col in g.columns and g[cfg.initial_inventory_col].notna().any() else max(np.mean(demand[:7]) * 3, 0)
        pipeline = []  # list of (arrival_index, quantity)
        for t in range(len(g)):
            # receive orders
            arrivals = [q for idx, q in pipeline if idx == t]
            if arrivals:
                inv += sum(arrivals)
            pipeline = [(idx, q) for idx, q in pipeline if idx > t]
            # demand consumes inventory
            sold = min(inv, demand[t])
            inv -= sold
            stockout = 1 if inv <= 1e-8 and demand[t] > sold else 0
            # order-up-to level
            forecast_window = forecast[t: min(len(forecast), t + cfg.horizon)]
            demand_window = demand[max(0, t - cfg.horizon): t]
            sigma = float(np.std(demand_window)) if len(demand_window) else 0.0
            target_level = float(np.sum(forecast_window) + cfg.z_service * sigma * np.sqrt(cfg.lead_time + cfg.horizon))
            target_level = min(target_level, capacity)
            q_order = max(0.0, target_level - inv)
            q_order = min(q_order, max(0.0, capacity - inv))
            pipeline.append((t + cfg.lead_time, q_order))
            overstock = max(0.0, inv - target_level)
            rows.append({
                **{col: val for col, val in zip(cfg.id_cols, key)},
                cfg.date_col: g.loc[t, cfg.date_col],
                "inventory_after_demand": inv,
                "order_quantity": q_order,
                "stockout_event": stockout,
                "overstock_exposure": overstock,
                "stock_aware_loss": cfg.understock_weight * max(demand[t] - forecast[t], 0) + cfg.overstock_weight * max(forecast[t] - demand[t], 0),
            })
    return pd.DataFrame(rows)

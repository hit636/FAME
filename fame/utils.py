# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if out[date_col].isna().any():
        bad = out[out[date_col].isna()].head(3)
        raise ValueError(f"日期列 {date_col} 存在无法解析的值，例如: {bad.to_dict('records')}")
    return out


def make_series_id(df: pd.DataFrame, id_cols: Sequence[str], out_col: str = "series_id") -> pd.DataFrame:
    out = df.copy()
    out[out_col] = out[list(id_cols)].astype(str).agg("__".join, axis=1)
    return out


def valid_demand_mask(df: pd.DataFrame, cfg_or_cols) -> pd.Series:
    """Return rows that are valid for demand-error evaluation.

    In industrial sales logs, observed sales can be censored by stockout,
    machine outage, suspension, or unavailability. FAME excludes these rows
    from oracle-loss construction and evaluation so the router does not learn
    from artificially truncated demand.
    """
    if hasattr(cfg_or_cols, "availability_cols"):
        cols = list(cfg_or_cols.availability_cols)
    else:
        cols = list(cfg_or_cols)
    mask = pd.Series(True, index=df.index)
    if "is_available" in df.columns:
        mask &= pd.to_numeric(df["is_available"], errors="coerce").fillna(1).astype(int).eq(1)
    for c in ["stockout_flag", "outage_flag", "suspension_flag"]:
        if c in df.columns:
            mask &= pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).eq(0)
    # Also support custom availability columns: 1 means available for columns whose
    # name includes available; otherwise 1 means censored/invalid.
    for c in cols:
        if c not in df.columns or c in {"is_available", "stockout_flag", "outage_flag", "suspension_flag"}:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if "available" in c:
            mask &= vals.fillna(1).astype(int).eq(1)
        else:
            mask &= vals.fillna(0).astype(int).eq(0)
    return mask


def complete_daily_grid(
    df: pd.DataFrame,
    id_cols: Sequence[str],
    date_col: str,
    target_col: str,
    fill_target: float = 0.0,
    static_cols: Sequence[str] | None = None,
    dynamic_cols: Sequence[str] | None = None,
    allow_dynamic_bfill: bool = False,
) -> pd.DataFrame:
    """Complete missing dates inside each series' observed min/max range.

    Leakage control: static metadata can be forward/backward filled within a
    split, but dynamic context such as weather, promotion and availability should
    only be forward-filled unless explicitly allowed. This avoids leaking future
    context values into earlier training rows when a complete grid is built.
    """
    df = ensure_datetime(df, date_col).sort_values(list(id_cols) + [date_col]).copy()
    static_set = set(static_cols or [])
    dynamic_set = set(dynamic_cols or [])
    completed = []
    for _, g in df.groupby(list(id_cols), sort=False):
        start, end = g[date_col].min(), g[date_col].max()
        grid = pd.DataFrame({date_col: pd.date_range(start, end, freq="D")})
        for c in id_cols:
            grid[c] = g[c].iloc[0]
        merged = grid.merge(g, on=list(id_cols) + [date_col], how="left")
        merged[target_col] = merged[target_col].fillna(fill_target)
        non_keys = [c for c in merged.columns if c not in list(id_cols) + [date_col, target_col]]
        if non_keys:
            static = [c for c in non_keys if c in static_set]
            dynamic = [c for c in non_keys if c in dynamic_set]
            other = [c for c in non_keys if c not in static_set and c not in dynamic_set]
            if static:
                merged[static] = merged[static].ffill().bfill()
            if dynamic:
                merged[dynamic] = merged[dynamic].ffill()
                if allow_dynamic_bfill:
                    merged[dynamic] = merged[dynamic].bfill()
            if other:
                merged[other] = merged[other].ffill()
        completed.append(merged)
    return pd.concat(completed, ignore_index=True) if completed else pd.DataFrame(columns=df.columns)


def chronological_split_dates(
    df: pd.DataFrame,
    date_col: str,
    validation_ratio: float,
    test_ratio: float,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    dates = np.array(sorted(pd.to_datetime(df[date_col].unique())))
    if len(dates) < 5:
        raise ValueError("数据日期过少，无法进行 train/validation/test 切分")
    n = len(dates)
    val_start_idx = max(1, int(n * (1.0 - validation_ratio - test_ratio)))
    test_start_idx = max(val_start_idx + 1, int(n * (1.0 - test_ratio)))
    return pd.Timestamp(dates[val_start_idx]), pd.Timestamp(dates[test_start_idx])


def split_validation_window(
    validation_df: pd.DataFrame,
    date_col: str,
    oracle_fraction: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(pd.to_datetime(validation_df[date_col].unique())))
    if len(dates) < 2:
        return validation_df.copy(), validation_df.copy()
    cut = max(1, min(len(dates) - 1, int(len(dates) * oracle_fraction)))
    oracle_dates = set(dates[:cut])
    oracle_df = validation_df[validation_df[date_col].isin(oracle_dates)].copy()
    calib_df = validation_df[~validation_df[date_col].isin(oracle_dates)].copy()
    if calib_df.empty:
        calib_df = oracle_df.copy()
    return oracle_df, calib_df


def mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return np.nan
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)))


def wape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)))


def safe_float(x, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def json_dump(obj, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def json_load(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_torch_device(device: str = "auto"):
    """Resolve a torch device from configuration.

    Parameters
    ----------
    device:
        ``"auto"`` selects CUDA when available and otherwise CPU. A concrete
        value such as ``"cpu"``, ``"cuda"`` or ``"cuda:1"`` is respected. If
        CUDA is requested but unavailable, CPU is used as a safe fallback.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for device-aware modules.") from exc

    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = str(device).strip().lower()
    if dev.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        return torch.device(dev)
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data-quality checks for FAME training, deployment and validation.

These checks are intentionally lightweight. They do not replace enterprise data
quality platforms, but they make the conference artifact closer to an industrial
pipeline: failures are explicit, audit reports are saved, and warning items can
be used by monitoring systems.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .config import FAMEConfig
from .utils import json_dump


@dataclass
class DataAuditResult:
    """Structured audit result returned by data validation functions."""

    name: str
    rows: int
    columns: int
    n_series: int = 0
    min_date: str = ""
    max_date: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, float | int | str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def _missing_columns(df: pd.DataFrame, required: Sequence[str]) -> List[str]:
    return [c for c in required if c not in df.columns]


def _series_count(df: pd.DataFrame, id_cols: Sequence[str]) -> int:
    if not all(c in df.columns for c in id_cols) or df.empty:
        return 0
    return int(df[list(id_cols)].drop_duplicates().shape[0])


def _date_bounds(df: pd.DataFrame, date_col: str) -> tuple[str, str]:
    if date_col not in df.columns or df.empty:
        return "", ""
    d = pd.to_datetime(df[date_col], errors="coerce")
    d = d[d.notna()]
    if d.empty:
        return "", ""
    return str(d.min().date()), str(d.max().date())


def audit_history(df: pd.DataFrame, cfg: FAMEConfig, name: str = "history") -> DataAuditResult:
    """Validate historical training/inference data.

    The function checks schema, date parsing, duplicate id-date rows, negative
    demand, NaN ratios, short histories and date gaps.
    """
    id_cols = list(cfg.id_cols)
    required = id_cols + [cfg.date_col, cfg.target_col]
    result = DataAuditResult(
        name=name,
        rows=int(len(df)),
        columns=int(len(df.columns)),
        n_series=_series_count(df, id_cols),
    )
    result.min_date, result.max_date = _date_bounds(df, cfg.date_col)

    missing = _missing_columns(df, required)
    if missing:
        result.errors.append(f"Missing required columns: {missing}")
        return result

    work = df.copy()
    dates = pd.to_datetime(work[cfg.date_col], errors="coerce")
    bad_dates = int(dates.isna().sum())
    if bad_dates:
        result.errors.append(f"Invalid date values in {cfg.date_col}: {bad_dates} rows")
    target = pd.to_numeric(work[cfg.target_col], errors="coerce")
    result.stats["target_nan_rows"] = int(target.isna().sum())
    result.stats["target_negative_rows"] = int((target < 0).sum())
    if (target < 0).any():
        result.errors.append("Target contains negative values; sales demand must be non-negative.")

    duplicate_rows = int(work.duplicated(subset=id_cols + [cfg.date_col]).sum())
    result.stats["duplicate_id_date_rows"] = duplicate_rows
    if duplicate_rows:
        result.warnings.append(f"Duplicate id-date rows found: {duplicate_rows}. Aggregate before production training.")

    # Missing ratio by column.
    na_ratio = work.isna().mean().sort_values(ascending=False)
    for c, r in na_ratio.items():
        if r > 0.2:
            result.warnings.append(f"Column {c} has high missing ratio: {r:.1%}")

    # Short series and internal date-gap diagnostics.
    lengths = work.groupby(id_cols, dropna=False).size()
    short_count = int((lengths < cfg.min_history).sum())
    result.stats["short_series_count"] = short_count
    if short_count:
        result.warnings.append(f"Series shorter than min_history={cfg.min_history}: {short_count}")

    gap_rows = 0
    try:
        for _, g in work.assign(__date__=dates).dropna(subset=["__date__"]).groupby(id_cols, sort=False):
            unique_days = g["__date__"].dt.normalize().drop_duplicates().sort_values()
            if len(unique_days) >= 2:
                span = (unique_days.max() - unique_days.min()).days + 1
                gap_rows += max(0, span - len(unique_days))
        result.stats["estimated_missing_calendar_days"] = int(gap_rows)
        if gap_rows:
            result.warnings.append(f"Estimated missing internal calendar days: {gap_rows}; FAME can complete grid if enabled.")
    except Exception as exc:
        result.warnings.append(f"Date-gap audit skipped due to error: {exc}")

    availability_cols = [c for c in cfg.availability_cols if c in work.columns]
    if availability_cols:
        invalid_avail = int(work[availability_cols].isna().sum().sum())
        result.stats["availability_missing_cells"] = invalid_avail

    return result


def audit_future_context(df: pd.DataFrame, cfg: FAMEConfig, name: str = "future") -> DataAuditResult:
    """Validate future frame or future context data.

    A future frame can be full id-date rows or date/city-level context. The check
    only requires a date column, then reports which merge style appears likely.
    """
    id_cols = list(cfg.id_cols)
    result = DataAuditResult(name=name, rows=int(len(df)), columns=int(len(df.columns)), n_series=_series_count(df, id_cols))
    result.min_date, result.max_date = _date_bounds(df, cfg.date_col)
    if cfg.date_col not in df.columns:
        result.errors.append(f"Missing required date column: {cfg.date_col}")
        return result
    dates = pd.to_datetime(df[cfg.date_col], errors="coerce")
    if dates.isna().any():
        result.errors.append(f"Invalid future date values: {int(dates.isna().sum())} rows")
    if all(c in df.columns for c in id_cols):
        result.stats["future_mode"] = "full_id_date_frame"
    elif "city_name" in df.columns:
        result.stats["future_mode"] = "date_city_context"
    else:
        result.stats["future_mode"] = "date_level_context"
        result.warnings.append("Future context has neither id columns nor city_name; it will be merged only by date if supported.")
    return result


def write_audit_report(results: Iterable[DataAuditResult], path: str | Path) -> None:
    """Persist audit results as JSON for reproducibility and deployment logs."""
    json_dump({r.name: r.to_dict() for r in results}, path)

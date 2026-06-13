# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd


def expert_usage_entropy(explanation_df: pd.DataFrame, expert_col: str = "expert", weight_col: str = "weight") -> float:
    if explanation_df.empty:
        return 0.0
    usage = explanation_df.groupby(expert_col)[weight_col].sum()
    p = usage.to_numpy(dtype=float)
    p = p / max(p.sum(), 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))


def topk_oracle_recall(prob_df: pd.DataFrame, hard_oracle_df: pd.DataFrame, expert_names: Sequence[str], id_cols: Sequence[str], k: int = 2) -> float:
    df = prob_df.merge(hard_oracle_df[list(id_cols) + ["oracle_expert"]], on=list(id_cols), how="inner")
    if df.empty:
        return 0.0
    hits = []
    for _, row in df.iterrows():
        ranked = list(row[list(expert_names)].astype(float).sort_values(ascending=False).index[:k])
        hits.append(row["oracle_expert"] in ranked)
    return float(np.mean(hits))


def oracle_gap(loss_matrix: pd.DataFrame, selected_expert_col: str, expert_names: Sequence[str]) -> float:
    if loss_matrix.empty or selected_expert_col not in loss_matrix.columns:
        return np.nan
    oracle_loss = loss_matrix[list(expert_names)].min(axis=1).to_numpy(dtype=float)
    selected_loss = []
    for _, row in loss_matrix.iterrows():
        e = row[selected_expert_col]
        selected_loss.append(float(row[e]) if e in expert_names else np.nan)
    selected_loss = np.asarray(selected_loss, dtype=float)
    return float(np.nanmean((selected_loss - oracle_loss) / np.maximum(oracle_loss, 1e-8)))



def retained_oracle_mass(prob_df: pd.DataFrame, soft_oracle_df: pd.DataFrame, expert_names: Sequence[str], id_cols: Sequence[str], k: int = 2) -> float:
    """Average soft-oracle probability mass retained by the router's Top-k set."""
    df = prob_df.merge(soft_oracle_df[list(id_cols) + list(expert_names)], on=list(id_cols), how="inner", suffixes=("_router", "_oracle"))
    if df.empty:
        return 0.0
    masses = []
    for _, row in df.iterrows():
        router_scores = {name: float(row[f"{name}_router"]) if f"{name}_router" in row else float(row[name]) for name in expert_names}
        top = sorted(expert_names, key=lambda n: router_scores[n], reverse=True)[:k]
        mass = 0.0
        for name in top:
            col = f"{name}_oracle"
            if col in row:
                mass += float(row[col])
        masses.append(mass)
    return float(np.mean(masses)) if masses else 0.0

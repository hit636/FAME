# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import FAMEConfig
from .utils import resolve_torch_device, set_seed


class RouterMLP(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_size: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class RouterTrainResult:
    train_loss: List[float]
    best_loss: float
    epochs: int


class SparseRouter:
    """Forecastability-aware sparse router.

    It learns p_i = softmax(MLP(z_i)) using FAME's objective:
    L_total = L_pred + lambda * KL(q||p) + beta * L_balance + gamma * L_cost.
    """

    def __init__(self, config: FAMEConfig, expert_names: Sequence[str], costs: Sequence[float]):
        self.config = config
        self.expert_names = list(expert_names)
        self.costs = np.asarray(costs, dtype=np.float32)
        self.input_dim_: Optional[int] = None
        self.model: Optional[RouterMLP] = None
        self.device = resolve_torch_device(config.device)
        self.history_: Optional[RouterTrainResult] = None

    def _ensure_model(self, input_dim: int) -> None:
        if self.model is None or self.input_dim_ != input_dim:
            self.input_dim_ = input_dim
            self.model = RouterMLP(
                input_dim=input_dim,
                num_experts=len(self.expert_names),
                hidden_size=self.config.hidden_size,
                dropout=self.config.dropout,
            ).to(self.device)

    def fit(
        self,
        fingerprints: pd.DataFrame,
        soft_targets: pd.DataFrame,
        pred_tensor: Optional[np.ndarray] = None,
        target_tensor: Optional[np.ndarray] = None,
    ) -> RouterTrainResult:
        """Train router.

        Parameters
        ----------
        fingerprints:
            DataFrame with id columns + fingerprint features.
        soft_targets:
            DataFrame with id columns + expert suitability q.
        pred_tensor:
            Optional [N, M, H] expert forecasts for L_pred. If omitted, only
            router supervision, balance and cost are used.
        target_tensor:
            Optional [N, H] oracle-window targets for L_pred.
        """
        set_seed(self.config.seed)
        id_cols = list(self.config.id_cols)
        df = fingerprints.merge(soft_targets, on=id_cols, how="inner")
        if df.empty:
            raise ValueError("No matched rows between fingerprints and oracle soft targets.")
        feature_cols = [c for c in fingerprints.columns if c not in id_cols]
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        q = df[self.expert_names].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
        self._ensure_model(X.shape[1])
        assert self.model is not None

        # Align prediction/target tensors by the merge order. In the standard
        # pipeline, tensors already follow soft_targets/fingerprint order. If a
        # custom caller passes tensors, they should keep this order.
        use_pred_loss = pred_tensor is not None and target_tensor is not None
        if use_pred_loss:
            pred_tensor = np.asarray(pred_tensor, dtype=np.float32)
            target_tensor = np.asarray(target_tensor, dtype=np.float32)
            n = min(len(X), pred_tensor.shape[0], target_tensor.shape[0])
            X, q = X[:n], q[:n]
            pred_tensor, target_tensor = pred_tensor[:n], target_tensor[:n]
            # Keep a validity mask for censored / missing oracle-window targets.
            # Do not turn NaN targets into zeros without masking, otherwise the
            # prediction loss would incorrectly reward low forecasts on censored days.
            valid_target_mask = np.isfinite(target_tensor).astype(np.float32)
            pred_tensor = np.nan_to_num(pred_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            target_tensor = np.nan_to_num(target_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            valid_target_mask = None

        ds_tensors = [torch.from_numpy(X), torch.from_numpy(q)]
        if use_pred_loss:
            ds_tensors += [torch.from_numpy(pred_tensor), torch.from_numpy(target_tensor), torch.from_numpy(valid_target_mask)]
        dataset = TensorDataset(*ds_tensors)
        dl = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.router_lr, weight_decay=self.config.weight_decay)
        costs = torch.from_numpy(self.costs).to(self.device)
        best_state = None
        best_loss = float("inf")
        patience_left = self.config.patience
        losses = []

        for epoch in range(self.config.router_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_seen = 0
            for batch in dl:
                xb = batch[0].to(self.device)
                qb = batch[1].to(self.device)
                logits = self.model(xb)
                p = torch.softmax(logits, dim=-1)
                logp = torch.log(p + 1e-12)
                router_loss = torch.mean(torch.sum(qb * (torch.log(qb + 1e-12) - logp), dim=-1))
                pbar = p.mean(dim=0)
                balance_loss = len(self.expert_names) * torch.sum(pbar ** 2)
                cost_loss = torch.mean(torch.sum(p * costs.view(1, -1), dim=-1))
                pred_loss = torch.tensor(0.0, device=self.device)
                if use_pred_loss:
                    predb = batch[2].to(self.device)  # [B, M, H]
                    targetb = batch[3].to(self.device)  # [B, H]
                    maskb = batch[4].to(self.device)  # [B, H]
                    mix = torch.sum(p.unsqueeze(-1) * predb, dim=1)
                    denom = torch.clamp(maskb.sum(), min=1.0)
                    pred_loss = torch.sum(((mix - targetb) ** 2) * maskb) / denom
                loss = (
                    pred_loss
                    + self.config.lambda_router * router_loss
                    + self.config.beta_balance * balance_loss
                    + self.config.gamma_cost * cost_loss
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu()) * len(xb)
                n_seen += len(xb)
            epoch_loss = epoch_loss / max(n_seen, 1)
            losses.append(epoch_loss)
            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_left = self.config.patience
            else:
                patience_left -= 1
            if patience_left <= 0:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.history_ = RouterTrainResult(train_loss=losses, best_loss=best_loss, epochs=len(losses))
        return self.history_

    def predict_proba(self, fingerprints: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.input_dim_ is None:
            raise RuntimeError("SparseRouter is not fitted.")
        id_cols = list(self.config.id_cols)
        feature_cols = [c for c in fingerprints.columns if c not in id_cols]
        X = fingerprints[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            p = torch.softmax(self.model(torch.from_numpy(X).to(self.device)), dim=-1).cpu().numpy()
        out = fingerprints[id_cols].copy()
        for j, name in enumerate(self.expert_names):
            out[name] = p[:, j]
        return out

    def select_active_experts(
        self,
        prob_row: pd.Series,
        available_mask: Optional[np.ndarray] = None,
        top_r: Optional[int] = None,
        delta: Optional[float] = None,
    ) -> Tuple[List[str], np.ndarray]:
        top_r = int(top_r if top_r is not None else self.config.top_r)
        delta = float(delta if delta is not None else self.config.delta)
        probs = prob_row[self.expert_names].to_numpy(dtype=float)
        if available_mask is None:
            available_mask = np.ones(len(self.expert_names), dtype=bool)
        available_mask = np.asarray(available_mask, dtype=bool)
        masked = np.where(available_mask, probs, -np.inf)
        if not np.isfinite(masked).any():
            masked = probs
            available_mask = np.ones(len(self.expert_names), dtype=bool)
        top_idx = np.argsort(masked)[::-1][: max(1, min(top_r, len(masked)))]
        top_probs = probs[top_idx]
        norm_top = top_probs / max(top_probs.sum(), 1e-12)
        keep = norm_top >= delta
        if not keep.any():
            keep[np.argmax(norm_top)] = True
        active_idx = top_idx[keep]
        active_probs = probs[active_idx]
        weights = active_probs / max(active_probs.sum(), 1e-12)
        return [self.expert_names[i] for i in active_idx], weights.astype(float)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "SparseRouter":
        return joblib.load(path)

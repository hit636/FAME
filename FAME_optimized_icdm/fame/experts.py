from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from .config import ExpertSpec, FAMEConfig
from .features import TimeSeriesFeatureBuilder
from .utils import resolve_torch_device

warnings.filterwarnings("ignore")


try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch is optional until DLinear is used
    torch = None
    nn = None


class TinyDLinear(nn.Module if nn is not None else object):
    """Pickle-safe compact DLinear-style module used by sequence expert wrappers."""

    def __init__(self, seq_len: int, horizon: int):
        if nn is None:
            raise ImportError("TinyDLinear requires PyTorch.")
        super().__init__()
        self.trend = nn.Linear(seq_len, horizon)
        self.seasonal = nn.Linear(seq_len, horizon)
        self.avg = nn.AvgPool1d(kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        trend = self.avg(x.unsqueeze(1)).squeeze(1)
        seasonal = x - trend
        return self.trend(trend) + self.seasonal(seasonal)


class BaseExpert:
    """Common expert interface.

    All experts must implement ``fit(df)`` and ``predict(history_df, future_df)``.
    ``predict`` returns one horizon-length vector for each series in future_df.
    """

    def __init__(self, spec: ExpertSpec, config: FAMEConfig):
        self.spec = spec
        self.config = config
        self.name = spec.name
        self.cost = float(spec.cost)
        self.fitted_ = False

    def fit(self, df: pd.DataFrame) -> "BaseExpert":
        raise NotImplementedError

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def is_available(self, history_df: pd.DataFrame) -> bool:
        return len(history_df) >= self.config.min_history


class StatisticalExpert(BaseExpert):
    def fit(self, df: pd.DataFrame) -> "StatisticalExpert":
        # Statistical experts are fitted lazily per series at inference/evaluation.
        self.fitted_ = True
        return self

    def _predict_one(self, y: np.ndarray, horizon: int) -> np.ndarray:
        kind = self.spec.kind
        params = self.spec.params
        y = np.asarray(y, dtype=float)
        if len(y) == 0:
            return np.zeros(horizon)
        if kind == "naive":
            return np.full(horizon, y[-1])
        if kind == "seasonal_naive":
            period = int(params.get("seasonal_period", self.config.seasonal_period))
            if len(y) >= period:
                reps = int(np.ceil(horizon / period))
                return np.tile(y[-period:], reps)[:horizon]
            return np.full(horizon, np.mean(y))
        if kind == "moving_average":
            window = int(params.get("window", 7))
            return np.full(horizon, np.mean(y[-window:]))
        if kind == "croston":
            return self._croston(y, horizon, alpha=float(params.get("alpha", 0.1)))
        if kind == "tsb":
            return self._tsb(y, horizon, alpha=float(params.get("alpha", 0.1)), beta=float(params.get("beta", 0.1)))
        if kind == "prophet":
            # Prophet is handled in predict() when the optional package is installed.
            # The fallback keeps this expert active in lightweight environments.
            return self._ets(y, horizon)
        if kind == "ets":
            return self._ets(y, horizon)
        if kind == "sarima":
            return self._sarima(y, horizon)
        raise ValueError(f"Unsupported statistical expert kind: {kind}")

    def _croston(self, y: np.ndarray, horizon: int, alpha: float = 0.1) -> np.ndarray:
        demand = y[y > 0]
        if len(demand) == 0:
            return np.zeros(horizon)
        q = demand[0]
        a = 1.0
        last = 0
        first = True
        for t, val in enumerate(y, start=1):
            if val > 0:
                interval = 1 if first else t - last
                q = alpha * val + (1 - alpha) * q
                a = alpha * interval + (1 - alpha) * a
                last = t
                first = False
        forecast = q / max(a, 1e-8)
        return np.full(horizon, max(0.0, forecast))

    def _tsb(self, y: np.ndarray, horizon: int, alpha: float = 0.1, beta: float = 0.1) -> np.ndarray:
        """Teunter-Syntetos-Babai intermittent-demand forecast.

        TSB separately smooths demand size and demand occurrence probability,
        making it suitable for products whose demand can become obsolete or highly
        intermittent.
        """
        y = np.asarray(y, dtype=float)
        if len(y) == 0:
            return np.zeros(horizon)
        first_pos = y[y > 0]
        z = float(first_pos[0]) if len(first_pos) else 0.0
        p = 1.0 if y[0] > 0 else 0.0
        for val in y:
            occ = 1.0 if val > 0 else 0.0
            p = beta * occ + (1.0 - beta) * p
            if val > 0:
                z = alpha * float(val) + (1.0 - alpha) * z
        return np.full(horizon, max(0.0, p * z))

    def _prophet(self, h: pd.DataFrame, f: pd.DataFrame) -> np.ndarray:
        """Optional Prophet wrapper with robust fallback.

        Prophet is not a hard dependency for the open artifact. If unavailable or
        unstable on a short/intermittent series, the expert falls back to ETS.
        """
        cfg = self.config
        y = pd.to_numeric(h[cfg.target_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if not bool(self.spec.params.get("use_prophet", False)):
            return self._ets(y, len(f))
        try:
            from prophet import Prophet
            train = pd.DataFrame({"ds": pd.to_datetime(h[cfg.date_col]), "y": y})
            if len(train) < max(10, cfg.seasonal_period * 2):
                raise ValueError("history too short for Prophet")
            model = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False)
            model.fit(train)
            fut = pd.DataFrame({"ds": pd.to_datetime(f[cfg.date_col])})
            out = model.predict(fut)["yhat"].to_numpy(dtype=float)
            return out
        except Exception:
            return self._ets(y, len(f))

    def _ets(self, y: np.ndarray, horizon: int) -> np.ndarray:
        period = int(self.spec.params.get("seasonal_period", self.config.seasonal_period))
        if not bool(self.spec.params.get("use_statsmodels", False)):
            if len(y) >= period * 2:
                level = np.mean(y[-period:])
                pattern = y[-period:] - level
                reps = int(np.ceil(horizon / period))
                return np.tile(level + pattern, reps)[:horizon]
            return np.full(horizon, np.mean(y[-min(7, len(y)):]))
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            if len(y) >= period * 2:
                model = ExponentialSmoothing(y, trend=None, seasonal="add", seasonal_periods=period)
            else:
                model = ExponentialSmoothing(y, trend=None, seasonal=None)
            fit = model.fit(optimized=True)
            return np.asarray(fit.forecast(horizon), dtype=float)
        except Exception:
            return np.full(horizon, np.mean(y[-min(7, len(y)):]))

    def _sarima(self, y: np.ndarray, horizon: int) -> np.ndarray:
        # A full SARIMAX fit per series can be expensive in artifact evaluation.
        # The portable default uses a fast seasonal ARIMA proxy unless the user
        # explicitly sets params={"use_statsmodels": True}.
        period = int(self.spec.params.get("seasonal_period", self.config.seasonal_period))
        if not bool(self.spec.params.get("use_statsmodels", False)):
            if len(y) >= period:
                reps = int(np.ceil(horizon / period))
                seasonal = np.tile(y[-period:], reps)[:horizon]
                level = np.mean(y[-min(len(y), period):])
                return 0.7 * seasonal + 0.3 * level
            return np.full(horizon, np.mean(y[-min(7, len(y)):]))
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            seasonal_order = (1, 1, 1, period) if len(y) >= period * 2 else (0, 0, 0, 0)
            model = SARIMAX(y, order=(1, 1, 1), seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=int(self.spec.params.get("maxiter", 50)))
            return np.asarray(fit.get_forecast(horizon).predicted_mean, dtype=float)
        except Exception:
            return np.full(horizon, np.mean(y[-min(7, len(y)):]))

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        out = []
        for key, f in future_df.groupby(list(cfg.id_cols), sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            mask = np.ones(len(history_df), dtype=bool)
            for col, val in zip(cfg.id_cols, key):
                mask &= history_df[col].astype(str).to_numpy() == str(val)
            h = history_df.loc[mask].sort_values(cfg.date_col)
            f = f.sort_values(cfg.date_col)
            y = pd.to_numeric(h[cfg.target_col], errors="coerce").fillna(0).to_numpy()
            if self.spec.kind == "prophet":
                pred = self._prophet(h, f)
            else:
                pred = self._predict_one(y, len(f))
            tmp = f[list(cfg.id_cols) + [cfg.date_col]].copy()
            tmp["prediction"] = np.asarray(pred, dtype=float)[:len(f)]
            tmp["expert"] = self.name
            out.append(tmp)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


class MLGlobalExpert(BaseExpert):
    """Global tabular supervised expert, e.g. Linear/RF/LightGBM/XGBoost."""

    def __init__(self, spec: ExpertSpec, config: FAMEConfig):
        super().__init__(spec, config)
        self.builder = TimeSeriesFeatureBuilder(config)
        self.model = None
        self.scaler = None

    def _new_model(self):
        params = dict(self.spec.params)
        kind = self.spec.kind
        if kind == "linear":
            return Ridge(alpha=float(params.get("alpha", 1.0)))
        if kind == "random_forest":
            return RandomForestRegressor(
                n_estimators=int(params.get("n_estimators", 120)),
                max_depth=params.get("max_depth", 12),
                random_state=self.config.seed,
                n_jobs=params.get("n_jobs", -1),
            )
        if kind == "lightgbm":
            if not bool(params.get("use_external_lightgbm", False)):
                # Portable artifact fallback with LightGBM-like tabular semantics.
                return Ridge(alpha=float(params.get("alpha", 1.0)))
            try:
                from lightgbm import LGBMRegressor
                return LGBMRegressor(
                    n_estimators=int(params.get("n_estimators", 300)),
                    learning_rate=float(params.get("learning_rate", 0.03)),
                    num_leaves=int(params.get("num_leaves", 64)),
                    subsample=float(params.get("subsample", 0.8)),
                    colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                    objective=params.get("objective", "regression"),
                    random_state=self.config.seed,
                    n_jobs=params.get("n_jobs", -1),
                )
            except Exception:
                return ExtraTreesRegressor(n_estimators=30, max_depth=10, random_state=self.config.seed, n_jobs=1)
        if kind == "xgboost":
            if not bool(params.get("use_external_xgboost", False)):
                # Portable artifact fallback with XGBoost-like tree-ensemble semantics.
                return ExtraTreesRegressor(
                    n_estimators=int(params.get("n_estimators", 80)),
                    max_depth=int(params.get("max_depth", 12)),
                    random_state=self.config.seed,
                    n_jobs=int(params.get("n_jobs", 1)),
                )
            try:
                from xgboost import XGBRegressor
                return XGBRegressor(
                    n_estimators=int(params.get("n_estimators", 300)),
                    learning_rate=float(params.get("learning_rate", 0.03)),
                    max_depth=int(params.get("max_depth", 8)),
                    subsample=float(params.get("subsample", 0.8)),
                    colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                    objective="reg:squarederror",
                    random_state=self.config.seed,
                    n_jobs=params.get("n_jobs", -1),
                )
            except Exception:
                return ExtraTreesRegressor(n_estimators=60, max_depth=12, random_state=self.config.seed, n_jobs=1)
        raise ValueError(f"Unsupported ML expert kind: {kind}")

    def fit(self, df: pd.DataFrame) -> "MLGlobalExpert":
        cfg = self.config
        feat = self.builder.fit_transform(df)
        # Only use rows whose target is historical/known.
        feat = feat.dropna(subset=[cfg.target_col]).copy()
        X = feat[self.builder.feature_cols_].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = pd.to_numeric(feat[cfg.target_col], errors="coerce").fillna(0.0).to_numpy()
        if len(X) == 0:
            raise ValueError(f"Expert {self.name}: no training rows after feature building.")
        self.scaler = None
        if self.spec.kind == "linear":
            self.scaler = StandardScaler()
            X_fit = self.scaler.fit_transform(X)
        else:
            X_fit = X
        self.model = self._new_model()
        self.model.fit(X_fit, y)
        self.fitted_ = True
        return self

    def _predict_matrix(self, feat: pd.DataFrame) -> np.ndarray:
        """Predict from a feature frame using the fitted tabular model."""
        X = feat[self.builder.feature_cols_].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return np.asarray(self.model.predict(X), dtype=float)

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        """Recursive multi-step tabular forecasting.

        Future rows often carry a placeholder target value. Building lag features
        on all future rows at once would contaminate horizon t>1 with placeholder
        zeros. Therefore, this method predicts each series recursively: after the
        forecast for horizon t is produced, it is written back as the target value
        used to construct lag/window features for horizon t+1.
        """
        if not self.fitted_ or self.model is None:
            raise RuntimeError(f"Expert {self.name} is not fitted.")
        cfg = self.config
        id_cols = list(cfg.id_cols)
        out = []

        if not getattr(cfg, "recursive_ml_prediction", True):
            combo = pd.concat([history_df, future_df], ignore_index=True, sort=False)
            feat = self.builder.transform(combo)
            future_key = future_df[id_cols + [cfg.date_col]].copy()
            future_key["__row_order__"] = np.arange(len(future_key))
            merged = feat.merge(future_key, on=id_cols + [cfg.date_col], how="inner")
            pred = self._predict_matrix(merged)
            res = merged[id_cols + [cfg.date_col, "__row_order__"]].copy()
            res["prediction"] = pred
            res["expert"] = self.name
            return res.sort_values("__row_order__").drop(columns=["__row_order__"]).reset_index(drop=True)

        history_groups = {}
        for key, h in history_df.groupby(id_cols, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            history_groups[key] = h.sort_values(cfg.date_col).copy()

        for key, f in future_df.groupby(id_cols, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            h = history_groups.get(key, pd.DataFrame(columns=history_df.columns)).sort_values(cfg.date_col).copy()
            f_work = f.sort_values(cfg.date_col).copy()
            if cfg.target_col not in f_work.columns:
                f_work[cfg.target_col] = np.nan
            f_work[cfg.target_col] = np.nan
            preds = []
            for pos, idx in enumerate(f_work.index):
                prefix = f_work.iloc[:pos + 1].copy()
                prefix.loc[prefix.index[-1], cfg.target_col] = 0.0
                combo = pd.concat([h, prefix], ignore_index=True, sort=False)
                feat = self.builder.transform(combo)
                row_key = f_work.loc[[idx], id_cols + [cfg.date_col]].copy()
                current = feat.merge(row_key, on=id_cols + [cfg.date_col], how="inner")
                if current.empty:
                    pred_value = 0.0
                else:
                    pred_value = float(self._predict_matrix(current)[-1])
                pred_value = max(0.0, pred_value) if cfg.clip_non_negative else pred_value
                f_work.loc[idx, cfg.target_col] = pred_value
                preds.append(pred_value)
            tmp = f_work[id_cols + [cfg.date_col]].copy()
            tmp["prediction"] = np.asarray(preds, dtype=float)
            tmp["expert"] = self.name
            out.append(tmp)

        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


class DLinearExpert(BaseExpert):
    """A compact DLinear-style expert for sequence-only forecasting.

    This is a practical, optional expert.  It trains a global sequence model over
    product-terminal series and predicts horizon steps from recent target values.
    """

    def __init__(self, spec: ExpertSpec, config: FAMEConfig):
        super().__init__(spec, config)
        self.model = None
        self.scaler = None
        self.device = None

    def fit(self, df: pd.DataFrame) -> "DLinearExpert":
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except Exception as e:
            raise ImportError("DLinearExpert requires PyTorch.") from e

        cfg = self.config
        seq_len = int(self.spec.params.get("seq_len", 49))
        horizon = cfg.horizon
        xy_blocks = []
        total_window = seq_len + horizon
        for _, g in df.groupby(list(cfg.id_cols), sort=False):
            y = pd.to_numeric(
                g.sort_values(cfg.date_col)[cfg.target_col],
                errors="coerce",
            ).fillna(0).to_numpy(dtype=np.float32)
            if len(y) < total_window:
                continue
            # Vectorized, low-fragmentation sliding windows. sliding_window_view
            # creates a view before the final concatenate, avoiding Python-list
            # per-window appends on long industrial series.
            windows = np.lib.stride_tricks.sliding_window_view(y, total_window)
            xy_blocks.append(windows)
        if not xy_blocks:
            # no enough data; fitted as unavailable with mean fallback
            self.mean_ = float(pd.to_numeric(df[cfg.target_col], errors="coerce").fillna(0).mean())
            self.fitted_ = True
            self.model = None
            return self
        XY = np.concatenate(xy_blocks, axis=0).astype(np.float32, copy=False)
        max_windows = cfg.max_dlinear_windows
        if max_windows is not None and len(XY) > int(max_windows):
            rng = np.random.default_rng(cfg.seed)
            idx = rng.choice(len(XY), size=int(max_windows), replace=False)
            XY = XY[idx]
        X = XY[:, :seq_len]
        Y = XY[:, seq_len:seq_len + horizon]
        self.y_mean_ = float(X.mean())
        self.y_std_ = float(X.std() + 1e-6)
        X = (X - self.y_mean_) / self.y_std_
        Y = (Y - self.y_mean_) / self.y_std_

        self.device = resolve_torch_device(cfg.device)
        self.model = TinyDLinear(seq_len, horizon).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=float(self.spec.params.get("lr", 1e-3)))
        loss_fn = nn.MSELoss()
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        dl = DataLoader(ds, batch_size=int(self.spec.params.get("batch_size", 128)), shuffle=True)
        epochs = int(self.spec.params.get("epochs", 60))
        self.model.train()
        for _ in range(epochs):
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
        self.fitted_ = True
        return self

    def is_available(self, history_df: pd.DataFrame) -> bool:
        seq_len = int(self.spec.params.get("seq_len", 49))
        return len(history_df) >= seq_len

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        import torch
        cfg = self.config
        seq_len = int(self.spec.params.get("seq_len", 49))
        out = []
        batch_items = []
        fallback_items = []

        history_groups = {}
        for key, h in history_df.groupby(list(cfg.id_cols), sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            history_groups[key] = h.sort_values(cfg.date_col)

        for key, f in future_df.groupby(list(cfg.id_cols), sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            h = history_groups.get(key)
            y = np.array([], dtype=float)
            if h is not None and not h.empty:
                y = pd.to_numeric(h[cfg.target_col], errors="coerce").fillna(0).to_numpy(dtype=float)
            horizon = len(f)
            if self.model is None or len(y) < seq_len:
                fallback_items.append((key, f, y, horizon))
            else:
                x = y[-seq_len:]
                x = ((x - self.y_mean_) / self.y_std_).astype(np.float32)
                batch_items.append((key, f, x, horizon))

        # One batched forward pass for all DLinear-assigned series.
        batch_preds = []
        if batch_items:
            X = np.stack([item[2] for item in batch_items], axis=0).astype(np.float32)
            bs = int(self.spec.params.get("predict_batch_size", self.spec.params.get("batch_size", 128)))
            self.model.eval()
            preds = []
            with torch.no_grad():
                for start in range(0, len(X), bs):
                    xb = torch.from_numpy(X[start:start + bs]).to(self.device)
                    yb = self.model(xb).cpu().numpy()
                    preds.append(yb)
            batch_preds = np.concatenate(preds, axis=0) if preds else np.empty((0, cfg.horizon))
            batch_preds = batch_preds * self.y_std_ + self.y_mean_

        for item_idx, (key, f, _x, horizon) in enumerate(batch_items):
            pred = batch_preds[item_idx].reshape(-1)
            if horizon != len(pred):
                pred = np.resize(pred, horizon)
            tmp = f[list(cfg.id_cols) + [cfg.date_col]].copy()
            tmp["prediction"] = pred[:horizon]
            tmp["expert"] = self.name
            out.append(tmp)

        for key, f, y, horizon in fallback_items:
            fallback = getattr(self, "mean_", np.mean(y[-min(7, len(y)):]) if len(y) else 0.0)
            pred = np.full(horizon, fallback)
            tmp = f[list(cfg.id_cols) + [cfg.date_col]].copy()
            tmp["prediction"] = pred[:horizon]
            tmp["expert"] = self.name
            out.append(tmp)

        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def build_expert(spec: ExpertSpec, config: FAMEConfig) -> BaseExpert:
    if spec.kind in {"naive", "seasonal_naive", "moving_average", "croston", "tsb", "ets", "sarima", "prophet"}:
        return StatisticalExpert(spec, config)
    if spec.kind in {"linear", "random_forest", "lightgbm", "xgboost"}:
        return MLGlobalExpert(spec, config)
    if spec.kind in {"dlinear", "timemixer", "timesnet"}:
        # timemixer/timesnet keep paper-aligned expert names; this portable
        # artifact uses a DLinear-compatible sequence wrapper unless the user
        # plugs in external implementations.
        return DLinearExpert(spec, config)
    raise ValueError(f"Unknown expert kind: {spec.kind}")


class ExpertPool:
    def __init__(self, config: FAMEConfig):
        self.config = config
        self.experts: List[BaseExpert] = [build_expert(spec, config) for spec in config.enabled_experts()]
        self.expert_names = [e.name for e in self.experts]
        self.costs = np.asarray([e.cost for e in self.experts], dtype=float)

    def fit(self, train_df: pd.DataFrame) -> "ExpertPool":
        fitted = []
        for expert in self.experts:
            try:
                fitted.append(expert.fit(train_df))
            except Exception as e:
                # Keep the pool usable; disabled expert gets dropped.
                print(f"[WARN] Expert {expert.name} fitting failed and will be disabled: {e}")
        self.experts = fitted
        self.expert_names = [e.name for e in self.experts]
        self.costs = np.asarray([e.cost for e in self.experts], dtype=float)
        if not self.experts:
            raise RuntimeError("No expert fitted successfully.")
        return self

    def predict_all(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        preds: Dict[str, pd.DataFrame] = {}
        for expert in self.experts:
            try:
                if getattr(self.config, "debug_expert_timing", False):
                    import time
                    _t0 = time.time()
                    print(f"[DEBUG] predict_all start {expert.name}", flush=True)
                pred = expert.predict(history_df, future_df)
                if getattr(self.config, "debug_expert_timing", False):
                    print(f"[DEBUG] predict_all done {expert.name} in {time.time() - _t0:.3f}s", flush=True)
                if self.config.clip_non_negative and not pred.empty:
                    pred["prediction"] = pd.to_numeric(pred["prediction"], errors="coerce").fillna(0).clip(lower=0)
                preds[expert.name] = pred
            except Exception as e:
                print(f"[WARN] Expert {expert.name} prediction failed: {e}", flush=True)
        return preds

    def available_mask_for_series(self, history_df: pd.DataFrame) -> np.ndarray:
        mask = []
        for e in self.experts:
            try:
                mask.append(bool(e.is_available(history_df)))
            except Exception:
                mask.append(False)
        arr = np.asarray(mask, dtype=bool)
        if not arr.any() and len(arr):
            arr[0] = True
        return arr

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ExpertPool":
        return joblib.load(path)

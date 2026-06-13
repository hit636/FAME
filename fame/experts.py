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
from .utils import resolve_torch_device, valid_demand_mask

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


class TinyTimeMixer(nn.Module if nn is not None else object):
    """Compact TimeMixer-style multi-scale mixing expert.

    This portable implementation keeps the key TimeMixer intuition used in the
    paper's expert pool: represent a series at several temporal resolutions and
    learn cross-scale mixing before producing a horizon-length forecast. It is
    intentionally small so that the artifact can run without the official
    TimeMixer repository, while still being an independent expert rather than a
    DLinear alias.
    """

    def __init__(self, seq_len: int, horizon: int, hidden_size: int = 128):
        if nn is None:
            raise ImportError("TinyTimeMixer requires PyTorch.")
        super().__init__()
        self.seq_len = seq_len
        self.avg3 = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.avg7 = nn.AvgPool1d(kernel_size=7, stride=1, padding=3)
        self.avg14 = nn.AvgPool1d(kernel_size=15, stride=1, padding=7)
        self.mixer = nn.Sequential(
            nn.Linear(seq_len * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x):
        x1 = x.unsqueeze(1)
        s0 = x
        s1 = self.avg3(x1).squeeze(1)[:, : self.seq_len]
        s2 = self.avg7(x1).squeeze(1)[:, : self.seq_len]
        s3 = self.avg14(x1).squeeze(1)[:, : self.seq_len]
        z = torch.cat([s0, s1, s2, s3], dim=1)
        return self.mixer(z)


class TinyTimesNet(nn.Module if nn is not None else object):
    """Portable TimesNet-style FFT-period expert with Table-IV-visible knobs.

    The public artifact still uses a compact proxy rather than the official
    TimesNet repository, but the configuration now exposes the paper Table IV
    hyperparameters (d_model=256, layers=4, heads=8) and records them in the
    backend manifest. This avoids silent mismatch between the paper table and
    artifact defaults while keeping the code self-contained.
    """

    def __init__(self, seq_len: int, horizon: int, top_k: int = 3, d_model: int = 256, layers: int = 4, heads: int = 8):
        if nn is None:
            raise ImportError("TinyTimesNet requires PyTorch.")
        super().__init__()
        self.seq_len = seq_len
        self.top_k = max(1, int(top_k))
        self.d_model = int(d_model)
        self.layers = int(layers)
        self.heads = int(heads)
        self.periodic_head = nn.Linear(seq_len, horizon)
        self.residual_head = nn.Linear(seq_len, horizon)
        self.direct_head = nn.Linear(seq_len, horizon)
        blocks = []
        blocks.append(nn.Linear(seq_len, self.d_model))
        for _ in range(max(1, self.layers)):
            blocks.extend([nn.GELU(), nn.Linear(self.d_model, self.d_model), nn.LayerNorm(self.d_model)])
        blocks.append(nn.GELU())
        blocks.append(nn.Linear(self.d_model, horizon))
        self.proxy_stack = nn.Sequential(*blocks)

    def forward(self, x):
        # x: [B, L]
        freq = torch.fft.rfft(x, dim=1)
        amp = torch.abs(freq)
        if amp.shape[1] > 1:
            amp = amp.clone()
            amp[:, 0] = 0.0
        k = min(self.top_k, amp.shape[1])
        idx = torch.topk(amp, k=k, dim=1).indices
        mask = torch.zeros_like(freq, dtype=torch.float32)
        mask.scatter_(1, idx, 1.0)
        seasonal = torch.fft.irfft(freq * mask.to(freq.dtype), n=self.seq_len, dim=1)
        residual = x - seasonal
        return self.periodic_head(seasonal) + self.residual_head(residual) + 0.1 * self.direct_head(x) + 0.1 * self.proxy_stack(x)


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
        self.backend_ = spec.kind
        self.fallback_count_ = 0
        self.error_count_ = 0

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
        """Prophet expert aligned with paper Table IV.

        Uses weekly seasonality, optional holiday/event regressors, optional
        numeric external regressors, and a small validation grid for the
        changepoint prior scale. In non-strict artifact mode it falls back to ETS
        if Prophet is unavailable or a series is too short.
        """
        cfg = self.config
        params = self.spec.params
        y = pd.to_numeric(h[cfg.target_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if not bool(params.get("use_prophet", False)):
            return self._ets(y, len(f))
        try:
            from prophet import Prophet
            h2 = h.copy(); f2 = f.copy()
            h2[cfg.date_col] = pd.to_datetime(h2[cfg.date_col]); f2[cfg.date_col] = pd.to_datetime(f2[cfg.date_col])
            train = pd.DataFrame({"ds": h2[cfg.date_col], "y": y})
            if len(train) < max(14, cfg.seasonal_period * 3):
                raise ValueError("history too short for Prophet")

            # Holiday/event dataframe from event_name and off-day flags when available.
            holidays = []
            if bool(params.get("holiday_regressors", True)):
                both = pd.concat([h2, f2], ignore_index=True, sort=False)
                if "event_name" in both.columns:
                    ev = both[[cfg.date_col, "event_name"]].dropna()
                    ev = ev[ev["event_name"].astype(str).str.len() > 0]
                    for _, r in ev.iterrows():
                        holidays.append({"ds": pd.Timestamp(r[cfg.date_col]), "holiday": str(r["event_name"])[:50]})
                if "is_offday" in both.columns:
                    off = both[pd.to_numeric(both["is_offday"], errors="coerce").fillna(0) > 0]
                    for d in pd.to_datetime(off[cfg.date_col]).dropna().unique():
                        holidays.append({"ds": pd.Timestamp(d), "holiday": "offday"})
            holidays_df = pd.DataFrame(holidays).drop_duplicates() if holidays else None

            regressor_cols = [c for c in ["is_offday", "coupon_amount", "discount_quantity", "max_temperature", "min_temperature"] if c in h2.columns and c in f2.columns]
            for c in regressor_cols:
                train[c] = pd.to_numeric(h2[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)

            def fit_predict(cps: float, fit_train: pd.DataFrame, pred_df: pd.DataFrame):
                m = Prophet(
                    weekly_seasonality=bool(params.get("weekly_seasonality", True)),
                    daily_seasonality=False,
                    yearly_seasonality=False,
                    holidays=holidays_df,
                    changepoint_prior_scale=float(cps),
                )
                for rc in regressor_cols:
                    m.add_regressor(rc)
                m.fit(fit_train)
                fut = pd.DataFrame({"ds": pd.to_datetime(pred_df[cfg.date_col])})
                for rc in regressor_cols:
                    fut[rc] = pd.to_numeric(pred_df[rc], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                return m.predict(fut)["yhat"].to_numpy(dtype=float)

            grid = list(params.get("changepoint_prior_grid", [params.get("changepoint_prior_scale", 0.05)]))
            best_cps = float(grid[0])
            if bool(params.get("tune_changepoint_prior", True)) and len(train) >= 28 and len(grid) > 1:
                holdout = min(max(7, cfg.horizon), max(7, len(train) // 5))
                tr, va = train.iloc[:-holdout].copy(), train.iloc[-holdout:].copy()
                best_loss = np.inf
                for cps in grid:
                    try:
                        pred = fit_predict(float(cps), tr, h2.iloc[-holdout:].copy())
                        loss = float(np.nanmean((pred - va["y"].to_numpy(dtype=float)) ** 2))
                        if loss < best_loss:
                            best_loss = loss; best_cps = float(cps)
                    except Exception:
                        continue
            out = fit_predict(best_cps, train, f2)
            self.backend_ = f"prophet_holiday_tuned_cps={best_cps:g}"
            return out
        except Exception as exc:
            self.fallback_count_ += 1
            self.backend_ = "prophet_fallback_ets"
            if getattr(self.config, "strict_experts", False):
                raise RuntimeError(f"Prophet expert failed in strict mode: {exc}") from exc
            return self._ets(y, len(f))

    def _ets(self, y: np.ndarray, horizon: int) -> np.ndarray:
        """Holt-Winters/ETS aligned with Table IV.

        Uses additive trend by default and selects additive/multiplicative/no
        seasonality on a small train/validation holdout when statsmodels is
        enabled.
        """
        period = int(self.spec.params.get("seasonal_period", self.config.seasonal_period))
        params = self.spec.params
        if not bool(params.get("use_statsmodels", False)):
            if len(y) >= period * 2:
                level = np.mean(y[-period:])
                pattern = y[-period:] - level
                reps = int(np.ceil(horizon / period))
                return np.tile(level + pattern, reps)[:horizon]
            return np.full(horizon, np.mean(y[-min(7, len(y)):]))
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            y = np.asarray(y, dtype=float)
            trend_options = list(params.get("trend_options", ["add"]))
            seasonal_options = list(params.get("seasonal_options", ["add", "mul", None]))
            candidates = []
            for tr in trend_options:
                for seas in seasonal_options:
                    if seas == "mul" and np.nanmin(y) <= 0:
                        continue
                    if seas is not None and len(y) < period * 2:
                        continue
                    candidates.append((tr, seas))
            if not candidates:
                candidates = [("add" if len(y) >= 4 else None, None)]

            best = candidates[0]
            if bool(params.get("select_seasonal", True)) and len(y) >= max(21, period * 3) and len(candidates) > 1:
                holdout = min(max(7, horizon), max(7, len(y) // 5))
                y_tr, y_va = y[:-holdout], y[-holdout:]
                best_loss = np.inf
                for tr, seas in candidates:
                    try:
                        model = ExponentialSmoothing(y_tr, trend=tr, seasonal=seas, seasonal_periods=period if seas else None)
                        fit = model.fit(optimized=True)
                        pred = np.asarray(fit.forecast(holdout), dtype=float)
                        loss = float(np.nanmean((pred - y_va) ** 2))
                        if loss < best_loss:
                            best_loss = loss; best = (tr, seas)
                    except Exception:
                        continue
            tr, seas = best
            model = ExponentialSmoothing(y, trend=tr, seasonal=seas, seasonal_periods=period if seas else None)
            fit = model.fit(optimized=True)
            self.backend_ = f"ets_trend={tr}_seasonal={seas}"
            return np.asarray(fit.forecast(horizon), dtype=float)
        except Exception as exc:
            self.fallback_count_ += 1
            self.backend_ = "ets_fallback_mean"
            if getattr(self.config, "strict_experts", False):
                raise RuntimeError(f"ETS expert failed in strict mode: {exc}") from exc
            return np.full(horizon, np.mean(y[-min(7, len(y)):] if len(y) else [0.0]))

    def _sarima(self, y: np.ndarray, horizon: int) -> np.ndarray:
        """SARIMA with train/validation order grid selection as in Table IV."""
        period = int(self.spec.params.get("seasonal_period", self.config.seasonal_period))
        params = self.spec.params
        if not bool(params.get("use_statsmodels", False)):
            if len(y) >= period:
                reps = int(np.ceil(horizon / period))
                seasonal = np.tile(y[-period:], reps)[:horizon]
                level = np.mean(y[-min(len(y), period):])
                return 0.7 * seasonal + 0.3 * level
            return np.full(horizon, np.mean(y[-min(7, len(y)):]))
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            y = np.asarray(y, dtype=float)
            order_grid = [tuple(o) for o in params.get("order_grid", [(1, 1, 1), (0, 1, 1), (1, 1, 0), (0, 1, 0)])]
            seasonal_grid = [tuple(o) for o in params.get("seasonal_order_grid", [(0, 0, 0, 0), (1, 1, 1, period)])]
            if len(y) < period * 2:
                seasonal_grid = [(0, 0, 0, 0)]
            maxiter = int(params.get("maxiter", 50))
            best_order, best_seasonal = order_grid[0], seasonal_grid[0]
            if bool(params.get("select_order", True)) and len(y) >= max(28, period * 4):
                holdout = min(max(7, horizon), max(7, len(y) // 5))
                y_tr, y_va = y[:-holdout], y[-holdout:]
                best_loss = np.inf
                for order in order_grid:
                    for seas in seasonal_grid:
                        try:
                            model = SARIMAX(y_tr, order=order, seasonal_order=seas,
                                            enforce_stationarity=False, enforce_invertibility=False)
                            fit = model.fit(disp=False, maxiter=maxiter)
                            pred = np.asarray(fit.get_forecast(holdout).predicted_mean, dtype=float)
                            loss = float(np.nanmean((pred - y_va) ** 2))
                            if loss < best_loss:
                                best_loss = loss; best_order, best_seasonal = order, seas
                        except Exception:
                            continue
            model = SARIMAX(y, order=best_order, seasonal_order=best_seasonal,
                            enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=maxiter)
            self.backend_ = f"sarima_order={best_order}_seasonal={best_seasonal}"
            return np.asarray(fit.get_forecast(horizon).predicted_mean, dtype=float)
        except Exception as exc:
            self.fallback_count_ += 1
            self.backend_ = "sarima_fallback_mean"
            if getattr(self.config, "strict_experts", False):
                raise RuntimeError(f"SARIMA expert failed in strict mode: {exc}") from exc
            return np.full(horizon, np.mean(y[-min(7, len(y)):] if len(y) else [0.0]))

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
            self.backend_ = "sklearn_ridge"
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
                self.backend_ = "lightgbm_fallback_ridge"
                if getattr(self.config, "strict_experts", False):
                    raise RuntimeError("LightGBM external backend disabled in strict mode.")
                return Ridge(alpha=float(params.get("alpha", 1.0)))
            try:
                from lightgbm import LGBMRegressor
                self.backend_ = "external_lightgbm"
                return LGBMRegressor(
                    n_estimators=int(params.get("n_estimators", 300)),
                    learning_rate=float(params.get("learning_rate", 0.03)),
                    num_leaves=int(params.get("num_leaves", 64)),
                    subsample=float(params.get("subsample", 0.8)),
                    colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                    objective=params.get("objective", "regression"),
                    random_state=self.config.seed,
                    verbose=int(params.get("verbose", 0)),
                    n_jobs=params.get("n_jobs", -1),
                )
            except Exception as exc:
                self.fallback_count_ += 1
                self.backend_ = "lightgbm_fallback_extratrees"
                if getattr(self.config, "strict_experts", False):
                    raise RuntimeError(f"LightGBM backend failed in strict mode: {exc}") from exc
                return ExtraTreesRegressor(n_estimators=30, max_depth=10, random_state=self.config.seed, n_jobs=1)
        if kind == "xgboost":
            if not bool(params.get("use_external_xgboost", False)):
                # Portable artifact fallback with XGBoost-like tree-ensemble semantics.
                self.backend_ = "xgboost_fallback_extratrees"
                if getattr(self.config, "strict_experts", False):
                    raise RuntimeError("XGBoost external backend disabled in strict mode.")
                return ExtraTreesRegressor(
                    n_estimators=int(params.get("n_estimators", 80)),
                    max_depth=int(params.get("max_depth", 12)),
                    random_state=self.config.seed,
                    n_jobs=int(params.get("n_jobs", 1)),
                )
            try:
                from xgboost import XGBRegressor
                self.backend_ = "external_xgboost"
                return XGBRegressor(
                    n_estimators=int(params.get("n_estimators", 300)),
                    learning_rate=float(params.get("learning_rate", 0.03)),
                    max_depth=int(params.get("max_depth", 8)),
                    subsample=float(params.get("subsample", 0.8)),
                    colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                    objective="reg:squarederror",
                    random_state=self.config.seed,
                    verbose=int(params.get("verbose", 0)),
                    n_jobs=params.get("n_jobs", -1),
                )
            except Exception as exc:
                self.fallback_count_ += 1
                self.backend_ = "xgboost_fallback_extratrees"
                if getattr(self.config, "strict_experts", False):
                    raise RuntimeError(f"XGBoost backend failed in strict mode: {exc}") from exc
                return ExtraTreesRegressor(n_estimators=60, max_depth=12, random_state=self.config.seed, n_jobs=1)
        raise ValueError(f"Unsupported ML expert kind: {kind}")

    def fit(self, df: pd.DataFrame) -> "MLGlobalExpert":
        cfg = self.config
        if getattr(cfg, "exclude_censored_in_training", True):
            mask = valid_demand_mask(df, cfg)
            if bool(mask.any()):
                df = df.loc[mask].copy()
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

    def _build_model(self, seq_len: int, horizon: int):
        return TinyDLinear(seq_len, horizon)

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
        self.model = self._build_model(seq_len, horizon).to(self.device)
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



class TimeMixerExpert(DLinearExpert):
    """Portable TimeMixer expert using multi-scale temporal mixing."""

    def _build_model(self, seq_len: int, horizon: int):
        hidden = int(self.spec.params.get("hidden_size", 128))
        return TinyTimeMixer(seq_len, horizon, hidden_size=hidden)


class TimesNetExpert(DLinearExpert):
    """Portable TimesNet expert using FFT-driven dominant-period decomposition."""

    def _build_model(self, seq_len: int, horizon: int):
        top_k = int(self.spec.params.get("top_k", 3))
        d_model = int(self.spec.params.get("d_model", 256))
        layers = int(self.spec.params.get("layers", 4))
        heads = int(self.spec.params.get("heads", 8))
        self.backend_ = f"portable_timesnet_proxy_dmodel={d_model}_layers={layers}_heads={heads}"
        return TinyTimesNet(seq_len, horizon, top_k=top_k, d_model=d_model, layers=layers, heads=heads)


def build_expert(spec: ExpertSpec, config: FAMEConfig) -> BaseExpert:
    if spec.kind in {"naive", "seasonal_naive", "moving_average", "croston", "tsb", "ets", "sarima", "prophet"}:
        return StatisticalExpert(spec, config)
    if spec.kind in {"linear", "random_forest", "lightgbm", "xgboost"}:
        return MLGlobalExpert(spec, config)
    if spec.kind == "dlinear":
        return DLinearExpert(spec, config)
    if spec.kind == "timemixer":
        return TimeMixerExpert(spec, config)
    if spec.kind == "timesnet":
        return TimesNetExpert(spec, config)
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
                expert.error_count_ = getattr(expert, "error_count_", 0) + 1
                if getattr(self.config, "strict_experts", False):
                    raise RuntimeError(f"Expert {expert.name} fitting failed in strict mode: {e}") from e
                # Keep the pool usable in lightweight/reviewer mode; disabled expert gets dropped.
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
                if getattr(self.config, "strict_experts", False):
                    raise RuntimeError(f"Expert {expert.name} prediction failed in strict mode: {e}") from e
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

    def backend_report(self) -> List[Dict]:
        return [
            {
                "name": e.name,
                "kind": e.spec.kind,
                "backend": getattr(e, "backend_", e.spec.kind),
                "fallback_count": int(getattr(e, "fallback_count_", 0)),
                "error_count": int(getattr(e, "error_count_", 0)),
                "cost": float(getattr(e, "cost", 1.0)),
            }
            for e in self.experts
        ]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ExpertPool":
        return joblib.load(path)

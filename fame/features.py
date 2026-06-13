# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

from .config import FAMEConfig
from .utils import valid_demand_mask

EPS = 1e-8


def _linear_slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    try:
        return float(np.polyfit(x, y.astype(float), deg=1)[0])
    except Exception:
        return 0.0


def _acf(y: np.ndarray, lag: int) -> float:
    if len(y) <= lag or lag <= 0:
        return 0.0
    a = y[:-lag].astype(float)
    b = y[lag:].astype(float)
    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _seasonal_strength(y: np.ndarray, period: int = 7) -> float:
    """Robust seasonal strength proxy without depending on statsmodels STL."""
    y = np.asarray(y, dtype=float)
    if len(y) < period * 2:
        return 0.0
    resid = y.copy()
    pattern = np.zeros(period, dtype=float)
    for k in range(period):
        vals = y[np.arange(len(y)) % period == k]
        pattern[k] = np.nanmean(vals) if len(vals) else 0.0
    seasonal = np.array([pattern[i % period] for i in range(len(y))])
    resid = y - seasonal
    denom = np.var(y) + EPS
    return float(np.clip(1.0 - np.var(resid) / denom, 0.0, 1.0))


def _trend_strength(y: np.ndarray, window: int = 7) -> float:
    y = np.asarray(y, dtype=float)
    if len(y) < 4:
        return 0.0
    s = pd.Series(y)
    trend = s.rolling(window=min(window, len(y)), min_periods=1, center=True).mean().to_numpy()
    resid = y - trend
    return float(np.clip(1.0 - np.var(resid) / (np.var(y) + EPS), 0.0, 1.0))


def _spectral_features(y: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    if len(y) < 4 or np.allclose(y, y[0]):
        return {
            "spectral_entropy": 0.0,
            "dominant_frequency": 0.0,
            "band_energy_low": 1.0,
            "band_energy_mid": 0.0,
            "band_energy_high": 0.0,
        }
    centered = y - np.mean(y)
    power = np.abs(np.fft.rfft(centered)) ** 2
    if len(power) <= 1:
        return {
            "spectral_entropy": 0.0,
            "dominant_frequency": 0.0,
            "band_energy_low": 1.0,
            "band_energy_mid": 0.0,
            "band_energy_high": 0.0,
        }
    power = power[1:]  # remove DC component
    total = power.sum() + EPS
    p = power / total
    entropy = -np.sum(p * np.log(p + EPS)) / (np.log(len(p) + EPS) + EPS)
    freqs = np.fft.rfftfreq(len(y))[1:]
    dominant = float(freqs[int(np.argmax(p))]) if len(freqs) else 0.0
    n = len(p)
    low = p[: max(1, n // 3)].sum()
    mid = p[max(1, n // 3): max(2, 2 * n // 3)].sum()
    high = p[max(2, 2 * n // 3):].sum()
    return {
        "spectral_entropy": float(entropy),
        "dominant_frequency": dominant,
        "band_energy_low": float(low),
        "band_energy_mid": float(mid),
        "band_energy_high": float(high),
    }


class ForecastabilityFingerprintExtractor:
    """Extract the interpretable forecastability fingerprint z_i.

    The extractor follows FAME's groups: lifecycle, sparsity/intermittency,
    volatility, trend, seasonality, spectral, metadata and context sensitivity.
    """

    def __init__(self, config: FAMEConfig):
        self.config = config
        self.metadata_cols = list(config.metadata_cols)
        self.context_cols = list(config.context_cols)
        self.feature_names_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.encoder_: Optional[OrdinalEncoder] = None
        self.scaler_: Optional[RobustScaler] = None
        self.fitted_: bool = False

    def _one_series_features(self, g: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> Dict[str, float | str]:
        cfg = self.config
        g = g.sort_values(cfg.date_col).copy()
        y = pd.to_numeric(g[cfg.target_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        dates = pd.to_datetime(g[cfg.date_col])
        active_mask = y > 0
        nonzero = y[active_mask]
        diffs = np.diff(y) if len(y) > 1 else np.array([0.0])
        ref_date = pd.Timestamp(reference_date) if reference_date is not None else pd.Timestamp(dates.max())

        duration = max(1, int((dates.max() - dates.min()).days) + 1)
        active_days = int(active_mask.sum())
        days_since_first = int((ref_date - dates.min()).days)
        last_sale_date = dates[active_mask].max() if active_mask.any() else dates.min()
        days_since_last = int((ref_date - last_sale_date).days)

        mean_y = float(np.mean(y)) if len(y) else 0.0
        std_y = float(np.std(y)) if len(y) else 0.0
        mean_nonzero = float(np.mean(nonzero)) if len(nonzero) else 0.0
        std_nonzero = float(np.std(nonzero)) if len(nonzero) else 0.0
        zero_ratio = float(np.mean(y == 0)) if len(y) else 1.0
        adi = float(len(y) / (active_days + EPS)) if len(y) else 0.0
        cv2 = float((std_nonzero / (mean_nonzero + EPS)) ** 2) if len(nonzero) else 0.0
        cv = float(std_y / (mean_y + EPS)) if mean_y > 0 else 0.0
        rolling_var = float(pd.Series(y).rolling(7, min_periods=1).var().fillna(0).mean()) if len(y) else 0.0
        q1, q3 = np.percentile(y, [25, 75]) if len(y) else (0.0, 0.0)
        iqr = q3 - q1
        outlier_ratio = float(np.mean((y > q3 + 1.5 * iqr) | (y < q1 - 1.5 * iqr))) if len(y) else 0.0
        burstiness = float((np.std(diffs) - np.mean(np.abs(diffs))) / (np.std(diffs) + np.mean(np.abs(diffs)) + EPS))
        slope = _linear_slope(y)
        trend_strength = _trend_strength(y)
        drift = float(y[-1] - y[0]) if len(y) >= 2 else 0.0
        ma_change = float(pd.Series(y).rolling(7, min_periods=1).mean().iloc[-1] - pd.Series(y).rolling(7, min_periods=1).mean().iloc[0]) if len(y) else 0.0
        seasonal_strength = _seasonal_strength(y, cfg.seasonal_period)
        acf_7 = _acf(y, cfg.seasonal_period)
        acf_14 = _acf(y, 2 * cfg.seasonal_period)
        spec = _spectral_features(y)

        feat: Dict[str, float | str] = {
            "duration": float(duration),
            "active_days": float(active_days),
            "days_since_first_sale": float(days_since_first),
            "days_since_last_sale": float(days_since_last),
            "zero_ratio": zero_ratio,
            "adi": adi,
            "nonzero_mean": mean_nonzero,
            "cv2": cv2,
            "cv": cv,
            "rolling_var": rolling_var,
            "outlier_ratio": outlier_ratio,
            "burstiness": burstiness,
            "trend_slope": slope,
            "trend_strength": trend_strength,
            "drift": drift,
            "moving_average_change": ma_change,
            "seasonal_strength": seasonal_strength,
            "acf_peak_7": acf_7,
            "acf_peak_14": acf_14,
            **spec,
        }

        # Metadata: keep first non-null or last known value.
        for col in self.metadata_cols:
            if col in g.columns:
                val = g[col].dropna().iloc[-1] if g[col].notna().any() else "missing"
                if pd.api.types.is_numeric_dtype(g[col]):
                    feat[f"meta_{col}"] = float(pd.to_numeric(pd.Series([val]), errors="coerce").fillna(0).iloc[0])
                else:
                    feat[f"meta_{col}"] = str(val)

        # Context sensitivity: derive robust response summaries from historical data.
        if "is_offday" in g.columns:
            off = pd.to_numeric(g["is_offday"], errors="coerce").fillna(0).to_numpy()
            feat["holiday_lift"] = float((y[off > 0].mean() if np.any(off > 0) else mean_y) - (y[off <= 0].mean() if np.any(off <= 0) else mean_y))
        if "coupon_amount" in g.columns:
            promo = pd.to_numeric(g["coupon_amount"], errors="coerce").fillna(0).to_numpy()
            feat["promotion_lift"] = float((y[promo > 0].mean() if np.any(promo > 0) else mean_y) - (y[promo <= 0].mean() if np.any(promo <= 0) else mean_y))
        if "discount_quantity" in g.columns:
            promo = pd.to_numeric(g["discount_quantity"], errors="coerce").fillna(0).to_numpy()
            feat["discount_lift"] = float((y[promo > 0].mean() if np.any(promo > 0) else mean_y) - (y[promo <= 0].mean() if np.any(promo <= 0) else mean_y))
        if "max_temperature" in g.columns:
            temp = pd.to_numeric(g["max_temperature"], errors="coerce").fillna(method="ffill").fillna(method="bfill").fillna(0).to_numpy()
            feat["weather_sensitivity"] = float(np.corrcoef(temp, y)[0, 1]) if len(temp) > 2 and np.std(temp) > EPS and np.std(y) > EPS else 0.0
        if "city_name" in g.columns:
            feat["ctx_city_name"] = str(g["city_name"].dropna().iloc[-1]) if g["city_name"].notna().any() else "missing"
        if "weather" in g.columns:
            feat["ctx_weather_mode"] = str(g["weather"].mode().iloc[0]) if not g["weather"].dropna().empty else "missing"
        return feat

    def raw_transform(self, df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        cfg = self.config
        records = []
        for key, g in df.groupby(list(cfg.id_cols), sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            g_use = g
            if getattr(cfg, "exclude_censored_in_fingerprint", True):
                mask = valid_demand_mask(g, cfg)
                # If every row is censored, fall back to the original group so that
                # a route can still be produced, but keep the common case leakage-safe.
                if bool(mask.any()):
                    g_use = g.loc[mask].copy()
            feat = {col: val for col, val in zip(cfg.id_cols, key)}
            feat.update(self._one_series_features(g_use, reference_date=reference_date))
            records.append(feat)
        return pd.DataFrame(records)

    def fit(self, df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> "ForecastabilityFingerprintExtractor":
        raw = self.raw_transform(df, reference_date)
        id_set = set(self.config.id_cols)
        feature_cols = [c for c in raw.columns if c not in id_set]
        disabled = tuple(getattr(self.config, "disabled_fingerprint_keywords", ()) or ())
        if disabled:
            feature_cols = [c for c in feature_cols if not any(k in c for k in disabled)]
        self.categorical_cols_ = [c for c in feature_cols if raw[c].dtype == object]
        self.numeric_cols_ = [c for c in feature_cols if c not in self.categorical_cols_]
        self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if self.categorical_cols_:
            self.encoder_.fit(raw[self.categorical_cols_].astype(str).fillna("missing"))
        self.scaler_ = RobustScaler()
        if self.numeric_cols_:
            self.scaler_.fit(raw[self.numeric_cols_].replace([np.inf, -np.inf], np.nan).fillna(0.0))
        self.feature_names_ = self.numeric_cols_ + self.categorical_cols_
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("ForecastabilityFingerprintExtractor must be fitted before transform().")
        raw = self.raw_transform(df, reference_date)
        ids = raw[list(self.config.id_cols)].copy()
        parts = []
        if self.numeric_cols_:
            num = raw.reindex(columns=self.numeric_cols_).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            num_arr = self.scaler_.transform(num) if self.scaler_ is not None else num.to_numpy()
            parts.append(pd.DataFrame(num_arr, columns=self.numeric_cols_, index=raw.index))
        if self.categorical_cols_:
            cat = raw.reindex(columns=self.categorical_cols_).astype(str).fillna("missing")
            cat_arr = self.encoder_.transform(cat) if self.encoder_ is not None else cat.to_numpy()
            parts.append(pd.DataFrame(cat_arr, columns=self.categorical_cols_, index=raw.index))
        feats = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=raw.index)
        return pd.concat([ids.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    def fit_transform(self, df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        return self.fit(df, reference_date).transform(df, reference_date)


class TimeSeriesFeatureBuilder:
    """Build per-day supervised features for ML experts.

    This corresponds to the existing rule-code's date, lag and rolling features,
    but is expert-agnostic and does not require a fixed class_num.
    """

    def __init__(self, config: FAMEConfig):
        self.config = config
        self.categorical_cols_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.encoder_: Optional[OrdinalEncoder] = None
        self.feature_cols_: List[str] = []
        self.fitted_: bool = False

    def _build_one(self, g: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        g = g.sort_values(cfg.date_col).copy()
        g[cfg.date_col] = pd.to_datetime(g[cfg.date_col])
        y = pd.to_numeric(g[cfg.target_col], errors="coerce").fillna(0.0)
        g[cfg.target_col] = y
        g["year"] = g[cfg.date_col].dt.year
        g["month"] = g[cfg.date_col].dt.month
        g["week"] = g[cfg.date_col].dt.isocalendar().week.astype(int)
        g["day_of_week"] = g[cfg.date_col].dt.dayofweek
        g["day_of_month"] = g[cfg.date_col].dt.day
        g["day_of_year"] = g[cfg.date_col].dt.dayofyear
        g["weekend"] = (g[cfg.date_col].dt.dayofweek >= 5).astype(int)
        g["month_start"] = g[cfg.date_col].dt.is_month_start.astype(int)
        g["month_end"] = g[cfg.date_col].dt.is_month_end.astype(int)
        g["quarter_start"] = g[cfg.date_col].dt.is_quarter_start.astype(int)
        g["quarter_end"] = g[cfg.date_col].dt.is_quarter_end.astype(int)
        # Cyclical date features.
        for col, max_val in [("day_of_week", 7), ("month", 12), ("day_of_year", 366)]:
            g[f"{col}_sin"] = np.sin(2 * np.pi * g[col] / max_val)
            g[f"{col}_cos"] = np.cos(2 * np.pi * g[col] / max_val)
        # Lag and rolling features. Last-horizon future rows should have target set to 0;
        # lags are based on previous known rows only.
        for lag in [1, 2, 3, 7, 14, 28]:
            g[f"{cfg.target_col}_lag_{lag}"] = y.shift(lag)
        for lag in [1, 7]:
            base = g[f"{cfg.target_col}_lag_{lag}"]
            for window in [3, 5, 7, 14]:
                g[f"{cfg.target_col}_lag_{lag}_roll_{window}_mean"] = base.rolling(window, min_periods=1).mean()
                g[f"{cfg.target_col}_lag_{lag}_roll_{window}_median"] = base.rolling(window, min_periods=1).median()
                g[f"{cfg.target_col}_lag_{lag}_roll_{window}_max"] = base.rolling(window, min_periods=1).max()
                g[f"{cfg.target_col}_lag_{lag}_roll_{window}_min"] = base.rolling(window, min_periods=1).min()
                g[f"{cfg.target_col}_lag_{lag}_roll_{window}_std"] = base.rolling(window, min_periods=1).std().fillna(0)
        return g

    def build_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        frames = [self._build_one(g) for _, g in df.groupby(list(cfg.id_cols), sort=False)]
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        out = out.replace([np.inf, -np.inf], np.nan)
        return out

    def fit(self, df: pd.DataFrame) -> "TimeSeriesFeatureBuilder":
        raw = self.build_raw(df)
        exclude = set(list(self.config.id_cols) + [self.config.date_col, self.config.target_col])
        candidates = [c for c in raw.columns if c not in exclude]
        self.categorical_cols_ = [c for c in candidates if raw[c].dtype == object]
        self.numeric_cols_ = [c for c in candidates if c not in self.categorical_cols_]
        self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if self.categorical_cols_:
            self.encoder_.fit(raw[self.categorical_cols_].astype(str).fillna("missing"))
        self.feature_cols_ = self.numeric_cols_ + self.categorical_cols_
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("TimeSeriesFeatureBuilder must be fitted before transform().")
        raw = self.build_raw(df)
        meta = raw[list(self.config.id_cols) + [self.config.date_col, self.config.target_col]].copy()
        parts = []
        if self.numeric_cols_:
            # Never backward-fill supervised lag/rolling features. A global bfill can
            # leak future values and even values from another series into early rows.
            # Missing lag/context features are represented by 0 after upstream
            # leakage-safe future-context filling.
            num = raw.reindex(columns=self.numeric_cols_).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            parts.append(num.astype(float))
        if self.categorical_cols_:
            cat = raw.reindex(columns=self.categorical_cols_).astype(str).fillna("missing")
            enc = self.encoder_.transform(cat) if self.encoder_ is not None else cat.to_numpy()
            parts.append(pd.DataFrame(enc, columns=self.categorical_cols_, index=raw.index))
        feats = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=raw.index)
        return pd.concat([meta.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import FAMEConfig
from .experts import ExpertPool
from .features import ForecastabilityFingerprintExtractor
from .oracle import OracleMiningResult, mine_oracle
from .router import SparseRouter
from .utils import chronological_split_dates, complete_daily_grid, ensure_datetime, resolve_torch_device, set_seed, split_validation_window, valid_demand_mask


class FAMEModel:
    """End-to-end FAME implementation.

    The pipeline implements the paper's offline/online workflow:
    1. align series and covariates;
    2. extract forecastability fingerprints;
    3. train fixed heterogeneous expert pool;
    4. mine validation-loss oracle suitability;
    5. train sparse cost-aware router;
    6. online Top-r expert execution and weighted fusion.
    """

    def __init__(self, config: Optional[FAMEConfig] = None):
        self.config = config or FAMEConfig()
        self.fingerprint_extractor = ForecastabilityFingerprintExtractor(self.config)
        self.expert_pool: Optional[ExpertPool] = None
        self.router: Optional[SparseRouter] = None
        self.oracle_: Optional[OracleMiningResult] = None
        self.train_cutoff_: Optional[pd.Timestamp] = None
        self.test_cutoff_: Optional[pd.Timestamp] = None
        self.fitted_: bool = False
        self.calibration_: Optional[Dict] = None


    def set_device(self, device: str) -> "FAMEModel":
        """Move torch-based router/deep experts to a configured runtime device."""
        self.config.device = device
        torch_device = resolve_torch_device(device)
        if self.router is not None:
            self.router.device = torch_device
            if self.router.model is not None:
                self.router.model.to(torch_device)
        if self.expert_pool is not None:
            for expert in self.expert_pool.experts:
                if hasattr(expert, "device"):
                    expert.device = torch_device
                model = getattr(expert, "model", None)
                if model is not None and hasattr(model, "to"):
                    model.to(torch_device)
        return self

    def _prepare(self, df: pd.DataFrame, complete_grid: bool = True) -> pd.DataFrame:
        cfg = self.config
        df = ensure_datetime(df, cfg.date_col).copy()
        required = list(cfg.id_cols) + [cfg.date_col, cfg.target_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"输入数据缺少必要列: {missing}")
        df[cfg.target_col] = pd.to_numeric(df[cfg.target_col], errors="coerce").fillna(0.0)
        df = df.sort_values(list(cfg.id_cols) + [cfg.date_col]).reset_index(drop=True)
        if complete_grid:
            df = complete_daily_grid(
                df, cfg.id_cols, cfg.date_col, cfg.target_col,
                static_cols=cfg.metadata_cols,
                dynamic_cols=list(cfg.context_cols) + list(cfg.availability_cols),
                allow_dynamic_bfill=cfg.allow_dynamic_context_bfill,
            )
        return df

    def chronological_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        val_start, test_start = chronological_split_dates(df, cfg.date_col, cfg.validation_ratio, cfg.test_ratio)
        self.train_cutoff_, self.test_cutoff_ = val_start, test_start
        train_df = df[df[cfg.date_col] < val_start].copy()
        val_df = df[(df[cfg.date_col] >= val_start) & (df[cfg.date_col] < test_start)].copy()
        test_df = df[df[cfg.date_col] >= test_start].copy()
        return train_df, val_df, test_df

    def fit(
        self,
        df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
        validation_df: Optional[pd.DataFrame] = None,
        complete_grid: bool = True,
        loss_metric: str = "mse",
    ) -> "FAMEModel":
        """Fit FAME from raw daily sales data.

        If ``train_df`` and ``validation_df`` are omitted, chronological split is
        built from ``df`` using config ratios. The validation window is further
        split into oracle-mining and router-calibration sections.
        """
        set_seed(self.config.seed)
        cfg = self.config
        # Split before grid completion to avoid leaking dynamic context from
        # validation/test dates into training rows through backward fill. Each
        # split is completed independently with leakage-safe fill rules.
        raw_data = self._prepare(df, complete_grid=False)
        if train_df is None or validation_df is None:
            train_raw, val_raw, _ = self.chronological_split(raw_data)
            train_df = self._prepare(train_raw, complete_grid=complete_grid)
            val_df = self._prepare(val_raw, complete_grid=complete_grid)
            oracle_df, calib_df = split_validation_window(val_df, cfg.date_col, cfg.oracle_fraction_in_validation)
        else:
            train_df = self._prepare(train_df, complete_grid=complete_grid)
            val_df = self._prepare(validation_df, complete_grid=complete_grid)
            oracle_df, calib_df = split_validation_window(val_df, cfg.date_col, cfg.oracle_fraction_in_validation)

        if train_df.empty or oracle_df.empty:
            raise ValueError("train_df 或 oracle-mining validation window 为空，无法训练 FAME。")

        # 1) Expert pool fitting.
        self.expert_pool = ExpertPool(cfg).fit(train_df)

        # 2) Fingerprint fitting on training history only to prevent leakage.
        self.fingerprint_extractor.fit(train_df, reference_date=train_df[cfg.date_col].max())

        # 3) Oracle mining from validation-loss matrix.
        self.oracle_ = mine_oracle(self.expert_pool, train_df, oracle_df, cfg, metric=loss_metric)

        # 4) Router training. Fingerprints are extracted from history up to oracle origin.
        fp = self.fingerprint_extractor.transform(train_df, reference_date=train_df[cfg.date_col].max())
        # Reorder by soft target rows to keep tensor alignment.
        ordered_fp = self.oracle_.soft_targets[list(cfg.id_cols)].merge(fp, on=list(cfg.id_cols), how="inner")
        ordered_soft = ordered_fp[list(cfg.id_cols)].merge(self.oracle_.soft_targets, on=list(cfg.id_cols), how="left")
        n = min(len(ordered_fp), self.oracle_.prediction_tensor.shape[0])
        ordered_fp = ordered_fp.iloc[:n].reset_index(drop=True)
        ordered_soft = ordered_soft.iloc[:n].reset_index(drop=True)
        pred_tensor = self.oracle_.prediction_tensor[:n]
        target_tensor = self.oracle_.target_tensor[:n]
        self.router = SparseRouter(cfg, self.expert_pool.expert_names, self.expert_pool.costs)
        self.router.fit(ordered_fp, ordered_soft, pred_tensor=pred_tensor, target_tensor=target_tensor)

        # 5) Router calibration on the held-out later validation subwindow.
        calib_history = pd.concat([train_df, oracle_df], ignore_index=True, sort=False)
        self.fitted_ = True
        self.calibration_ = self._calibrate_router_delta(calib_history, calib_df)

        self.fitted_ = True
        return self


    def _calibrate_router_delta(self, history_df: pd.DataFrame, calib_df: pd.DataFrame) -> Dict:
        """Select the Top-r pruning threshold delta on the calibration window.

        The paper separates oracle mining and router calibration. This routine
        evaluates candidate deltas on the later validation subwindow using the
        already trained router and fitted experts. It freezes the best delta for
        deployment and stores a calibration report for artifact inspection.
        """
        cfg = self.config
        if calib_df is None or calib_df.empty or not getattr(cfg, "enable_router_calibration", True):
            return {"enabled": False, "reason": "empty calibration window or disabled"}
        original_delta = float(cfg.delta)
        best = {"delta": original_delta, "mse": float("inf"), "mae": float("inf"), "valid_rows": 0}
        scores = []
        # History available at the calibration origin includes training and the
        # earlier oracle-mining validation part.
        calib_start = pd.to_datetime(calib_df[cfg.date_col]).min()
        hist = history_df[history_df[cfg.date_col] < calib_start].copy()
        if hist.empty:
            hist = history_df.copy()
        for delta in list(getattr(cfg, "calibration_delta_grid", (original_delta,))):
            cfg.delta = float(delta)
            try:
                eval_rows = self.evaluate_window(hist, calib_df)
                valid = valid_demand_mask(eval_rows, cfg)
                eval_valid = eval_rows.loc[valid].copy()
                if eval_valid.empty:
                    mse_value = float("inf")
                    mae_value = float("inf")
                else:
                    err = pd.to_numeric(eval_valid[cfg.target_col], errors="coerce").fillna(0.0) - pd.to_numeric(eval_valid["predicted_sales"], errors="coerce").fillna(0.0)
                    mse_value = float(np.mean(np.square(err)))
                    mae_value = float(np.mean(np.abs(err)))
                row = {"delta": float(delta), "mse": mse_value, "mae": mae_value, "valid_rows": int(len(eval_valid))}
            except Exception as exc:
                row = {"delta": float(delta), "mse": float("inf"), "mae": float("inf"), "valid_rows": 0, "error": str(exc)}
            scores.append(row)
            if row["mse"] < best["mse"]:
                best = dict(row)
        cfg.delta = float(best.get("delta", original_delta))
        report = {
            "enabled": True,
            "selected_delta": cfg.delta,
            "original_delta": original_delta,
            "criterion": "calibration_mse_excluding_censored_days",
            "scores": scores,
        }
        return report

    def _normalize_group_key(self, key):
        return key if isinstance(key, tuple) else (key,)

    def _fill_future_missing(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        """Fill future covariate holes from latest history and within-series ffill/bfill.

        This is a production safeguard for partially missing weather, holiday,
        promotion or metadata frames. It keeps ML expert matrices NaN-free before
        the feature builder is called.
        """
        cfg = self.config
        if future.empty:
            return future
        id_cols = list(cfg.id_cols)
        key_cols = id_cols + [cfg.date_col]
        fill_cols = [
            c for c in list(cfg.metadata_cols) + list(cfg.context_cols) + list(cfg.availability_cols)
            if c in future.columns and c not in key_cols and c != cfg.target_col
        ]
        if not fill_cols:
            return future

        future = future.sort_values(key_cols).copy()
        hist = history.sort_values(key_cols).copy()
        last_known = hist.groupby(id_cols, sort=False).tail(1)[id_cols + fill_cols].copy()
        future = future.merge(last_known, on=id_cols, how="left", suffixes=("", "_hist_fill"))
        for c in fill_cols:
            hc = f"{c}_hist_fill"
            if hc in future.columns:
                future[c] = future[c].combine_first(future[hc])
                future.drop(columns=[hc], inplace=True)

        # ffill/bfill within each series, then deterministic final fallback.
        future[fill_cols] = future.groupby(id_cols, sort=False)[fill_cols].ffill()
        future[fill_cols] = future.groupby(id_cols, sort=False)[fill_cols].bfill()
        for c in fill_cols:
            if pd.api.types.is_numeric_dtype(future[c]):
                future[c] = pd.to_numeric(future[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            else:
                future[c] = future[c].astype(object).where(future[c].notna(), "missing")
        return future

    def make_future_frame(
        self,
        history_df: pd.DataFrame,
        start_date: Optional[pd.Timestamp] = None,
        periods: Optional[int] = None,
        future_context_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate a future frame for inference.

        If future covariates such as weather/holiday are available, pass them in
        ``future_context_df`` with ``date_col`` and optionally id/city columns.
        Otherwise the latest known metadata/context values are forward-filled.
        """
        cfg = self.config
        periods = periods or cfg.horizon
        history_df = self._prepare(history_df, complete_grid=False)
        rows = []
        for _, g in history_df.groupby(list(cfg.id_cols), sort=False):
            g = g.sort_values(cfg.date_col)
            last = g.iloc[-1].copy()
            sdate = pd.Timestamp(start_date) if start_date is not None else pd.Timestamp(last[cfg.date_col]) + pd.Timedelta(days=1)
            dates = pd.date_range(sdate, periods=periods, freq="D")
            for d in dates:
                row = last.copy()
                row[cfg.date_col] = d
                row[cfg.target_col] = 0.0
                rows.append(row)
        future = pd.DataFrame(rows)
        if future_context_df is not None and not future_context_df.empty:
            ctx = ensure_datetime(future_context_df, cfg.date_col)
            merge_cols = [cfg.date_col]
            # Prefer id-level future covariates; otherwise date + city covariates.
            if all(c in ctx.columns for c in cfg.id_cols):
                merge_cols = list(cfg.id_cols) + [cfg.date_col]
            elif "city_name" in ctx.columns and "city_name" in future.columns:
                merge_cols = [cfg.date_col, "city_name"]
            replace_cols = [c for c in ctx.columns if c not in merge_cols]
            future = future.merge(ctx[merge_cols + replace_cols], on=merge_cols, how="left", suffixes=("", "_future"))
            for c in replace_cols:
                fc = f"{c}_future"
                if fc in future.columns:
                    future[c] = future[fc].combine_first(future.get(c))
                    future.drop(columns=[fc], inplace=True)
        future = self._fill_future_missing(history_df, future)
        future = future.sort_values(list(cfg.id_cols) + [cfg.date_col]).reset_index(drop=True)
        return future

    def predict(
        self,
        history_df: pd.DataFrame,
        future_df: Optional[pd.DataFrame] = None,
        future_context_df: Optional[pd.DataFrame] = None,
        return_explanations: bool = True,
    ) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
        """Sparse Top-r inference with group-by-expert batching.

        The router is first evaluated for all series. Then an inverted index
        collects all series assigned to each expert and executes each expert once
        on its assigned batch. This keeps the FAME sparse semantics but avoids
        expensive per-series ``expert.predict`` calls for tree/deep experts.
        """
        if not self.fitted_ or self.expert_pool is None or self.router is None:
            raise RuntimeError("FAMEModel must be fitted or loaded before predict().")
        cfg = self.config
        id_cols = list(cfg.id_cols)
        key_cols = id_cols + [cfg.date_col]
        history = self._prepare(history_df, complete_grid=True)
        if future_df is None:
            future = self.make_future_frame(history, future_context_df=future_context_df)
        else:
            future_raw = future_df.copy()
            if cfg.target_col not in future_raw.columns:
                future_raw[cfg.target_col] = 0.0
            future = self._prepare(future_raw, complete_grid=False)
            future[cfg.target_col] = 0.0
            future = self._fill_future_missing(history, future)

        fp = self.fingerprint_extractor.transform(history, reference_date=history[cfg.date_col].max())
        probs = self.router.predict_proba(fp)

        history_groups = {self._normalize_group_key(k): g.copy() for k, g in history.groupby(id_cols, sort=False)}
        future_groups = {self._normalize_group_key(k): g.copy() for k, g in future.groupby(id_cols, sort=False)}
        prob_rows = {self._normalize_group_key(k): g.iloc[0] for k, g in probs.groupby(id_cols, sort=False)}

        expert_by_name = {e.name: e for e in self.expert_pool.experts}
        expert_to_keys: Dict[str, List[Tuple]] = {name: [] for name in expert_by_name}
        route_rows = []

        # Stage 1: route all series and build expert-wise inverted index.
        for key, f in future_groups.items():
            h = history_groups.get(key)
            prob_row = prob_rows.get(key)
            if h is None or h.empty or prob_row is None:
                continue
            available = self.expert_pool.available_mask_for_series(h)
            active_names, weights = self.router.select_active_experts(prob_row, available_mask=available)
            for name, w in zip(active_names, weights):
                if name not in expert_to_keys:
                    continue
                expert_to_keys[name].append(key)
                route_rows.append({
                    **{col: val for col, val in zip(id_cols, key)},
                    "expert": name,
                    "weight": float(w),
                    "raw_probability": float(prob_row[name]),
                    "top_r": cfg.top_r,
                    "delta": cfg.delta,
                })

        if not route_rows:
            empty_pred = pd.DataFrame(columns=key_cols + ["predicted_sales"])
            empty_exp = pd.DataFrame(columns=id_cols + ["expert", "weight", "raw_probability", "top_r", "delta"])
            return (empty_pred, empty_exp) if return_explanations else empty_pred

        route_df = pd.DataFrame(route_rows)
        prediction_parts = []

        def run_expert_batch(expert_name: str, keys: List[Tuple]) -> pd.DataFrame:
            expert = expert_by_name[expert_name]
            h_batch = pd.concat([history_groups[k] for k in keys if k in history_groups], ignore_index=True)
            f_batch = pd.concat([future_groups[k] for k in keys if k in future_groups], ignore_index=True)
            if h_batch.empty or f_batch.empty:
                return pd.DataFrame()
            try:
                return expert.predict(h_batch, f_batch)
            except Exception as batch_exc:
                if getattr(cfg, "strict_experts", False):
                    raise RuntimeError(f"Expert {expert_name} batch prediction failed in strict mode: {batch_exc}") from batch_exc
                # Robust fallback: isolate bad series without losing the whole expert batch.
                frames = []
                for k in keys:
                    try:
                        frames.append(expert.predict(history_groups[k], future_groups[k]))
                    except Exception as single_exc:
                        if getattr(cfg, "strict_experts", False):
                            raise RuntimeError(f"Expert {expert_name} failed for series {k} in strict mode: {single_exc}") from single_exc
                        print(f"[WARN] Expert {expert_name} failed for series {k}: {single_exc}")
                return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        # Stage 2: execute selected experts by batch and attach router weights.
        for expert_name, keys in expert_to_keys.items():
            if not keys:
                continue
            pred = run_expert_batch(expert_name, keys)
            if pred.empty:
                continue
            pred = pred.copy()
            pred["expert"] = expert_name
            w = route_df[route_df["expert"] == expert_name][id_cols + ["expert", "weight"]]
            pred = pred.merge(w, on=id_cols + ["expert"], how="inner")
            pred["prediction"] = pd.to_numeric(pred["prediction"], errors="coerce").fillna(0.0)
            pred["weighted_prediction"] = pred["prediction"] * pred["weight"].astype(float)
            prediction_parts.append(pred[key_cols + ["expert", "weight", "prediction", "weighted_prediction"]])

        if prediction_parts:
            allp = pd.concat(prediction_parts, ignore_index=True)
            # Re-normalize weights over successfully executed experts for each
            # id-date. This prevents under-forecasting when a non-strict run skips
            # a failed selected expert after routing.
            denom = allp.groupby(key_cols)["weight"].transform("sum").replace(0, np.nan)
            allp["renorm_weight"] = (allp["weight"] / denom).fillna(0.0)
            allp["weighted_prediction"] = allp["prediction"] * allp["renorm_weight"]
            final = allp.groupby(key_cols, as_index=False)["weighted_prediction"].sum()
            final.rename(columns={"weighted_prediction": "predicted_sales"}, inplace=True)
        else:
            final = pd.DataFrame(columns=key_cols + ["predicted_sales"])

        if cfg.clip_non_negative and not final.empty:
            final["predicted_sales"] = final["predicted_sales"].clip(lower=0)
        if cfg.round_output and not final.empty:
            final["predicted_sales"] = np.round(final["predicted_sales"]).astype(int)
        final = final.sort_values(key_cols).reset_index(drop=True)
        explanation = route_df.sort_values(id_cols + ["weight"], ascending=[True] * len(id_cols) + [False]).reset_index(drop=True)
        if return_explanations:
            return final, explanation
        return final

    def evaluate_window(self, history_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
        """Convenience evaluation on a known future window."""
        cfg = self.config
        future = eval_df.copy()
        actual = future[list(cfg.id_cols) + [cfg.date_col, cfg.target_col]].copy()
        pred, _ = self.predict(history_df, future_df=future, return_explanations=True)
        merged = actual.merge(pred, on=list(cfg.id_cols) + [cfg.date_col], how="left")
        merged["predicted_sales"] = merged["predicted_sales"].fillna(0.0)
        valid = valid_demand_mask(merged, cfg)
        err = merged[cfg.target_col] - merged["predicted_sales"]
        merged["is_valid_demand"] = valid.astype(int)
        merged["abs_error"] = np.where(valid, err.abs(), np.nan)
        merged["sq_error"] = np.where(valid, err ** 2, np.nan)
        return merged

    def save(self, directory: str) -> None:
        """Persist the full FAME model.

        Torch modules are moved to CPU before serialization so that artifacts
        trained on GPU can be loaded on CPU-only validation or deployment
        machines. After saving, the model is moved back to the configured
        runtime device when possible.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        original_device = self.config.device
        try:
            self.set_device("cpu")
        except Exception:
            pass
        joblib.dump(self, path / "fame_model.joblib")
        try:
            import json
            meta = {
                "top_r": self.config.top_r,
                "delta": self.config.delta,
                "tau": self.config.tau,
                "gamma_cost": self.config.gamma_cost,
                "expert_names": self.expert_pool.expert_names if self.expert_pool is not None else [],
                "expert_backends": self.expert_pool.backend_report() if self.expert_pool is not None else [],
                "calibration": self.calibration_,
            }
            with open(path / "model_metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            if self.calibration_ is not None:
                with open(path / "router_calibration.json", "w", encoding="utf-8") as f:
                    json.dump(self.calibration_, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        try:
            self.set_device(original_device)
        except Exception:
            pass

    @staticmethod
    def load(directory: str, device: Optional[str] = None) -> "FAMEModel":
        """Load a saved FAME model with CPU-safe CUDA fallback.

        Joblib artifacts may contain PyTorch tensors. If the artifact was saved
        from a CUDA machine and is later loaded on a CPU-only server, standard
        unpickling can fail. The fallback below maps torch storages to CPU and
        then applies ``set_device`` to the requested runtime device.
        """
        path = Path(directory)
        file = path / "fame_model.joblib"
        if not file.exists():
            raise FileNotFoundError(f"Saved FAME model not found: {file}")
        try:
            model = joblib.load(file)
        except RuntimeError as exc:
            msg = str(exc)
            if "deserialize object on a CUDA device" not in msg and "CUDA" not in msg:
                raise
            import io
            import torch
            import torch.storage

            original_loader = torch.storage._load_from_bytes

            def _load_from_bytes_cpu(b):
                return torch.load(io.BytesIO(b), map_location=torch.device("cpu"), weights_only=False)

            torch.storage._load_from_bytes = _load_from_bytes_cpu
            try:
                model = joblib.load(file)
            finally:
                torch.storage._load_from_bytes = original_loader
        if device is not None:
            model.set_device(device)
        else:
            model.set_device(model.config.device)
        return model

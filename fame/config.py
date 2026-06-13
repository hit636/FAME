# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class ExpertSpec:
    """Definition of one expert in the FAME expert pool.

    Parameters
    ----------
    name:
        Unique expert name, e.g. ``lightgbm`` or ``ets``.
    kind:
        Expert implementation identifier. Supported built-ins include
        ``sarima``, ``ets``, ``prophet``, ``tsb``, ``linear``,
        ``lightgbm``, ``xgboost``, ``dlinear``, ``timemixer`` and ``timesnet``.
        Lightweight baseline experts such as ``naive`` and ``moving_average`` are
        also supported for ablation and debugging.
    cost:
        Normalized inference cost. 1.0 may be set to the strongest cheap expert
        such as LightGBM; high-cost experts should use larger values.
    enabled:
        Whether this expert is active.
    params:
        Expert-specific hyperparameters.
    """

    name: str
    kind: str
    cost: float = 1.0
    enabled: bool = True
    params: Dict = field(default_factory=dict)


@dataclass
class FAMEConfig:
    """Top-level FAME configuration."""

    # Column names
    id_cols: Sequence[str] = ("vem_id", "merc_id")
    date_col: str = "date"
    target_col: str = "daily_quantity"

    # Common optional feature columns in vending-machine data.
    metadata_cols: Sequence[str] = (
        "merc_brand_code",
        "merc_type_code",
        "machine_type",
        "capacity",
        "scene_code",
        "city_name",
        "merc_sale_price",
    )
    context_cols: Sequence[str] = (
        "max_temperature",
        "min_temperature",
        "weather",
        "wind_level",
        "event_name",
        "is_offday",
        "coupon_amount",
        "discount_quantity",
    )
    availability_cols: Sequence[str] = (
        "is_available",
        "stockout_flag",
        "outage_flag",
        "suspension_flag",
    )

    # Forecasting protocol
    horizon: int = 14
    lookback: int = 56
    min_history: int = 15
    seasonal_period: int = 7
    validation_ratio: float = 0.10
    test_ratio: float = 0.20
    oracle_fraction_in_validation: float = 0.50

    # Router and sparse inference
    top_r: int = 2
    delta: float = 0.05
    tau: float = 0.30
    eta_oracle_cost: float = 0.0
    lambda_router: float = 1.0
    beta_balance: float = 0.01
    gamma_cost: float = 0.01
    hidden_size: int = 128
    dropout: float = 0.10
    router_epochs: int = 200
    router_lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    patience: int = 30
    seed: int = 42


    # Runtime / performance
    # ``auto`` uses CUDA when available; set to ``cpu`` or ``cuda:1`` in production schedulers.
    device: str = "auto"
    # Group-by-expert inference runs each selected expert on a batch of assigned series.
    # This is much faster than per-series expert execution for tree/deep experts.
    group_by_expert_inference: bool = True
    # Safety cap for DLinear training windows. None means using all windows.
    max_dlinear_windows: Optional[int] = 200_000

    # Paper-protocol calibration. The validation window is split into oracle mining
    # and router calibration windows. In this portable implementation, router
    # training uses the oracle-mining window, while the calibration window selects
    # the deployment-time pruning threshold delta. Full tau/gamma retraining grids
    # can be enabled by external experiment runners when compute budget permits.
    enable_router_calibration: bool = True
    calibration_delta_grid: Sequence[float] = (0.0, 0.02, 0.05, 0.10)

    # Multi-step ML forecasting protocol. Recursive prediction prevents future
    # horizon lag features from being polluted by placeholder zeros.
    recursive_ml_prediction: bool = True

    # Grid completion / leakage control. Static metadata may be backfilled within
    # a split, but dynamic context is forward-filled only to avoid future leakage.
    allow_dynamic_context_bfill: bool = False

    # Censored demand handling. In industrial logs, observed sales may be truncated
    # by stockout/outage/suspension. Excluding these rows from supervised expert
    # training and fingerprint statistics keeps the implementation aligned with
    # the paper's leakage/censoring protocol.
    exclude_censored_in_training: bool = True
    exclude_censored_in_fingerprint: bool = True

    # Artifact / reviewer controls. ``strict_experts`` disables silent fallback for
    # paper-style reproduction; lightweight reviewer runs can keep it False.
    strict_experts: bool = False
    record_expert_backend: bool = True
    debug_expert_timing: bool = False

    # Fingerprint ablation support. Each keyword removes matching feature columns
    # after raw fingerprint extraction, e.g. ["zero_ratio", "adi"] or ["spectral"].
    disabled_fingerprint_keywords: Sequence[str] = ()

    # Post-processing
    clip_non_negative: bool = True
    round_output: bool = False

    # Paper-aligned fixed expert pool. The production paper uses ten experts:
    # SARIMA, Holt-Winters/ETS, Prophet, Croston/TSB, Linear Regression,
    # XGBoost, LightGBM, DLinear, TimeMixer and TimesNet.
    #
    # By default, this artifact tries to use the full recommended dependencies
    # (statsmodels, prophet, xgboost and lightgbm). Robust fallbacks remain inside
    # each expert so that the workflow can still run when a dependency is missing
    # or a model fails on a short/intermittent series. TimeMixer and TimesNet are
    # implemented as independent portable neural experts: TimeMixer uses
    # multi-scale temporal mixing, while TimesNet uses FFT-driven dominant-period
    # decomposition. Users can still plug official implementations into the same
    # BaseExpert interface for exact production reproduction.
    expert_specs: List[ExpertSpec] = field(default_factory=lambda: [
        ExpertSpec("sarima", "sarima", cost=1.2, params={"seasonal_period": 7, "use_statsmodels": True, "select_order": True, "order_grid": [(1,1,1), (0,1,1), (1,1,0), (0,1,0)], "seasonal_order_grid": [(0,0,0,0), (1,1,1,7)], "maxiter": 50}),
        ExpertSpec("ets", "ets", cost=0.8, params={"seasonal_period": 7, "use_statsmodels": True, "trend_options": ["add"], "seasonal_options": ["add", "mul", None], "select_seasonal": True}),
        ExpertSpec("prophet", "prophet", cost=1.0, params={"seasonal_period": 7, "use_prophet": True, "weekly_seasonality": True, "holiday_regressors": True, "tune_changepoint_prior": True, "changepoint_prior_grid": [0.01, 0.05, 0.10]}),
        ExpertSpec("croston_tsb", "tsb", cost=0.6, params={"alpha": 0.1, "beta": 0.1}),
        ExpertSpec("linear", "linear", cost=0.6),
        ExpertSpec("xgboost", "xgboost", cost=1.1, params={"n_estimators": 300, "learning_rate": 0.015, "max_depth": 10, "subsample": 0.65, "colsample_bytree": 0.8, "use_external_xgboost": True}),
        ExpertSpec("lightgbm", "lightgbm", cost=1.0, params={"n_estimators": 300, "objective": "tweedie", "learning_rate": 0.015, "num_leaves": 256, "subsample": 0.65, "colsample_bytree": 0.8, "use_external_lightgbm": True}),
        ExpertSpec("dlinear", "dlinear", cost=3.0, params={"seq_len": 49, "epochs": 5}),
        ExpertSpec("timemixer", "timemixer", cost=3.6, params={"seq_len": 49, "epochs": 5, "hidden_size": 256, "batch_size": 128}),
        ExpertSpec("timesnet", "timesnet", cost=4.1, params={"seq_len": 49, "epochs": 5, "d_model": 256, "layers": 4, "heads": 8, "top_k": 3, "batch_size": 128}),
    ])

    def enabled_experts(self) -> List[ExpertSpec]:
        return [e for e in self.expert_specs if e.enabled]

    def cost_vector(self) -> List[float]:
        return [float(e.cost) for e in self.enabled_experts()]

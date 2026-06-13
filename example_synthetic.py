from __future__ import annotations

import numpy as np
import pandas as pd

from fame import FAMEConfig, FAMEModel
from fame.utils import mse, mae, wape


def build_synthetic(n_series: int = 6, n_days: int = 70, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for s in range(n_series):
        vem_id = f"V{s // 5:03d}"
        merc_id = f"M{s:04d}"
        city = "威海市" if s % 2 else "西安市"
        brand = f"B{s % 4}"
        scene = str(s % 3)
        base = rng.uniform(0.5, 8.0)
        seasonal = rng.uniform(0, 3) * np.sin(2 * np.pi * np.arange(n_days) / 7)
        trend = rng.uniform(-0.01, 0.03) * np.arange(n_days)
        sparse_mask = rng.random(n_days) < (0.55 if s % 5 == 0 else 0.1)
        noise = rng.normal(0, 1.0 + (s % 4) * 0.4, n_days)
        y = np.maximum(0, base + seasonal + trend + noise)
        y[sparse_mask] = 0
        y = np.round(y)
        for d, val in zip(dates, y):
            temp = 20 + 10 * np.sin(2 * np.pi * d.dayofyear / 365) + rng.normal(0, 2)
            rows.append({
                "vem_id": vem_id,
                "merc_id": merc_id,
                "date": d,
                "daily_quantity": val,
                "merc_brand_code": brand,
                "merc_type_code": f"T{s % 6}",
                "machine_type": f"machine_{s % 2}",
                "capacity": 30 + s % 5,
                "scene_code": scene,
                "city_name": city,
                "merc_sale_price": 3.0 + s % 8,
                "max_temperature": temp,
                "min_temperature": temp - 6,
                "weather": "晴" if rng.random() > 0.25 else "雨",
                "wind_level": rng.integers(1, 4),
                "event_name": "无" if rng.random() > 0.1 else "节假日",
                "is_offday": int(d.dayofweek >= 5),
                "coupon_amount": float(rng.choice([0, 0, 1], p=[0.8, 0.1, 0.1])),
            })
    return pd.DataFrame(rows)


def main():
    df = build_synthetic()
    df.to_csv('sales.csv')   # 注意括号是英文半角
    # Disable heavy experts for a quick demo.
    cfg = FAMEConfig(horizon=7, top_r=2, router_epochs=5, round_output=True, min_history=10)
    for spec in cfg.expert_specs:
        # Keep the demo lightweight. Enable more experts in real experiments.
        if spec.kind in {"sarima", "ets", "dlinear", "xgboost", "random_forest", "lightgbm"}:
            spec.enabled = False
    model = FAMEModel(cfg).fit(df)
    train_cut, test_cut = model.train_cutoff_, model.test_cutoff_
    history = df[df["date"] < test_cut].copy()
    test = df[df["date"] >= test_cut].groupby(["vem_id", "merc_id"], as_index=False).head(14)
    pred, explain = model.predict(history, future_df=test, return_explanations=True)
    merged = test.merge(pred, on=["vem_id", "merc_id", "date"], how="left")
    print(merged.head())
    print("MSE:", mse(merged["daily_quantity"], merged["predicted_sales"]))
    print("MAE:", mae(merged["daily_quantity"], merged["predicted_sales"]))
    print("WAPE:", wape(merged["daily_quantity"], merged["predicted_sales"]))
    print("Routing explanation:")
    print(explain.head(10))


if __name__ == "__main__":
    main()

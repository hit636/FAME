#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate demo input CSVs for FAME predict_fame.py:

python generate_fame_csv.py

Outputs:
- latest_history.csv: historical product-terminal daily sales.
- future_weather.csv: future context/weather data by date + city_name.

Usage with FAME:
python predict_fame.py \
  --model ./fame_model \
  --history latest_history.csv \
  --future future_weather.csv \
  --out prediction.csv

Note:
- This script creates demo data. In production, replace the data source with MySQL/API exports.
- latest_history.csv must contain the same id/date/target schema used during training.
- future_weather.csv is context-only; predict_fame.py will merge it by date + city_name.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def build_latest_history(
    n_vem: int = 4,
    n_merc: int = 6,
    n_days: int = 84,
    end_date: str = "2025-04-14",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build latest_history.csv.

    Required columns for FAME:
    - vem_id, merc_id, date, daily_quantity

    Optional but recommended columns:
    - metadata: merc_brand_code, merc_type_code, machine_type, capacity,
                scene_code, city_name, merc_sale_price
    - context: max_temperature, min_temperature, weather, wind_level,
               event_name, is_offday, coupon_amount, discount_quantity
    - availability: is_available, stockout_flag, outage_flag, suspension_flag
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp(end_date), periods=n_days, freq="D")

    rows = []
    cities = ["西安市", "威海市"]
    weather_pool = ["晴", "多云", "小雨", "阴"]
    machine_types = ["standard", "cooler"]

    for v in range(n_vem):
        vem_id = f"VEM{v + 1:04d}"
        city_name = cities[v % len(cities)]
        machine_type = machine_types[v % len(machine_types)]
        scene_code = str(v % 4)

        for m in range(n_merc):
            merc_id = f"MERC{m + 1:05d}"
            merc_brand_code = f"BRAND{m % 4}"
            merc_type_code = f"TYPE{m % 5}"
            capacity = int(24 + (m % 5) * 6)
            merc_sale_price = float(3.0 + (m % 8) * 0.8)

            base = rng.uniform(0.8, 8.0)
            weekly_amp = rng.uniform(0.0, 3.0)
            trend = rng.uniform(-0.01, 0.04)
            sparse_prob = 0.45 if m % 6 == 0 else (0.25 if m % 4 == 0 else 0.08)

            for t, d in enumerate(dates):
                # Context.
                seasonal_temp = 18 + 10 * np.sin(2 * np.pi * d.dayofyear / 365)
                max_temperature = seasonal_temp + rng.normal(0, 2.2)
                min_temperature = max_temperature - rng.uniform(5, 9)
                is_offday = int(d.dayofweek >= 5)
                event_name = "节假日" if rng.random() < 0.06 else "无"
                coupon_amount = float(rng.choice([0, 0, 0, 1, 2], p=[0.55, 0.20, 0.10, 0.10, 0.05]))
                discount_quantity = int(coupon_amount > 0)

                # Demand generation.
                weekly = weekly_amp * np.sin(2 * np.pi * t / 7)
                weekend_lift = 1.0 + 0.15 * is_offday
                temp_lift = 1.0 + max(0.0, max_temperature - 25.0) * 0.025
                promo_lift = 1.0 + 0.20 * discount_quantity
                holiday_lift = 1.20 if event_name != "无" else 1.0
                noise = rng.normal(0, 1.2 + 0.2 * (m % 3))

                demand = (base + weekly + trend * t + noise) * weekend_lift * temp_lift * promo_lift * holiday_lift
                if rng.random() < sparse_prob:
                    demand = 0.0
                daily_quantity = float(max(0, round(demand)))

                # Availability flags. Keep most rows valid.
                stockout_flag = int(rng.random() < 0.015)
                outage_flag = int(rng.random() < 0.005)
                suspension_flag = int(rng.random() < 0.003)
                is_available = int(not (outage_flag or suspension_flag))

                rows.append({
                    "vem_id": vem_id,
                    "merc_id": merc_id,
                    "date": d.strftime("%Y-%m-%d"),
                    "daily_quantity": daily_quantity,
                    "merc_brand_code": merc_brand_code,
                    "merc_type_code": merc_type_code,
                    "machine_type": machine_type,
                    "capacity": capacity,
                    "scene_code": scene_code,
                    "city_name": city_name,
                    "merc_sale_price": merc_sale_price,
                    "max_temperature": round(float(max_temperature), 2),
                    "min_temperature": round(float(min_temperature), 2),
                    "weather": str(rng.choice(weather_pool, p=[0.45, 0.30, 0.15, 0.10])),
                    "wind_level": int(rng.integers(1, 5)),
                    "event_name": event_name,
                    "is_offday": is_offday,
                    "coupon_amount": coupon_amount,
                    "discount_quantity": discount_quantity,
                    "is_available": is_available,
                    "stockout_flag": stockout_flag,
                    "outage_flag": outage_flag,
                    "suspension_flag": suspension_flag,
                })

    df = pd.DataFrame(rows)
    df = df.sort_values(["vem_id", "merc_id", "date"]).reset_index(drop=True)
    return df


def build_future_weather(
    history_df: pd.DataFrame,
    horizon: int = 14,
    seed: int = 2026,
) -> pd.DataFrame:
    """
    Build future_weather.csv as future context table.

    This version is context-only and should be passed without --future-is-full-frame.
    It will be merged by predict_fame.py using ['date', 'city_name'].
    """
    rng = np.random.default_rng(seed)

    history_df = history_df.copy()
    history_df["date"] = pd.to_datetime(history_df["date"])
    start_date = history_df["date"].max() + pd.Timedelta(days=1)
    future_dates = pd.date_range(start_date, periods=horizon, freq="D")

    cities = sorted(history_df["city_name"].dropna().astype(str).unique().tolist())
    weather_pool = ["晴", "多云", "小雨", "阴"]

    rows = []
    for city_name in cities:
        city_offset = 0.0 if city_name == "西安市" else -2.0
        for d in future_dates:
            seasonal_temp = 18 + 10 * np.sin(2 * np.pi * d.dayofyear / 365) + city_offset
            max_temperature = seasonal_temp + rng.normal(0, 1.8)
            min_temperature = max_temperature - rng.uniform(5, 9)
            is_offday = int(d.dayofweek >= 5)
            event_name = "节假日" if rng.random() < 0.08 else "无"
            coupon_amount = float(rng.choice([0, 0, 1, 2], p=[0.70, 0.15, 0.10, 0.05]))
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "city_name": city_name,
                "max_temperature": round(float(max_temperature), 2),
                "min_temperature": round(float(min_temperature), 2),
                "weather": str(rng.choice(weather_pool, p=[0.45, 0.30, 0.15, 0.10])),
                "wind_level": int(rng.integers(1, 5)),
                "event_name": event_name,
                "is_offday": is_offday,
                "coupon_amount": coupon_amount,
                "discount_quantity": int(coupon_amount > 0),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["city_name", "date"]).reset_index(drop=True)
    return df


def main() -> None:
    out_dir = Path("./data")
    history_df = build_latest_history()
    future_df = build_future_weather(history_df, horizon=14)

    history_path = out_dir / "latest_history.csv"
    future_path = out_dir / "future_weather.csv"

    # UTF-8 is recommended on Linux servers. Use utf-8-sig only if you need Excel-friendly Chinese display.
    history_df.to_csv(history_path, index=False, encoding="utf-8")
    future_df.to_csv(future_path, index=False, encoding="utf-8")

    print(f"Saved: {history_path.resolve()} rows={len(history_df)}")
    print(f"Saved: {future_path.resolve()} rows={len(future_df)}")


if __name__ == "__main__":
    main()

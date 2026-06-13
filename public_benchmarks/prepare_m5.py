#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert M5 Accuracy files to FAME schema.

Required files from the Kaggle/M5 release:
  sales_train_evaluation.csv, calendar.csv, sell_prices.csv
Example:
  python public_benchmarks/prepare_m5.py --m5-dir /data/m5 --out data/m5_fame.csv --max-series 5000
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m5-dir', required=True)
    ap.add_argument('--out', default='./data/m5_fame.csv')
    ap.add_argument('--max-series', type=int, default=0, help='0 means all series')
    ap.add_argument('--start-day', type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args(); root = Path(args.m5_dir)
    sales = pd.read_csv(root / 'sales_train_evaluation.csv')
    cal = pd.read_csv(root / 'calendar.csv')
    prices = pd.read_csv(root / 'sell_prices.csv')
    id_cols = ['id','item_id','dept_id','cat_id','store_id','state_id']
    day_cols = [c for c in sales.columns if c.startswith('d_')]
    day_cols = [c for c in day_cols if int(c.split('_')[1]) >= args.start_day]
    if args.max_series and args.max_series > 0:
        sales = sales.head(args.max_series).copy()
    long = sales[id_cols + day_cols].melt(id_vars=id_cols, value_vars=day_cols, var_name='d', value_name='daily_quantity')
    long = long.merge(cal[['d','date','wm_yr_wk','event_name_1','snap_CA','snap_TX','snap_WI']], on='d', how='left')
    long = long.merge(prices, on=['store_id','item_id','wm_yr_wk'], how='left')
    out = pd.DataFrame({
        'vem_id': long['store_id'].astype(str),
        'merc_id': long['item_id'].astype(str),
        'date': long['date'],
        'daily_quantity': pd.to_numeric(long['daily_quantity'], errors='coerce').fillna(0).clip(lower=0),
        'merc_brand_code': long['cat_id'].astype(str),
        'merc_type_code': long['dept_id'].astype(str),
        'machine_type': long['state_id'].astype(str),
        'capacity': 9999,
        'scene_code': long['store_id'].astype(str),
        'city_name': long['state_id'].astype(str),
        'merc_sale_price': pd.to_numeric(long.get('sell_price', 0), errors='coerce').fillna(0),
        'event_name': long.get('event_name_1', 'none').fillna('none').astype(str),
        'is_offday': 0,
        'coupon_amount': 0.0,
        'discount_quantity': 0,
        'is_available': 1,
        'stockout_flag': 0,
        'outage_flag': 0,
        'suspension_flag': 0,
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding='utf-8')
    print(f'Saved {args.out} rows={len(out)} series={out[["vem_id","merc_id"]].drop_duplicates().shape[0]}')


if __name__ == '__main__':
    main()

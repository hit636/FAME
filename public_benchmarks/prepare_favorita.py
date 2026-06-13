#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert Corporacion Favorita files to FAME schema.

Required files from Kaggle:
  train.csv, stores.csv, items.csv
Optional:
  holidays_events.csv, oil.csv
Example:
  python public_benchmarks/prepare_favorita.py --favorita-dir /data/favorita --out data/favorita_fame.csv --max-series 5000
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--favorita-dir', required=True)
    ap.add_argument('--out', default='./data/favorita_fame.csv')
    ap.add_argument('--max-series', type=int, default=0)
    ap.add_argument('--min-date', default=None)
    return ap.parse_args()


def main():
    args = parse_args(); root = Path(args.favorita_dir)
    train = pd.read_csv(root / 'train.csv')
    stores = pd.read_csv(root / 'stores.csv')
    items = pd.read_csv(root / 'items.csv')
    train['date'] = pd.to_datetime(train['date'])
    if args.min_date:
        train = train[train['date'] >= pd.Timestamp(args.min_date)].copy()
    train['unit_sales'] = pd.to_numeric(train['unit_sales'], errors='coerce').fillna(0).clip(lower=0)
    if args.max_series and args.max_series > 0:
        keys = train[['store_nbr','item_nbr']].drop_duplicates().head(args.max_series)
        train = train.merge(keys, on=['store_nbr','item_nbr'], how='inner')
    df = train.merge(stores, on='store_nbr', how='left').merge(items, on='item_nbr', how='left')
    if (root / 'oil.csv').exists():
        oil = pd.read_csv(root / 'oil.csv')
        oil['date'] = pd.to_datetime(oil['date'])
        df = df.merge(oil, on='date', how='left')
    if (root / 'holidays_events.csv').exists():
        hol = pd.read_csv(root / 'holidays_events.csv')
        hol['date'] = pd.to_datetime(hol['date'])
        hol = hol.groupby('date', as_index=False).agg(event_name=('description', 'first'))
        df = df.merge(hol, on='date', how='left')
    out = pd.DataFrame({
        'vem_id': df['store_nbr'].astype(str),
        'merc_id': df['item_nbr'].astype(str),
        'date': df['date'].dt.strftime('%Y-%m-%d'),
        'daily_quantity': df['unit_sales'],
        'merc_brand_code': df.get('family', 'missing').astype(str),
        'merc_type_code': df.get('class', 'missing').astype(str),
        'machine_type': df.get('type', 'missing').astype(str),
        'capacity': 9999,
        'scene_code': df.get('cluster', 0).astype(str),
        'city_name': df.get('city', 'missing').astype(str),
        'merc_sale_price': 0.0,
        'max_temperature': 0.0,
        'min_temperature': 0.0,
        'weather': 'missing',
        'wind_level': 0,
        'event_name': df.get('event_name', 'none').fillna('none').astype(str),
        'is_offday': 0,
        'coupon_amount': df.get('onpromotion', 0).astype(float),
        'discount_quantity': df.get('onpromotion', 0).astype(float),
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

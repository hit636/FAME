#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic public-retail benchmark runner for data already converted to FAME schema."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='CSV in FAME schema: vem_id, merc_id, date, daily_quantity')
    ap.add_argument('--name', default='public_retail')
    ap.add_argument('--out-root', default='./output/public')
    ap.add_argument('--model-root', default='./fame_model_public')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--fast-artifact', action='store_true')
    ap.add_argument('--strict-experts', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    data = Path(args.data)
    if not data.exists():
        raise FileNotFoundError(data)
    out_dir = Path(args.out_root) / args.name
    model_dir = Path(args.model_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    exp_cmd = [sys.executable, 'run_icdm_experiment.py', '--data', str(data), '--out-dir', str(out_dir / 'experiment'), '--model-out', str(model_dir), '--epochs', str(args.epochs), '--device', args.device]
    if args.fast_artifact:
        exp_cmd.append('--fast-artifact')
    if args.strict_experts:
        exp_cmd.append('--strict-experts')
    print('[RUN]', ' '.join(exp_cmd), flush=True)
    subprocess.check_call(exp_cmd)
    # Create history-for-table-v = train + validation split from experiment.
    train = pd.read_csv(out_dir / 'experiment' / 'split_train.csv')
    val = pd.read_csv(out_dir / 'experiment' / 'split_validation.csv')
    hist_path = out_dir / 'history_for_table_v.csv'
    pd.concat([train, val], ignore_index=True).to_csv(hist_path, index=False)
    table_cmd = [sys.executable, 'run_table_v.py', '--model', str(model_dir), '--history', str(hist_path), '--test', str(out_dir / 'experiment' / 'split_test.csv'), '--out-dir', str(out_dir / 'table_v'), '--device', args.device]
    print('[RUN]', ' '.join(table_cmd), flush=True)
    subprocess.check_call(table_cmd)
    print(f'Public benchmark artifacts saved under {out_dir}')


if __name__ == '__main__':
    main()

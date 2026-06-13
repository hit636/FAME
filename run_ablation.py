#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, subprocess, sys, os
from pathlib import Path
import pandas as pd
from fame.utils import json_dump

VARIANTS = [
    "default_fame_top2", "wo_sparsity_features", "wo_seasonality_features", "wo_spectral_features",
    "wo_metadata_context", "wo_balance_loss", "fame_acc_top2", "fame_costaware",
]

def parse_args():
    ap=argparse.ArgumentParser(description="Run FAME ablation study, one subprocess per variant for GPU/torch stability.")
    ap.add_argument("--data", default="./data/latest_history.csv")
    ap.add_argument("--out-dir", default="./output/ablation")
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--top-r", type=int, default=2)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast-artifact", action="store_true")
    ap.add_argument("--strict-experts", action="store_true")
    return ap.parse_args()

def main():
    args=parse_args(); out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    frames=[]
    for v in VARIANTS:
        cmd=[sys.executable,"run_ablation_variant.py","--data",args.data,"--out-dir",str(out_dir),"--horizon",str(args.horizon),"--top-r",str(args.top_r),"--delta",str(args.delta),"--tau",str(args.tau),"--epochs",str(args.epochs),"--device",args.device,"--seed",str(args.seed),"--variant",v]
        if args.fast_artifact: cmd.append("--fast-artifact")
        if args.strict_experts: cmd.append("--strict-experts")
        print("[RUN]", " ".join(cmd), flush=True)
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        subprocess.check_call(cmd, env=env)
        frames.append(pd.read_csv(out_dir / f"ablation_{v}.csv"))
    result=pd.concat(frames, ignore_index=True)
    result.to_csv(out_dir/"ablation_metrics.csv", index=False, encoding="utf-8")
    json_dump({"variants": result.to_dict("records")}, out_dir/"ablation_metrics.json")
    print(result.to_string(index=False))

if __name__=="__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline training entrypoint for the FAME paper artifact.

This script matches the offline part of FAME:
1. load and audit product-terminal daily history;
2. fit heterogeneous experts on the training window;
3. mine validation loss matrix and oracle suitability targets;
4. train the forecastability-aware sparse router;
5. save model, oracle artifacts, data audit and training manifest.

Default paths are current-directory relative so that deployment on a server can
use the project root directly:
    python train_fame.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from fame import FAMEConfig, FAMEModel
from fame.data_checks import audit_history, write_audit_report
from fame.logging_utils import setup_logger
from fame.utils import json_dump, resolve_torch_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FAME with industrial data checks.")
    parser.add_argument("--data", default="./data/latest_history.csv", help="Training history CSV. Default: ./data/latest_history.csv")
    parser.add_argument("--out", default="./fame_model", help="Output model directory. Default: ./fame_model")
    parser.add_argument("--log-dir", default="./logs", help="Log directory. Default: ./logs")
    parser.add_argument("--horizon", type=int, default=14, help="Forecast horizon used by replenishment planning.")
    parser.add_argument("--lookback", type=int, default=56, help="Look-back length for expert features.")
    parser.add_argument("--min-history", type=int, default=15, help="Minimum active history length per series.")
    parser.add_argument("--top-r", type=int, default=2, help="Maximum active experts in sparse inference.")
    parser.add_argument("--delta", type=float, default=0.05, help="Top-r normalized probability pruning threshold.")
    parser.add_argument("--tau", type=float, default=0.30, help="Soft oracle temperature.")
    parser.add_argument("--eta-oracle-cost", type=float, default=0.0, help="Cost coefficient used when mining cost-aware oracle targets.")
    parser.add_argument("--gamma-cost", type=float, default=0.01, help="Router cost regularization coefficient.")
    parser.add_argument("--epochs", type=int, default=200, help="Router training epochs.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, cuda:1, ...")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round-output", action="store_true", help="Round final predictions to integer counts.")
    parser.add_argument("--no-complete-grid", action="store_true", help="Disable missing-date grid completion.")
    parser.add_argument("--max-dlinear-windows", type=int, default=200000, help="Safety cap for DLinear windows.")
    return parser.parse_args()


def _safe_asdict(cfg: FAMEConfig) -> Dict[str, Any]:
    data = asdict(cfg)
    # dataclass ExpertSpec values are already converted by asdict; keep JSON simple.
    return data


def main() -> int:
    args = parse_args()
    logger = setup_logger("FAME_TRAIN", log_dir=args.log_dir, log_file="train_fame.log")
    started = time.time()
    set_seed(args.seed)

    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path.resolve()}")

        logger.info("Loading training data: %s", data_path)
        df = pd.read_csv(data_path, encoding="utf-8")
        logger.info("Loaded rows=%d columns=%d", len(df), len(df.columns))

        cfg = FAMEConfig(
            horizon=args.horizon,
            lookback=args.lookback,
            min_history=args.min_history,
            top_r=args.top_r,
            delta=args.delta,
            tau=args.tau,
            eta_oracle_cost=args.eta_oracle_cost,
            gamma_cost=args.gamma_cost,
            router_epochs=args.epochs,
            round_output=args.round_output,
            device=args.device,
            seed=args.seed,
            max_dlinear_windows=args.max_dlinear_windows,
        )

        audit = audit_history(df, cfg, name="training_history")
        write_audit_report([audit], out_dir / "training_data_audit.json")
        for w in audit.warnings:
            logger.warning(w)
        if not audit.ok:
            for e in audit.errors:
                logger.error(e)
            raise ValueError("Training data audit failed; see fame_model/training_data_audit.json")

        resolved_device = str(resolve_torch_device(cfg.device))
        logger.info("Resolved device: %s", resolved_device)
        logger.info("Fitting FAME: horizon=%d top_r=%d delta=%.3f tau=%.3f", cfg.horizon, cfg.top_r, cfg.delta, cfg.tau)

        model = FAMEModel(cfg)
        model.fit(df, complete_grid=not args.no_complete_grid)
        model.save(str(out_dir))
        logger.info("Model saved to: %s", out_dir.resolve())

        # Persist oracle artifacts for paper-style analysis.
        if model.oracle_ is not None:
            model.oracle_.loss_matrix.to_csv(out_dir / "oracle_loss_matrix.csv", index=False, encoding="utf-8")
            model.oracle_.soft_targets.to_csv(out_dir / "oracle_soft_targets.csv", index=False, encoding="utf-8")
            model.oracle_.hard_targets.to_csv(out_dir / "oracle_hard_targets.csv", index=False, encoding="utf-8")
            logger.info("Oracle artifacts saved: loss matrix, soft targets, hard targets")

        manifest = {
            "stage": "offline_training",
            "paper_component": "forecastability fingerprint + expert pool + oracle mining + sparse router",
            "data_path": str(data_path),
            "model_dir": str(out_dir),
            "resolved_device": resolved_device,
            "runtime_seconds": round(time.time() - started, 3),
            "config": _safe_asdict(cfg),
            "data_audit": audit.to_dict(),
            "router_training": asdict(model.router.history_) if model.router is not None and model.router.history_ is not None else None,
            "router_calibration": model.calibration_,
            "expert_names": model.expert_pool.expert_names if model.expert_pool is not None else [],
        }
        json_dump(manifest, out_dir / "training_manifest.json")
        logger.info("Training manifest saved to: %s", out_dir / "training_manifest.json")
        logger.info("Training finished in %.2f seconds", time.time() - started)
        return 0
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

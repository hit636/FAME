#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logging helpers for FAME industrial experiments and deployment.

The conference artifact is intended to be both reproducible and deployable.
This module centralizes logging so that training, batch inference and validation
produce timestamped logs in ``./logs`` without changing the core algorithm.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "FAME",
    log_dir: str | Path = "./logs",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a console+file logger.

    Parameters
    ----------
    name:
        Logger name.
    log_dir:
        Directory used for persistent logs.
    log_file:
        Optional file name. If omitted, ``<name>.log`` is used.
    level:
        Python logging level.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers when a script is re-run in notebooks.
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_name = log_file or f"{name.lower()}.log"
    file_handler = logging.FileHandler(Path(log_dir) / file_name, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger

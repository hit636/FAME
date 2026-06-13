# -*- coding: utf-8 -*-
"""FAME: Forecastability-Aware Mixture of Experts.

A modular, production-oriented reference implementation for heterogeneous
retail / industrial time-series forecasting.
"""

from .config import FAMEConfig, ExpertSpec
from .pipeline import FAMEModel

__all__ = ["FAMEConfig", "ExpertSpec", "FAMEModel"]

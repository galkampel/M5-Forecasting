"""
Preprocessing package for M5 Forecasting.

This package provides comprehensive preprocessing capabilities for time series
forecasting, including feature engineering, model configuration, evaluation,
and optimization.
"""

__version__ = "0.1.0"
__author__ = "M5 Forecasting Team"

from .config import PreprocessingConfig
from .data_loader import DataLoader
from .main import main

__all__ = [
    "main",
    "PreprocessingConfig",
    "DataLoader",
]

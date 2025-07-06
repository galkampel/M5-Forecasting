"""
Dataset preprocessing package for Walmart sales forecasting.

This package provides tools for preprocessing sales data, including
calendar processing, sales data filtering, and feature engineering.
"""

__version__ = "0.1.0"
__author__ = "Walmart Forecasting Team"

from .config import Config
from .main import DatasetProcessor

__all__ = ["DatasetProcessor", "Config"]

"""
Fourier feature engineering for time series preprocessing.

This module provides Fourier-based seasonality feature engineering
using the existing FourierModes transformer.
"""

import logging
from typing import List

import pandas as pd

from ..transformers.fourier import MultiSeasonalFourier


class FourierFeatureEngineer:
    """
    Fourier feature engineering class.

    This class orchestrates Fourier feature generation for time series data.
    """

    def __init__(
        self,
        periods: List[float],
        orders: List[int],
        time_cols: List[str],
        drop_time_cols: bool = True,
    ):
        """
        Initialize FourierFeatureEngineer.

        Args:
            periods: List of seasonal periods
            orders: List of harmonic orders for each period
            time_cols: List of time columns to use
            drop_time_cols: Whether to drop original time columns
        """
        self.periods = periods
        self.orders = orders
        self.time_cols = time_cols
        self.drop_time_cols = drop_time_cols
        self.logger = logging.getLogger(__name__)
        self.transformers = {}

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the data with Fourier features.

        Args:
            X: Input features DataFrame

        Returns:
            DataFrame with Fourier features added
        """
        X_transformed = X.copy()

        for time_col in self.time_cols:
            if time_col not in X.columns:
                self.logger.warning(f"Time column {time_col} not found in data")
                continue

            # Create transformer for this time column
            transformer = MultiSeasonalFourier(
                periods=self.periods,
                orders=self.orders,
                time_col=time_col,
                drop_time_col=self.drop_time_cols,
            )

            # Fit and transform
            X_transformed = transformer.fit_transform(X_transformed)
            self.transformers[time_col] = transformer

            self.logger.info(f"Added Fourier features for {time_col}")

        return X_transformed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data with fitted Fourier features.

        Args:
            X: Input features DataFrame

        Returns:
            DataFrame with Fourier features added
        """
        X_transformed = X.copy()

        for time_col, transformer in self.transformers.items():
            if time_col in X.columns:
                X_transformed = transformer.transform(X_transformed)

        return X_transformed

    def get_feature_names(self) -> List[str]:
        """
        Get names of generated Fourier features.

        Returns:
            List of feature names
        """
        feature_names = []
        for transformer in self.transformers.values():
            feature_names.extend(transformer.get_feature_names_out())
        return feature_names

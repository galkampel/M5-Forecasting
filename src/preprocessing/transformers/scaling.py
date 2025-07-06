"""
Scaling transformers for preprocessing pipeline.

This module provides scaling transformers including threshold-based
standard scaling.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import BaseTransformer


class StandardScalerwithThreshold(BaseTransformer):
    """
    StandardScaler that scales only values above a threshold.

    This transformer applies standard scaling only to values that are
    greater than the specified threshold. Values <= threshold remain unchanged.
    """

    def __init__(
        self, threshold: float = 0.0, with_mean: bool = True, with_std: bool = True
    ):
        """
        Initialize StandardScalerwithThreshold.

        Args:
            threshold: Only values > threshold will be scaled
            with_mean: Whether to center the data
            with_std: Whether to scale to unit variance
        """
        super().__init__()
        self.threshold = threshold
        self.with_mean = with_mean
        self.with_std = with_std
        self.scalers = {}
        self.feature_names = None

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        Fit the transformer.

        Args:
            X: Input features DataFrame
            y: Target values (optional)
        """
        self.feature_names = list(X.columns)

        # Fit scaler for each column
        for col in X.columns:
            # Get values above threshold for this column
            col_values = X[col]
            above_threshold = col_values > self.threshold

            if above_threshold.any():
                # Fit scaler on values above threshold
                scaler = StandardScaler(
                    with_mean=self.with_mean, with_std=self.with_std
                )
                scaler.fit(col_values[above_threshold].values.reshape(-1, 1))
                self.scalers[col] = scaler

    def _transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Transform the input data.

        Args:
            X: Input features DataFrame
            y: Target values (optional)

        Returns:
            DataFrame with values above threshold scaled
        """
        if self.feature_names is None:
            raise ValueError("Transformer not fitted. Call fit first.")

        X_transformed = X.copy()

        # Transform each column
        for col in X.columns:
            if col in self.scalers:
                col_values = X[col]
                above_threshold = col_values > self.threshold

                if above_threshold.any():
                    # Scale only values above threshold
                    values_to_scale = col_values[above_threshold].values.reshape(-1, 1)
                    scaled_values = self.scalers[col].transform(values_to_scale)
                    X_transformed.loc[above_threshold, col] = scaled_values.flatten()

        return X_transformed

    def get_feature_names_out(self) -> list:
        """Get output feature names."""
        if self.feature_names is None:
            return []
        return self.feature_names

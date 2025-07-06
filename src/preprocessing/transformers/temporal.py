"""
Temporal transformers for preprocessing pipeline.

This module provides temporal transformers for time series data.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .base import BaseTransformer


class ZeroPredTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that sets negative or near-zero values to zero in prediction columns.

    This transformer is used to clean up predictions by zeroing out
    small or negative values that don't make sense in the context of sales data.
    """

    def __init__(
        self,
        negative_mask: bool = True,
        is_close_mask: bool = True,
        threshold: float = 0.1,
    ):
        """
        Initialize ZeroPredTransformer.

        Args:
            negative_mask: Whether to zero out negative values
            is_close_mask: Whether to zero out values close to zero
            threshold: Threshold for values close to zero
        """
        self.negative_mask = negative_mask
        self.is_close_mask = is_close_mask
        self.threshold = threshold

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "ZeroPredTransformer":
        """
        Fit the transformer.

        Args:
            X: Input DataFrame
            y: Target values (optional)
        """
        # No fitting required for zero prediction transformer
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by zeroing out values.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with zeroed values
        """
        X_transformed = X.copy()

        # Get prediction columns (exclude metadata columns)
        pred_cols = [
            col for col in X.columns if col not in ["unique_id", "ds", "cutoff", "y"]
        ]

        for col in pred_cols:
            if col in X_transformed.columns:
                # Zero out negative values
                if self.negative_mask:
                    X_transformed.loc[X_transformed[col] < 0, col] = 0.0

                # Zero out values close to zero
                if self.is_close_mask:
                    X_transformed.loc[X_transformed[col] <= self.threshold, col] = 0.0

        return X_transformed

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit and transform the data.

        Args:
            X: Input DataFrame
            y: Target values (optional)

        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)


class ZeroPredTransformerLegacy(BaseTransformer):
    """
    Legacy ZeroPredTransformer for backward compatibility.

    This transformer is used to clean up predictions or raw data by zeroing out
    small or negative values that don't make sense in the context of sales data.
    """

    def __init__(self, target_col: str, threshold: float = 0.0):
        """
        Initialize ZeroPredTransformer.

        Args:
            target_col: Name of the column to zero out
            threshold: Values <= threshold will be set to zero
        """
        super().__init__()
        self.target_col = target_col
        self.threshold = threshold
        self.required_columns = [target_col]

    def _fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "ZeroPredTransformerLegacy":
        """
        Fit the transformer.

        Args:
            X: Input features DataFrame
            y: Target values (optional)
        """
        # No fitting required for zero prediction transformer
        return self

    def _transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Transform the input data by zeroing out values <= threshold.

        Args:
            X: Input features DataFrame
            y: Target values (optional)

        Returns:
            DataFrame with zeroed values
        """
        X_transformed = X.copy()
        mask = X_transformed[self.target_col] <= self.threshold
        X_transformed.loc[mask, self.target_col] = 0.0
        return X_transformed

    def get_feature_names_out(self) -> list:
        """Get output feature names."""
        return []

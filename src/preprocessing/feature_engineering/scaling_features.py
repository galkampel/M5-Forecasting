"""
Scaling feature engineering for time series preprocessing.

This module provides feature scaling and normalization utilities.
"""

import logging
from typing import List

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class ScalingFeatureEngineer:
    """
    Scaling feature engineering class.

    This class handles feature scaling and normalization with
    threshold-based filtering.
    """

    def __init__(
        self,
        method: str = "standard",
        threshold: float = 0.0,
        with_mean: bool = True,
        with_std: bool = True,
    ):
        """
        Initialize ScalingFeatureEngineer.

        Args:
            method: Scaling method ('standard', 'robust', 'minmax')
            threshold: Threshold for variance filtering
            with_mean: Whether to center the data (for StandardScaler)
            with_std: Whether to scale to unit variance (for StandardScaler)
        """
        self.method = method
        self.threshold = threshold
        self.with_mean = with_mean
        self.with_std = with_std
        self.logger = logging.getLogger(__name__)
        self.scaler = None
        self.feature_names = None

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the data with scaling features.

        Args:
            X: Input features DataFrame

        Returns:
            DataFrame with scaled features
        """
        # Filter features based on variance threshold
        X_filtered = self._filter_low_variance_features(X)

        # Create and fit scaler
        self.scaler = self._create_scaler()
        X_scaled = self.scaler.fit_transform(X_filtered)

        # Convert back to DataFrame
        X_transformed = pd.DataFrame(
            X_scaled, columns=X_filtered.columns, index=X_filtered.index
        )

        self.feature_names = list(X_transformed.columns)

        self.logger.info(
            f"Scaled {len(self.feature_names)} features using {self.method}"
        )

        return X_transformed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data with fitted scaling features.

        Args:
            X: Input features DataFrame

        Returns:
            DataFrame with scaled features
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform first.")

        # Filter features to match training data
        X_filtered = X[self.feature_names]

        # Transform using numpy array
        X_scaled = self.scaler.transform(X_filtered.to_numpy())

        # Convert back to DataFrame with proper type casting
        X_transformed = pd.DataFrame(
            data=X_scaled, columns=self.feature_names, index=X_filtered.index
        )

        return X_transformed

    def _filter_low_variance_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Filter features with low variance.

        Args:
            X: Input features DataFrame

        Returns:
            DataFrame with low variance features removed
        """
        if self.threshold <= 0:
            return X

        # Calculate variance for numeric columns
        numeric_cols = X.select_dtypes(include=["number"]).columns
        variances = X[numeric_cols].var()

        # Filter features above threshold
        high_var_cols = variances[variances > self.threshold].index
        X_filtered = X[high_var_cols]

        self.logger.info(
            f"Filtered {len(numeric_cols) - len(high_var_cols)} "
            f"low variance features"
        )

        return X_filtered

    def _create_scaler(self):
        """
        Create the appropriate scaler based on method.

        Returns:
            Scaler instance
        """
        if self.method == "standard":
            return StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        elif self.method == "robust":
            return RobustScaler()
        elif self.method == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

    def get_feature_names(self) -> List[str]:
        """
        Get names of scaled features.

        Returns:
            List of feature names
        """
        if self.feature_names is None:
            return []
        return self.feature_names

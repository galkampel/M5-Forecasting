"""
Encoding feature engineering for time series preprocessing.

This module provides categorical encoding feature engineering
using the existing BaselineEncoder transformer.
"""

import logging
from typing import Dict, List, Union

import pandas as pd

from ..transformers.encoding import BaselineEncoder


class EncodingFeatureEngineer:
    """
    Encoding feature engineering class.

    This class handles categorical encoding with baseline category control.
    """

    def __init__(
        self, baseline_categories: Dict[str, Union[str, int]], drop_first: bool = True
    ):
        """
        Initialize EncodingFeatureEngineer.

        Args:
            baseline_categories: Dictionary mapping column names to baseline
                categories
            drop_first: Whether to drop the first category
        """
        self.baseline_categories = baseline_categories
        self.drop_first = drop_first
        self.logger = logging.getLogger(__name__)
        self.encoder = None

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the data with encoding features.

        Args:
            X: Input features DataFrame

        Returns:
            DataFrame with encoded features
        """
        # Create encoder
        self.encoder = BaselineEncoder(
            baseline_categories=self.baseline_categories, drop_first=self.drop_first
        )

        # Fit and transform
        X_transformed = self.encoder.fit_transform(X)

        self.logger.info(
            f"Encoded {len(self.baseline_categories)} categorical features"
        )

        return X_transformed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data with fitted encoding features.

        Args:
            X: Input features DataFrame

        Returns:
            DataFrame with encoded features
        """
        if self.encoder is None:
            raise ValueError("Encoder not fitted. Call fit_transform first.")

        return self.encoder.transform(X)

    def get_feature_names(self) -> List[str]:
        """
        Get names of generated encoded features.

        Returns:
            List of feature names
        """
        if self.encoder is None:
            return []

        return self.encoder.get_feature_names_out()

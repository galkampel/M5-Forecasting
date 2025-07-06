"""
Encoding transformers for categorical features.

This module provides encoding transformers including OneHotEncoder
with baseline category control.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from .base import BaseTransformer


class OHEwithBaseline(BaseTransformer):
    """
    OneHotEncoder with baseline category control.

    This transformer extends sklearn's OneHotEncoder to allow
    specification of a baseline category that will be placed first
    in the encoded features.
    """

    def __init__(self, baseline_category: Union[str, int], drop_first: bool = False):
        """
        Initialize OHEwithBaseline transformer.

        Args:
            baseline_category: Category to use as baseline
            drop_first: Whether to drop the first category
        """
        super().__init__()
        self.baseline_category = baseline_category
        self.drop_first = drop_first
        self.ohe = OneHotEncoder(
            drop="first" if self.drop_first else None, sparse_output=False
        )

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Optional[pd.Series] = None
    ) -> "OHEwithBaseline":
        """
        Fit the transformer.

        Args:
            X: Input features
            y: Target values (optional)

        Returns:
            Self for method chaining
        """
        self.ohe.fit(X)
        categories_ = self.ohe.categories_[0].tolist()

        # Move baseline category to first position
        if self.baseline_category in categories_:
            categories_.remove(self.baseline_category)
            categories_.insert(0, self.baseline_category)
            self.ohe.categories_[0] = np.array(categories_)

        # Handle the case where there's only one category and drop_first=True
        # In this case, we should not drop the category to avoid empty output
        if len(categories_) == 1 and self.drop_first:
            self.logger.warning(
                f"Only one category '{categories_[0]}' found. "
                f"Setting drop_first=False to avoid empty output."
            )
            # Recreate the OneHotEncoder without dropping first
            self.ohe = OneHotEncoder(drop=None, sparse_output=False)
            self.ohe.fit(X)
            # Reapply baseline category ordering
            categories_ = self.ohe.categories_[0].tolist()
            if self.baseline_category in categories_:
                categories_.remove(self.baseline_category)
                categories_.insert(0, self.baseline_category)
                self.ohe.categories_[0] = np.array(categories_)

        return self

    def transform(
        self, X: Union[np.ndarray, pd.DataFrame], y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Transform the input data.

        Args:
            X: Input features
            y: Target values (optional)

        Returns:
            Encoded DataFrame
        """
        X_encoded = self.ohe.transform(X)
        # Preserve the original index if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                X_encoded, 
                columns=self.get_feature_names_out(), 
                index=X.index
            )
        else:
            return pd.DataFrame(
                X_encoded, 
                columns=self.get_feature_names_out()
            )

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the encoded data.

        Args:
            X: Encoded DataFrame

        Returns:
            Original categorical data
        """
        X_inverse = self.ohe.inverse_transform(X)
        return pd.DataFrame(X_inverse)

    def get_feature_names_out(self, *args, **params) -> list:
        """
        Get output feature names.

        Returns:
            List of feature names
        """
        return self.ohe.get_feature_names_out()


class BaselineEncoder(BaseTransformer):
    """
    Generic baseline encoder for categorical features.

    This transformer provides a more flexible baseline encoding
    approach with additional validation and error handling.
    """

    def __init__(self, baseline_categories: dict, drop_first: bool = True):
        """
        Initialize BaselineEncoder.

        Args:
            baseline_categories: Dictionary mapping column names to baseline categories
            drop_first: Whether to drop the first category
        """
        super().__init__()
        self.baseline_categories = baseline_categories
        self.drop_first = drop_first
        self.encoders = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaselineEncoder":
        """
        Fit the transformer.

        Args:
            X: Input features DataFrame
            y: Target values (optional)
        """
        for col, baseline in self.baseline_categories.items():
            if col in X.columns:
                encoder = OHEwithBaseline(
                    baseline_category=baseline, drop_first=self.drop_first
                )
                encoder.fit(X[[col]])
                self.encoders[col] = encoder
        return self

    def _transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Transform the input data.

        Args:
            X: Input features DataFrame
            y: Target values (optional)

        Returns:
            Encoded DataFrame
        """
        X_transformed = X.copy()

        for col, encoder in self.encoders.items():
            if col in X_transformed.columns:
                # Encode the column
                encoded = encoder.transform(X_transformed[[col]])

                # Drop original column and add encoded columns
                X_transformed = X_transformed.drop(columns=[col])
                X_transformed = pd.concat([X_transformed, encoded], axis=1)

        # Store feature names
        self.feature_names_out_ = list(X_transformed.columns)

        return X_transformed

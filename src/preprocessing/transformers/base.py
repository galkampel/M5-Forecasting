"""
Base transformer classes for preprocessing pipeline.

This module provides base classes and common functionality
for all transformers in the preprocessing pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..utils.logging import LoggerMixin


class BaseTransformer(BaseEstimator, TransformerMixin, LoggerMixin):
    """
    Base transformer class with common functionality.

    This class provides a foundation for all transformers in the
    preprocessing pipeline with logging, validation, and error handling.
    """

    def __init__(self):
        """Initialize base transformer."""
        super().__init__()
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseTransformer":
        """
        Fit the transformer.

        Args:
            X: Input features DataFrame
            y: Target values (optional)

        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting {self.__class__.__name__}")

        try:
            self._validate_input(X)
            self._fit(X, y)
            self._is_fitted = True
            self.logger.info(f"Successfully fitted {self.__class__.__name__}")
            return self

        except Exception as e:
            self.logger.error(f"Error fitting {self.__class__.__name__}: {str(e)}")
            raise

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transform the input data.

        Args:
            X: Input features DataFrame
            y: Target values (optional)

        Returns:
            Transformed DataFrame
        """
        self.logger.info(f"Transforming with {self.__class__.__name__}")

        try:
            check_is_fitted(self, "_is_fitted")
            self._validate_input(X)
            result = self._transform(X, y)
            self.logger.info(f"Successfully transformed with {self.__class__.__name__}")
            return result

        except Exception as e:
            self.logger.error(
                f"Error transforming with {self.__class__.__name__}: {str(e)}"
            )
            raise

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit the transformer and transform the data.

        Args:
            X: Input features DataFrame
            y: Target values (optional)

        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X, y)

    def _validate_input(self, X: pd.DataFrame) -> None:
        """
        Validate input data.

        Args:
            X: Input DataFrame to validate
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame")

        if X.empty:
            raise ValueError("Input DataFrame cannot be empty")

        # Check for required columns if specified
        if hasattr(self, "required_columns"):
            missing_cols = [
                col for col in self.required_columns if col not in X.columns
            ]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        Internal fit method to be implemented by subclasses.

        Args:
            X: Input features DataFrame
            y: Target values (optional)
        """
        pass

    @abstractmethod
    def _transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Internal transform method to be implemented by subclasses.

        Args:
            X: Input features DataFrame
            y: Target values (optional)

        Returns:
            Transformed DataFrame
        """
        pass

    def get_feature_names_out(self) -> list:
        """
        Get output feature names.

        Returns:
            List of output feature names
        """
        if hasattr(self, "feature_names_out_"):
            return self.feature_names_out_
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement get_feature_names_out"
            )

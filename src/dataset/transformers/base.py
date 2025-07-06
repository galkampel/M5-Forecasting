"""
Base transformer classes for dataset preprocessing.

This module provides base classes and common functionality for all transformers.
"""

from typing import Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils.logging import LoggerMixin


class BaseTransformer(BaseEstimator, TransformerMixin, LoggerMixin):
    """
    Base transformer class with enhanced functionality.
    
    This class extends sklearn's BaseEstimator and TransformerMixin with
    additional validation, logging, and error handling capabilities.
    """
    
    def __init__(self, **kwargs):
        """Initialize base transformer."""
        super().__init__()
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseTransformer':
        """
        Fit the transformer.
        
        Args:
            X: Input DataFrame
            y: Target series (optional)
            
        Returns:
            Self for method chaining
        """
        self._validate_input(X)
        self._fit(X, y)
        self._is_fitted = True
        self.log_info(f"Fitted {self.__class__.__name__}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Transformed DataFrame
            
        Raises:
            ValueError: If transformer not fitted
        """
        if not self._is_fitted:
            raise ValueError(f"{self.__class__.__name__} must be fitted before transform")
        
        self._validate_input(X)
        result = self._transform(X)
        self._validate_output(result)
        self.log_info(f"Transformed data: {X.shape} -> {result.shape}")
        return result
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform the data.
        
        Args:
            X: Input DataFrame
            y: Target series (optional)
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """
        Validate input data.
        
        Args:
            X: Input DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if X.empty:
            raise ValueError("Input DataFrame is empty")
    
    def _validate_output(self, X: pd.DataFrame) -> None:
        """
        Validate output data.
        
        Args:
            X: Output DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Output must be a pandas DataFrame")
        
        if X.empty:
            self.log_warning("Output DataFrame is empty")
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        Internal fit method to be implemented by subclasses.
        
        Args:
            X: Input DataFrame
            y: Target series (optional)
        """
        pass
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Internal transform method to be implemented by subclasses.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        raise NotImplementedError("Subclasses must implement _transform")
    
    def get_feature_names_out(self) -> list[str]:
        """
        Get output feature names.
        
        Returns:
            List of feature names
        """
        if not self._is_fitted:
            raise ValueError("Transformer must be fitted before getting feature names")
        
        return self._get_feature_names_out()
    
    def _get_feature_names_out(self) -> list[str]:
        """
        Internal method to get feature names.
        
        Returns:
            List of feature names
        """
        return []
    
    @property
    def is_fitted(self) -> bool:
        """Check if transformer is fitted."""
        return self._is_fitted 
"""
Fourier mode transformers for seasonality features.

This module provides Fourier-based seasonality feature generation
for time series preprocessing.
"""
from typing import Optional
import numpy as np
import pandas as pd
from .base import BaseTransformer


class FourierModes(BaseTransformer):
    """
    Fourier mode feature generator for seasonality.
    
    This transformer creates Fourier mode features for capturing
    seasonal patterns in time series data.
    """
    
    def __init__(self, T: float = 365.25, k: int = 1, 
                 time_col: str = "day_of_year", 
                 drop_time_col: bool = False):
        """
        Initialize FourierModes transformer.
        
        Args:
            T: Period of the seasonality
            k: Number of harmonics to generate
            time_col: Name of the time column
            drop_time_col: Whether to drop the original time column
        """
        super().__init__()
        self.T = T
        self.k = k
        self.time_col = time_col
        self.drop_time_col = drop_time_col
        self.required_columns = [time_col]
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FourierModes':
        """
        Fit the transformer (no fitting required for Fourier modes).
        
        Args:
            X: Input features DataFrame
            y: Target values (optional)
        """
        # Fourier modes don't require fitting
        return self
    
    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transform the input data by adding Fourier mode features.
        
        Args:
            X: Input features DataFrame
            y: Target values (optional)
            
        Returns:
            DataFrame with Fourier mode features added
        """
        X_transformed = X.copy()
        
        # Generate Fourier modes
        for order in range(1, self.k + 1):
            # Sine component
            sin_col = f"fourier_sin_{order}"
            X_transformed[sin_col] = np.sin(
                2 * np.pi * (order * X_transformed[self.time_col] / self.T)
            )
            
            # Cosine component
            cos_col = f"fourier_cos_{order}"
            X_transformed[cos_col] = np.cos(
                2 * np.pi * (order * X_transformed[self.time_col] / self.T)
            )
        
        # Drop original time column if requested
        if self.drop_time_col and self.time_col in X_transformed.columns:
            X_transformed = X_transformed.drop(columns=[self.time_col])
        
        # Store feature names
        self.feature_names_out_ = list(X_transformed.columns)
        
        return X_transformed
    
    def get_feature_names_out(self) -> list:
        """Get output feature names."""
        if hasattr(self, 'feature_names_out_'):
            return self.feature_names_out_
        else:
            return []


class MultiSeasonalFourier(BaseTransformer):
    """
    Multi-seasonal Fourier mode generator.
    
    This transformer creates Fourier mode features for multiple
    seasonal periods simultaneously.
    """
    
    def __init__(self, periods: list, orders: list, 
                 time_col: str = "day_of_year",
                 drop_time_col: bool = False):
        """
        Initialize MultiSeasonalFourier transformer.
        
        Args:
            periods: List of seasonal periods
            orders: List of harmonic orders for each period
            time_col: Name of the time column
            drop_time_col: Whether to drop the original time column
        """
        super().__init__()
        if len(periods) != len(orders):
            raise ValueError("Periods and orders must have the same length")
        
        self.periods = periods
        self.orders = orders
        self.time_col = time_col
        self.drop_time_col = drop_time_col
        self.required_columns = [time_col]
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MultiSeasonalFourier':
        """
        Fit the transformer (no fitting required).
        
        Args:
            X: Input features DataFrame
            y: Target values (optional)
        """
        return self
    
    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transform the input data by adding multi-seasonal Fourier features.
        
        Args:
            X: Input features DataFrame
            y: Target values (optional)
            
        Returns:
            DataFrame with multi-seasonal Fourier features added
        """
        X_transformed = X.copy()
        
        # Generate Fourier modes for each period
        for i, (period, order) in enumerate(zip(self.periods, self.orders)):
            for harmonic in range(1, order + 1):
                # Sine component
                sin_col = f"fourier_sin_p{i+1}_h{harmonic}"
                X_transformed[sin_col] = np.sin(
                    2 * np.pi * (harmonic * X_transformed[self.time_col] / period)
                )
                
                # Cosine component
                cos_col = f"fourier_cos_p{i+1}_h{harmonic}"
                X_transformed[cos_col] = np.cos(
                    2 * np.pi * (harmonic * X_transformed[self.time_col] / period)
                )
        
        # Drop original time column if requested
        if self.drop_time_col and self.time_col in X_transformed.columns:
            X_transformed = X_transformed.drop(columns=[self.time_col])
        
        # Store feature names
        self.feature_names_out_ = list(X_transformed.columns)
        
        return X_transformed
    
    def get_feature_names_out(self) -> list:
        """Get output feature names."""
        if hasattr(self, 'feature_names_out_'):
            return self.feature_names_out_
        else:
            return [] 
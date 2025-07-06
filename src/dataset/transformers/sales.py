"""
Sales data transformers.

This module contains transformers for processing sales data including
filtering, wide-to-long conversion, and interval features.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseTransformer


class SparseTSFilter(BaseTransformer):
    """
    Filter sparse time series based on mean values.

    This transformer filters time series where the mean target value
    is between specified lower and upper bounds.
    """

    def __init__(
        self, index_cols: List[str], mean_lb: float = 0.15, mean_ub: float = 0.5
    ):
        """
        Initialize SparseTSFilter.

        Args:
            index_cols: Index columns for grouping
            mean_lb: Lower bound for mean sales
            mean_ub: Upper bound for mean sales
        """
        super().__init__()
        self.index_cols = index_cols
        self.mean_lb = mean_lb
        self.mean_ub = mean_ub
        self._sparse_series: pd.DataFrame = pd.DataFrame()

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input data."""
        super()._validate_input(X)

        missing_cols = [col for col in self.index_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing index columns: {missing_cols}")

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Fit the filter."""
        cols_to_drop = X.select_dtypes("O").columns.difference(set(self.index_cols))
        sparse_series = (
            X.drop(columns=cols_to_drop)
            .set_index(self.index_cols)
            .mean(axis=1)[
                lambda mean_demand: mean_demand.between(self.mean_lb, self.mean_ub)
            ]
        )
        self._sparse_series = sparse_series.reset_index()[self.index_cols]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filter sparse time series."""
        X = X.copy()
        X = pd.merge(X, self._sparse_series, on=self.index_cols)
        return X


class PredictabilityFilter(BaseTransformer):
    """
    Filter non-predictable series.

    This transformer filters out series where the q-quantile is 0,
    indicating too many zeros for prediction.
    """

    def __init__(
        self, q: float, start_date: pd.Timestamp, end_date: pd.Timestamp, date2day: Dict
    ):
        """
        Initialize PredictabilityFilter.

        Args:
            q: Quantile for filtering
            start_date: Start date for analysis
            end_date: End date for analysis
            date2day: Date to day mapping
        """
        super().__init__()
        self.q = q
        self.start_date = start_date
        self.end_date = end_date
        self.date2day = date2day
        self._df_q_sales: pd.DataFrame = pd.DataFrame()
        self._df_predictable_series: pd.DataFrame = pd.DataFrame()

    @property
    def q_percent(self) -> int:
        """Get quantile percentage."""
        return int(self.q * 100)

    @property
    def n_days(self) -> int:
        """Get number of days in range."""
        return (self.end_date - self.start_date).days

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Fit the filter."""
        col_name = f"q{self.q_percent}_sales"
        id_cols = X.select_dtypes("O").columns.tolist()

        self._df_q_sales = (
            X.set_index(id_cols)
            .loc[:, self.date2day[self.start_date] : self.date2day[self.end_date]]
            .apply(lambda s: pd.Series({col_name: s.quantile(q=self.q)}), axis=1)
        )

        self._df_predictable_series = self._df_q_sales.query(
            f"{col_name} > 0"
        ).reset_index()[id_cols]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filter predictable series."""
        X = X.copy()
        id_cols = X.select_dtypes("O").columns.tolist()
        X = pd.merge(X, self._df_predictable_series, on=id_cols)
        return X


class WideToLongTransformer(BaseTransformer):
    """
    Convert wide format to long format.

    This transformer converts wide format sales data to long format
    for time series analysis.
    """

    def __init__(
        self,
        index_cols: List[str],
        long_dict: Dict,
        wide_prefix: str = "d_",
        long_col: str = "date",
        target_col: str = "sales",
    ):
        """
        Initialize WideToLongTransformer.

        Args:
            index_cols: Index columns for grouping
            long_dict: Mapping from wide to long values
            wide_prefix: Prefix for wide format columns
            long_col: Column name in long format
            target_col: Target column name
        """
        super().__init__()
        self.index_cols = index_cols
        self.long_dict = long_dict
        self.wide_prefix = wide_prefix
        self.long_col = long_col
        self.target_col = target_col
        self.cols_to_drop: List[str] = []

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Fit the transformer."""
        self.cols_to_drop = (
            X.select_dtypes("O").columns.difference(set(self.index_cols)).tolist()
        )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert wide to long format."""
        X = X.copy()
        X = pd.wide_to_long(
            X.drop(columns=self.cols_to_drop),
            stubnames=self.wide_prefix,
            i=self.index_cols,
            j=self.long_col,
        ).rename(columns={self.wide_prefix: self.target_col})
        X = X.reset_index()
        X[self.long_col] = X[self.long_col].map(
            lambda el: self.long_dict[f"{self.wide_prefix}{el}"]
        )
        return X


class IntervalTransformer(BaseTransformer):
    """
    Create interval features for time series.

    This transformer creates features for periods since last sale
    and periods since last zero sale.
    """

    def __init__(
        self,
        groupby_cols: List[str],
        target_col: str,
        set_non_zero_intervals: bool = True,
        set_zero_intervals: bool = False,
    ):
        """
        Initialize IntervalTransformer.

        Args:
            groupby_cols: Columns for grouping
            target_col: Target column name
            set_non_zero_intervals: Whether to create non-zero interval features
            set_zero_intervals: Whether to create zero interval features
        """
        super().__init__()
        self.groupby_cols = groupby_cols
        self.target_col = target_col
        self.set_non_zero_intervals = set_non_zero_intervals
        self.set_zero_intervals = set_zero_intervals

    @staticmethod
    def create_non_zero_interval_series(s: pd.Series) -> np.ndarray:
        """
        Create non-zero interval series.

        Args:
            s: Input series

        Returns:
            Array of interval values
        """
        vals = s.values
        is_zero_val_series = np.where(vals == 0, 1, 0)
        non_zero_vals_idxes = np.where(vals > 0)[0]
        non_zero_interval_vals = np.concatenate(
            [
                np.where(sub_series == 1, sub_series.cumsum(), 0)
                for sub_series in np.split(is_zero_val_series, non_zero_vals_idxes)
            ]
        )
        return non_zero_interval_vals

    @staticmethod
    def create_zero_interval_series(s: pd.Series) -> np.ndarray:
        """
        Create zero interval series.

        Args:
            s: Input series

        Returns:
            Array of interval values
        """
        vals = s.values
        is_non_zero_val_series = np.where(vals > 0, 1, 0)
        zero_vals_idxes = np.where(vals == 0)[0]
        zero_interval_vals = np.concatenate(
            [
                np.where(sub_series == 1, sub_series.cumsum(), 0)
                for sub_series in np.split(is_non_zero_val_series, zero_vals_idxes)
            ]
        )
        return zero_interval_vals

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interval features."""
        X = X.copy()

        if self.set_non_zero_intervals:
            X[f"periods since last {self.target_col}"] = X.groupby(self.groupby_cols)[
                self.target_col
            ].transform(self.create_non_zero_interval_series)

        if self.set_zero_intervals:
            X[f"periods since last 0 {self.target_col}"] = X.groupby(self.groupby_cols)[
                self.target_col
            ].transform(self.create_zero_interval_series)

        return X

"""
Feature engineering transformers.

This module contains transformers for feature engineering including
momentum features and ID transformations.
"""

from typing import Dict, List, Optional

import pandas as pd

from .base import BaseTransformer


class MomentumTransformer(BaseTransformer):
    """
    Create momentum features for time series.

    This transformer creates momentum features by calculating the difference
    between current values and rolling means.
    """

    def __init__(
        self,
        groupby_cols: List[str],
        target_col: str,
        window_size: int,
        closed: str = "left",
    ):
        """
        Initialize MomentumTransformer.

        Args:
            groupby_cols: Columns for grouping
            target_col: Target column name
            window_size: Window size for rolling calculation
            closed: Window closure method
        """
        super().__init__()
        self.groupby_cols = groupby_cols
        self.target_col = target_col
        self.window_size = window_size
        self.closed = closed

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create momentum features."""
        X = X.copy()
        X[f"{self.target_col}_momentum"] = X.groupby(self.groupby_cols)[
            self.target_col
        ].transform(
            lambda s: s - s.rolling(window=self.window_size, closed=self.closed).mean()
        )
        return X


class IdTransformer(BaseTransformer):
    """
    Transform ID columns into a single unique ID.

    This transformer combines multiple ID columns into a single unique ID
    column for easier processing.
    """

    def __init__(
        self,
        id_cols: List[str],
        id_col_name: str = "unique_id",
        sep: str = "__",
        drop: bool = False,
    ):
        """
        Initialize IdTransformer.

        Args:
            id_cols: ID columns to combine
            id_col_name: Name for the combined ID column
            sep: Separator for combining IDs
            drop: Whether to drop original ID columns
        """
        super().__init__()
        self.id_cols = id_cols
        self.sep = sep
        self.drop = drop
        self.id_col_name = id_col_name
        self._id_col2loc: Dict[str, int] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Fit the transformer."""
        self._id_col2loc = {
            id_col: (X.columns == id_col).argmax().item() for id_col in self.id_cols
        }

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform ID columns."""
        X = X.copy()
        X = self._add_id_col(X)
        if self.drop:
            X = X.drop(columns=self.id_cols)
        return X

    def _add_id_col(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add combined ID column."""
        id_col_name = self.id_col_name
        id_series = X[self.id_cols[0]]
        for id_col in self.id_cols[1:]:
            id_series = id_series.str.cat(X[id_col], sep=self.sep)
        X.insert(0, id_col_name, id_series)
        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply inverse transformation.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with original ID columns
        """
        X = X.copy()
        df_id_cols = X[self.id_col_name].str.split(pat=self.sep, expand=True)
        X = X.drop(columns=self.id_col_name)
        for i, (id_col, id_loc) in enumerate(self._id_col2loc.items()):
            X.insert(id_loc, id_col, df_id_cols[i])
        return X

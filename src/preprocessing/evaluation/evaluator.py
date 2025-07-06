"""
Evaluator module for time series forecasting evaluation.

This module provides evaluation functionality for time series forecasting
models.
"""

from typing import List

import numpy as np
import pandas as pd


def mase(y_true: pd.Series, y_pred: pd.Series, y_baseline: pd.Series) -> float:
    """Mean Absolute Scaled Error."""
    mae = np.mean(np.abs(y_true - y_pred))
    mae_baseline = np.mean(np.abs(y_true - y_baseline))
    return float(mae / mae_baseline if mae_baseline > 0 else np.inf)


def rmsse(
    y_true: pd.Series, y_pred: pd.Series, y_baseline: pd.Series
) -> float:
    """Root Mean Squared Scaled Error."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    rmse_baseline = np.sqrt(np.mean((y_true - y_baseline) ** 2))
    return float(rmse / rmse_baseline if rmse_baseline > 0 else np.inf)


def mrae(y_true: pd.Series, y_pred: pd.Series, y_baseline: pd.Series) -> float:
    """Mean Relative Absolute Error."""
    mae = np.mean(np.abs(y_true - y_pred))
    mae_baseline = np.mean(np.abs(y_true - y_baseline))
    return float(mae / mae_baseline if mae_baseline > 0 else np.inf)


def f1_zero(y_true: pd.Series, y_pred: pd.Series) -> float:
    """F1 score for zero predictions."""
    y_true_zero = y_true == 0
    y_pred_zero = y_pred == 0

    tp = np.sum(y_true_zero & y_pred_zero)
    fp = np.sum(~y_true_zero & y_pred_zero)
    fn = np.sum(y_true_zero & ~y_pred_zero)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return float(
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )


def non_zero_mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Error for non-zero values."""
    non_zero_mask = (y_true > 0) & (y_pred > 0)
    if np.sum(non_zero_mask) == 0:
        return np.inf
    return float(
        np.mean(np.abs(y_true[non_zero_mask] - y_pred[non_zero_mask]))
    )


class Evaluator:
    """Evaluator for time series forecasting models."""

    def __init__(
        self,
        target_col: str = "y",
        seasonal_baseline_col: str = "SeasonalNaive",
        zero_baseline_col: str = "ZeroModel",
    ):
        """
        Initialize the evaluator.

        Args:
            target_col: Name of the target column
            seasonal_baseline_col: Name of the seasonal baseline column
            zero_baseline_col: Name of the zero baseline column
        """
        self.target_col = target_col
        self.seasonal_baseline_col = seasonal_baseline_col
        self.zero_baseline_col = zero_baseline_col

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluate predictions for a single group.

        Args:
            df: DataFrame with target and prediction columns

        Returns:
            Series with evaluation metrics
        """
        y_true = df[self.target_col]
        y_seasonal_baseline = df[self.seasonal_baseline_col]
        y_zero_baseline = df[self.zero_baseline_col]

        # Get prediction columns (exclude target and baseline columns)
        pred_cols = [
            col
            for col in df.columns
            if col
            not in [
                self.target_col,
                self.seasonal_baseline_col,
                self.zero_baseline_col,
                "unique_id",
                "ds",
                "cutoff",
            ]
        ]

        results = {}

        for col in pred_cols:
            y_pred = df[col]

            # Calculate metrics
            results[f"{col}_mae"] = np.mean(np.abs(y_true - y_pred))
            results[f"{col}_rmse"] = np.sqrt(np.mean((y_true - y_pred) ** 2))
            results[f"{col}_mase"] = mase(y_true, y_pred, y_seasonal_baseline)
            results[f"{col}_rmsse"] = rmsse(
                y_true, y_pred, y_seasonal_baseline
            )
            results[f"{col}_mrae"] = mrae(y_true, y_pred, y_zero_baseline)
            results[f"{col}_f1_zero"] = f1_zero(y_true, y_pred)
            results[f"{col}_non_zero_mae"] = non_zero_mae(y_true, y_pred)

        return pd.Series(results)

    def evaluate_by_group(
        self, df: pd.DataFrame, group_cols: List[str]
    ) -> pd.DataFrame:
        """
        Evaluate predictions grouped by specified columns.

        Args:
            df: DataFrame with target and prediction columns
            group_cols: Columns to group by

        Returns:
            DataFrame with evaluation metrics for each group
        """
        return df.groupby(by=group_cols).apply(self.evaluate).reset_index()

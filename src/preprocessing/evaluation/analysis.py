"""
Analysis module for evaluation results.

This module provides analysis functionality for evaluation results.
"""

from typing import List, Union

import pandas as pd


def get_top_k_results(
    df_metrics: pd.DataFrame,
    model_name: str,
    group_index: Union[str, List[str]],
    metric: str = "mean_absolute_error",
    k: int = 10,
    return_best: bool = True,
) -> pd.DataFrame:
    """
    Get top k results for a specific model and metric.

    Args:
        df_metrics: DataFrame with evaluation metrics
        model_name: Name of the model to analyze
        group_index: Column(s) to group by
        metric: Metric column to sort by
        k: Number of top results to return
        return_best: If True, return best results; if False, return worst results

    Returns:
        DataFrame with top k results
    """
    # Find the metric column for the model
    metric_col = None
    for col in df_metrics.columns:
        if col.startswith(f"{model_name}_") and col.endswith(f"_{metric}"):
            metric_col = col
            break

    if metric_col is None:
        # Try alternative metric names
        alt_metrics = ["mae", "rmse", "mse", "mape"]
        for alt_metric in alt_metrics:
            for col in df_metrics.columns:
                if col.startswith(f"{model_name}_") and col.endswith(f"_{alt_metric}"):
                    metric_col = col
                    break
            if metric_col:
                break

    if metric_col is None:
        raise ValueError(f"Could not find metric column for model {model_name}")

    # Sort by the metric
    if return_best:
        # For error metrics, lower is better
        df_sorted = df_metrics.sort_values(by=metric_col, ascending=True)
    else:
        # For error metrics, higher is worse
        df_sorted = df_metrics.sort_values(by=metric_col, ascending=False)

    # Get top k results
    top_k = df_sorted.head(k)

    return top_k

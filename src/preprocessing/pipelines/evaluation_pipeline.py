"""
Evaluation pipeline for time series forecasting results.

This module provides an evaluation pipeline that applies postprocessing
and evaluates results using custom metrics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..evaluation import Evaluator, get_top_k_results
from ..transformers.temporal import ZeroPredTransformer
from ..utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationPipeline:
    """
    Evaluation pipeline for time series forecasting results.

    This pipeline applies postprocessing to predictions and evaluates
    results using custom metrics on different segments.
    """

    def __init__(
        self,
        output_dir: str = "outputs/evaluation",
        target_col: str = "y",
        baseline_col: Optional[str] = None,
        df_fitted: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize EvaluationPipeline.

        Args:
            output_dir: Directory to save evaluation results
            target_col: Column name for target values
            baseline_col: Column name for baseline predictions
            df_fitted: DataFrame with fitted values for scaled metrics
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_col = target_col
        self.baseline_col = baseline_col
        self.df_fitted = df_fitted

        # Initialize postprocessing transformer
        self.zero_pred_transformer = ZeroPredTransformer(
            negative_mask=True, is_close_mask=True
        )

        # Initialize evaluator
        self.evaluator = Evaluator(
            target_col=target_col, baseline_col=baseline_col, df_fitted=df_fitted
        )

    def run(self, df_results: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the evaluation pipeline.

        Args:
            df_results: DataFrame with predictions and true values

        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting evaluation pipeline")

        # Apply postprocessing
        logger.info("Applying postprocessing with ZeroPredTransformer")
        df_results_processed = self.zero_pred_transformer.fit_transform(df_results)

        # Save processed results
        processed_path = self.output_dir / "processed_results.csv"
        df_results_processed.to_csv(processed_path, index=False)
        logger.info(f"Saved processed results to {processed_path}")

        # Evaluate all results
        logger.info("Evaluating all results")
        df_metrics_all = self.evaluator.evaluate_all(df_results_processed)

        # Evaluate by unique_id
        logger.info("Evaluating by unique_id")
        df_metrics_item = self.evaluator.evaluate_by_group(
            df_results_processed, ["unique_id"]
        )

        # Evaluate by cutoff
        logger.info("Evaluating by cutoff")
        df_metrics_cutoff = self.evaluator.evaluate_by_group(
            df_results_processed, ["cutoff"]
        )

        # Evaluate by unique_id and cutoff
        logger.info("Evaluating by unique_id and cutoff")
        df_metrics_item_cutoff = self.evaluator.evaluate_by_group(
            df_results_processed, ["unique_id", "cutoff"]
        )

        # Save evaluation results
        results = {
            "processed_results": df_results_processed,
            "metrics_all": df_metrics_all,
            "metrics_item": df_metrics_item,
            "metrics_cutoff": df_metrics_cutoff,
            "metrics_item_cutoff": df_metrics_item_cutoff,
        }

        self._save_results(results)

        logger.info("Evaluation pipeline completed successfully")
        return results

    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Save evaluation results to files.

        Args:
            results: Dictionary with evaluation results
        """
        # Save metrics
        for name, df in results.items():
            if isinstance(df, pd.DataFrame):
                file_path = self.output_dir / f"{name}.csv"
                df.to_csv(file_path)
                logger.info(f"Saved {name} to {file_path}")

    def get_top_k_analysis(
        self,
        df_metrics: pd.DataFrame,
        models: List[str],
        group_index: Union[str, List[str]] = "unique_id",
        k: int = 10,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get top k analysis for multiple models.

        Args:
            df_metrics: DataFrame with evaluation metrics
            models: List of model names to analyze
            group_index: Column(s) to group by (e.g., 'unique_id', 'cutoff', or both)
            k: Number of top results to return

        Returns:
            Dictionary with top k results for each model
        """
        analysis_results = {}

        # Reset index if unique_id is the index
        if (
            "unique_id" not in df_metrics.columns
            and df_metrics.index.name == "unique_id"
        ):
            df_metrics = df_metrics.reset_index()

        for model in models:
            # Get best results
            best_results = get_top_k_results(
                df_metrics=df_metrics,
                model_name=model,
                group_index=group_index,
                metric="mean_absolute_error",
                k=k,
                return_best=True,
            )

            # Get worst results
            worst_results = get_top_k_results(
                df_metrics=df_metrics,
                model_name=model,
                group_index=group_index,
                metric="mean_absolute_error",
                k=k,
                return_best=False,
            )

            analysis_results[f"{model}_best"] = best_results
            analysis_results[f"{model}_worst"] = worst_results

            # Save analysis results
            best_path = self.output_dir / f"{model}_best_{k}.csv"
            worst_path = self.output_dir / f"{model}_worst_{k}.csv"

            best_results.to_csv(best_path)
            worst_results.to_csv(worst_path)

            logger.info(f"Saved {model} analysis to {best_path} and {worst_path}")

        return analysis_results

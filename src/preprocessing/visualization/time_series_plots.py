"""
Time series visualization for preprocessing pipeline.

This module provides visualization utilities for time series analysis,
including aggregated series plots and individual series analysis.
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class TimeSeriesPlotter:
    """
    Time series visualization class.

    This class provides methods for visualizing time series data,
    aggregated series, and individual series analysis.
    """

    def __init__(self, output_dir: str = "outputs/visualization"):
        """
        Initialize TimeSeriesPlotter.

        Args:
            output_dir: Output directory for plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def plot_aggregated_series(
        self,
        df: pd.DataFrame,
        agg_index: str,
        agg_freq: str,
        models: List[str],
        alphas: List[float],
        linestyles: List[str],
    ) -> None:
        """
        Plot aggregated time series.

        Args:
            df: DataFrame with time series data
            agg_index: Column to aggregate by (e.g., 'unique_id')
            agg_freq: Frequency for aggregation (e.g., 'D' for daily, 'W' for weekly)
            models: List of model columns to plot
            alphas: List of alpha values for transparency
            linestyles: List of line styles
        """
        # Convert date column to datetime if needed
        if "ds" in df.columns:
            df["ds"] = pd.to_datetime(df["ds"])

        # Aggregate data
        agg_data = (
            df.groupby([agg_index, pd.Grouper(key="ds", freq=agg_freq)])[models]
            .mean()
            .reset_index()
        )

        # Create plot
        plt.figure(figsize=(15, 8))

        for i, model in enumerate(models):
            alpha = alphas[i] if i < len(alphas) else 0.7
            linestyle = linestyles[i] if i < len(linestyles) else "-"

            # Plot aggregated series
            plt.plot(
                agg_data["ds"],
                agg_data[model],
                label=model,
                alpha=alpha,
                linestyle=linestyle,
                linewidth=2,
            )

        plt.title(f"Aggregated Time Series by {agg_index} ({agg_freq})")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / f"aggregated_series_{agg_index}_{agg_freq}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Saved aggregated series plot to {plot_path}")

    def series_analysis_plots(
        self, df: pd.DataFrame, df_preds: pd.DataFrame, unique_id: str, model_col: str
    ) -> None:
        """
        Create comprehensive series analysis plots.

        Args:
            df: Original time series data
            df_preds: DataFrame with predictions
            unique_id: Unique identifier for the series
            model_col: Model column name
        """
        # Filter data for the specific series
        series_data = df[df["unique_id"] == unique_id].copy()
        series_preds = df_preds[df_preds["unique_id"] == unique_id].copy()

        if series_data.empty or series_preds.empty:
            self.logger.warning(f"No data found for series {unique_id}")
            return

        # Convert date column to datetime if needed
        if "ds" in series_data.columns:
            series_data["ds"] = pd.to_datetime(series_data["ds"])
        if "ds" in series_preds.columns:
            series_preds["ds"] = pd.to_datetime(series_preds["ds"])

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Series Analysis: {unique_id} - {model_col}", fontsize=16)

        # Plot 1: Time series with predictions
        ax1 = axes[0, 0]
        ax1.plot(series_data["ds"], series_data["y"], label="Actual", linewidth=2)
        ax1.plot(
            series_preds["ds"],
            series_preds[model_col],
            label=f"{model_col} Predictions",
            linewidth=2,
            alpha=0.8,
        )
        ax1.set_title("Time Series with Predictions")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="x", rotation=45)

        # Plot 2: Scatter plot of actual vs predicted
        ax2 = axes[0, 1]
        ax2.scatter(series_preds["y"], series_preds[model_col], alpha=0.6)
        ax2.plot(
            [series_preds["y"].min(), series_preds["y"].max()],
            [series_preds["y"].min(), series_preds["y"].max()],
            "r--",
            label="Perfect Prediction",
        )
        ax2.set_title("Actual vs Predicted")
        ax2.set_xlabel("Actual Values")
        ax2.set_ylabel("Predicted Values")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Residuals
        ax3 = axes[1, 0]
        residuals = series_preds["y"] - series_preds[model_col]
        ax3.scatter(series_preds["ds"], residuals, alpha=0.6)
        ax3.axhline(y=0, color="r", linestyle="--", label="Zero Residual")
        ax3.set_title("Residuals Over Time")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Residuals")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis="x", rotation=45)

        # Plot 4: Residuals distribution
        ax4 = axes[1, 1]
        ax4.hist(residuals, bins=30, alpha=0.7, edgecolor="black")
        ax4.axvline(x=0, color="r", linestyle="--", label="Zero Residual")
        ax4.set_title("Residuals Distribution")
        ax4.set_xlabel("Residuals")
        ax4.set_ylabel("Frequency")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / f"series_analysis_{unique_id}_{model_col}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Saved series analysis plot to {plot_path}")

    def plot_multiple_series(
        self,
        df: pd.DataFrame,
        unique_ids: List[str],
        model_col: str,
        max_series: int = 6,
    ) -> None:
        """
        Plot multiple series in a grid layout.

        Args:
            df: DataFrame with time series data
            unique_ids: List of unique identifiers to plot
            model_col: Model column name
            max_series: Maximum number of series to plot
        """
        # Limit number of series
        unique_ids = unique_ids[:max_series]

        # Calculate grid dimensions
        n_series = len(unique_ids)
        n_cols = min(3, n_series)
        n_rows = (n_series + n_cols - 1) // n_cols

        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, unique_id in enumerate(unique_ids):
            if i >= len(axes):
                break

            # Filter data for the specific series
            series_data = df[df["unique_id"] == unique_id].copy()

            if series_data.empty:
                continue

            # Convert date column to datetime if needed
            if "ds" in series_data.columns:
                series_data["ds"] = pd.to_datetime(series_data["ds"])

            # Plot series
            axes[i].plot(series_data["ds"], series_data["y"], linewidth=2)
            if model_col in series_data.columns:
                axes[i].plot(
                    series_data["ds"],
                    series_data[model_col],
                    linewidth=2,
                    alpha=0.8,
                    label=f"{model_col} Predictions",
                )

            axes[i].set_title(f"Series: {unique_id}")
            axes[i].set_xlabel("Date")
            axes[i].set_ylabel("Value")
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis="x", rotation=45)

            if model_col in series_data.columns:
                axes[i].legend()

        # Hide empty subplots
        for i in range(n_series, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / f"multiple_series_{model_col}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Saved multiple series plot to {plot_path}")

    def generate_visualization_report(
        self, df: pd.DataFrame, df_preds: pd.DataFrame, models: List[str]
    ) -> None:
        """
        Generate comprehensive visualization report.

        Args:
            df: Original time series data
            df_preds: DataFrame with predictions
            models: List of model columns
        """
        self.logger.info("Generating comprehensive visualization report")

        # Get unique series
        unique_ids = df["unique_id"].unique()[:10]  # Limit to first 10 series

        # Create aggregated plots
        self.plot_aggregated_series(
            df, "unique_id", "D", models, [0.7, 0.8], ["-", "--"]
        )
        self.plot_aggregated_series(
            df, "unique_id", "W", models, [0.7, 0.8], ["-", "--"]
        )

        # Create series analysis plots for a few series
        for i, unique_id in enumerate(unique_ids[:3]):
            for model in models[:2]:  # Limit to first 2 models
                self.series_analysis_plots(df, df_preds, unique_id, model)

        # Create multiple series plot
        self.plot_multiple_series(df, unique_ids[:6], models[0] if models else "y")

        self.logger.info("Visualization report completed")

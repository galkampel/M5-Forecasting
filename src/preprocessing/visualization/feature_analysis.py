"""
Feature analysis visualization for preprocessing pipeline.

This module provides visualization utilities for analyzing
feature distributions, correlations, and importance.
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class FeatureAnalyzer:
    """
    Feature analysis visualization class.

    This class provides methods for visualizing feature distributions,
    correlations, and importance scores.
    """

    def __init__(self, output_dir: str = "../outputs/preprocessing"):
        """
        Initialize FeatureAnalyzer.

        Args:
            output_dir: Output directory for plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def plot_feature_distributions(
        self, features_df: pd.DataFrame, n_features: int = 20
    ) -> None:
        """
        Plot feature distributions.

        Args:
            features_df: Features DataFrame
            n_features: Number of features to plot
        """
        # Select numeric features
        numeric_features = features_df.select_dtypes(include=["number"]).columns[
            :n_features
        ]

        if len(numeric_features) == 0:
            self.logger.warning("No numeric features found for distribution plots")
            return

        # Create subplots
        n_cols = 4
        n_rows = (len(numeric_features) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for i, feature in enumerate(numeric_features):
            if i < len(axes):
                axes[i].hist(features_df[feature].dropna(), bins=30, alpha=0.7)
                axes[i].set_title(f"Distribution of {feature}")
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel("Frequency")

        # Hide empty subplots
        for i in range(len(numeric_features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "feature_distributions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Saved feature distributions plot to {plot_path}")

    def plot_correlation_matrix(
        self, features_df: pd.DataFrame, max_features: int = 50
    ) -> None:
        """
        Plot correlation matrix.

        Args:
            features_df: Features DataFrame
            max_features: Maximum number of features to include
        """
        # Select numeric features
        numeric_features = features_df.select_dtypes(include=["number"])

        if len(numeric_features.columns) == 0:
            self.logger.warning("No numeric features found for correlation plot")
            return

        # Limit number of features for visualization
        if len(numeric_features.columns) > max_features:
            # Select features with highest variance
            variances = numeric_features.var().sort_values(ascending=False)
            selected_features = variances.head(max_features).index
            numeric_features = numeric_features[selected_features]

        # Calculate correlation matrix
        corr_matrix = numeric_features.corr()

        # Create plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
        )
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "feature_correlation_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Saved correlation matrix plot to {plot_path}")

    def plot_feature_importance(self, importance_scores: dict) -> None:
        """
        Plot feature importance scores.

        Args:
            importance_scores: Dictionary of feature importance scores
        """
        if not importance_scores:
            self.logger.warning("No importance scores provided")
            return

        # Sort features by importance
        sorted_features = sorted(
            importance_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Take top 20 features
        top_features = sorted_features[:20]

        features, scores = zip(*top_features)

        # Create plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel("Importance Score")
        plt.title("Top 20 Feature Importance Scores")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "feature_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Saved feature importance plot to {plot_path}")

    def generate_feature_report(self, features_df: pd.DataFrame) -> None:
        """
        Generate comprehensive feature analysis report.

        Args:
            features_df: Features DataFrame
        """
        self.logger.info("Generating feature analysis report")

        # Create plots
        self.plot_feature_distributions(features_df)
        self.plot_correlation_matrix(features_df)

        # Generate feature statistics
        feature_stats = {
            "total_features": int(len(features_df.columns)),
            "numeric_features": int(
                len(features_df.select_dtypes(include=["number"]).columns)
            ),
            "categorical_features": int(
                len(features_df.select_dtypes(include=["object"]).columns)
            ),
            "missing_values": int(features_df.isnull().sum().sum()),
            "memory_usage_mb": float(
                features_df.memory_usage(deep=True).sum() / 1024 / 1024
            ),
        }

        # Save statistics
        import json

        stats_path = self.output_dir / "feature_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(feature_stats, f, indent=2)

        self.logger.info(f"Saved feature statistics to {stats_path}")
        self.logger.info("Feature analysis report completed")

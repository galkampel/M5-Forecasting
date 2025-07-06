"""
Data loading utilities for preprocessing pipeline.

This module provides data loading functionality with validation,
error handling, and memory-efficient processing.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for preprocessing pipeline."""

    def __init__(self, config):
        """
        Initialize data loader.

        Args:
            config: PreprocessingConfig instance
        """
        self.config = config
        self.features_df = None
        self.targets_df = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load features and targets data.

        Returns:
            Tuple of (features_df, targets_df)
        """
        logger.info("Loading preprocessing data...")

        try:
            # Load features
            self.features_df = self._load_features()
            logger.info(f"Loaded features: {self.features_df.shape}")

            # Load targets
            self.targets_df = self._load_targets()
            logger.info(f"Loaded targets: {self.targets_df.shape}")

            # Validate data
            self._validate_data()

            return self.features_df, self.targets_df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _load_features(self) -> pd.DataFrame:
        """Load features data."""
        features_path = self.config.data.features_path

        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")

        try:
            # Try to infer date columns
            date_columns = ["date", "ds", "timestamp"]
            features_df = pd.read_csv(features_path)

            # Convert date columns to datetime
            for col in date_columns:
                if col in features_df.columns:
                    features_df[col] = pd.to_datetime(features_df[col])
                    break

            logger.info(f"Features columns: {list(features_df.columns)}")
            return features_df

        except Exception as e:
            logger.error(f"Error loading features: {str(e)}")
            raise

    def _load_targets(self) -> pd.DataFrame:
        """Load targets data."""
        targets_path = self.config.data.targets_path

        if not os.path.exists(targets_path):
            raise FileNotFoundError(f"Targets file not found: {targets_path}")

        try:
            targets_df = pd.read_csv(targets_path)

            # Convert date columns to datetime
            date_columns = ["date", "ds", "timestamp"]
            for col in date_columns:
                if col in targets_df.columns:
                    targets_df[col] = pd.to_datetime(targets_df[col])
                    break

            logger.info(f"Targets columns: {list(targets_df.columns)}")
            return targets_df

        except Exception as e:
            logger.error(f"Error loading targets: {str(e)}")
            raise

    def _validate_data(self) -> None:
        """Validate loaded data."""
        if self.features_df is None or self.targets_df is None:
            raise ValueError("Both features and targets must be loaded")

        # Check for required columns
        required_feature_cols = ["unique_id", "ds"]
        required_target_cols = ["unique_id", "ds", "y"]

        missing_feature_cols = [
            col for col in required_feature_cols if col not in self.features_df.columns
        ]
        missing_target_cols = [
            col for col in required_target_cols if col not in self.targets_df.columns
        ]

        if missing_feature_cols:
            raise ValueError(
                f"Missing required feature columns: {missing_feature_cols}"
            )

        if missing_target_cols:
            raise ValueError(f"Missing required target columns: {missing_target_cols}")

        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(self.features_df["ds"]):
            raise ValueError("Feature 'ds' column must be datetime")

        if not pd.api.types.is_datetime64_any_dtype(self.targets_df["ds"]):
            raise ValueError("Target 'ds' column must be datetime")

        # Check for missing values
        feature_missing = self.features_df.isnull().sum().sum()
        target_missing = self.targets_df.isnull().sum().sum()

        if feature_missing > 0:
            logger.warning(f"Features contain {feature_missing} missing values")

        if target_missing > 0:
            logger.warning(f"Targets contain {target_missing} missing values")

        # Check data consistency
        feature_ids = set(self.features_df["unique_id"].unique())
        target_ids = set(self.targets_df["unique_id"].unique())

        if feature_ids != target_ids:
            logger.warning(
                f"Feature and target IDs don't match. "
                f"Feature IDs: {len(feature_ids)}, "
                f"Target IDs: {len(target_ids)}"
            )

        logger.info("Data validation completed successfully")

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about loaded data.

        Returns:
            Dictionary with data information
        """
        if self.features_df is None or self.targets_df is None:
            raise ValueError("Data not loaded yet")

        return {
            "features_shape": self.features_df.shape,
            "targets_shape": self.targets_df.shape,
            "feature_columns": list(self.features_df.columns),
            "target_columns": list(self.targets_df.columns),
            "unique_ids": len(self.features_df["unique_id"].unique()),
            "date_range": {
                "start": self.features_df["ds"].min(),
                "end": self.features_df["ds"].max(),
            },
            "memory_usage": {
                "features_mb": self.features_df.memory_usage(deep=True).sum() / 1024**2,
                "targets_mb": self.targets_df.memory_usage(deep=True).sum() / 1024**2,
            },
        }

    def save_processed_data(
        self, features_df: pd.DataFrame, targets_df: pd.DataFrame
    ) -> None:
        """
        Save processed data to output directory.

        Args:
            features_df: Processed features DataFrame
            targets_df: Processed targets DataFrame
        """
        output_dir = self.config.data.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save features
        features_path = os.path.join(output_dir, "processed_features.csv")
        features_df.to_csv(features_path, index=False)
        logger.info(f"Saved processed features to {features_path}")

        # Save targets
        targets_path = os.path.join(output_dir, "processed_targets.csv")
        targets_df.to_csv(targets_path, index=False)
        logger.info(f"Saved processed targets to {targets_path}")

    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load processed data from output directory.

        Returns:
            Tuple of (features_df, targets_df)
        """
        output_dir = self.config.data.output_dir

        features_path = os.path.join(output_dir, "processed_features.csv")
        targets_path = os.path.join(output_dir, "processed_targets.csv")

        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Processed features not found: {features_path}")

        if not os.path.exists(targets_path):
            raise FileNotFoundError(f"Processed targets not found: {targets_path}")

        try:
            features_df = pd.read_csv(features_path)
            targets_df = pd.read_csv(targets_path)

            # Convert date columns back to datetime
            for df in [features_df, targets_df]:
                if "ds" in df.columns:
                    df["ds"] = pd.to_datetime(df["ds"])

            logger.info("Loaded processed data successfully")
            return features_df, targets_df

        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise

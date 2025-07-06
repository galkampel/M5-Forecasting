"""
Feature engineering pipeline.

This module orchestrates feature engineering including merging
calendar, sales, and price data.
"""

from typing import Tuple

import pandas as pd

from ..config import Config
from ..transformers.features import IdTransformer
from ..utils.logging import LoggerMixin


class FeaturePipeline(LoggerMixin):
    """
    Pipeline for feature engineering.

    This pipeline handles merging calendar, sales, and price data
    to create final feature and target DataFrames.
    """

    def __init__(self, config: Config):
        """
        Initialize feature pipeline.

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config

        # Create transformers
        self.id_transformer = IdTransformer(
            id_cols=config.columns.index_cols, drop=True
        )

    def process(
        self, calendar_df: pd.DataFrame, sales_df: pd.DataFrame, prices_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process features and targets.

        Args:
            calendar_df: Processed calendar DataFrame
            sales_df: Processed sales DataFrame
            prices_df: Raw prices DataFrame

        Returns:
            Tuple of (features_df, targets_df)
        """
        self.log_info("Processing features and targets")

        try:
            # Step 1: Create features by merging calendar and sales data
            self.log_info("Step 1: Merging calendar and sales data")
            index_cols_with_date = self.config.columns.index_cols + [
                self.config.columns.date_col
            ]
            features_df = pd.merge(
                sales_df[index_cols_with_date],
                calendar_df,
                on=self.config.columns.date_col,
            )

            # Step 2: Merge with price data
            self.log_info("Step 2: Merging with price data")
            features_df = pd.merge(
                features_df,
                prices_df,
                on=self.config.columns.index_cols + ["wm_yr_wk"],
                how="left",
            )

            # Step 3: Handle missing prices
            self.log_info("Step 3: Handling missing prices")
            features_df = features_df.fillna(value={"sell_price": 0.0})
            features_df["is_item_exists"] = (features_df["sell_price"] > 0).astype(int)

            # Step 4: Create unique ID
            self.log_info("Step 4: Creating unique ID")
            features_df = self.id_transformer.fit_transform(features_df)

            # Step 5: Rename columns for compatibility
            self.log_info("Step 5: Renaming columns")
            nixtla_mapper = {
                self.config.columns.date_col: "ds",
                self.config.columns.target_col: "y",
            }
            features_df = features_df.rename(columns=nixtla_mapper)

            # Step 6: Create targets DataFrame
            self.log_info("Step 6: Creating targets DataFrame")
            targets_df = sales_df.copy()
            targets_df = self.id_transformer.fit_transform(targets_df)
            targets_df = targets_df.rename(columns=nixtla_mapper)

            self.log_info(f"Feature processing completed: {features_df.shape}")
            self.log_info(f"Target processing completed: {targets_df.shape}")

            return features_df, targets_df

        except Exception as e:
            self.log_error(f"Feature processing failed: {str(e)}")
            raise

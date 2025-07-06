"""
Data loading utilities for dataset preprocessing.

This module provides centralized data loading with validation and error handling.
"""

import os
from typing import Any, Dict, Optional

import pandas as pd
from tqdm import tqdm

from .config import Config
from .utils.helpers import get_memory_usage, validate_file_exists
from .utils.logging import LoggerMixin, get_logger


class DataLoader(LoggerMixin):
    """Data loader with validation and error handling."""

    def __init__(self, config: Config):
        """
        Initialize data loader.

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        # self.logger = get_logger("DataLoader")

    def load_calendar_data(self) -> pd.DataFrame:
        """
        Load calendar data with validation.

        Returns:
            Calendar DataFrame

        Raises:
            FileNotFoundError: If calendar file not found
            ValueError: If data validation fails
        """
        self.log_info(f"Loading calendar data from {self.config.data.calendar_path}")

        if not validate_file_exists(self.config.data.calendar_path):
            raise FileNotFoundError(
                f"Calendar file not found: {self.config.data.calendar_path}"
            )

        try:
            df = pd.read_csv(
                self.config.data.calendar_path,
                parse_dates=[self.config.columns.date_col],
            )

            # Validate required columns
            required_cols = ["d", self.config.columns.date_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            self.log_info(f"Loaded calendar data: {df.shape}")
            self.log_info(f"Memory usage: {get_memory_usage(df)}")

            return df

        except Exception as e:
            self.log_error(f"Failed to load calendar data: {str(e)}")
            raise

    def load_sales_data(self) -> pd.DataFrame:
        """
        Load sales data with validation.

        Returns:
            Sales DataFrame

        Raises:
            FileNotFoundError: If sales file not found
            ValueError: If data validation fails
        """
        self.log_info(f"Loading sales data from {self.config.data.sales_path}")

        if not validate_file_exists(self.config.data.sales_path):
            raise FileNotFoundError(
                f"Sales file not found: {self.config.data.sales_path}"
            )

        try:
            df = pd.read_csv(self.config.data.sales_path)

            # Validate required columns
            required_cols = ["dept_id", "store_id"] + self.config.columns.index_cols
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            self.log_info(f"Loaded sales data: {df.shape}")
            self.log_info(f"Memory usage: {get_memory_usage(df)}")

            return df

        except Exception as e:
            self.log_error(f"Failed to load sales data: {str(e)}")
            raise

    def load_prices_data(self) -> pd.DataFrame:
        """
        Load prices data with validation.

        Returns:
            Prices DataFrame

        Raises:
            FileNotFoundError: If prices file not found
            ValueError: If data validation fails
        """
        self.log_info(f"Loading prices data from {self.config.data.prices_path}")

        if not validate_file_exists(self.config.data.prices_path):
            raise FileNotFoundError(
                f"Prices file not found: {self.config.data.prices_path}"
            )

        try:
            df = pd.read_csv(self.config.data.prices_path)

            # Validate required columns
            required_cols = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            self.log_info(f"Loaded prices data: {df.shape}")
            self.log_info(f"Memory usage: {get_memory_usage(df)}")

            return df

        except Exception as e:
            self.log_error(f"Failed to load prices data: {str(e)}")
            raise

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all data sources.

        Returns:
            Dictionary containing all loaded DataFrames

        Raises:
            Exception: If any data loading fails
        """
        self.log_info("Loading all data sources...")

        try:
            data = {
                "calendar": self.load_calendar_data(),
                "sales": self.load_sales_data(),
                "prices": self.load_prices_data(),
            }

            self.log_info("Successfully loaded all data sources")
            return data

        except Exception as e:
            self.log_error(f"Failed to load all data: {str(e)}")
            raise

    def validate_data_integrity(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate data integrity across all sources.

        Args:
            data: Dictionary of loaded DataFrames

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        self.log_info("Validating data integrity...")

        try:
            # Check for empty DataFrames
            for name, df in data.items():
                if df.empty:
                    raise ValueError(f"{name} DataFrame is empty")

            # Validate date ranges
            calendar_df = data["calendar"]
            start_date = pd.to_datetime(self.config.dates.start_date)
            end_date = pd.to_datetime(self.config.dates.end_date)

            calendar_dates = calendar_df[self.config.columns.date_col]
            if calendar_dates.min() > start_date or calendar_dates.max() < end_date:
                raise ValueError("Calendar data doesn't cover required date range")

            # Validate store and department consistency
            sales_df = data["sales"]
            if self.config.filters.store_id not in sales_df["store_id"].values:
                raise ValueError(
                    f"Store {self.config.filters.store_id} not found in sales data"
                )

            if self.config.filters.dept_id not in sales_df["dept_id"].values:
                raise ValueError(
                    f"Department {self.config.filters.dept_id} not found in sales data"
                )

            self.log_info("Data integrity validation passed")
            return True

        except Exception as e:
            self.log_error(f"Data integrity validation failed: {str(e)}")
            raise ValueError(f"Data integrity validation failed: {str(e)}")

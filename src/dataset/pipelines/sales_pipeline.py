"""
Sales data processing pipeline.

This module orchestrates sales data processing including filtering,
wide-to-long conversion, and feature creation.
"""

import pandas as pd

from ..config import Config
from ..transformers.sales import (
    IntervalTransformer,
    PredictabilityFilter,
    SparseTSFilter,
    WideToLongTransformer,
)
from ..utils.helpers import create_date_mappings
from ..utils.logging import LoggerMixin


class SalesPipeline(LoggerMixin):
    """
    Pipeline for processing sales data.

    This pipeline handles sales data filtering, wide-to-long conversion,
    and interval feature creation.
    """

    def __init__(self, config: Config):
        """
        Initialize sales pipeline.

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config

        # Create transformers that don't need date mappings
        self.sparse_filter = SparseTSFilter(
            index_cols=config.columns.index_cols,
            mean_lb=config.filters.mean_lb,
            mean_ub=config.filters.mean_ub,
        )

        self.interval_transformer = IntervalTransformer(
            groupby_cols=config.columns.index_cols,
            target_col=config.columns.target_col,
            set_non_zero_intervals=config.processing.set_non_zero_intervals,
            set_zero_intervals=config.processing.set_zero_intervals,
        )

    def process(
        self, sales_df: pd.DataFrame, calendar_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process sales data.

        Args:
            sales_df: Raw sales DataFrame
            calendar_df: Calendar DataFrame for date mappings

        Returns:
            Processed sales DataFrame
        """
        self.log_info("Processing sales data")

        try:
            # Step 1: Create date mappings from calendar data
            self.log_info("Step 1: Creating date mappings")
            day2date, date2day = create_date_mappings(calendar_df)

            # Step 2: Create transformers that need date mappings
            self.log_info("Step 2: Creating date-dependent transformers")

            # Calculate adjusted start date (subtract max_lag_size days)
            adjusted_start_date = pd.to_datetime(
                self.config.dates.start_date
            ) - pd.Timedelta(days=self.config.dates.max_lag_size)

            predictability_filter = PredictabilityFilter(
                q=self.config.filters.predictability_q,
                start_date=pd.to_datetime(self.config.dates.test_date),
                end_date=pd.to_datetime(self.config.dates.end_date),
                date2day=date2day,
            )

            wide_to_long = WideToLongTransformer(
                index_cols=self.config.columns.index_cols,
                long_dict=day2date,
                long_col=self.config.columns.date_col,
                target_col=self.config.columns.target_col,
            )

            # Step 3: Filter by department and store
            self.log_info("Step 3: Filtering by department and store")
            sales_df = sales_df.query(
                f"dept_id == '{self.config.filters.dept_id}' and "
                f"store_id == '{self.config.filters.store_id}'"
            )

            # Step 4: Apply sparse time series filter
            self.log_info("Step 4: Applying sparse time series filter")
            sales_df = self.sparse_filter.fit_transform(sales_df)

            # Step 5: Apply predictability filter
            self.log_info("Step 5: Applying predictability filter")
            sales_df = predictability_filter.fit_transform(sales_df)

            # Step 6: Convert wide to long format
            self.log_info("Step 6: Converting wide to long format")
            sales_df = wide_to_long.fit_transform(sales_df)

            # Step 7: Create interval features
            self.log_info("Step 7: Creating interval features")
            sales_df = self.interval_transformer.fit_transform(sales_df)

            # Step 8: Filter by date range
            self.log_info("Step 8: Filtering by date range")
            sales_df = sales_df.query(
                f"{self.config.columns.date_col}.between("
                f"'{adjusted_start_date.date()}', "
                f"'{self.config.dates.end_date}')"
            )

            self.log_info(f"Sales processing completed: {sales_df.shape}")
            return sales_df

        except Exception as e:
            self.log_error(f"Sales processing failed: {str(e)}")
            raise

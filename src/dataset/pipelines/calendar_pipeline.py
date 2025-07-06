"""
Calendar data processing pipeline.

This module orchestrates calendar data processing including cleaning,
event processing, and filtering.
"""

import pandas as pd

from ..config import Config
from ..transformers.calendar import CalendarCleaner, EventTransformer, SNAPFilter
from ..utils.logging import LoggerMixin


class CalendarPipeline(LoggerMixin):
    """
    Pipeline for processing calendar data.

    This pipeline handles calendar data cleaning, event processing,
    and filtering operations.
    """

    def __init__(self, config: Config):
        """
        Initialize calendar pipeline.

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config

        # Calculate adjusted start date (subtract max_lag_size days)
        adjusted_start_date = pd.to_datetime(config.dates.start_date) - pd.Timedelta(
            days=config.dates.max_lag_size
        )

        # Create transformers
        self.event_transformer = EventTransformer(
            event_cols=config.columns.event_cols,
            event_baseline="No Event",
            date_col=config.columns.date_col,
            drop_baseline=config.processing.drop_baseline_events,
            drop_event_cols=config.processing.drop_event_cols,
        )

        self.calendar_cleaner = CalendarCleaner(
            date_col=config.columns.date_col,
            start_date=adjusted_start_date.strftime("%Y-%m-%d"),
            end_date=config.dates.end_date,
            drop_cols=["d"],  # Drop day column as it's not needed
        )

        self.snap_filter = SNAPFilter(state_id=config.get_state_id())

    def process(self, calendar_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process calendar data.

        Args:
            calendar_df: Raw calendar DataFrame

        Returns:
            Processed calendar DataFrame
        """
        self.log_info("Processing calendar data")

        try:
            # Step 1: Clean calendar data
            self.log_info("Step 1: Cleaning calendar data")
            calendar_df = self.calendar_cleaner.fit_transform(calendar_df)

            # Step 2: Filter SNAP columns
            self.log_info("Step 2: Filtering SNAP columns")
            calendar_df = self.snap_filter.fit_transform(calendar_df)

            # Step 3: Process events
            self.log_info("Step 3: Processing events")
            calendar_df = self.event_transformer.fit_transform(calendar_df)

            self.log_info(f"Calendar processing completed: {calendar_df.shape}")
            return calendar_df

        except Exception as e:
            self.log_error(f"Calendar processing failed: {str(e)}")
            raise

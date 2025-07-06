"""
Calendar data transformers.

This module contains transformers for processing calendar data including
events, SNAP filters, and calendar cleaning.
"""

from typing import Optional, List
import pandas as pd
from functools import partial

from .base import BaseTransformer


class EventTransformer(BaseTransformer):
    """
    Transform event columns into wide format.
    
    This transformer processes event columns and creates wide format
    event indicators.
    """

    def __init__(
        self,
        event_cols: List[str],
        event_baseline: str = "No Event",
        date_col: str = "date",
        drop_baseline: bool = False,
        drop_event_cols: bool = False
    ):
        """
        Initialize EventTransformer.
        
        Args:
            event_cols: List of event column names
            event_baseline: Baseline event name
            date_col: Date column name
            drop_baseline: Whether to drop baseline events
            drop_event_cols: Whether to drop original event columns
        """
        super().__init__()
        self.event_cols = event_cols
        self.event_baseline = event_baseline
        self.date_col = date_col
        self.drop_baseline = drop_baseline
        self.drop_event_cols = drop_event_cols
        self._merge_events = partial(
            self.merge_events, event_baseline=self.event_baseline
        )
        self._unpivot_events = partial(
            self.unpivot_events,
            event_baseline=self.event_baseline,
            max_events=len(self.event_cols)
        )
        self._wide_event_cols: List[str] = []

    @staticmethod
    def merge_events(event_series: pd.Series, event_baseline: str) -> List[str]:
        """
        Merge events into a list.
        
        Args:
            event_series: Series of events
            event_baseline: Baseline event name
            
        Returns:
            List of events
        """
        events = event_series[pd.notnull(event_series)].tolist()
        return events if events else [event_baseline]

    @staticmethod
    def unpivot_events(s: pd.Series, event_baseline: str, max_events: int) -> List[str]:
        """
        Unpivot events from wide format.
        
        Args:
            s: Series of event indicators
            event_baseline: Baseline event name
            max_events: Maximum number of events
            
        Returns:
            List of events
        """
        event_lst = s.index[s == 1].tolist() if (s == 1).sum() > 0 else [event_baseline] * max_events
        if len(event_lst) < max_events:
            event_lst += [event_baseline] * (max_events - len(event_lst))
        return event_lst

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input data."""
        super()._validate_input(X)
        
        # Check required columns
        missing_cols = [col for col in self.event_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing event columns: {missing_cols}")
        
        if self.date_col not in X.columns:
            raise ValueError(f"Date column '{self.date_col}' not found")

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform event columns."""
        X = X.copy()
        event_col = "all_events"
        
        # Create a column with a list of all events
        X_events = X[self.event_cols].apply(self._merge_events, axis=1).rename(event_col)
        X_events.index = X.index
        
        # Create a long representation
        X_events = pd.concat([X[self.date_col], X_events], axis=1).explode(column=event_col)
        X_wide_events = pd.crosstab(index=X_events[self.date_col], columns=X_events[event_col])
        X_wide_events.index = X.index
        
        if self.drop_baseline:
            X_wide_events = X_wide_events.drop(columns=self.event_baseline)
            self._wide_event_cols = X_wide_events.columns.tolist()
        
        X = pd.concat([X, X_wide_events.set_index(X.index)], axis=1)
        
        if self.drop_event_cols:
            X = X.drop(columns=self.event_cols)
        
        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply inverse transformation.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with original event columns
        """
        X = X.copy()
        if self.drop_event_cols:
            X[self.event_cols] = X[self._wide_event_cols].apply(
                self._unpivot_events, axis=1, result_type="expand"
            )
        X = X.drop(columns=self._wide_event_cols)
        return X


class CalendarCleaner(BaseTransformer):
    """
    Clean and filter calendar data.
    
    This transformer removes irrelevant columns and filters dates.
    """

    def __init__(
        self,
        date_col: str = "date",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        drop_cols: Optional[List[str]] = None
    ):
        """
        Initialize CalendarCleaner.
        
        Args:
            date_col: Date column name
            start_date: Start date for filtering
            end_date: End date for filtering
            drop_cols: Columns to drop
        """
        super().__init__()
        self.date_col = date_col
        self.start_date = start_date
        self.end_date = end_date
        self.drop_cols = drop_cols or []

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean calendar data."""
        X = X.copy()
        
        # Drop specified columns
        if self.drop_cols:
            X = X.drop(columns=self.drop_cols, errors='ignore')
        
        # Filter dates
        if self.start_date or self.end_date:
            if self.start_date and self.end_date:
                X = X.query(f"{self.date_col}.between(@self.start_date, @self.end_date)")
            elif self.start_date:
                X = X.query(f"{self.date_col} >= @self.start_date")
            elif self.end_date:
                X = X.query(f"{self.date_col} <= @self.end_date")
        
        return X


class SNAPFilter(BaseTransformer):
    """
    Filter SNAP columns based on state.
    
    This transformer removes SNAP columns for states other than the specified one.
    """

    def __init__(self, state_id: str):
        """
        Initialize SNAPFilter.
        
        Args:
            state_id: State ID to keep SNAP columns for
        """
        super().__init__()
        self.state_id = state_id

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filter SNAP columns."""
        X = X.copy()
        
        # Find SNAP columns to remove (those not ending with state_id)
        snap_cols = [
            col for col in X.columns 
            if col.startswith("snap_") and not col.endswith(self.state_id)
        ]
        
        if snap_cols:
            X = X.drop(columns=snap_cols)
            self.log_info(f"Dropped SNAP columns: {snap_cols}")
        
        return X
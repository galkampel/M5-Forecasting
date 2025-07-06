"""
Calendar feature engineering for time series preprocessing.

This module provides calendar-based feature engineering including
event features, SNAP features, and seasonal patterns.
"""

import logging
from typing import List

import pandas as pd


class CalendarFeatureEngineer:
    """
    Calendar feature engineering class.

    This class handles calendar-based feature generation including
    events, SNAP benefits, and seasonal patterns.
    """

    def __init__(self, include_events: bool = True, include_snap: bool = True):
        """
        Initialize CalendarFeatureEngineer.

        Args:
            include_events: Whether to include event features
            include_snap: Whether to include SNAP features
        """
        self.include_events = include_events
        self.include_snap = include_snap
        self.logger = logging.getLogger(__name__)
        self.event_features = None
        self.snap_features = None

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the data with calendar features.

        Args:
            X: Input features DataFrame

        Returns:
            DataFrame with calendar features added
        """
        X_transformed = X.copy()

        # Add event features if enabled
        if self.include_events:
            X_transformed = self._add_event_features(X_transformed)

        # Add SNAP features if enabled
        if self.include_snap:
            X_transformed = self._add_snap_features(X_transformed)

        return X_transformed

    def _add_event_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add event-related features.

        Args:
            X: Input features DataFrame

        Returns:
            DataFrame with event features added
        """
        # Check if event column exists
        if "event" not in X.columns:
            self.logger.warning("Event column not found, skipping event features")
            return X

        # Create event type features
        X["is_event"] = (X["event"] != "No Event").astype(int)

        # Create specific event type features
        event_types = X["event"].unique()
        for event_type in event_types:
            if event_type != "No Event":
                col_name = f"event_{event_type.lower().replace(' ', '_')}"
                X[col_name] = (X["event"] == event_type).astype(int)

        self.logger.info(f"Added event features for {len(event_types)} event types")
        return X

    def _add_snap_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add SNAP benefit features.

        Args:
            X: Input features DataFrame

        Returns:
            DataFrame with SNAP features added
        """
        # Check if SNAP column exists
        if "snap" not in X.columns:
            self.logger.warning("SNAP column not found, skipping SNAP features")
            return X

        # Create SNAP features
        X["is_snap"] = X["snap"].astype(int)

        # Create SNAP day of month features
        if "day" in X.columns:
            X["snap_day_1"] = ((X["day"] == 1) & (X["is_snap"] == 1)).astype(int)
            X["snap_day_2"] = ((X["day"] == 2) & (X["is_snap"] == 1)).astype(int)
            X["snap_day_3"] = ((X["day"] == 3) & (X["is_snap"] == 1)).astype(int)

        self.logger.info("Added SNAP features")
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data with fitted calendar features.

        Args:
            X: Input features DataFrame

        Returns:
            DataFrame with calendar features added
        """
        # Calendar features don't require fitting, so transform is same as
        # fit_transform
        return self.fit_transform(X)

    def get_feature_names(self) -> List[str]:
        """
        Get names of generated calendar features.

        Returns:
            List of feature names
        """
        feature_names = []

        if self.include_events:
            feature_names.extend(["is_event"])

        if self.include_snap:
            feature_names.extend(["is_snap", "snap_day_1", "snap_day_2", "snap_day_3"])

        return feature_names

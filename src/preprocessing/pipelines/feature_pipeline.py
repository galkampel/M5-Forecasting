"""
Feature engineering pipeline for preprocessing.

This module provides the main feature engineering pipeline that
orchestrates Fourier features, calendar features, encoding, and scaling.
"""

import logging
from typing import List
import pandas as pd
from pathlib import Path

from ..feature_engineering import (
    FourierFeatureEngineer,
    CalendarFeatureEngineer,
    EncodingFeatureEngineer,
    ScalingFeatureEngineer
)
from ..config import PreprocessingConfig


class FeatureEngineeringPipeline:
    """
    Feature engineering pipeline.
    
    This class orchestrates the complete feature engineering process
    including Fourier features, calendar features, encoding, and scaling.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize FeatureEngineeringPipeline.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize feature engineers
        self.fourier_engineer = None
        self.calendar_engineer = None
        self.encoding_engineer = None
        self.scaling_engineer = None
        
        # Output paths
        self.output_dir = Path(config.data.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, features_df: pd.DataFrame, 
            targets_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            features_df: Input features DataFrame
            targets_df: Input targets DataFrame
            
        Returns:
            Tuple of (processed_features, processed_targets)
        """
        self.logger.info("Starting feature engineering pipeline")
        
        # Step 1: Fourier features
        if self.config.features.fourier.enabled:
            self.logger.info("Adding Fourier features")
            features_df = self._add_fourier_features(features_df)
        
        # Step 2: Calendar features
        if self.config.features.calendar.enabled:
            self.logger.info("Adding calendar features")
            features_df = self._add_calendar_features(features_df)
        
        # Step 3: Encoding features
        if self.config.features.encoding:
            self.logger.info("Adding encoding features")
            features_df = self._add_encoding_features(features_df)
        
        # Step 4: Scaling features
        if self.config.features.scaling.enabled:
            self.logger.info("Adding scaling features")
            features_df = self._add_scaling_features(features_df)
        
        # Save processed features
        self._save_processed_features(features_df, targets_df)
        
        self.logger.info(
            f"Feature engineering completed. "
            f"Final features shape: {features_df.shape}"
        )
        
        return features_df, targets_df
    
    def _add_fourier_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Fourier features to the dataset.
        
        Args:
            features_df: Input features DataFrame
            
        Returns:
            DataFrame with Fourier features added
        """
        fourier_config = self.config.features.fourier
        
        self.fourier_engineer = FourierFeatureEngineer(
            periods=fourier_config.periods,
            orders=fourier_config.orders,
            time_cols=fourier_config.time_cols,
            drop_time_cols=fourier_config.drop_time_cols
        )
        
        return self.fourier_engineer.fit_transform(features_df)
    
    def _add_calendar_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar features to the dataset.
        
        Args:
            features_df: Input features DataFrame
            
        Returns:
            DataFrame with calendar features added
        """
        calendar_config = self.config.features.calendar
        
        self.calendar_engineer = CalendarFeatureEngineer(
            include_events=calendar_config.include_events,
            include_snap=calendar_config.include_snap
        )
        
        return self.calendar_engineer.fit_transform(features_df)
    
    def _add_encoding_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add encoding features to the dataset.
        
        Args:
            features_df: Input features DataFrame
            
        Returns:
            DataFrame with encoding features added
        """
        encoding_config = self.config.features.encoding
        
        self.encoding_engineer = EncodingFeatureEngineer(
            baseline_categories=encoding_config.baseline_categories,
            drop_first=encoding_config.drop_first
        )
        
        return self.encoding_engineer.fit_transform(features_df)
    
    def _add_scaling_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add scaling features to the dataset.
        
        Args:
            features_df: Input features DataFrame
            
        Returns:
            DataFrame with scaling features added
        """
        scaling_config = self.config.features.scaling
        
        self.scaling_engineer = ScalingFeatureEngineer(
            method=scaling_config.method,
            threshold=scaling_config.threshold,
            with_mean=scaling_config.with_mean,
            with_std=scaling_config.with_std
        )
        
        return self.scaling_engineer.fit_transform(features_df)
    
    def _save_processed_features(self, features_df: pd.DataFrame, 
                                targets_df: pd.DataFrame) -> None:
        """
        Save processed features and targets.
        
        Args:
            features_df: Processed features DataFrame
            targets_df: Processed targets DataFrame
        """
        # Save features
        features_path = self.output_dir / "features_2012-01-01_2015-12-31.csv"
        features_df.to_csv(features_path, index=False)
        self.logger.info(f"Saved processed features to {features_path}")
        
        # Save targets
        targets_path = self.output_dir / "target_features_2012-01-01_2015-12-31.csv"
        targets_df.to_csv(targets_path, index=False)
        self.logger.info(f"Saved processed targets to {targets_path}")
        
        # Save feature information
        self._save_feature_info(features_df)
    
    def _save_feature_info(self, features_df: pd.DataFrame) -> None:
        """
        Save feature information for analysis.
        
        Args:
            features_df: Processed features DataFrame
        """
        feature_info = {
            "total_features": len(features_df.columns),
            "feature_names": list(features_df.columns),
            "data_shape": features_df.shape,
            "memory_usage": features_df.memory_usage(deep=True).sum()
        }
        
        # Save as JSON
        import json
        feature_info_path = self.output_dir / "feature_info.json"
        with open(feature_info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        self.logger.info(f"Saved feature info to {feature_info_path}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all generated features.
        
        Returns:
            List of feature names
        """
        feature_names = []
        
        if self.fourier_engineer:
            feature_names.extend(self.fourier_engineer.get_feature_names())
        
        if self.calendar_engineer:
            feature_names.extend(self.calendar_engineer.get_feature_names())
        
        if self.encoding_engineer:
            feature_names.extend(self.encoding_engineer.get_feature_names())
        
        if self.scaling_engineer:
            feature_names.extend(self.scaling_engineer.get_feature_names())
        
        return feature_names 
#!/usr/bin/env python3
"""
Test script for AutoMLForecast pipeline implementation.

This script tests the complete AutoMLForecast pipeline with Ridge and LGBM models.
"""

import pandas as pd
import numpy as np
import tempfile
import sys

# Add src to path
sys.path.append('src')

from preprocessing.config import PreprocessingConfig
from preprocessing.pipelines.modeling_pipeline import ModelTrainingPipeline


def create_sample_data(n_series: int = 10, 
                      n_days: int = 365) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create sample data for testing.
    
    Args:
        n_series: Number of time series
        n_days: Number of days per series
        
    Returns:
        Tuple of (features_df, targets_df)
    """
    # Create sample features
    np.random.seed(42)
    
    # Generate dates
    start_date = pd.Timestamp('2020-01-01')
    dates = pd.date_range(start_date, periods=n_days, freq='D')
    
    # Create features DataFrame
    features_data = []
    for i in range(n_series):
        for date in dates:
            features_data.append({
                'unique_id': f'series_{i}',
                'ds': date,
                'feature_1': np.random.normal(0, 1),
                'feature_2': np.random.normal(0, 1),
                'feature_3': np.random.normal(0, 1),
                'weekday': date.weekday(),
                'month': date.month,
                'year': date.year,
                'event_name': np.random.choice(['none', 'holiday', 'promotion'], 
                                             p=[0.8, 0.1, 0.1])
            })
    
    features_df = pd.DataFrame(features_data)
    
    # Create targets DataFrame with some time series patterns
    targets_data = []
    for i in range(n_series):
        # Create a trend + seasonality pattern
        trend = np.linspace(10, 20, n_days)
        seasonality = 5 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        noise = np.random.normal(0, 1, n_days)
        
        for j, date in enumerate(dates):
            targets_data.append({
                'unique_id': f'series_{i}',
                'ds': date,
                'y': max(0, trend[j] + seasonality[j] + noise[j])
            })
    
    targets_df = pd.DataFrame(targets_data)
    
    return features_df, targets_df


def test_automlforecast_pipeline():
    """Test the AutoMLForecast pipeline."""
    print("Testing AutoMLForecast pipeline...")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample data
        features_df, targets_df = create_sample_data(n_series=5, n_days=100)
        print(f"Created sample data: {features_df.shape}, {targets_df.shape}")
        
        # Create configuration
        config = PreprocessingConfig()
        config.data.output_dir = temp_dir
        config.data.cache_dir = temp_dir
        config.models.ridge.enabled = True
        config.models.lgbm.enabled = True
        config.optimization.enabled = True
        config.optimization.n_trials = 5  # Small number for testing
        config.cv.n_windows = 2  # Small number for testing
        config.cv.forecast_horizon = 7
        
        # Create and run pipeline
        pipeline = ModelTrainingPipeline(config)
        
        try:
            results = pipeline.run(features_df, targets_df)
            print("✅ AutoMLForecast pipeline completed successfully!")
            print(f"Results: {results}")
            
            # Test prediction
            predictions = pipeline.predict(features_df, horizon=7)
            print(f"✅ Predictions generated: {predictions.shape}")
            
        except Exception as e:
            print(f"❌ AutoMLForecast pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


if __name__ == "__main__":
    success = test_automlforecast_pipeline()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1) 
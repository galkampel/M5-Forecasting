"""
Data processing module for Streamlit app.

This module handles dataset creation and loading using the src/dataset pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.dataset.main import main as dataset_main
    from src.dataset.config import Config
    from src.dataset.data_loader import DataLoader
except ImportError as e:
    st.error(f"Error importing dataset modules: {e}")


@st.cache_data
def create_dataset_interface():
    """
    Creates the dataset processing interface:
    - Dataset creation using src/dataset pipelines
    - Data quality assessment
    - Intermittent series identification
    - Data preview and statistics
    """
    try:
        # Load configuration
        config_path = Path("config/default.yaml")
        if not config_path.exists():
            st.error("Configuration file not found. Please ensure config/default.yaml exists.")
            return None, None
        
        config = Config.from_file(str(config_path))
        
        # Create dataset using the dataset pipeline
        dataset_main(config)
        
        # Load processed data
        features_df, targets_df = load_processed_data()
        
        return features_df, targets_df
        
    except Exception as e:
        st.error(f"Error creating dataset: {str(e)}")
        return None, None


@st.cache_data
def load_processed_data():
    """
    Load processed data from the data/processed directory.
    
    Returns:
        tuple: (features_df, targets_df) - Processed features and targets dataframes
    """
    try:
        processed_dir = Path("data/processed")
        
        # Look for processed files
        features_file = processed_dir / "features_2012-01-01_2015-12-31.csv"
        targets_file = processed_dir / "target_features_2012-01-01_2015-12-31.csv"
        
        # If specific files don't exist, look for any CSV files
        if not features_file.exists():
            csv_files = list(processed_dir.glob("*.csv"))
            if csv_files:
                features_file = csv_files[0]
            else:
                st.error("No processed data files found in data/processed/")
                return None, None
        
        if not targets_file.exists():
            csv_files = list(processed_dir.glob("*target*.csv"))
            if csv_files:
                targets_file = csv_files[0]
            else:
                # Use the same file for both if no target file found
                targets_file = features_file
        
        # Load data
        features_df = pd.read_csv(features_file)
        targets_df = pd.read_csv(targets_file)
        
        # Convert 'ds' column to datetime in both dataframes
        if 'ds' in features_df.columns:
            features_df['ds'] = pd.to_datetime(features_df['ds'])
        if 'ds' in targets_df.columns:
            targets_df['ds'] = pd.to_datetime(targets_df['ds'])
        
        st.success(f"Loaded data: {features_df.shape} features, {targets_df.shape} targets")
        
        return features_df, targets_df
        
    except Exception as e:
        st.error(f"Error loading processed data: {str(e)}")
        return None, None


def identify_intermittent_series(targets_df):
    """
    Identify intermittent time series based on zero percentage.
    
    Args:
        targets_df: DataFrame with target values
        
    Returns:
        DataFrame: Series with intermittent classification
    """
    if targets_df is None or targets_df.empty:
        return pd.DataFrame()
    
    # Calculate zero percentage for each series
    if 'unique_id' in targets_df.columns and 'y' in targets_df.columns:
        zero_stats = targets_df.groupby('unique_id').agg({
            'y': lambda x: (x == 0).sum() / len(x) * 100
        }).rename(columns={'y': 'zero_percentage'})
        
        # Classify as intermittent if > 50% zeros
        zero_stats['is_intermittent'] = zero_stats['zero_percentage'] > 50
        zero_stats['demand_type'] = zero_stats['is_intermittent'].map({
            True: 'Intermittent',
            False: 'Regular'
        })
        
        return zero_stats
    
    return pd.DataFrame()


def calculate_demand_statistics(targets_df):
    """
    Calculate demand statistics for each series.
    
    Args:
        targets_df: DataFrame with target values
        
    Returns:
        DataFrame: Series with demand statistics
    """
    if targets_df is None or targets_df.empty:
        return pd.DataFrame()
    
    if 'unique_id' in targets_df.columns and 'y' in targets_df.columns:
        # Calculate demand statistics
        demand_stats = targets_df.groupby('unique_id').agg({
            'y': ['mean', 'std', 'min', 'max', 'count']
        }).round(2)
        
        # Flatten column names
        demand_stats.columns = ['_'.join(col).strip() for col in demand_stats.columns]
        
        # Calculate coefficient of variation
        demand_stats['cv'] = (demand_stats['y_std'] / demand_stats['y_mean']).round(3)
        
        # Calculate zero percentage
        zero_counts = targets_df.groupby('unique_id')['y'].apply(lambda x: (x == 0).sum())
        total_counts = targets_df.groupby('unique_id')['y'].count()
        demand_stats['zero_percentage'] = (zero_counts / total_counts * 100).round(2)
        
        return demand_stats
    
    return pd.DataFrame()


def analyze_price_correlation(features_df, targets_df):
    """
    Analyze correlation between prices and demand.
    
    Args:
        features_df: DataFrame with features including prices
        targets_df: DataFrame with target values
        
    Returns:
        DataFrame: Price-demand correlation analysis
    """
    if features_df is None or targets_df is None:
        return pd.DataFrame()
    
    try:
        # Merge features and targets
        merged_df = features_df.merge(targets_df, on=['unique_id', 'ds'], how='inner')
        
        # Find price columns
        price_cols = [col for col in merged_df.columns if 'price' in col.lower()]
        
        if not price_cols:
            return pd.DataFrame()
        
        # Calculate correlations
        correlations = {}
        for price_col in price_cols:
            if price_col in merged_df.columns and 'y' in merged_df.columns:
                corr = merged_df[price_col].corr(merged_df['y'])
                correlations[price_col] = corr
        
        return pd.DataFrame(list(correlations.items()), columns=['Price_Column', 'Correlation'])
        
    except Exception as e:
        st.error(f"Error analyzing price correlation: {str(e)}")
        return pd.DataFrame()


def create_data_quality_report(features_df, targets_df):
    """
    Create a comprehensive data quality report.
    
    Args:
        features_df: DataFrame with features
        targets_df: DataFrame with targets
        
    Returns:
        dict: Data quality metrics
    """
    quality_report = {}
    
    try:
        # Features quality
        if features_df is not None:
            quality_report['features'] = {
                'shape': features_df.shape,
                'missing_values': features_df.isnull().sum().sum(),
                'missing_percentage': (features_df.isnull().sum().sum() / 
                                     (features_df.shape[0] * features_df.shape[1])) * 100,
                'duplicate_rows': features_df.duplicated().sum(),
                'numeric_columns': features_df.select_dtypes(include=[np.number]).shape[1],
                'categorical_columns': features_df.select_dtypes(include=['object']).shape[1]
            }
        
        # Targets quality
        if targets_df is not None:
            quality_report['targets'] = {
                'shape': targets_df.shape,
                'missing_values': targets_df.isnull().sum().sum(),
                'missing_percentage': (targets_df.isnull().sum().sum() / 
                                     (targets_df.shape[0] * targets_df.shape[1])) * 100,
                'duplicate_rows': targets_df.duplicated().sum(),
                'unique_series': len(targets_df['unique_id'].unique()) if 'unique_id' in targets_df.columns else 0,
                'date_range': {
                    'start': targets_df['ds'].min() if 'ds' in targets_df.columns else None,
                    'end': targets_df['ds'].max() if 'ds' in targets_df.columns else None
                }
            }
        
        return quality_report
        
    except Exception as e:
        st.error(f"Error creating data quality report: {str(e)}")
        return {} 
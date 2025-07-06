"""
Helper functions for dataset preprocessing.

This module contains utility functions used throughout the preprocessing pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any
import pandas as pd


def create_dict_from_df(df: pd.DataFrame, key_col: str, val_col: str) -> Dict[Any, Any]:
    """
    Create a dictionary from DataFrame columns.
    
    Args:
        df: Input DataFrame
        key_col: Column to use as dictionary keys
        val_col: Column to use as dictionary values
        
    Returns:
        Dictionary mapping key_col values to val_col values
        
    Raises:
        ValueError: If key_col or val_col not found in DataFrame
    """
    if key_col not in df.columns:
        raise ValueError(f"Key column '{key_col}' not found in DataFrame")
    if val_col not in df.columns:
        raise ValueError(f"Value column '{val_col}' not found in DataFrame")
    
    return df.set_index(key_col)[val_col].to_dict()


def ensure_directory(path: str) -> str:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Normalized path string
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(Path(path).resolve())


def validate_file_exists(file_path: str) -> bool:
    """
    Validate that a file exists.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file exists, False otherwise
    """
    return os.path.exists(file_path)


def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Get memory usage of DataFrame in human readable format.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Memory usage string (e.g., "1.2 GB")
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / 1024 / 1024
    if memory_mb > 1024:
        return f"{memory_mb / 1024:.1f} GB"
    else:
        return f"{memory_mb:.1f} MB"


def chunk_dataframe(df: pd.DataFrame, chunk_size: int):
    """
    Generator to yield DataFrame chunks.
    
    Args:
        df: DataFrame to chunk
        chunk_size: Size of each chunk
        
    Yields:
        DataFrame chunks
    """
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size]


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Safely merge DataFrames with error handling.
    
    Args:
        left: Left DataFrame
        right: Right DataFrame
        **kwargs: Additional arguments for pd.merge
        
    Returns:
        Merged DataFrame
        
    Raises:
        ValueError: If merge fails or produces unexpected results
    """
    try:
        result = pd.merge(left, right, **kwargs)
        if len(result) == 0:
            raise ValueError("Merge produced empty result")
        return result
    except Exception as e:
        raise ValueError(f"Merge failed: {str(e)}")


def validate_date_range(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Validate and parse date range.
    
    Args:
        start_date: Start date string
        end_date: End date string
        
    Returns:
        Tuple of (start_timestamp, end_timestamp)
        
    Raises:
        ValueError: If dates are invalid or start_date >= end_date
    """
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}")
    
    if start >= end:
        raise ValueError("start_date must be before end_date")
    
    return start, end


def create_date_mappings(calendar_df: pd.DataFrame) -> tuple[Dict[str, pd.Timestamp], Dict[pd.Timestamp, str]]:
    """
    Create date mappings from calendar DataFrame.
    
    Args:
        calendar_df: Calendar DataFrame with 'd' and 'date' columns
        
    Returns:
        Tuple of (day2date, date2day) mappings
        
    Raises:
        ValueError: If required columns not found
    """
    required_cols = ['d', 'date']
    missing_cols = [col for col in required_cols if col not in calendar_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    day2date = create_dict_from_df(calendar_df, 'd', 'date')
    date2day = {date: day for day, date in day2date.items()}
    
    return day2date, date2day
"""
Configuration management for dataset preprocessing.

This module provides configuration loading, validation, and management
using Pydantic models for type safety and validation.
"""

import os
from pathlib import Path
from typing import List

import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Configuration for data paths."""

    calendar_path: str = Field(..., description="Path to calendar CSV file")
    sales_path: str = Field(..., description="Path to sales CSV file")
    prices_path: str = Field(..., description="Path to prices CSV file")
    output_dir: str = Field(..., description="Output directory for processed data")

    @field_validator("calendar_path", "sales_path", "prices_path")
    @classmethod
    def validate_file_paths(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"File does not exist: {v}")
        return v

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class FiltersConfig(BaseModel):
    """Configuration for data filtering."""

    dept_id: str = Field(..., description="Department ID to filter")
    store_id: str = Field(..., description="Store ID to filter")
    mean_lb: float = Field(..., ge=0, description="Lower bound for mean sales")
    mean_ub: float = Field(..., ge=0, description="Upper bound for mean sales")
    predictability_q: float = Field(
        ..., ge=0, le=1, description="Quantile for predictability filter"
    )

    @field_validator("mean_ub")
    @classmethod
    def validate_mean_bounds(cls, v, info):
        if "mean_lb" in info.data and v <= info.data["mean_lb"]:
            raise ValueError("mean_ub must be greater than mean_lb")
        return v


class DatesConfig(BaseModel):
    """Configuration for date ranges."""

    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    test_date: str = Field(..., description="Test date for predictability filter")
    max_lag_size: int = Field(..., ge=0, description="Maximum lag size for features")

    @field_validator("start_date", "end_date", "test_date")
    @classmethod
    def validate_date_format(cls, v):
        try:
            pd.to_datetime(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Use YYYY-MM-DD format.")
        return v

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, v, info):
        if "start_date" in info.data:
            start = pd.to_datetime(info.data["start_date"])
            end = pd.to_datetime(v)
            if end <= start:
                raise ValueError("end_date must be after start_date")
        return v

    @field_validator("test_date")
    @classmethod
    def validate_test_date(cls, v, info):
        if "start_date" in info.data and "end_date" in info.data:
            start = pd.to_datetime(info.data["start_date"])
            end = pd.to_datetime(info.data["end_date"])
            test = pd.to_datetime(v)
            if test < start or test > end:
                raise ValueError("test_date must be between start_date and end_date")
        return v


class ColumnsConfig(BaseModel):
    """Configuration for column mappings."""

    target_col: str = Field(..., description="Target column name")
    date_col: str = Field(..., description="Date column name")
    index_cols: List[str] = Field(..., description="Index columns for grouping")
    event_cols: List[str] = Field(..., description="Event column names")


class ProcessingConfig(BaseModel):
    """Configuration for processing options."""

    drop_baseline_events: bool = Field(
        ..., description="Whether to drop baseline events"
    )
    drop_event_cols: bool = Field(
        ..., description="Whether to drop original event columns"
    )
    set_non_zero_intervals: bool = Field(
        ..., description="Whether to create non-zero interval features"
    )
    set_zero_intervals: bool = Field(
        ..., description="Whether to create zero interval features"
    )


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = Field(..., description="Logging level")
    format: str = Field(..., description="Log format string")
    file: str = Field(..., description="Log file path")

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


class PerformanceConfig(BaseModel):
    """Configuration for performance settings."""

    chunk_size: int = Field(..., ge=1, description="Chunk size for processing")
    max_memory_usage: str = Field(..., description="Maximum memory usage")
    parallel_processing: bool = Field(
        ..., description="Whether to use parallel processing"
    )
    num_workers: int = Field(
        ..., ge=1, description="Number of workers for parallel processing"
    )


class Config(BaseModel):
    """Main configuration class."""

    data: DataConfig
    filters: FiltersConfig
    dates: DatesConfig
    columns: ColumnsConfig
    processing: ProcessingConfig
    logging: LoggingConfig
    performance: PerformanceConfig

    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def save(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def get_state_id(self) -> str:
        """Extract state ID from store ID."""
        return self.filters.store_id.split("_")[0]

    def get_date_range_str(self) -> str:
        """Get date range string for file naming."""
        return f"{self.dates.start_date}_{self.dates.end_date}"

    def get_output_paths(self) -> tuple[str, str]:
        """Get output file paths for features and targets."""
        date_range = self.get_date_range_str()
        features_path = os.path.join(self.data.output_dir, f"features_{date_range}.csv")
        targets_path = os.path.join(
            self.data.output_dir, f"target_features_{date_range}.csv"
        )
        return features_path, targets_path

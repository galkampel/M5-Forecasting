"""
Tests for configuration module.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.dataset.config import Config, DataConfig, FiltersConfig


class TestDataConfig:
    """Test DataConfig class."""

    def test_valid_config(self):
        """Test valid data configuration."""
        config = DataConfig(
            calendar_path="data/calendar.csv",
            sales_path="data/sales_train_validation.csv",
            prices_path="data/sell_prices.csv",
            output_dir="data/processed",
        )

        assert config.calendar_path == "data/calendar.csv"
        assert config.sales_path == "data/sales_train_validation.csv"
        assert config.prices_path == "data/sell_prices.csv"
        assert config.output_dir == "data/processed"

    def test_invalid_file_path(self):
        """Test invalid file path validation."""
        with pytest.raises(ValueError, match="File does not exist"):
            DataConfig(
                calendar_path="nonexistent.csv",
                sales_path="data/sales_train_validation.csv",
                prices_path="data/sell_prices.csv",
                output_dir="data/processed",
            )


class TestFiltersConfig:
    """Test FiltersConfig class."""

    def test_valid_filters(self):
        """Test valid filters configuration."""
        config = FiltersConfig(
            dept_id="FOODS_3",
            store_id="CA_1",
            mean_lb=0.15,
            mean_ub=0.5,
            predictability_q=0.85,
        )

        assert config.dept_id == "FOODS_3"
        assert config.store_id == "CA_1"
        assert config.mean_lb == 0.15
        assert config.mean_ub == 0.5
        assert config.predictability_q == 0.85

    def test_invalid_mean_bounds(self):
        """Test invalid mean bounds validation."""
        with pytest.raises(ValueError, match="mean_ub must be greater than mean_lb"):
            FiltersConfig(
                dept_id="FOODS_3",
                store_id="CA_1",
                mean_lb=0.5,
                mean_ub=0.15,  # Invalid: upper bound < lower bound
                predictability_q=0.85,
            )


class TestConfig:
    """Test main Config class."""

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "data": {
                "calendar_path": "data/calendar.csv",
                "sales_path": "data/sales_train_validation.csv",
                "prices_path": "data/sell_prices.csv",
                "output_dir": "data/processed",
            },
            "filters": {
                "dept_id": "FOODS_3",
                "store_id": "CA_1",
                "mean_lb": 0.15,
                "mean_ub": 0.5,
                "predictability_q": 0.85,
            },
            "dates": {
                "start_date": "2012-01-01",
                "end_date": "2015-12-31",
                "test_date": "2015-07-01",
                "max_lag_size": 30,
            },
            "columns": {
                "target_col": "sales",
                "date_col": "date",
                "index_cols": ["store_id", "item_id"],
                "event_cols": ["event_name_1", "event_name_2"],
            },
            "processing": {
                "drop_baseline_events": True,
                "drop_event_cols": False,
                "set_non_zero_intervals": True,
                "set_zero_intervals": True,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/dataset_processing.log",
            },
            "performance": {
                "chunk_size": 10000,
                "max_memory_usage": "4GB",
                "parallel_processing": True,
                "num_workers": 4,
            },
        }

        config = Config.from_dict(config_dict)

        assert config.data.calendar_path == "data/calendar.csv"
        assert config.filters.dept_id == "FOODS_3"
        assert config.dates.start_date == "2012-01-01"
        assert config.columns.target_col == "sales"

    def test_from_file(self):
        """Test creating config from YAML file."""
        config_dict = {
            "data": {
                "calendar_path": "data/calendar.csv",
                "sales_path": "data/sales_train_validation.csv",
                "prices_path": "data/sell_prices.csv",
                "output_dir": "data/processed",
            },
            "filters": {
                "dept_id": "FOODS_3",
                "store_id": "CA_1",
                "mean_lb": 0.15,
                "mean_ub": 0.5,
                "predictability_q": 0.85,
            },
            "dates": {
                "start_date": "2012-01-01",
                "end_date": "2015-12-31",
                "test_date": "2015-07-01",
                "max_lag_size": 30,
            },
            "columns": {
                "target_col": "sales",
                "date_col": "date",
                "index_cols": ["store_id", "item_id"],
                "event_cols": ["event_name_1", "event_name_2"],
            },
            "processing": {
                "drop_baseline_events": True,
                "drop_event_cols": False,
                "set_non_zero_intervals": True,
                "set_zero_intervals": True,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/dataset_processing.log",
            },
            "performance": {
                "chunk_size": 10000,
                "max_memory_usage": "4GB",
                "parallel_processing": True,
                "num_workers": 4,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            temp_file = f.name

        try:
            config = Config.from_file(temp_file)

            assert config.data.calendar_path == "data/calendar.csv"
            assert config.filters.dept_id == "FOODS_3"
            assert config.dates.start_date == "2012-01-01"
            assert config.columns.target_col == "sales"

        finally:
            Path(temp_file).unlink()

    def test_get_state_id(self):
        """Test get_state_id method."""
        config_dict = {
            "data": {
                "calendar_path": "data/calendar.csv",
                "sales_path": "data/sales_train_validation.csv",
                "prices_path": "data/sell_prices.csv",
                "output_dir": "data/processed",
            },
            "filters": {
                "dept_id": "FOODS_3",
                "store_id": "CA_1",
                "mean_lb": 0.15,
                "mean_ub": 0.5,
                "predictability_q": 0.85,
            },
            "dates": {
                "start_date": "2012-01-01",
                "end_date": "2015-12-31",
                "test_date": "2015-07-01",
                "max_lag_size": 30,
            },
            "columns": {
                "target_col": "sales",
                "date_col": "date",
                "index_cols": ["store_id", "item_id"],
                "event_cols": ["event_name_1", "event_name_2"],
            },
            "processing": {
                "drop_baseline_events": True,
                "drop_event_cols": False,
                "set_non_zero_intervals": True,
                "set_zero_intervals": True,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/dataset_processing.log",
            },
            "performance": {
                "chunk_size": 10000,
                "max_memory_usage": "4GB",
                "parallel_processing": True,
                "num_workers": 4,
            },
        }

        config = Config.from_dict(config_dict)
        assert config.get_state_id() == "CA"

    def test_get_date_range_str(self):
        """Test get_date_range_str method."""
        config_dict = {
            "data": {
                "calendar_path": "data/calendar.csv",
                "sales_path": "data/sales_train_validation.csv",
                "prices_path": "data/sell_prices.csv",
                "output_dir": "data/processed",
            },
            "filters": {
                "dept_id": "FOODS_3",
                "store_id": "CA_1",
                "mean_lb": 0.15,
                "mean_ub": 0.5,
                "predictability_q": 0.85,
            },
            "dates": {
                "start_date": "2012-01-01",
                "end_date": "2015-12-31",
                "test_date": "2015-07-01",
                "max_lag_size": 30,
            },
            "columns": {
                "target_col": "sales",
                "date_col": "date",
                "index_cols": ["store_id", "item_id"],
                "event_cols": ["event_name_1", "event_name_2"],
            },
            "processing": {
                "drop_baseline_events": True,
                "drop_event_cols": False,
                "set_non_zero_intervals": True,
                "set_zero_intervals": True,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/dataset_processing.log",
            },
            "performance": {
                "chunk_size": 10000,
                "max_memory_usage": "4GB",
                "parallel_processing": True,
                "num_workers": 4,
            },
        }

        config = Config.from_dict(config_dict)
        assert config.get_date_range_str() == "2012-01-01_2015-12-31"

"""
Configuration management for preprocessing pipeline.

This module provides configuration classes and utilities for managing
preprocessing parameters, model settings, and pipeline configurations.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class DataConfig:
    """Configuration for data loading and paths."""

    features_path: str = "data/processed/features_2012-01-01_2015-12-31.csv"
    targets_path: str = "data/processed/target_features_2012-01-01_2015-12-31.csv"
    output_dir: str = "outputs/preprocessing"
    cache_dir: str = "cache"

    def __post_init__(self):
        """Validate paths after initialization."""
        for path_attr in ["features_path", "targets_path"]:
            path = getattr(self, path_attr)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")


@dataclass
class FourierConfig:
    """Configuration for Fourier feature engineering."""

    enabled: bool = True
    periods: List[float] = field(default_factory=lambda: [365.25, 7])
    orders: List[int] = field(default_factory=lambda: [1, 1])
    time_cols: List[str] = field(
        default_factory=lambda: ["day_of_year", "week_of_year"]
    )
    drop_time_cols: bool = True


@dataclass
class CalendarConfig:
    """Configuration for calendar feature engineering."""

    enabled: bool = True
    include_events: bool = True
    include_snap: bool = True


@dataclass
class EncodingConfig:
    """Configuration for categorical encoding."""

    baseline_categories: Dict[str, Union[str, int]] = field(
        default_factory=lambda: {
            "weekday": "Monday",
            "month": 1,
            "year": 2012,
            "event": "No Event",
        }
    )
    drop_first: bool = True


@dataclass
class ScalingConfig:
    """Configuration for feature scaling."""

    enabled: bool = True
    method: str = "standard"
    threshold: float = 0.0
    with_mean: bool = True
    with_std: bool = True


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    fourier: FourierConfig = field(default_factory=FourierConfig)
    calendar: CalendarConfig = field(default_factory=CalendarConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)


@dataclass
class ModelParams:
    """Base model parameters."""

    lags: List[int] = field(default_factory=lambda: [1, 7, 14, 30])
    lag_transforms: Dict[int, List[str]] = field(
        default_factory=lambda: {
            1: ["RollingMean", "RollingStd"],
            7: ["RollingMean", "RollingStd"],
            14: ["RollingMean"],
            30: ["RollingMean"],
        }
    )


@dataclass
class RidgeConfig(ModelParams):
    """Configuration for Ridge regression."""

    enabled: bool = True
    best_model_params: Dict[str, Any] = field(default_factory=lambda: {"alpha": 1.0})


@dataclass
class LGBMConfig(ModelParams):
    """Configuration for LightGBM."""

    enabled: bool = True
    best_model_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
        }
    )


@dataclass
class ModelsConfig:
    """Configuration for all models."""

    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    lgbm: LGBMConfig = field(default_factory=LGBMConfig)


@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation."""

    method: str = "expanding_window"
    initial_window: int = 730  # 2 years
    step_length: int = 30  # monthly steps
    forecast_horizon: int = 31  # 1 month ahead
    n_windows: int = 5  # number of cross-validation windows


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    metrics: List[str] = field(
        default_factory=lambda: [
            "mae",
            "rmse",
            "mape",
            "mase",
            "rmsse",
            "mrae",
            "f1_zero",
            "non_zero_mae",
        ]
    )
    save_predictions: bool = True
    save_models: bool = False
    confidence_intervals: bool = False
    feature_importance: bool = True


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""

    enabled: bool = False
    framework: str = "optuna"
    n_trials: int = 100
    timeout: int = 3600  # 1 hour
    study_name: str = "preprocessing_optimization"


@dataclass
class CachingConfig:
    """Configuration for caching."""

    enabled: bool = True
    cache_dir: str = "../cache"
    max_size: str = "2GB"
    ttl: int = 7200  # 2 hours


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    file: str = "../logs/preprocessing.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class PreprocessingConfig:
    """Main configuration class for preprocessing pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.data = DataConfig()
        self.features = FeatureConfig()
        self.models = ModelsConfig()
        self.cv = CrossValidationConfig()
        self.evaluation = EvaluationConfig()
        self.optimization = OptimizationConfig()
        self.caching = CachingConfig()
        self.logging = LoggingConfig()

        if config_path:
            self.load_from_file(config_path)

    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        self._update_from_dict(config_data)

    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.

        Args:
            config_data: Configuration dictionary
        """
        # Update data configuration
        if "data" in config_data:
            for key, value in config_data["data"].items():
                setattr(self.data, key, value)

        # Update features configuration
        if "features" in config_data:
            self._update_features_config(config_data["features"])

        # Update models configuration
        if "models" in config_data:
            self._update_models_config(config_data["models"])

        # Update other configurations
        config_names = ["cv", "evaluation", "optimization", "caching", "logging"]
        for config_name in config_names:
            if config_name in config_data:
                config_obj = getattr(self, config_name)
                for key, value in config_data[config_name].items():
                    setattr(config_obj, key, value)

    def _update_features_config(self, features_data: Dict[str, Any]) -> None:
        """Update features configuration."""
        if "fourier" in features_data:
            for key, value in features_data["fourier"].items():
                setattr(self.features.fourier, key, value)

        if "calendar" in features_data:
            for key, value in features_data["calendar"].items():
                setattr(self.features.calendar, key, value)

        if "encoding" in features_data:
            for key, value in features_data["encoding"].items():
                setattr(self.features.encoding, key, value)

        if "scaling" in features_data:
            for key, value in features_data["scaling"].items():
                setattr(self.features.scaling, key, value)

    def _update_models_config(self, models_data: Dict[str, Any]) -> None:
        """Update models configuration."""
        if "ridge" in models_data:
            for key, value in models_data["ridge"].items():
                setattr(self.models.ridge, key, value)

        if "lgbm" in models_data:
            for key, value in models_data["lgbm"].items():
                setattr(self.models.lgbm, key, value)

    def save_to_file(self, config_path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            config_path: Path to save configuration file
        """
        config_dict = self.to_dict()

        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "data": self.data.__dict__,
            "features": {
                "fourier": self.features.fourier.__dict__,
                "calendar": self.features.calendar.__dict__,
                "encoding": self.features.encoding.__dict__,
                "scaling": self.features.scaling.__dict__,
            },
            "models": {
                "ridge": self.models.ridge.__dict__,
                "lgbm": self.models.lgbm.__dict__,
            },
            "cv": self.cv.__dict__,
            "evaluation": self.evaluation.__dict__,
            "optimization": self.optimization.__dict__,
            "caching": self.caching.__dict__,
            "logging": self.logging.__dict__,
        }

    def validate(self) -> None:
        """Validate configuration parameters."""
        # Validate data paths
        if not os.path.exists(self.data.features_path):
            raise ValueError(f"Features file not found: {self.data.features_path}")

        if not os.path.exists(self.data.targets_path):
            raise ValueError(f"Targets file not found: {self.data.targets_path}")

        # Validate feature engineering parameters
        if self.features.fourier.enabled:
            if len(self.features.fourier.periods) != len(self.features.fourier.orders):
                raise ValueError("Fourier periods and orders must have same length")

        # Validate model parameters
        if not self.models.ridge.enabled and not self.models.lgbm.enabled:
            raise ValueError("At least one model must be enabled")

        # Validate cross-validation parameters
        if self.cv.initial_window <= 0:
            raise ValueError("Initial window must be positive")

        if self.cv.step_length <= 0:
            raise ValueError("Step length must be positive")

        if self.cv.forecast_horizon <= 0:
            raise ValueError("Forecast horizon must be positive")

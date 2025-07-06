"""
MLForecast parameters and best model configuration.

This module provides classes for managing MLForecast parameters
and extracting best models from AutoMLForecast results.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from sklearn.linear_model import Ridge


@dataclass
class MLForecastParams:
    """
    MLForecast parameters for model configuration.

    This class manages parameters for MLForecast models including
    lags, lag transforms, and model-specific configurations.
    """

    model_name: str
    freq: str = "D"
    lags: List[int] = None
    lag_transforms: Dict[int, List[str]] = None
    auto_mlf_results_dict: Dict[str, Any] = None
    pipeline: Any = None

    def __post_init__(self):
        """Set default values after initialization."""
        if self.lags is None:
            self.lags = [1, 7, 14, 30]

        if self.lag_transforms is None:
            self.lag_transforms = {
                1: ["RollingMean", "RollingStd"],
                7: ["RollingMean", "RollingStd"],
                14: ["RollingMean"],
                30: ["RollingMean"],
            }

    def get_best_model(self):
        """
        Get the best model from AutoMLForecast results.

        Returns:
            Best model instance
        """
        if self.auto_mlf_results_dict is None:
            # Return default model if no results available
            if self.model_name == "Ridge":
                return Ridge(fit_intercept=True, random_state=42)
            elif self.model_name == "LGBMRegressor":
                return LGBMRegressor(
                    objective="tweedie", tweedie_variance_power=1.8, random_state=42
                )
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

        # Extract best parameters from AutoMLForecast results
        best_params = self.auto_mlf_results_dict.get("best_params", {})

        if self.model_name == "Ridge":
            model = Ridge(fit_intercept=True, random_state=42)
            if "alpha" in best_params:
                model.alpha = best_params["alpha"]
        elif self.model_name == "LGBMRegressor":
            model = LGBMRegressor(objective="tweedie", random_state=42)
            if "learning_rate" in best_params:
                model.learning_rate = best_params["learning_rate"]
            if "n_estimators" in best_params:
                model.n_estimators = best_params["n_estimators"]
            if "max_depth" in best_params:
                model.max_depth = best_params["max_depth"]
            if "tweedie_variance_power" in best_params:
                model.tweedie_variance_power = best_params["tweedie_variance_power"]
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        return model

    def create_mlforecast(self) -> MLForecast:
        """
        Create MLForecast instance with best model.

        Returns:
            MLForecast instance
        """
        best_model = self.get_best_model()

        return MLForecast(
            models=best_model,
            freq=self.freq,
            lags=self.lags,
            lag_transforms=self.lag_transforms,
        )

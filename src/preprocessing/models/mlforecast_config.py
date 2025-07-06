"""
MLForecast configuration classes for preprocessing pipeline.

This module provides configuration classes for MLForecast-based models
including Ridge regression and LightGBM.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .base import BaseModelConfig


class MLForecastParams(BaseModel):
    """
    MLForecast parameters configuration.

    This class defines the parameters for MLForecast models.
    """

    lags: List[int] = Field(default=[1, 7, 14, 30])
    lag_transforms: Dict[int, List[str]] = Field(
        default={
            1: ["RollingMean", "RollingStd"],
            7: ["RollingMean", "RollingStd"],
            14: ["RollingMean"],
            30: ["RollingMean"],
        }
    )
    date_col: str = Field(default="date")
    id_col: str = Field(default="id")
    target_col: str = Field(default="target")
    freq: str = Field(default="D")
    num_threads: int = Field(default=1)
    random_state: int = Field(default=42)


class RidgeConfig(BaseModelConfig):
    """
    Configuration for Ridge regression model.
    """

    def __init__(
        self,
        enabled: bool = True,
        lags: Optional[List[int]] = None,
        lag_transforms: Optional[Dict[int, List[str]]] = None,
        best_model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RidgeConfig.

        Args:
            enabled: Whether the model is enabled
            lags: List of lag values
            lag_transforms: Dictionary of lag transformations
            best_model_params: Best model parameters
        """
        super().__init__("ridge", enabled)

        self.lags = lags or [1, 7, 14, 30]
        self.lag_transforms = lag_transforms or {
            1: ["RollingMean", "RollingStd"],
            7: ["RollingMean", "RollingStd"],
            14: ["RollingMean"],
            30: ["RollingMean"],
        }
        self.best_model_params = best_model_params or {"alpha": 1.0}

    def get_params(self) -> Dict[str, Any]:
        """
        Get Ridge model parameters.

        Returns:
            Dictionary of Ridge parameters
        """
        return {
            "lags": self.lags,
            "lag_transforms": self.lag_transforms,
            "best_model_params": self.best_model_params,
        }

    def validate(self) -> bool:
        """
        Validate Ridge configuration.

        Returns:
            True if configuration is valid
        """
        try:
            # Validate lags
            if not isinstance(self.lags, list) or len(self.lags) == 0:
                self.logger.error("Lags must be a non-empty list")
                return False

            # Validate lag_transforms
            if not isinstance(self.lag_transforms, dict):
                self.logger.error("Lag transforms must be a dictionary")
                return False

            # Validate best_model_params
            if not isinstance(self.best_model_params, dict):
                self.logger.error("Best model params must be a dictionary")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False


class LGBMConfig(BaseModelConfig):
    """
    Configuration for LightGBM model.
    """

    def __init__(
        self,
        enabled: bool = True,
        lags: Optional[List[int]] = None,
        lag_transforms: Optional[Dict[int, List[str]]] = None,
        best_model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LGBMConfig.

        Args:
            enabled: Whether the model is enabled
            lags: List of lag values
            lag_transforms: Dictionary of lag transformations
            best_model_params: Best model parameters
        """
        super().__init__("lgbm", enabled)

        self.lags = lags or [1, 7, 14, 30]
        self.lag_transforms = lag_transforms or {
            1: ["RollingMean", "RollingStd", "ExponentiallyWeightedMean"],
            7: ["RollingMean", "RollingStd"],
            14: ["RollingMean"],
            30: ["RollingMean"],
        }
        self.best_model_params = best_model_params or {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
        }

    def get_params(self) -> Dict[str, Any]:
        """
        Get LightGBM model parameters.

        Returns:
            Dictionary of LightGBM parameters
        """
        return {
            "lags": self.lags,
            "lag_transforms": self.lag_transforms,
            "best_model_params": self.best_model_params,
        }

    def validate(self) -> bool:
        """
        Validate LightGBM configuration.

        Returns:
            True if configuration is valid
        """
        try:
            # Validate lags
            if not isinstance(self.lags, list) or len(self.lags) == 0:
                self.logger.error("Lags must be a non-empty list")
                return False

            # Validate lag_transforms
            if not isinstance(self.lag_transforms, dict):
                self.logger.error("Lag transforms must be a dictionary")
                return False

            # Validate best_model_params
            if not isinstance(self.best_model_params, dict):
                self.logger.error("Best model params must be a dictionary")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False

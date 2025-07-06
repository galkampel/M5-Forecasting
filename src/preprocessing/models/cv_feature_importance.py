"""
Cross-validation feature importance analysis.

This module provides the CVFeatureImportance class for analyzing
feature importance across cross-validation folds.
"""

from functools import cached_property
from typing import Optional, Dict

import numpy as np
import pandas as pd
from mlforecast import MLForecast


class CVFeatureImportance:
    """
    Cross-validation feature importance analysis.

    This class analyzes feature importance across cross-validation folds
    for MLForecast models.
    """

    def __init__(self, model_name: str, ml_forecaster: MLForecast):
        """
        Initialize CVFeatureImportance.

        Args:
            model_name: Name of the model (e.g., 'Ridge', 'LGBMRegressor')
            ml_forecaster: Trained MLForecast instance
        """
        self.model_name = model_name
        self.ml_forecaster = ml_forecaster

    @cached_property
    def feature_names_in_(self) -> np.ndarray:
        """
        Get feature names from the model.

        Returns:
            Array of feature names
        """
        if (
            not hasattr(self.ml_forecaster, "cv_models_")
            or not self.ml_forecaster.cv_models_
        ):
            raise ValueError(
                "MLForecast must have cv_models_ attribute with cross-validation models"
            )

        model = self.ml_forecaster.cv_models_[0][self.model_name]["model"]

        if self.model_name == "Ridge":
            feature_names_in_ = model.feature_names_in_
            if hasattr(model, "intercept_"):
                feature_names_in_ = np.concatenate([["intercept"], feature_names_in_])
        elif self.model_name == "LGBMRegressor":
            feature_names_in_ = model.feature_names_in_
        else:
            # For other models, try to get feature names
            if hasattr(model, "feature_names_in_"):
                feature_names_in_ = model.feature_names_in_
            else:
                # Fallback: create generic feature names
                n_features = len(self.get_fold_feature_importance(0))
                feature_names_in_ = np.array(
                    [f"feature_{i}" for i in range(n_features)]
                )

        return feature_names_in_

    def get_fold_feature_importance(self, fold: int) -> np.ndarray:
        """
        Get feature importance for a specific fold.

        Args:
            fold: Fold index

        Returns:
            Array of feature importance values
        """
        if not hasattr(self.ml_forecaster, "cv_models_") or fold >= len(
            self.ml_forecaster.cv_models_
        ):
            raise ValueError(f"Fold {fold} not found in cv_models_")

        model = self.ml_forecaster.cv_models_[fold][self.model_name]["model"]

        if self.model_name == "Ridge":
            coef = model.coef_
            if hasattr(model, "intercept_"):
                coef = np.concatenate([[model.intercept_], coef])
        elif self.model_name == "LGBMRegressor":
            # Split-based Importance - measures how many times a feature is used to split nodes across all trees
            coef = model.feature_importances_
        else:
            # For other models, try to get coefficients or feature importance
            if hasattr(model, "coef_"):
                coef = model.coef_
                if hasattr(model, "intercept_"):
                    coef = np.concatenate([[model.intercept_], coef])
            elif hasattr(model, "feature_importances_"):
                coef = model.feature_importances_
            else:
                raise ValueError(
                    f"Model {self.model_name} does not support feature importance analysis"
                )

        return coef

    def create_feature_importance_table(
        self, cv_results: pd.DataFrame, h: int = 30
    ) -> pd.DataFrame:
        """
        Create feature importance table across all folds.

        Args:
            cv_results: Cross-validation results DataFrame
            h: Forecast horizon

        Returns:
            DataFrame with feature importance for each fold
        """
        cutoffs = cv_results["cutoff"].unique()
        df_fi = pd.DataFrame(index=self.feature_names_in_)

        for i, cutoff in enumerate(cutoffs):
            try:
                # Handle different cutoff data types
                if hasattr(cutoff, 'date'):
                    # If cutoff is already a datetime-like object
                    start_date = cutoff + pd.Timedelta(days=1)
                    end_date = cutoff + pd.Timedelta(days=h)
                    fold_range_str = f"{start_date.date()}_{end_date.date()}"
                elif hasattr(cutoff, 'days'):
                    # If cutoff is a Timedelta object, convert to datetime
                    # We need to get the actual date from the cv_results
                    fold_data = cv_results[cv_results["cutoff"] == cutoff]
                    if not fold_data.empty:
                        # Get the first date from this fold
                        first_date = pd.to_datetime(fold_data["ds"].iloc[0])
                        start_date = first_date + pd.Timedelta(days=1)
                        end_date = first_date + pd.Timedelta(days=h)
                        fold_range_str = f"{start_date.date()}_{end_date.date()}"
                    else:
                        fold_range_str = f"fold_{i}"
                else:
                    # Fallback to simple fold identifier
                    fold_range_str = f"fold_{i}"
                
                df_fi[fold_range_str] = self.get_fold_feature_importance(fold=i)
            except Exception as e:
                # If there's an error with date conversion, use a simple fold identifier
                fold_range_str = f"fold_{i}"
                df_fi[fold_range_str] = self.get_fold_feature_importance(fold=i)

        return df_fi

    def get_mean_feature_importance(
        self, cv_results: pd.DataFrame, h: int = 30
    ) -> pd.Series:
        """
        Get mean feature importance across all folds.

        Args:
            cv_results: Cross-validation results DataFrame
            h: Forecast horizon

        Returns:
            Series with mean feature importance
        """
        df_fi = self.create_feature_importance_table(cv_results, h)
        return df_fi.mean(axis=1)

    def get_std_feature_importance(
        self, cv_results: pd.DataFrame, h: int = 30
    ) -> pd.Series:
        """
        Get standard deviation of feature importance across all folds.

        Args:
            cv_results: Cross-validation results DataFrame
            h: Forecast horizon

        Returns:
            Series with standard deviation of feature importance
        """
        df_fi = self.create_feature_importance_table(cv_results, h)
        return df_fi.std(axis=1)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance as a dictionary.

        Returns:
            Dictionary with feature names as keys and importance scores as values
        """
        try:
            # Check if cv_models_ exists and has the required structure
            if not hasattr(self.ml_forecaster, "cv_models_") or not self.ml_forecaster.cv_models_:
                return {"error": "No cross-validation models found. Run cross_validation first."}
            
            # Check if the model exists in cv_models_
            if self.model_name not in self.ml_forecaster.cv_models_[0]:
                return {"error": f"Model {self.model_name} not found in cross-validation results."}
            
            # Get feature names
            feature_names = self.feature_names_in_
            
            # Get feature importance from the first fold as a baseline
            # In a more sophisticated implementation, you might want to average across folds
            importance_scores = self.get_fold_feature_importance(0)
            
            # Create dictionary
            feature_importance = {}
            for i, feature_name in enumerate(feature_names):
                if i < len(importance_scores):
                    feature_importance[str(feature_name)] = float(importance_scores[i])
            
            return feature_importance
            
        except Exception as e:
            # Return error message if feature importance analysis fails
            return {"error": f"Feature importance analysis failed: {str(e)}"}

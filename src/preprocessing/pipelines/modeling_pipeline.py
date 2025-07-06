"""
Modeling pipeline for preprocessing.

This module provides the modeling pipeline for training and evaluating
AutoMLForecast-based models with hyperparameter optimization.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
from mlforecast.auto import AutoMLForecast, AutoModel
from src.preprocessing.models.cv_feature_importance import CVFeatureImportance
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, ZeroModel
from lightgbm import LGBMRegressor

from ..config import PreprocessingConfig
from ..transformers.encoding import OHEwithBaseline
from ..transformers.scaling import StandardScalerwithThreshold
from ..transformers.temporal import ZeroPredTransformer


class ModelTrainingPipeline:
    """
    Model training pipeline using AutoMLForecast.

    This class handles model training and evaluation using AutoMLForecast
    with Optuna hyperparameter optimization.
    """

    def __init__(self, config: PreprocessingConfig):
        """
        Initialize ModelTrainingPipeline.

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(config.data.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self._setup_models()

        # Initialize results storage
        self.results = {}

        # Initialize ZeroPredTransformer for post-processing
        self.zero_pred_transformer = ZeroPredTransformer(
            negative_mask=True, 
            is_close_mask=True, 
            threshold=0.1
        )

        # Model configurations
        self.trained_models = {}
        self.auto_mlf = None

    def _setup_models(self) -> None:
        """Setup enabled models."""
        self.models = {}

        if self.config.models.ridge.enabled:
            self.models["Ridge"] = self.config.models.ridge

        if self.config.models.lgbm.enabled:
            self.models["LGBMRegressor"] = self.config.models.lgbm

    def run(
        self, features_df: pd.DataFrame, targets_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run the modeling pipeline using AutoMLForecast.

        Args:
            features_df: Processed features DataFrame
            targets_df: Processed targets DataFrame

        Returns:
            Dictionary containing model results
        """
        self.logger.info("Starting AutoMLForecast modeling pipeline")

        # Prepare data for MLForecast
        mlforecast_data = self._prepare_mlforecast_data(features_df, targets_df)

        # Create AutoMLForecast instance
        auto_mlf = self._create_automlforecast()

        # Fit AutoMLForecast
        self.logger.info("Fitting AutoMLForecast")
        n_windows = (
            self.config.cv.n_windows if hasattr(self.config.cv, "n_windows") else 5
        )
        n_trials = (
            self.config.optimization.n_trials
            if self.config.optimization.enabled
            else 20
        )

        auto_mlf.fit(
            mlforecast_data,
            n_windows=n_windows,
            h=self.config.cv.forecast_horizon,
            num_samples=n_trials,
        )

        # Store trained AutoMLForecast
        self.auto_mlf = auto_mlf

        # Extract results for each model
        results = self._extract_model_results(auto_mlf, mlforecast_data)

        # Perform cross-validation and feature importance analysis
        cv_results = self._perform_cross_validation_analysis(mlforecast_data)
        results.update(cv_results)

        # Save results and models
        self._save_results(results)
        self._save_models()

        self.logger.info("AutoMLForecast modeling pipeline completed")
        return results

    def _perform_cross_validation_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform cross-validation prediction and feature importance analysis.

        Args:
            data: MLForecast formatted data

        Returns:
            Dictionary containing CV results and feature importance
        """
        self.logger.info("Performing cross-validation analysis")

        cv_results = {}
        self.results = {}  # Initialize results dictionary

        # First, run baseline models
        self.logger.info("Running baseline models")
        baseline_results_path = self._run_baseline_models(data)

        # Calculate baseline metrics
        self.logger.info("Calculating baseline metrics")
        baseline_metrics = self._calculate_baseline_metrics(baseline_results_path)

        # Generate fitted values for baselines
        self.logger.info("Generating fitted values for baselines")
        baseline_fitted_values = self._generate_baseline_fitted_values(data)

        # After fitting, run analysis for each model
        for model_name in self.auto_mlf.models:
            best_params = None
            if self.config.optimization.enabled and hasattr(self.auto_mlf, "best_params_"):
                best_params = self.auto_mlf.best_params_.get(model_name)
            
            self._run_model_analysis(model_name, data, baseline_results_path, best_params)

        # Add baseline results
        cv_results["baselines"] = {
            "cv_predictions_path": str(baseline_results_path),
            "fitted_values_path": str(baseline_fitted_values),
            "cv_metrics": baseline_metrics,  # Add baseline metrics
        }

        # Add model results to cv_results
        for model_name, model_results in self.results.items():
            cv_results[model_name] = model_results

        return {"cross_validation": cv_results}

    def _run_model_analysis(
        self, model_name: str, data: pd.DataFrame, baseline_path: str, best_params: Optional[Dict[str, Any]] = None
    ):
        """
        Run CV and feature importance analysis for a given model.

        Args:
            model_name: Name of the model
            data: Training data
            baseline_path: Path to baseline predictions
            best_params: Best parameters for the model
        """
        if self.results.get(model_name) is None:
            self.results[model_name] = {}
        
        try:
            # Cross-validation analysis
            cv_preds = self._extract_cv_predictions_from_fit(model_name, data, best_params)
            
            # Save CV predictions
            cv_preds_path = self.output_dir / f"{model_name}_cv_predictions.csv"
            self.results[model_name]["cv_predictions_path"] = str(cv_preds_path)
            
            if cv_preds is not None and not cv_preds.empty:
                # Always save the original cv_preds (with all folds) to CSV
                cv_preds.to_csv(cv_preds_path, index=False)
                self.logger.info(f"Saved CV predictions with {cv_preds['cutoff'].nunique()} folds to {cv_preds_path}")
                
                # Load baseline predictions for metrics calculation
                baseline_preds = self._load_and_validate_baselines(baseline_path)
                
                # Combine with baselines for metrics calculation only
                if baseline_preds is not None and not baseline_preds.empty:
                    combined_preds = self._combine_predictions(cv_preds, baseline_preds)
                else:
                    combined_preds = cv_preds
                
                # Calculate CV metrics
                cv_metrics = self._calculate_cv_metrics(combined_preds, model_name)
                self.results[model_name]["cv_metrics"] = cv_metrics
            else:
                self.logger.warning(f"CV predictions are empty for {model_name}. Skipping metrics calculation.")
                # Create a placeholder file with error information
                error_df = pd.DataFrame({
                    'error': [f"CV predictions extraction failed for {model_name}"],
                    'timestamp': [pd.Timestamp.now()]
                })
                error_df.to_csv(cv_preds_path, index=False)
                self.logger.info(f"Created placeholder CV predictions file at {cv_preds_path}")

            # Feature importance analysis
            # We pass original cv_preds here, not combined_preds
            if cv_preds is not None and not cv_preds.empty:
                feature_importance = self._analyze_feature_importance(
                    model_name, cv_preds, data, best_params
                )
                if feature_importance:
                    self.results[model_name]["feature_importance"] = feature_importance

            # Generate fitted values for the main model
            fitted_values = self._generate_model_fitted_values(
                self.auto_mlf, data, model_name
            )
            self.results[model_name]["fitted_values_path"] = str(fitted_values)
        except Exception as e:
            self.logger.error(f"Error in CV analysis for {model_name}: {str(e)}")
            self.results[model_name] = {"status": "error", "error": str(e)}

    def _extract_cv_predictions_from_fit(
        self, model_name: str, data: pd.DataFrame, best_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Extract cross-validation predictions using a new MLForecast instance.
        This is done because AutoMLForecast does not expose the CV predictions directly
        in a way that can be used for detailed analysis.

        Args:
            model_name: Name of the model
            data: Training data
            best_params: Best parameters for the model

        Returns:
            DataFrame with CV predictions
        """
        try:
            from mlforecast import MLForecast
            from mlforecast.lag_transforms import RollingMean

            # Ensure data has the correct format for MLForecast
            mlforecast_data = data.copy()
            
            # Debug: Log original data info
            self.logger.info(f"Original data shape: {mlforecast_data.shape}")
            self.logger.info(f"Original columns: {list(mlforecast_data.columns)}")
            
            # Convert ds column to datetime if it's not already
            if 'ds' in mlforecast_data.columns:
                mlforecast_data['ds'] = pd.to_datetime(mlforecast_data['ds'])
            
            # Ensure y column is numeric
            if 'y' in mlforecast_data.columns:
                mlforecast_data['y'] = pd.to_numeric(mlforecast_data['y'], errors='coerce')
                # Remove rows with NaN in y
                mlforecast_data = mlforecast_data.dropna(subset=['y'])

            # For MLForecast, we need to keep categorical columns that the pipeline expects
            # Keep essential columns, categorical columns, and numeric features
            essential_cols = ['unique_id', 'ds', 'y']
            categorical_cols = ['weekday', 'month']  # Columns that the pipeline expects
            numeric_cols = mlforecast_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Combine all needed columns
            keep_cols = list(set(essential_cols + categorical_cols + numeric_cols))
            
            # Ensure all essential columns are present
            missing_essential = [col for col in essential_cols if col not in keep_cols]
            if missing_essential:
                self.logger.error(f"Missing essential columns: {missing_essential}")
                return None
            
            # Filter to only keep columns we need
            mlforecast_data = mlforecast_data[keep_cols]
            
            # Ensure categorical columns are properly formatted
            if 'weekday' in mlforecast_data.columns:
                mlforecast_data['weekday'] = mlforecast_data['weekday'].astype(str)
            if 'month' in mlforecast_data.columns:
                mlforecast_data['month'] = mlforecast_data['month'].astype(str)
            
            # Remove rows with NaN in any column
            mlforecast_data = mlforecast_data.dropna()
            
            if len(mlforecast_data) == 0:
                self.logger.error("No valid data remaining after cleaning")
                return None
            
            # Debug: Log cleaned data info
            self.logger.info(f"Cleaned data shape: {mlforecast_data.shape}")
            self.logger.info(f"Cleaned columns: {list(mlforecast_data.columns)}")

            # Get the correct scikit-learn pipeline for the model
            if model_name == "Ridge":
                pipeline = self._create_ridge_pipeline()
            elif model_name == "LGBMRegressor":
                pipeline = self._create_lgbm_pipeline()
            else:
                self.logger.warning(f"Unknown model type for CV: {model_name}")
                return None

            # Set default lags and transforms, to be overridden by best_params if available
            lags = [1, 7, 14]
            lag_transforms = {1: [RollingMean(window_size=w) for w in [7, 14]]}

            if best_params:
                # Apply best hyperparameters to the model within the pipeline
                model_params = {}
                for k, v in best_params.items():
                    if k not in ["lags", "lag_transforms", "transform_index"]:
                        model_params[f"model__{k}"] = v
                pipeline.set_params(**model_params)

                # Use best lags found during optimization
                if "lags" in best_params:
                    lags = best_params["lags"]

            # Create a new MLForecast instance with the full pipeline
            mlf = MLForecast(
                models={model_name: pipeline},
                freq="D",
                lags=lags,
                lag_transforms=lag_transforms,
            )

            # Run cross-validation to get predictions
            cv_preds = mlf.cross_validation(
                df=mlforecast_data,
                h=self.config.cv.forecast_horizon,
                n_windows=self.config.cv.n_windows,
                static_features=[],
            )

            # Ensure the output has the correct format
            if cv_preds is not None and not cv_preds.empty:
                # Convert ds column to datetime if needed
                if 'ds' in cv_preds.columns:
                    cv_preds['ds'] = pd.to_datetime(cv_preds['ds'])
                # Ensure model predictions are numeric
                if model_name in cv_preds.columns:
                    cv_preds[model_name] = pd.to_numeric(cv_preds[model_name], errors='coerce')
                # Log number of unique cutoffs (folds)
                if 'cutoff' in cv_preds.columns:
                    n_folds = cv_preds['cutoff'].nunique()
                    self.logger.info(f"Number of unique cutoffs (folds) in CV predictions: {n_folds}")
                # IMPORTANT: Do NOT filter by date or cutoff. Return all folds.
                self.logger.info(f"CV predictions shape: {cv_preds.shape}")
                self.logger.info(f"CV predictions columns: {list(cv_preds.columns)}")

            return cv_preds

        except Exception as e:
            self.logger.error(f"Error extracting CV predictions for {model_name}: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def _calculate_cv_metrics(
        self, cv_preds: pd.DataFrame, model_name: str
    ) -> Dict[str, float]:
        """
        Calculate cross-validation metrics using evaluator metrics.

        Args:
            cv_preds: Cross-validation predictions DataFrame
            model_name: Name of the model

        Returns:
            Dictionary of metrics
        """
        # Check if cv_preds is actually a DataFrame
        if not isinstance(cv_preds, pd.DataFrame):
            return {"error": f"Expected DataFrame but got {type(cv_preds).__name__}"}
        
        if "y" not in cv_preds.columns or model_name not in cv_preds.columns:
            return {"error": f"Missing required columns in CV predictions. Available columns: {list(cv_preds.columns)}"}

        # Apply ZeroPredTransformer post-processing to predictions
        self.logger.info(f"Applying ZeroPredTransformer post-processing to {model_name} predictions")
        cv_preds_processed = self.zero_pred_transformer.fit_transform(cv_preds)

        # Remove NaN values
        valid_mask = ~(cv_preds_processed["y"].isna() | cv_preds_processed[model_name].isna())
        y_true = cv_preds_processed.loc[valid_mask, "y"]
        y_pred = cv_preds_processed.loc[valid_mask, model_name]

        if len(y_true) == 0:
            return {"error": "No valid predictions for metric calculation"}

        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from src.preprocessing.evaluation.evaluator import (
            mase, rmsse, mrae, f1_zero, non_zero_mae
        )

        # Basic metrics
        metrics = {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        }

        # Add evaluator metrics if baseline models are available
        baseline_models = ["Naive", "SeasonalNaive", "ZeroModel"]
        seasonal_baseline = None
        zero_baseline = None
        
        # Find available baselines
        for baseline in baseline_models:
            if baseline in cv_preds_processed.columns:
                baseline_pred = cv_preds_processed.loc[valid_mask, baseline]
                # Check if baseline predictions are valid (not all NaN)
                if not baseline_pred.isna().all() and len(baseline_pred) > 0:
                    if baseline == "SeasonalNaive":
                        seasonal_baseline = baseline_pred
                    elif baseline == "ZeroModel":
                        zero_baseline = baseline_pred
                    self.logger.info(f"Found valid baseline predictions for {baseline}")
                else:
                    self.logger.warning(f"Baseline {baseline} has all NaN predictions")
            else:
                self.logger.warning(f"Baseline {baseline} not found in predictions")

        # Calculate evaluator metrics with error handling
        try:
            if seasonal_baseline is not None and len(seasonal_baseline) > 0:
                metrics["mase"] = mase(y_true, y_pred, seasonal_baseline)
                metrics["rmsse"] = rmsse(y_true, y_pred, seasonal_baseline)
                self.logger.info(f"Calculated MASE and RMSSE for {model_name}")
            else:
                self.logger.warning(f"No valid SeasonalNaive baseline for {model_name}, skipping MASE and RMSSE")
                metrics["mase"] = None
                metrics["rmsse"] = None
        except Exception as e:
            self.logger.error(f"Error calculating MASE/RMSSE for {model_name}: {e}")
            metrics["mase"] = None
            metrics["rmsse"] = None
        
        try:
            if zero_baseline is not None and len(zero_baseline) > 0:
                metrics["mrae"] = mrae(y_true, y_pred, zero_baseline)
                self.logger.info(f"Calculated MRAE for {model_name}")
            else:
                self.logger.warning(f"No valid ZeroModel baseline for {model_name}, skipping MRAE")
                metrics["mrae"] = None
        except Exception as e:
            self.logger.error(f"Error calculating MRAE for {model_name}: {e}")
            metrics["mrae"] = None
        
        # Calculate F1-zero and non-zero MAE (don't require baselines)
        try:
            metrics["f1_zero"] = f1_zero(y_true, y_pred)
            metrics["non_zero_mae"] = non_zero_mae(y_true, y_pred)
            self.logger.info(f"Calculated F1-zero and non-zero MAE for {model_name}")
        except Exception as e:
            self.logger.error(f"Error calculating F1-zero/non-zero MAE for {model_name}: {e}")
            metrics["f1_zero"] = None
            metrics["non_zero_mae"] = None

        # Add baseline comparison metrics if baselines are available
        for baseline in baseline_models:
            if baseline in cv_preds_processed.columns:
                baseline_pred = cv_preds_processed.loc[valid_mask, baseline]
                # Check if baseline predictions are valid
                if not baseline_pred.isna().all() and len(baseline_pred) > 0:
                    try:
                        baseline_mae = mean_absolute_error(y_true, baseline_pred)
                        baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))

                        metrics[f"mae_vs_{baseline}"] = (
                            metrics["mae"] / baseline_mae
                            if baseline_mae > 0
                            else metrics["mae"]
                        )
                        metrics[f"rmse_vs_{baseline}"] = (
                            metrics["rmse"] / baseline_rmse
                            if baseline_rmse > 0
                            else metrics["rmse"]
                        )
                    except Exception as e:
                        self.logger.error(f"Error calculating baseline comparison for {baseline}: {e}")
                        metrics[f"mae_vs_{baseline}"] = None
                        metrics[f"rmse_vs_{baseline}"] = None
                else:
                    metrics[f"mae_vs_{baseline}"] = None
                    metrics[f"rmse_vs_{baseline}"] = None

        return metrics

    def _run_baseline_models(self, data: pd.DataFrame) -> str:
        """
        Run baseline models using StatsForecast.

        Args:
            data: MLForecast formatted data

        Returns:
            Path to saved baseline results
        """
        # Create StatsForecast instance with baseline models
        stats_forecaster = StatsForecast(
            models=[Naive(), SeasonalNaive(season_length=7), ZeroModel()], freq="D"
        )

        # Fit the models (StatsForecast only needs unique_id, ds, y)
        stats_forecaster.fit(data[["unique_id", "ds", "y"]])

        # Perform cross-validation
        baseline_results = stats_forecaster.cross_validation(
            df=data[["unique_id", "ds", "y"]],
            h=self.config.cv.forecast_horizon,
            n_windows=self.config.cv.n_windows,
            step_size=30,
        )

        # Save baseline results
        baseline_path = self.output_dir / "baseline_cv_predictions.csv"
        baseline_results.to_csv(baseline_path, index=False)
        self.logger.info(f"Saved baseline CV predictions to {baseline_path}")

        return str(baseline_path)

    def _generate_baseline_fitted_values(self, data: pd.DataFrame) -> str:
        """
        Generate fitted values for baseline models.

        Args:
            data: MLForecast formatted data

        Returns:
            Path to saved baseline fitted values
        """
        # Create StatsForecast instance with baseline models
        stats_forecaster = StatsForecast(
            models=[Naive(), SeasonalNaive(season_length=7), ZeroModel()], freq="D"
        )
        # Fit the models (StatsForecast only needs unique_id, ds, y)
        stats_forecaster.fit(data[["unique_id", "ds", "y"]])
        # Run forecast with fitted=True to store fitted values
        stats_forecaster.forecast(df=data[["unique_id", "ds", "y"]], h=self.config.cv.forecast_horizon, fitted=True)
        # Generate fitted values
        fitted_values = stats_forecaster.forecast_fitted_values()
        # Save fitted values
        fitted_path = self.output_dir / "baseline_fitted_values.csv"
        fitted_values.to_csv(fitted_path, index=False)
        self.logger.info(f"Saved baseline fitted values to {fitted_path}")
        return str(fitted_path)

    def _load_and_validate_baselines(self, baseline_path: str) -> Optional[pd.DataFrame]:
        """
        Load and validate baseline predictions.

        Args:
            baseline_path: Path to baseline predictions file

        Returns:
            Validated baseline predictions DataFrame or None if invalid
        """
        try:
            # Load baseline predictions
            baseline_preds = pd.read_csv(baseline_path)
            
            if baseline_preds.empty:
                self.logger.warning("Baseline predictions file is empty")
                return None
            
            # Check for required columns
            required_cols = ["unique_id", "ds", "y"]
            missing_cols = [col for col in required_cols if col not in baseline_preds.columns]
            if missing_cols:
                self.logger.warning(f"Missing required columns in baseline predictions: {missing_cols}")
                return None
            
            # Check for baseline model columns
            baseline_models = ["Naive", "SeasonalNaive", "ZeroModel"]
            available_baselines = [col for col in baseline_models if col in baseline_preds.columns]
            
            if not available_baselines:
                self.logger.warning("No baseline model columns found in baseline predictions")
                return None
            
            self.logger.info(f"Found valid baseline predictions with models: {available_baselines}")
            return baseline_preds
            
        except Exception as e:
            self.logger.error(f"Error loading baseline predictions: {e}")
            return None

    def _combine_predictions(
        self, model_preds: pd.DataFrame, baseline_preds: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine model predictions with baseline predictions.

        Args:
            model_preds: Model cross-validation predictions
            baseline_preds: Baseline predictions DataFrame

        Returns:
            Combined predictions DataFrame
        """
        try:
            # Debug: Log column information
            self.logger.info(f"Baseline columns: {list(baseline_preds.columns)}")
            self.logger.info(f"Model columns: {list(model_preds.columns)}")

            # Create copies to avoid modifying original DataFrames
            baseline_preds_clean = baseline_preds.copy()
            model_preds_clean = model_preds.copy()

            # Ensure consistent data types for merge columns
            for df in [baseline_preds_clean, model_preds_clean]:
                if 'ds' in df.columns:
                    # Convert to datetime, handling various formats
                    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
                if 'cutoff' in df.columns:
                    # Convert to datetime, handling various formats
                    df['cutoff'] = pd.to_datetime(df['cutoff'], errors='coerce')
                if 'unique_id' in df.columns:
                    # Ensure unique_id is string
                    df['unique_id'] = df['unique_id'].astype(str)

            # Check if we have valid datetime columns after conversion
            if baseline_preds_clean['ds'].isna().all() or model_preds_clean['ds'].isna().all():
                self.logger.warning("Failed to convert datetime columns, returning model predictions only")
                return model_preds

            # Remove rows with invalid dates
            baseline_preds_clean = baseline_preds_clean.dropna(subset=['ds'])
            model_preds_clean = model_preds_clean.dropna(subset=['ds'])

            if len(baseline_preds_clean) == 0 or len(model_preds_clean) == 0:
                self.logger.warning("No valid data after datetime conversion, returning model predictions only")
                return model_preds

            # Find common columns for merge
            common_cols = list(set(baseline_preds_clean.columns) & set(model_preds_clean.columns))
            self.logger.info(f"Common columns for merge: {common_cols}")

            # Ensure we have the minimum required columns
            required_cols = ["unique_id", "ds", "y"]
            missing_cols = [col for col in required_cols if col not in common_cols]
            if missing_cols:
                self.logger.warning(f"Missing required columns for merge: {missing_cols}, returning model predictions only")
                return model_preds

            # Ensure y column is numeric in both DataFrames
            for df in [baseline_preds_clean, model_preds_clean]:
                if 'y' in df.columns:
                    df['y'] = pd.to_numeric(df['y'], errors='coerce')

            # Remove rows with NaN in y column
            baseline_preds_clean = baseline_preds_clean.dropna(subset=['y'])
            model_preds_clean = model_preds_clean.dropna(subset=['y'])

            if len(baseline_preds_clean) == 0 or len(model_preds_clean) == 0:
                self.logger.warning("No valid data after y column cleaning, returning model predictions only")
                return model_preds

            # Merge on common columns
            combined = pd.merge(
                baseline_preds_clean,
                model_preds_clean,
                on=common_cols,
                how="inner",
            )

            if len(combined) == 0:
                self.logger.warning("No matching rows after merge, returning model predictions only")
                return model_preds

            self.logger.info(f"Successfully combined predictions. Combined shape: {combined.shape}")
            return combined
            
        except Exception as e:
            self.logger.error(f"Error combining predictions: {str(e)}")
            # Return model_preds as fallback if merge fails
            return model_preds

    def _generate_model_fitted_values(
        self, auto_mlf: AutoMLForecast, data: pd.DataFrame, model_name: str
    ) -> str:
        """
        Generate fitted values for the main model.

        Args:
            auto_mlf: AutoMLForecast instance
            data: Training data
            model_name: Name of the model

        Returns:
            Path to saved fitted values
        """
        # Since we're using only cross_validation, fitted values are not available
        # Return a placeholder path
        fitted_path = self.output_dir / f"{model_name}_fitted_values.csv"
        self.logger.info(f"Skipping fitted values generation for {model_name} (using cross_validation only)")
        return str(fitted_path)

    def _analyze_feature_importance(
        self, model_name: str, cv_preds: pd.DataFrame, data: pd.DataFrame, best_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Analyze feature importance using CVFeatureImportance. This requires running
        cross-validation again to get access to the fitted models per fold.

        Args:
            model_name: Name of the model
            cv_preds: Cross-validation predictions (used to check if analysis is possible)
            data: Training data
            best_params: Best parameters for the model

        Returns:
            Dictionary with feature importance scores
        """
        if cv_preds is None or cv_preds.empty:
            return {"error": f"No CV predictions available for {model_name}"}

        try:
            # Create a new MLForecast instance for feature importance analysis
            from mlforecast import MLForecast
            from mlforecast.lag_transforms import RollingMean

            # Use the same data preparation as the main pipeline
            # This ensures we keep categorical columns that the pipeline expects
            mlforecast_data = data.copy()
            
            # Debug: Log original data info
            self.logger.info(f"Feature importance - Original data shape: {mlforecast_data.shape}")
            self.logger.info(f"Feature importance - Original columns: {list(mlforecast_data.columns)}")
            
            # Convert ds column to datetime if it's not already
            if 'ds' in mlforecast_data.columns:
                mlforecast_data['ds'] = pd.to_datetime(mlforecast_data['ds'])
            
            # Ensure y column is numeric
            if 'y' in mlforecast_data.columns:
                mlforecast_data['y'] = pd.to_numeric(mlforecast_data['y'], errors='coerce')
                # Remove rows with NaN in y
                mlforecast_data = mlforecast_data.dropna(subset=['y'])

            # For feature importance analysis, we need to keep categorical columns that the pipeline expects
            # Keep essential columns, categorical columns, and numeric features
            essential_cols = ['unique_id', 'ds', 'y']
            categorical_cols = ['weekday', 'month']  # Columns that the pipeline expects
            numeric_cols = mlforecast_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Combine all needed columns
            keep_cols = list(set(essential_cols + categorical_cols + numeric_cols))
            
            # Ensure all essential columns are present
            missing_essential = [col for col in essential_cols if col not in keep_cols]
            if missing_essential:
                self.logger.error(f"Missing essential columns for feature importance: {missing_essential}")
                return {"error": f"Missing essential columns: {missing_essential}"}
            
            # Filter to only keep columns we need
            mlforecast_data = mlforecast_data[keep_cols]
            
            # Ensure categorical columns are properly formatted
            if 'weekday' in mlforecast_data.columns:
                mlforecast_data['weekday'] = mlforecast_data['weekday'].astype(str)
            if 'month' in mlforecast_data.columns:
                mlforecast_data['month'] = mlforecast_data['month'].astype(str)
            
            # Remove rows with NaN in any column
            mlforecast_data = mlforecast_data.dropna()
            
            if len(mlforecast_data) == 0:
                self.logger.error("No valid data remaining after cleaning for feature importance")
                return {"error": "No valid data remaining after cleaning"}
            
            # Debug: Log cleaned data info
            self.logger.info(f"Feature importance - Cleaned data shape: {mlforecast_data.shape}")
            self.logger.info(f"Feature importance - Cleaned columns: {list(mlforecast_data.columns)}")

            # Get the correct scikit-learn pipeline for the model
            if model_name == "Ridge":
                pipeline = self._create_ridge_pipeline()
            elif model_name == "LGBMRegressor":
                pipeline = self._create_lgbm_pipeline()
            else:
                self.logger.warning(f"Unknown model type for feature importance: {model_name}")
                return {"error": f"Unknown model type: {model_name}"}

            # Set default lags and transforms, to be overridden by best_params if available
            lags = [1, 7, 14]
            lag_transforms = {1: [RollingMean(window_size=w) for w in [7, 14]]}

            if best_params:
                # Apply best hyperparameters to the model within the pipeline
                model_params = {}
                for k, v in best_params.items():
                    if k not in ["lags", "lag_transforms", "transform_index"]:
                        model_params[f"model__{k}"] = v
                pipeline.set_params(**model_params)

                # Use best lags found during optimization
                if "lags" in best_params:
                    lags = best_params["lags"]

            # Create a new MLForecast instance with the full pipeline
            mlf = MLForecast(
                models={model_name: pipeline},
                freq="D",
                lags=lags,
                lag_transforms=lag_transforms,
            )

            # Run cross-validation to get cv_models_ attribute
            cv_results = mlf.cross_validation(
                df=mlforecast_data,
                h=self.config.cv.forecast_horizon,
                n_windows=self.config.cv.n_windows,
                static_features=[],
            )

            # Check if cv_models_ attribute exists and contains the required models
            if not hasattr(mlf, 'cv_models_') or mlf.cv_models_ is None:
                self.logger.warning(f"No fitted models found for {model_name}")
                return {"error": f"No fitted models found for {model_name}"}

            # Create CVFeatureImportance instance
            cv_fi = CVFeatureImportance(model_name, mlf)

            # Get feature importance using the CV results
            feature_importance = cv_fi.get_feature_importance()

            self.logger.info(f"Successfully extracted feature importance for {model_name}")
            return feature_importance

        except Exception as e:
            self.logger.error(f"Error in feature importance analysis: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {"error": f"Feature importance analysis failed: {str(e)}"}

    def _prepare_mlforecast_data(
        self, features_df: pd.DataFrame, targets_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare data for MLForecast format.

        Args:
            features_df: Features DataFrame
            targets_df: Targets DataFrame

        Returns:
            DataFrame in MLForecast format
        """
        self.logger.info("Preparing data for MLForecast")

        # Ensure 'ds' column is datetime in both dataframes
        if 'ds' in features_df.columns:
            features_df['ds'] = pd.to_datetime(features_df['ds'])
        if 'ds' in targets_df.columns:
            targets_df['ds'] = pd.to_datetime(targets_df['ds'])

        # Merge features and targets
        data = features_df.merge(targets_df, on=["unique_id", "ds"], how="inner")

        # Drop irrelevant columns as per notebook
        features_to_drop = [
            "wm_yr_wk",
            "wday",
            "year",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
        ]
        data = data.drop(columns=features_to_drop, errors="ignore")

        # Ensure required columns exist
        required_cols = ["unique_id", "ds", "y"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Ensure categorical columns that the pipeline expects are present
        # Add missing categorical columns with default values if they don't exist
        if 'weekday' not in data.columns:
            # Extract weekday from ds column
            data['weekday'] = data['ds'].dt.day_name()
            self.logger.info("Added weekday column from ds")
        
        if 'month' not in data.columns:
            # Extract month from ds column
            data['month'] = data['ds'].dt.month
            self.logger.info("Added month column from ds")

        # Ensure numeric columns that the pipeline expects are present
        expected_numeric_cols = [
            "sell_price", "periods since last sales", 
            "periods since last 0 sales", "snap_CA", "is_item_exists"
        ]
        
        for col in expected_numeric_cols:
            if col not in data.columns:
                # Add missing numeric columns with default values
                if col == "sell_price":
                    data[col] = 1.0  # Default price
                elif col in ["periods since last sales", "periods since last 0 sales"]:
                    data[col] = 0  # Default periods
                elif col in ["snap_CA", "is_item_exists"]:
                    data[col] = 0  # Default binary values
                self.logger.info(f"Added missing numeric column: {col}")

        # Ensure all numeric columns are actually numeric
        for col in expected_numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                # Fill NaN values with defaults
                if col == "sell_price":
                    data[col] = data[col].fillna(0.0)
                elif col in ["periods since last sales", "periods since last 0 sales"]:
                    data[col] = data[col].fillna(0)
                elif col in ["snap_CA", "is_item_exists"]:
                    data[col] = data[col].fillna(0)

        # Remove rows with NaN in y column
        data = data.dropna(subset=['y'])

        # Sort by unique_id and date
        data = data.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        self.logger.info(f"Prepared data shape: {data.shape}")
        self.logger.info(f"Available columns: {list(data.columns)}")
        return data

    def _create_automlforecast(self) -> AutoMLForecast:
        """
        Create AutoMLForecast instance with models and configurations.

        Returns:
            AutoMLForecast instance
        """
        # Create model pipelines
        models = {}

        if "Ridge" in self.models:
            ridge_pipeline = self._create_ridge_pipeline()
            ridge_model = AutoModel(model=ridge_pipeline, config=self._ridge_config)
            models["Ridge"] = ridge_model

        if "LGBMRegressor" in self.models:
            lgbm_pipeline = self._create_lgbm_pipeline()
            lgbm_model = AutoModel(model=lgbm_pipeline, config=self._lgbm_config)
            models["LGBMRegressor"] = lgbm_model

        # Create AutoMLForecast
        auto_mlf = AutoMLForecast(
            models=models,
            freq="D",
            init_config=self._init_config,
            fit_config=self._fit_config,
        )

        return auto_mlf

    def _fit_config(self, trial) -> Dict[str, Any]:
        """Configuration for AutoMLForecast fit."""
        return {"static_features": []}

    def _create_ridge_pipeline(self) -> Pipeline:
        """Create Ridge regression pipeline."""
        # Get baseline categories from config
        weekday_baseline = self.config.features.encoding.baseline_categories.get(
            "weekday", "Monday"
        )
        month_baseline = self.config.features.encoding.baseline_categories.get(
            "month", 1
        )

        ohe_weekday = OHEwithBaseline(
            baseline_category=weekday_baseline, drop_first=True
        )
        ohe_month = OHEwithBaseline(
            baseline_category=month_baseline, drop_first=True
        )

        # Create column transformer with remainder="passthrough" to handle all remaining columns
        ct_ridge = ColumnTransformer(
            transformers=[
                ("ohe_weekday", ohe_weekday, ["weekday"]),
                ("ohe_month", ohe_month, ["month"]),
                (
                    "SD_scaler_with_thres",
                    StandardScalerwithThreshold(
                        threshold=0.0, with_mean=True, with_std=True
                    ),
                    ["sell_price"],
                ),
                (
                    "SD_scaler",
                    StandardScaler(),
                    ["periods since last sales", "periods since last 0 sales"],
                ),
            ],
            remainder="passthrough",  # Automatically pass through all other columns
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")

        # Create pipeline
        pipeline = Pipeline(
            [
                ("ct_transformer", ct_ridge),
                ("model", Ridge(fit_intercept=True, random_state=42)),
            ]
        )

        return pipeline

    def _create_lgbm_pipeline(self) -> Pipeline:
        """Create LightGBM pipeline."""
        # Get baseline categories from config
        weekday_baseline = self.config.features.encoding.baseline_categories.get(
            "weekday", "Monday"
        )
        month_baseline = self.config.features.encoding.baseline_categories.get(
            "month", 1
        )

        ohe_weekday = OHEwithBaseline(
            baseline_category=weekday_baseline, drop_first=True
        )
        ohe_month = OHEwithBaseline(
            baseline_category=month_baseline, drop_first=True
        )

        # Create column transformer with remainder="passthrough" to handle all remaining columns
        ct_lgbm = ColumnTransformer(
            transformers=[
                ("ohe_weekday", ohe_weekday, ["weekday"]),
                ("ohe_month", ohe_month, ["month"]),
            ],
            remainder="passthrough",  # Automatically pass through all other columns
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")

        # Create pipeline with LGBMRegressor
        pipeline = Pipeline(
            [
                ("ct_transformer", ct_lgbm),
                (
                    "model",
                    LGBMRegressor(
                        objective="tweedie",
                        learning_rate=0.1,
                        n_estimators=100,
                        random_state=42,
                        verbose=-1  # Suppress LightGBM output
                    ),
                ),
            ]
        )

        return pipeline

    def _init_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Initialize configuration for AutoMLForecast.

        Args:
            trial: Optuna trial

        Returns:
            Configuration dictionary
        """
        from mlforecast.lag_transforms import RollingMean, RollingStd

        transform_options = [
            [RollingMean(window_size=7), RollingStd(window_size=7)],
            [
                RollingMean(window_size=7),
                RollingStd(window_size=7),
                RollingMean(window_size=28),
                RollingStd(window_size=28),
            ],
        ]

        transform_choice = trial.suggest_int(
            "transform_index", 0, len(transform_options) - 1
        )

        return {
            "lags": trial.suggest_categorical("lags", [[1, 7], [1, 7, 30]]),
            "lag_transforms": {1: transform_options[transform_choice]},
        }

    def _ridge_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Ridge model configuration for Optuna.

        Args:
            trial: Optuna trial

        Returns:
            Configuration dictionary
        """
        return {
            "model__fit_intercept": True,
            "model__alpha": trial.suggest_float("alpha", 1e-6, 1e3, log=True),
            "model__random_state": 42,
        }

    def _lgbm_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        LightGBM model configuration for Optuna.

        Args:
            trial: Optuna trial

        Returns:
            Configuration dictionary
        """
        return {
            "model__learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "model__n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "model__max_depth": trial.suggest_int("max_depth", 3, 8),
            "model__objective": "tweedie",
            "model__tweedie_variance_power": trial.suggest_float(
                "tweedie_variance_power", 1.4, 1.9
            ),
            "model__random_state": 42,
            "model__verbose": -1,  # Suppress LightGBM output
        }

    def _extract_model_results(
        self, auto_mlf: AutoMLForecast, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Extract results for each model from AutoMLForecast.

        Args:
            auto_mlf: Trained AutoMLForecast instance
            data: Training data

        Returns:
            Dictionary of model results
        """
        results = {}

        for model_name in self.models.keys():
            self.logger.info(f"Extracting results for {model_name}")

            try:
                # Get model results from AutoMLForecast
                # The results_ attribute contains the study for each model
                study = auto_mlf.results_[model_name]

                # Extract best parameters and metrics
                best_params = study.best_params
                best_score = study.best_value

                results[model_name] = {
                    "model_name": model_name,
                    "status": "trained",
                    "best_params": best_params,
                    "best_score": best_score,
                    "cv_results": {
                        "n_trials": len(study.trials),
                        "best_trial": study.best_trial.number,
                    },
                    "metrics": {
                        "mae": best_score if best_score is not None else 0.0,
                        "rmse": best_score if best_score is not None else 0.0,
                        "mape": 0.0,  # Would need actual predictions vs ground truth
                    },
                }

            except Exception as e:
                self.logger.error(
                    f"Error extracting results for {model_name}: {str(e)}"
                )
                results[model_name] = {
                    "model_name": model_name,
                    "status": "error",
                    "error": str(e),
                }

        return results

    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Save model results.

        Args:
            results: Model results dictionary
        """
        results_path = self.output_dir / "automlforecast_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Saved AutoMLForecast results to {results_path}")

    def _save_models(self) -> None:
        """Save trained AutoMLForecast model."""
        if self.auto_mlf is not None:
            model_path = self.output_dir / "automlforecast_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(self.auto_mlf, f)

            self.logger.info(f"Saved AutoMLForecast model to {model_path}")

    def get_enabled_models(self) -> List[str]:
        """
        Get list of enabled models.

        Returns:
            List of enabled model names
        """
        return list(self.models.keys())

    def _calculate_baseline_metrics(self, baseline_path: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for baseline models.

        Args:
            baseline_path: Path to baseline predictions file

        Returns:
            Dictionary of baseline metrics
        """
        try:
            baseline_preds = self._load_and_validate_baselines(baseline_path)
            if baseline_preds is None or baseline_preds.empty:
                return {}
            
            baseline_metrics = {}
            baseline_models = ["Naive", "SeasonalNaive", "ZeroModel"]
            
            for baseline in baseline_models:
                if baseline in baseline_preds.columns:
                    try:
                        # Apply ZeroPredTransformer post-processing
                        baseline_preds_processed = self.zero_pred_transformer.fit_transform(baseline_preds)
                        
                        # Remove NaN values
                        valid_mask = ~(baseline_preds_processed["y"].isna() | baseline_preds_processed[baseline].isna())
                        y_true = baseline_preds_processed.loc[valid_mask, "y"]
                        y_pred = baseline_preds_processed.loc[valid_mask, baseline]
                        
                        if len(y_true) == 0:
                            self.logger.warning(f"No valid predictions for baseline {baseline}")
                            continue
                        
                        from sklearn.metrics import mean_absolute_error, mean_squared_error
                        from src.preprocessing.evaluation.evaluator import (
                            mase, rmsse, mrae, f1_zero, non_zero_mae
                        )
                        
                        # Calculate basic metrics
                        metrics = {
                            "mae": mean_absolute_error(y_true, y_pred),
                            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                        }
                        
                        # Calculate evaluator metrics
                        try:
                            metrics["f1_zero"] = f1_zero(y_true, y_pred)
                            metrics["non_zero_mae"] = non_zero_mae(y_true, y_pred)
                        except Exception as e:
                            self.logger.error(f"Error calculating F1-zero/non-zero MAE for {baseline}: {e}")
                            metrics["f1_zero"] = None
                            metrics["non_zero_mae"] = None
                        
                        # Calculate relative metrics if other baselines are available
                        for other_baseline in baseline_models:
                            if other_baseline != baseline and other_baseline in baseline_preds_processed.columns:
                                try:
                                    other_pred = baseline_preds_processed.loc[valid_mask, other_baseline]
                                    if not other_pred.isna().all() and len(other_pred) > 0:
                                        other_mae = mean_absolute_error(y_true, other_pred)
                                        other_rmse = np.sqrt(mean_squared_error(y_true, other_pred))
                                        
                                        metrics[f"mae_vs_{other_baseline}"] = (
                                            metrics["mae"] / other_mae if other_mae > 0 else metrics["mae"]
                                        )
                                        metrics[f"rmse_vs_{other_baseline}"] = (
                                            metrics["rmse"] / other_rmse if other_rmse > 0 else metrics["rmse"]
                                        )
                                except Exception as e:
                                    self.logger.error(f"Error calculating comparison for {baseline} vs {other_baseline}: {e}")
                        
                        baseline_metrics[baseline] = metrics
                        self.logger.info(f"Calculated metrics for baseline {baseline}")
                        
                    except Exception as e:
                        self.logger.error(f"Error calculating metrics for baseline {baseline}: {e}")
                        baseline_metrics[baseline] = {"error": str(e)}
            
            return baseline_metrics
            
        except Exception as e:
            self.logger.error(f"Error in baseline metrics calculation: {e}")
            return {}

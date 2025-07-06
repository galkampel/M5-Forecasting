"""
Main entry point for preprocessing pipeline.

This module provides the main orchestration script that coordinates
the entire preprocessing pipeline with command-line interface.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import PreprocessingConfig
from .data_loader import DataLoader
from .pipelines import (
    FeatureEngineeringPipeline,
    HyperparameterOptimizationPipeline,
    EvaluationPipeline,
    ModelTrainingPipeline,
)
from .utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="M5 Forecasting Preprocessing Pipeline"
    )

    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration YAML file"
    )

    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for results"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration and data, don't run pipeline",
    )

    parser.add_argument(
        "--skip-feature-engineering",
        action="store_true",
        help="Skip feature engineering step",
    )

    parser.add_argument(
        "--skip-modeling", action="store_true", help="Skip modeling step"
    )

    parser.add_argument(
        "--skip-evaluation", action="store_true", help="Skip evaluation step"
    )

    parser.add_argument(
        "--skip-optimization", action="store_true", help="Skip optimization step"
    )

    return parser.parse_args()


def main(config_path: Optional[str] = None) -> int:
    """
    Main preprocessing pipeline.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse command line arguments
        args = parse_args()

        # Use provided config path or command line argument
        if config_path is None:
            config_path = args.config

        # Initialize configuration
        logger = logging.getLogger(__name__)
        logger.info("Initializing preprocessing configuration...")

        config = PreprocessingConfig(config_path)

        # Override output directory if specified
        if args.output_dir:
            config.data.output_dir = args.output_dir

        # Setup logging
        setup_logging(
            level=args.log_level,
            log_file=config.logging.file,
            log_format=config.logging.format,
        )

        # Validate configuration
        logger.info("Validating configuration...")
        config.validate()
        logger.info("Configuration validation passed")

        # If validate-only mode, exit here
        if args.validate_only:
            logger.info("Validation-only mode - exiting")
            return 0

        # Initialize data loader
        logger.info("Initializing data loader...")
        data_loader = DataLoader(config)

        # Load data
        logger.info("Loading data...")
        features_df, targets_df = data_loader.load_data()

        # Get data information
        data_info = data_loader.get_data_info()
        logger.info(f"Data loaded successfully: {data_info}")

        # Run preprocessing pipeline
        logger.info("Starting preprocessing pipeline...")

        # Feature engineering step
        if not args.skip_feature_engineering:
            logger.info("Running feature engineering...")
            feature_pipeline = FeatureEngineeringPipeline(config)
            features_df, targets_df = feature_pipeline.run(features_df, targets_df)
            logger.info("Feature engineering completed")
        else:
            logger.info("Skipping feature engineering")

        # Optimization step (if enabled)
        if not args.skip_optimization and config.optimization.enabled:
            logger.info("Running hyperparameter optimization...")
            optimization_pipeline = HyperparameterOptimizationPipeline(config)
            optimization_results = optimization_pipeline.run(features_df, targets_df)
            logger.info("Hyperparameter optimization completed")
        else:
            logger.info("Skipping hyperparameter optimization")

        # Modeling step
        if not args.skip_modeling:
            logger.info("Running modeling...")
            modeling_pipeline = ModelTrainingPipeline(config)
            model_results = modeling_pipeline.run(features_df, targets_df)
            logger.info("Modeling completed")
        else:
            logger.info("Skipping modeling")
            model_results = {}

        # Evaluation step
        if not args.skip_evaluation and model_results:
            logger.info("Running evaluation...")
            evaluation_pipeline = EvaluationPipeline(config)
            evaluation_results = evaluation_pipeline.run(model_results)
            logger.info("Evaluation completed")
        else:
            logger.info("Skipping evaluation")

        # Save final summary
        _save_pipeline_summary(config, data_info)

        logger.info("Preprocessing pipeline completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}")
        return 1


def _save_pipeline_summary(config: PreprocessingConfig, data_info: dict) -> None:
    """
    Save pipeline summary.

    Args:
        config: Preprocessing configuration
        data_info: Data information dictionary
    """
    import json
    from pathlib import Path

    output_dir = Path(config.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "pipeline_status": "completed",
        "config_summary": {
            "features_enabled": {
                "fourier": config.features.fourier.enabled,
                "calendar": config.features.calendar.enabled,
                "encoding": True,
                "scaling": config.features.scaling.enabled,
            },
            "models_enabled": {
                "ridge": config.models.ridge.enabled,
                "lgbm": config.models.lgbm.enabled,
            },
            "optimization_enabled": config.optimization.enabled,
        },
        "data_info": data_info,
        "output_directory": str(output_dir.absolute()),
    }

    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger = logging.getLogger(__name__)
    logger.info(f"Saved pipeline summary to {summary_path}")


if __name__ == "__main__":
    sys.exit(main())

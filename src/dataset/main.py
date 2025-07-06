"""
Main entry point for dataset preprocessing.

This module provides the main orchestration script and command-line interface
for the dataset preprocessing pipeline.
"""

import argparse
import sys
from typing import Optional

import pandas as pd

from .config import Config
from .data_loader import DataLoader
from .pipelines.calendar_pipeline import CalendarPipeline
from .pipelines.feature_pipeline import FeaturePipeline
from .pipelines.sales_pipeline import SalesPipeline
from .utils.logging import setup_logging


class DatasetProcessor:
    """
    Main processor for dataset preprocessing.

    This class orchestrates the entire preprocessing pipeline including
    data loading, transformation, and output generation.
    """

    def __init__(self, config: Config):
        """
        Initialize the dataset processor.

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = setup_logging(
            level=config.logging.level,
            format_str=config.logging.format,
            log_file=config.logging.file,
            logger_name="DatasetProcessor",
        )
        self.data_loader = DataLoader(config)

        # Initialize pipelines
        self.calendar_pipeline = CalendarPipeline(config)
        self.sales_pipeline = SalesPipeline(config)
        self.feature_pipeline = FeaturePipeline(config)

        # Store results
        self._features: Optional[pd.DataFrame] = None
        self._targets: Optional[pd.DataFrame] = None

    def process(self) -> None:
        """
        Run the complete preprocessing pipeline.

        Raises:
            Exception: If any step in the pipeline fails
        """
        try:
            self.logger.info("Starting dataset preprocessing pipeline")

            # Step 1: Load and validate data
            self.logger.info("Step 1: Loading data")
            data = self.data_loader.load_all_data()
            self.data_loader.validate_data_integrity(data)

            # Step 2: Process calendar data
            self.logger.info("Step 2: Processing calendar data")
            calendar_data = self.calendar_pipeline.process(data["calendar"])

            # Step 3: Process sales data
            self.logger.info("Step 3: Processing sales data")
            sales_data = self.sales_pipeline.process(data["sales"], data["calendar"])

            # Step 4: Create features
            self.logger.info("Step 4: Creating features")
            self._features, self._targets = self.feature_pipeline.process(
                calendar_data, sales_data, data["prices"]
            )

            # Step 5: Save results
            self.logger.info("Step 5: Saving results")
            self._save_results()

            self.logger.info("Dataset preprocessing completed successfully")

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    def _save_results(self) -> None:
        """Save processed features and targets."""
        features_path, targets_path = self.config.get_output_paths()

        if self._features is not None:
            self._features.to_csv(features_path, index=False)
            self.logger.info(f"Saved features to: {features_path}")

        if self._targets is not None:
            self._targets.to_csv(targets_path, index=False)
            self.logger.info(f"Saved targets to: {targets_path}")

    def get_features(self) -> Optional[pd.DataFrame]:
        """Get processed features."""
        return self._features

    def get_targets(self) -> Optional[pd.DataFrame]:
        """Get processed targets."""
        return self._targets

    def get_summary(self) -> dict:
        """Get processing summary."""
        summary = {
            "config": {
                "dept_id": self.config.filters.dept_id,
                "store_id": self.config.filters.store_id,
                "date_range": self.config.get_date_range_str(),
            },
            "data_shapes": {},
        }

        if self._features is not None:
            summary["data_shapes"]["features"] = self._features.shape

        if self._targets is not None:
            summary["data_shapes"]["targets"] = self._targets.shape

        return summary


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Dataset preprocessing for Walmart sales forecasting"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run validation only without processing"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = Config.from_file(args.config)

        # Override log level if verbose
        if args.verbose:
            config.logging.level = "DEBUG"

        # Create processor
        processor = DatasetProcessor(config)

        if args.dry_run:
            # Just validate configuration and data availability
            print("Dry run mode - validating configuration and data...")

            # Test data loading
            data_loader = DataLoader(config)
            data = data_loader.load_all_data()
            data_loader.validate_data_integrity(data)

            print("✓ Configuration and data validation passed")
            print(
                f"✓ Data shapes: {', '.join([f'{k}: {v.shape}' for k, v in data.items()])}"
            )

        else:
            # Run full processing
            processor.process()

            # Print summary
            summary = processor.get_summary()
            print("\nProcessing Summary:")
            print(f"  Department: {summary['config']['dept_id']}")
            print(f"  Store: {summary['config']['store_id']}")
            print(f"  Date Range: {summary['config']['date_range']}")
            print(f"  Features Shape: {summary['data_shapes'].get('features', 'N/A')}")
            print(f"  Targets Shape: {summary['data_shapes'].get('targets', 'N/A')}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Validation Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Processing Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

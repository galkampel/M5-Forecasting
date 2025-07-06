#!/usr/bin/env python3
"""
Automated pipeline runner with monitoring and validation.

This script provides a comprehensive automation tool for running the complete
M5 forecasting pipeline with built-in monitoring, validation, and error handling.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.config import PreprocessingConfig
from preprocessing.data_loader import DataLoader
from preprocessing.pipelines import (
    FeatureEngineeringPipeline,
    ModelTrainingPipeline,
    ModelEvaluationPipeline,
    HyperparameterOptimizationPipeline
)
from preprocessing.utils.monitoring import PipelineMonitor, get_system_info
from preprocessing.utils.validation import DataQualityMonitor
from preprocessing.utils.logging import setup_logging


class PipelineRunner:
    """
    Automated pipeline runner with comprehensive monitoring and validation.
    """
    
    def __init__(self, config_path: Optional[str] = None, 
                 output_dir: str = "outputs/automated_run"):
        """
        Initialize PipelineRunner.
        
        Args:
            config_path: Path to configuration file
            output_dir: Output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.output_dir / "pipeline_run.log"
        setup_logging(
            level="INFO",
            log_file=str(log_file),
            log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = PreprocessingConfig(config_path)
        self.config.data.output_dir = str(self.output_dir)
        
        # Initialize monitoring and validation
        self.monitor = PipelineMonitor(str(self.output_dir / "monitoring"))
        self.quality_monitor = DataQualityMonitor(str(self.output_dir / "data_quality"))
        
        # Pipeline results
        self.results: Dict[str, Any] = {}
        self.start_time = None
        self.end_time = None
    
    def run_complete_pipeline(self) -> bool:
        """
        Run the complete pipeline with monitoring and validation.
        
        Returns:
            True if successful, False otherwise
        """
        self.start_time = datetime.now()
        self.logger.info("Starting complete pipeline run")
        
        try:
            # Start monitoring
            self.monitor.start_monitoring()
            
            # Log system info
            system_info = get_system_info()
            self.logger.info(f"System info: {system_info}")
            
            # Initialize data loader
            with self.monitor.monitor_pipeline_stage("data_loading"):
                self.logger.info("Initializing data loader")
                data_loader = DataLoader(self.config)
                
                # Load raw data
                self.logger.info("Loading raw data")
                features_df, targets_df = data_loader.load_data()
                
                # Monitor data quality
                self.logger.info("Monitoring data quality")
                quality_result = self.quality_monitor.monitor_dataframe(
                    features_df, "raw_features"
                )
                self.results["data_quality"] = quality_result
            
            # Feature engineering
            with self.monitor.monitor_pipeline_stage("feature_engineering"):
                self.logger.info("Running feature engineering pipeline")
                feature_pipeline = FeatureEngineeringPipeline(self.config)
                feature_results = feature_pipeline.run(features_df, targets_df)
                self.results["feature_engineering"] = feature_results
            
            # Load processed data
            with self.monitor.monitor_pipeline_stage("load_processed_data"):
                self.logger.info("Loading processed data")
                processed_features, processed_targets = data_loader.load_processed_data()
                
                # Monitor processed data quality
                processed_quality = self.quality_monitor.monitor_dataframe(
                    processed_features, "processed_features"
                )
                self.results["processed_data_quality"] = processed_quality
            
            # Model training
            with self.monitor.monitor_pipeline_stage("model_training"):
                self.logger.info("Running model training pipeline")
                modeling_pipeline = ModelTrainingPipeline(self.config)
                modeling_results = modeling_pipeline.run(processed_features, processed_targets)
                self.results["model_training"] = modeling_results
            
            # Model evaluation
            with self.monitor.monitor_pipeline_stage("model_evaluation"):
                self.logger.info("Running model evaluation pipeline")
                evaluation_pipeline = ModelEvaluationPipeline(self.config)
                evaluation_results = evaluation_pipeline.run()
                self.results["model_evaluation"] = evaluation_results
            
            # Hyperparameter optimization (if enabled)
            if self.config.optimization.enabled:
                with self.monitor.monitor_pipeline_stage("hyperparameter_optimization"):
                    self.logger.info("Running hyperparameter optimization")
                    optimization_pipeline = HyperparameterOptimizationPipeline(self.config)
                    optimization_results = optimization_pipeline.run()
                    self.results["optimization"] = optimization_results
            
            # Generate final reports
            with self.monitor.monitor_pipeline_stage("report_generation"):
                self.logger.info("Generating final reports")
                self._generate_final_reports()
            
            self.end_time = datetime.now()
            self.logger.info("Pipeline completed successfully")
            return True
            
        except Exception as e:
            self.end_time = datetime.now()
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            self._save_error_report(e)
            return False
        
        finally:
            # Stop monitoring
            self.monitor.stop_monitoring()
    
    def _generate_final_reports(self) -> None:
        """Generate final comprehensive reports."""
        # Generate monitoring report
        monitoring_report_path = self.monitor.generate_report()
        self.results["monitoring_report"] = monitoring_report_path
        
        # Generate quality dashboard
        quality_dashboard = self.quality_monitor.generate_quality_dashboard()
        self.results["quality_dashboard"] = quality_dashboard
        
        # Generate pipeline summary
        pipeline_summary = {
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
            "config_used": {
                "data_config": self.config.data.__dict__,
                "models_config": self.config.models.__dict__,
                "cv_config": self.config.cv.__dict__,
                "evaluation_config": self.config.evaluation.__dict__,
                "optimization_config": self.config.optimization.__dict__
            },
            "system_info": get_system_info(),
            "pipeline_stages": list(self.results.keys())
        }
        
        self.results["pipeline_summary"] = pipeline_summary
        
        # Save comprehensive results
        results_path = self.output_dir / "pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Final reports saved to {results_path}")
    
    def _save_error_report(self, error: Exception) -> None:
        """Save error report."""
        error_report = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "pipeline_stage": "unknown",
            "system_info": get_system_info()
        }
        
        error_path = self.output_dir / "error_report.json"
        with open(error_path, 'w') as f:
            json.dump(error_report, f, indent=2, default=str)
        
        self.logger.error(f"Error report saved to {error_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated M5 Forecasting Pipeline Runner"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/automated_run",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration and data, don't run pipeline"
    )
    
    parser.add_argument(
        "--skip-monitoring",
        action="store_true",
        help="Skip performance monitoring"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip data validation"
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main function for automated pipeline execution.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_args()
    
    try:
        # Create pipeline runner
        runner = PipelineRunner(
            config_path=args.config,
            output_dir=args.output_dir
        )
        
        # Run pipeline
        success = runner.run_complete_pipeline()
        
        if success:
            print("✅ Pipeline completed successfully!")
            print(f"Results saved to: {runner.output_dir}")
            return 0
        else:
            print("❌ Pipeline failed!")
            print(f"Check error report in: {runner.output_dir}")
            return 1
            
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
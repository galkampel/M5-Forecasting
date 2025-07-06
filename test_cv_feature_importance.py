#!/usr/bin/env python3
"""
Test script for cross-validation prediction and feature importance analysis.

This script tests the implementation of cross-validation prediction
and feature importance analysis using CVFeatureImportance.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from preprocessing.config import PreprocessingConfig
from preprocessing.pipelines.modeling_pipeline import ModelTrainingPipeline
from preprocessing.data_loader import DataLoader


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/cv_feature_importance.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main function to run cross-validation and feature importance analysis."""
    print("Starting Cross-Validation and Feature Importance Analysis")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info("Loading configuration")
        config = PreprocessingConfig()
        
        # Load data
        logger.info("Loading data")
        data_loader = DataLoader(config)
        
        # Load processed features and targets
        features_df, targets_df = data_loader.load_processed_data()
        
        logger.info(f"Features shape: {features_df.shape}")
        logger.info(f"Targets shape: {targets_df.shape}")
        
        # Create and run modeling pipeline
        logger.info("Creating modeling pipeline")
        modeling_pipeline = ModelTrainingPipeline(config)
        
        # Run the pipeline with cross-validation and feature importance analysis
        logger.info("Running modeling pipeline with CV and feature importance")
        results = modeling_pipeline.run(features_df, targets_df)
        
        # Print results summary
        print("\nResults Summary:")
        print("-" * 40)
        
        # Print model results
        for model_name, model_results in results.items():
            if model_name == "cross_validation":
                continue
                
            print(f"\n{model_name.upper()} Model:")
            if "error" in model_results:
                print(f"  Error: {model_results['error']}")
            else:
                print(f"  Status: {model_results.get('status', 'N/A')}")
                print(f"  Best Score: {model_results.get('best_score', 'N/A')}")
                print(f"  Best Params: {model_results.get('best_params', {})}")
        
        # Print cross-validation results
        if "cross_validation" in results:
            print(f"\nCross-Validation Results:")
            print("-" * 40)
            
            cv_results = results["cross_validation"]
            for model_name, cv_result in cv_results.items():
                print(f"\n{model_name.upper()} CV Analysis:")
                
                if "error" in cv_result:
                    print(f"  Error: {cv_result['error']}")
                else:
                    print(f"  CV Predictions: {cv_result.get('cv_predictions_path', 'N/A')}")
                    
                    # Print CV metrics
                    cv_metrics = cv_result.get('cv_metrics', {})
                    if "error" in cv_metrics:
                        print(f"  CV Metrics Error: {cv_metrics['error']}")
                    else:
                        print(f"  MAE: {cv_metrics.get('mae', 'N/A'):.4f}")
                        print(f"  RMSE: {cv_metrics.get('rmse', 'N/A'):.4f}")
                        print(f"  MAPE: {cv_metrics.get('mape', 'N/A'):.4f}%")
                    
                    # Print feature importance
                    fi_result = cv_result.get('feature_importance', {})
                    if "error" in fi_result:
                        print(f"  Feature Importance Error: {fi_result['error']}")
                    else:
                        print(f"  Feature Importance: {fi_result.get('file_path', 'N/A')}")
                        
                        # Print top features
                        top_features = fi_result.get('top_features', {})
                        if top_features:
                            print("  Top 5 Features:")
                            for i, (feature, importance) in enumerate(
                                sorted(top_features.items(), 
                                      key=lambda x: x[1], reverse=True)[:5]
                            ):
                                print(f"    {i+1}. {feature}: {importance:.4f}")
        
        print(f"\nAll results saved to: {config.data.output_dir}")
        print("=" * 60)
        print("Cross-Validation and Feature Importance Analysis Completed!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
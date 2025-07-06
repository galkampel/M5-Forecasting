"""
Data validation and quality monitoring utilities.

This module provides comprehensive data validation capabilities using
Great Expectations and Pandera for ensuring data quality throughout the pipeline.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import great_expectations as ge
    from great_expectations.core.batch import RuntimeBatchRequest
    from great_expectations.data_context import BaseDataContext
    from great_expectations.data_context.types.resource_identifiers import (
        GeCloudIdentifier,
    )

    GREAT_EXPECTATIONS_AVAILABLE = True
except ImportError:
    GREAT_EXPECTATIONS_AVAILABLE = False

try:
    import pandera as pa
    from pandera.typing import DataFrame, Series

    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False


class DataValidator:
    """
    Data validation utility using Great Expectations and Pandera.

    This class provides comprehensive data validation capabilities for
    ensuring data quality throughout the preprocessing pipeline.
    """

    def __init__(self, context_path: Optional[str] = None):
        """
        Initialize DataValidator.

        Args:
            context_path: Path to Great Expectations context
        """
        self.logger = logging.getLogger(__name__)
        self.context_path = context_path or "great_expectations"
        self.context: Optional[BaseDataContext] = None
        self.schemas: Dict[str, Any] = {}

        if GREAT_EXPECTATIONS_AVAILABLE:
            self._setup_great_expectations()

    def _setup_great_expectations(self) -> None:
        """Setup Great Expectations context."""
        try:
            self.context = ge.get_context()
            self.logger.info("Great Expectations context initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize Great Expectations: {e}")
            self.context = None

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        validation_name: str,
        expectations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Validate a DataFrame using Great Expectations.

        Args:
            df: DataFrame to validate
            validation_name: Name for this validation
            expectations: List of expectations to apply

        Returns:
            Validation results
        """
        if not GREAT_EXPECTATIONS_AVAILABLE or self.context is None:
            self.logger.warning("Great Expectations not available")
            return {"success": False, "error": "Great Expectations not available"}

        try:
            # Create batch request
            batch_request = RuntimeBatchRequest(
                datasource_name="pandas_datasource",
                data_connector_name="default_runtime_data_connector_name",
                data_asset_name=validation_name,
                runtime_parameters={"batch_data": df},
                batch_identifiers={"default_identifier_name": validation_name},
            )

            # Create validator
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=f"{validation_name}_suite",
            )

            # Apply expectations
            if expectations:
                for expectation in expectations:
                    expectation_type = expectation.pop("expectation_type")
                    getattr(validator, expectation_type)(**expectation)

            # Run validation
            results = validator.validate()

            return {
                "success": results.success,
                "statistics": results.statistics,
                "results": [
                    {
                        "expectation_type": result.expectation_config.expectation_type,
                        "success": result.success,
                        "kwargs": result.expectation_config.kwargs,
                    }
                    for result in results.run_results
                ],
            }

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {"success": False, "error": str(e)}

    def create_schema(self, name: str, schema_definition: Dict[str, Any]) -> None:
        """
        Create a Pandera schema for data validation.

        Args:
            name: Schema name
            schema_definition: Schema definition dictionary
        """
        if not PANDERA_AVAILABLE:
            self.logger.warning("Pandera not available")
            return

        try:
            schema = pa.DataFrameSchema.from_dict(schema_definition)
            self.schemas[name] = schema
            self.logger.info(f"Schema '{name}' created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create schema '{name}': {e}")

    def validate_with_schema(
        self, df: pd.DataFrame, schema_name: str
    ) -> Dict[str, Any]:
        """
        Validate DataFrame against a Pandera schema.

        Args:
            df: DataFrame to validate
            schema_name: Name of the schema to use

        Returns:
            Validation results
        """
        if not PANDERA_AVAILABLE:
            return {"success": False, "error": "Pandera not available"}

        if schema_name not in self.schemas:
            return {"success": False, "error": f"Schema '{schema_name}' not found"}

        try:
            schema = self.schemas[schema_name]
            validated_df = schema.validate(df)

            return {
                "success": True,
                "validated_dataframe": validated_df,
                "schema_name": schema_name,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "schema_name": schema_name}

    def generate_data_quality_report(
        self, df: pd.DataFrame, report_name: str = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.

        Args:
            df: DataFrame to analyze
            report_name: Name for the report

        Returns:
            Data quality report
        """
        if report_name is None:
            report_name = (
                f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        report = {
            "timestamp": datetime.now().isoformat(),
            "report_name": report_name,
            "dataframe_info": {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            },
            "missing_values": {
                "total_missing": df.isnull().sum().sum(),
                "missing_per_column": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            },
            "duplicates": {
                "total_duplicates": df.duplicated().sum(),
                "duplicate_percentage": df.duplicated().sum() / len(df) * 100,
            },
            "statistical_summary": df.describe().to_dict(),
            "column_analysis": {},
        }

        # Analyze each column
        for column in df.columns:
            col_data = df[column]
            col_analysis = {
                "dtype": str(col_data.dtype),
                "unique_count": col_data.nunique(),
                "unique_percentage": col_data.nunique() / len(col_data) * 100,
                "missing_count": col_data.isnull().sum(),
                "missing_percentage": col_data.isnull().sum() / len(col_data) * 100,
            }

            # Add type-specific analysis
            if pd.api.types.is_numeric_dtype(col_data):
                col_analysis.update(
                    {
                        "min": (
                            float(col_data.min())
                            if not col_data.isnull().all()
                            else None
                        ),
                        "max": (
                            float(col_data.max())
                            if not col_data.isnull().all()
                            else None
                        ),
                        "mean": (
                            float(col_data.mean())
                            if not col_data.isnull().all()
                            else None
                        ),
                        "std": (
                            float(col_data.std())
                            if not col_data.isnull().all()
                            else None
                        ),
                        "zeros": int((col_data == 0).sum()),
                        "negative": (
                            int((col_data < 0).sum())
                            if not col_data.isnull().all()
                            else 0
                        ),
                    }
                )
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                col_analysis.update(
                    {
                        "min_date": (
                            str(col_data.min()) if not col_data.isnull().all() else None
                        ),
                        "max_date": (
                            str(col_data.max()) if not col_data.isnull().all() else None
                        ),
                        "date_range_days": (
                            (col_data.max() - col_data.min()).days
                            if not col_data.isnull().all()
                            else None
                        ),
                    }
                )
            elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_string_dtype(
                col_data
            ):
                col_analysis.update(
                    {
                        "top_values": col_data.value_counts().head(5).to_dict(),
                        "avg_length": (
                            col_data.astype(str).str.len().mean()
                            if not col_data.isnull().all()
                            else None
                        ),
                    }
                )

            report["column_analysis"][column] = col_analysis

        return report

    def save_validation_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Save validation report to file.

        Args:
            report: Validation report
            output_path: Path to save the report
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Validation report saved to {output_path}")


class DataQualityMonitor:
    """
    Data quality monitoring utility for tracking data quality over time.
    """

    def __init__(self, output_dir: str = "logs/data_quality"):
        """
        Initialize DataQualityMonitor.

        Args:
            output_dir: Output directory for quality reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.validator = DataValidator()
        self.logger = logging.getLogger(__name__)
        self.quality_history: List[Dict[str, Any]] = []

    def monitor_dataframe(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        validation_rules: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Monitor data quality of a DataFrame.

        Args:
            df: DataFrame to monitor
            dataset_name: Name of the dataset
            validation_rules: Optional validation rules

        Returns:
            Monitoring results
        """
        # Generate quality report
        quality_report = self.validator.generate_data_quality_report(df, dataset_name)

        # Apply validation rules if provided
        validation_results = {}
        if validation_rules:
            validation_results = self.validator.validate_dataframe(
                df, dataset_name, validation_rules.get("expectations")
            )

        # Combine results
        monitoring_result = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "quality_report": quality_report,
            "validation_results": validation_results,
            "overall_quality_score": self._calculate_quality_score(quality_report),
        }

        # Store in history
        self.quality_history.append(monitoring_result)

        # Save report
        report_path = self.output_dir / f"{dataset_name}_quality_report.json"
        self.validator.save_validation_report(monitoring_result, str(report_path))

        self.logger.info(f"Data quality monitoring completed for {dataset_name}")
        return monitoring_result

    def _calculate_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """
        Calculate overall data quality score.

        Args:
            quality_report: Quality report

        Returns:
            Quality score (0-100)
        """
        score = 100.0

        # Penalize for missing values
        missing_percentages = quality_report["missing_values"][
            "missing_per_column"
        ].values()
        avg_missing = np.mean(list(missing_percentages))
        score -= avg_missing * 0.5  # 0.5 penalty per % missing

        # Penalize for duplicates
        duplicate_percentage = quality_report["duplicates"]["duplicate_percentage"]
        score -= duplicate_percentage * 0.3  # 0.3 penalty per % duplicates

        # Penalize for low cardinality (potential data quality issues)
        for col_analysis in quality_report["column_analysis"].values():
            unique_percentage = col_analysis["unique_percentage"]
            if unique_percentage < 1.0:  # Very low cardinality
                score -= 5.0
            elif unique_percentage < 5.0:  # Low cardinality
                score -= 2.0

        return max(0.0, score)

    def get_quality_trends(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get quality trends for a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Quality trends
        """
        dataset_history = [
            record
            for record in self.quality_history
            if record["dataset_name"] == dataset_name
        ]

        if not dataset_history:
            return {"error": f"No history found for dataset '{dataset_name}'"}

        # Sort by timestamp
        dataset_history.sort(key=lambda x: x["timestamp"])

        # Extract trends
        timestamps = [record["timestamp"] for record in dataset_history]
        quality_scores = [record["overall_quality_score"] for record in dataset_history]
        missing_percentages = [
            np.mean(
                list(
                    record["quality_report"]["missing_values"][
                        "missing_per_column"
                    ].values()
                )
            )
            for record in dataset_history
        ]

        return {
            "dataset_name": dataset_name,
            "total_monitoring_events": len(dataset_history),
            "first_monitoring": timestamps[0],
            "last_monitoring": timestamps[-1],
            "quality_score_trend": {
                "timestamps": timestamps,
                "scores": quality_scores,
                "mean_score": np.mean(quality_scores),
                "std_score": np.std(quality_scores),
                "trend": (
                    "improving"
                    if quality_scores[-1] > quality_scores[0]
                    else "declining"
                ),
            },
            "missing_values_trend": {
                "timestamps": timestamps,
                "missing_percentages": missing_percentages,
                "mean_missing": np.mean(missing_percentages),
                "trend": (
                    "improving"
                    if missing_percentages[-1] < missing_percentages[0]
                    else "declining"
                ),
            },
        }

    def generate_quality_dashboard(self) -> Dict[str, Any]:
        """
        Generate comprehensive quality dashboard.

        Args:
            Dashboard data
        """
        if not self.quality_history:
            return {"error": "No quality history available"}

        # Get unique datasets
        datasets = list(set(record["dataset_name"] for record in self.quality_history))

        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "total_monitoring_events": len(self.quality_history),
            "datasets_monitored": datasets,
            "overall_statistics": {
                "avg_quality_score": np.mean(
                    [record["overall_quality_score"] for record in self.quality_history]
                ),
                "min_quality_score": min(
                    [record["overall_quality_score"] for record in self.quality_history]
                ),
                "max_quality_score": max(
                    [record["overall_quality_score"] for record in self.quality_history]
                ),
            },
            "dataset_trends": {},
        }

        # Generate trends for each dataset
        for dataset in datasets:
            dashboard["dataset_trends"][dataset] = self.get_quality_trends(dataset)

        return dashboard


# Predefined validation schemas for common data types
def get_sales_data_schema() -> Dict[str, Any]:
    """Get predefined schema for sales data."""
    return {
        "columns": {
            "unique_id": pa.Column(str, nullable=False),
            "ds": pa.Column("datetime64[ns]", nullable=False),
            "sales": pa.Column(float, nullable=True, ge=0),
            "weekday": pa.Column(int, nullable=True, ge=0, le=6),
            "month": pa.Column(int, nullable=True, ge=1, le=12),
            "year": pa.Column(int, nullable=True, ge=2010, le=2030),
        },
        "index": None,
        "strict": True,
        "coerce": True,
    }


def get_features_data_schema() -> Dict[str, Any]:
    """Get predefined schema for features data."""
    return {
        "columns": {
            "unique_id": pa.Column(str, nullable=False),
            "ds": pa.Column("datetime64[ns]", nullable=False),
            "feature_1": pa.Column(float, nullable=True),
            "feature_2": pa.Column(float, nullable=True),
            "feature_3": pa.Column(float, nullable=True),
            "weekday": pa.Column(int, nullable=True, ge=0, le=6),
            "month": pa.Column(int, nullable=True, ge=1, le=12),
            "year": pa.Column(int, nullable=True, ge=2010, le=2030),
        },
        "index": None,
        "strict": True,
        "coerce": True,
    }

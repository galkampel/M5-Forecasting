# Preprocessing Analysis & Conversion Plan

## Overview
This document outlines the plan for converting the `preprocessing.ipynb` notebook into a well-structured Python package for advanced time series preprocessing with proper error handling, validation, documentation, and configuration management.

## Current Notebook Analysis

### preprocessing.ipynb Analysis
The notebook performs the following main operations:
1. **Data Loading**: Loads processed features and targets from dataset pipeline
2. **Feature Engineering**: Fourier modes, calendar features, encoding with baseline categories
3. **Model Configuration**: MLForecast parameter management with Pydantic models
4. **Model Training**: Ridge regression and LightGBM with MLForecast framework
5. **Cross-Validation**: Time series cross-validation with expanding windows
6. **Feature Importance**: Cross-validation feature importance analysis
7. **Model Evaluation**: Comprehensive evaluation with multiple metrics
8. **Visualization**: Results visualization and analysis plots
9. **Hyperparameter Optimization**: Optuna-based hyperparameter tuning

### Key Components Identified

#### 1. Custom Transformers
- `FourierModes`: Fourier seasonality feature generation
- `OHEwithBaseline`: OneHotEncoder with baseline category control
- `StandardScalerwithThreshold`: StandardScaler with threshold-based filtering
- `ZeroPredTransformer`: Zero prediction transformer for intermittent demand

#### 2. Model Configuration
- `MLForecastParams`: Pydantic model for MLForecast parameter management
- `CVFeatureImportance`: Cross-validation feature importance analysis

#### 3. Evaluation Framework
- `Evaluator`: Comprehensive evaluation class with multiple metrics
- Custom metrics: MASE, RMSSE, MRAE, F1-zero, non-zero MAE

#### 4. Visualization Functions
- `plot_aggregated_series`: Time series aggregation plots
- `series_analysis_plots`: Series analysis and diagnostic plots

## Proposed File Structure

```
src/
├── preprocessing/
│   ├── __init__.py
│   ├── main.py                    # Main entry point for preprocessing pipeline
│   ├── config.py                  # Configuration management
│   ├── data_loader.py             # Data loading utilities
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   ├── fourier_features.py    # Fourier seasonality features
│   │   ├── calendar_features.py   # Calendar feature engineering
│   │   ├── encoding_features.py   # Categorical encoding features
│   │   └── scaling_features.py    # Feature scaling and normalization
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                # Base model configuration classes
│   │   ├── mlforecast_config.py   # MLForecast parameter management
│   │   ├── ridge_config.py        # Ridge regression configuration
│   │   ├── lgbm_config.py         # LightGBM configuration
│   │   └── model_factory.py       # Model instantiation factory
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── feature_pipeline.py    # Feature engineering pipeline
│   │   ├── modeling_pipeline.py   # Model training pipeline
│   │   ├── evaluation_pipeline.py # Model evaluation pipeline
│   │   └── optimization_pipeline.py # Hyperparameter optimization pipeline
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Custom evaluation metrics
│   │   ├── evaluator.py           # Main evaluation class
│   │   ├── feature_importance.py  # Feature importance analysis
│   │   └── cross_validation.py    # Cross-validation strategies
│   ├── transformers/
│   │   ├── __init__.py
│   │   ├── base.py                # Base transformer classes
│   │   ├── fourier.py             # Fourier mode transformers
│   │   ├── encoding.py            # Encoding transformers
│   │   ├── scaling.py             # Scaling transformers
│   │   └── temporal.py            # Temporal feature transformers
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── optuna_config.py       # Optuna configuration
│   │   ├── objective_functions.py # Objective functions for optimization
│   │   └── search_spaces.py       # Hyperparameter search spaces
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── results_plots.py       # Results visualization
│   │   ├── series_plots.py        # Time series plots
│   │   ├── diagnostic_plots.py    # Diagnostic plots
│   │   └── feature_plots.py       # Feature analysis plots
│   ├── validators/
│   │   ├── __init__.py
│   │   ├── data_validator.py      # Data validation utilities
│   │   ├── model_validator.py     # Model validation
│   │   └── config_validator.py    # Configuration validation
│   └── utils/
│       ├── __init__.py
│       ├── logging.py             # Logging configuration
│       ├── caching.py             # Caching utilities
│       └── helpers.py             # Helper functions
├── config/
│   ├── preprocessing_default.yaml # Default preprocessing configuration
│   ├── preprocessing_development.yaml # Development configuration
│   └── preprocessing_production.yaml # Production configuration
└── tests/
    ├── test_preprocessing/
    │   ├── test_feature_engineering/
    │   ├── test_models/
    │   ├── test_pipelines/
    │   ├── test_evaluation/
    │   ├── test_transformers/
    │   └── test_optimization/
    └── test_integration/
```

## Detailed File Breakdown

### 1. Main Entry Point (`src/preprocessing/main.py`)

**Purpose**: Main orchestration script that coordinates the entire preprocessing pipeline.

**Key Components**:
- Command-line interface with argparse
- Configuration loading and validation
- Pipeline execution with error handling
- Logging setup
- Progress tracking
- Model comparison and selection

**Error Handling**:
- Data validation before preprocessing
- Model fitting failures
- Memory usage monitoring
- Graceful failure handling with rollback
- Model convergence checks

### 2. Configuration Management (`src/preprocessing/config.py`)

**Purpose**: Centralized configuration management for preprocessing pipeline.

**Configuration Parameters**:
```yaml
# Data configuration
data:
  features_path: "../data/processed/features_2012-01-01_2015-12-31.csv"
  targets_path: "../data/processed/target_features_2012-01-01_2015-12-31.csv"
  output_dir: "../outputs/preprocessing"
  cache_dir: "../cache"

# Feature engineering
features:
  fourier:
    enabled: true
    periods: [365.25, 7]  # yearly and weekly seasonality
    orders: [1, 1]        # number of harmonics
    time_cols: ["day_of_year", "week_of_year"]
    drop_time_cols: true
  calendar:
    enabled: true
    include_events: true
    include_snap: true
  encoding:
    baseline_categories:
      weekday: "Monday"
      month: 1
      year: 2012
      event: "No Event"
    drop_first: true
  scaling:
    enabled: true
    method: "standard"
    threshold: 0.0
    with_mean: true
    with_std: true

# Model configuration
models:
  ridge:
    enabled: true
    lags: [1, 7, 14, 30]
    lag_transforms:
      1: ["RollingMean", "RollingStd"]
      7: ["RollingMean", "RollingStd"]
      14: ["RollingMean"]
      30: ["RollingMean"]
    best_model_params:
      alpha: 1.0
  lgbm:
    enabled: true
    lags: [1, 7, 14, 30]
    lag_transforms:
      1: ["RollingMean", "RollingStd", "ExponentiallyWeightedMean"]
      7: ["RollingMean", "RollingStd"]
      14: ["RollingMean"]
      30: ["RollingMean"]
    best_model_params:
      n_estimators: 100
      learning_rate: 0.1
      max_depth: 6

# Cross-validation
cv:
  method: "expanding_window"
  initial_window: 730  # 2 years
  step_length: 30      # monthly steps
  forecast_horizon: 31 # 1 month ahead

# Evaluation
evaluation:
  metrics: ["mae", "rmse", "mape", "mase", "rmsse", "mrae", "f1_zero", "non_zero_mae"]
  save_predictions: true
  save_models: false
  confidence_intervals: false
  feature_importance: true

# Hyperparameter optimization
optimization:
  enabled: false
  framework: "optuna"
  n_trials: 100
  timeout: 3600  # 1 hour
  study_name: "preprocessing_optimization"

# Caching
caching:
  enabled: true
  cache_dir: "../cache"
  max_size: "2GB"
  ttl: 7200  # 2 hours

# Logging
logging:
  level: "INFO"
  file: "../logs/preprocessing.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### 3. Feature Engineering

#### 3.1 Fourier Features (`src/preprocessing/feature_engineering/fourier_features.py`)

**Purpose**: Fourier-based seasonality feature engineering.

**Key Components**:
- `FourierModes`: Fourier mode feature generation
- `MultiSeasonalFourier`: Multiple seasonality support
- `AdaptiveFourier`: Adaptive seasonality detection

**Error Handling**:
- Invalid period values
- Memory-efficient computation
- Numerical stability checks
- Time column validation

#### 3.2 Calendar Features (`src/preprocessing/feature_engineering/calendar_features.py`)

**Purpose**: Calendar-based feature engineering.

**Key Components**:
- `CalendarFeatureEngineer`: Main calendar feature engineering class
- `EventFeatureEngineer`: Event and holiday feature engineering
- `SNAPFeatureEngineer`: SNAP benefit feature engineering
- `SeasonalFeatureEngineer`: Seasonal pattern features

**Error Handling**:
- Missing date columns
- Invalid date formats
- Event data validation
- State ID validation

#### 3.3 Encoding Features (`src/preprocessing/feature_engineering/encoding_features.py`)

**Purpose**: Categorical encoding with baseline control.

**Key Components**:
- `OHEwithBaseline`: OneHotEncoder with baseline category
- `BaselineEncoder`: Generic baseline encoding
- `TemporalEncoder`: Temporal feature encoding

**Error Handling**:
- Missing baseline categories
- Invalid category values
- Memory-efficient encoding
- Data type validation

#### 3.4 Scaling Features (`src/preprocessing/feature_engineering/scaling_features.py`)

**Purpose**: Feature scaling and normalization.

**Key Components**:
- `StandardScalerwithThreshold`: StandardScaler with threshold filtering
- `RobustScaler`: Robust scaling for outliers
- `MinMaxScaler`: Min-max scaling

**Error Handling**:
- Zero variance features
- Threshold validation
- Memory-efficient scaling
- Data type validation

### 4. Models

#### 4.1 Base Models (`src/preprocessing/models/base.py`)

**Purpose**: Base classes for model configuration.

**Key Components**:
- `BaseModelConfig`: Base model configuration class
- `ModelRegistry`: Model registration and management
- `ModelFactory`: Model instantiation factory

#### 4.2 MLForecast Configuration (`src/preprocessing/models/mlforecast_config.py`)

**Purpose**: MLForecast parameter management.

**Key Components**:
- `MLForecastParams`: Pydantic model for MLForecast parameters
- `RidgeConfig`: Ridge regression configuration
- `LGBMConfig`: LightGBM configuration

**Error Handling**:
- Parameter validation
- Model compatibility checks
- Memory requirements validation
- Configuration schema validation

### 5. Pipelines

#### 5.1 Feature Pipeline (`src/preprocessing/pipelines/feature_pipeline.py`)

**Purpose**: Feature engineering pipeline orchestration.

**Key Components**:
- `FeatureEngineeringPipeline`: Complete feature engineering pipeline
- `FourierPipeline`: Fourier feature pipeline
- `EncodingPipeline`: Encoding pipeline
- `ScalingPipeline`: Scaling pipeline

**Error Handling**:
- Missing value handling
- Data type validation
- Memory management
- Pipeline validation

#### 5.2 Modeling Pipeline (`src/preprocessing/pipelines/modeling_pipeline.py`)

**Purpose**: Model training and evaluation pipeline.

**Key Components**:
- `ModelTrainingPipeline`: Model training orchestration
- `CrossValidationPipeline`: Cross-validation pipeline
- `FeatureImportancePipeline`: Feature importance pipeline
- `ModelComparisonPipeline`: Model comparison pipeline

**Error Handling**:
- Training failures
- Memory overflow protection
- Model convergence monitoring
- Graceful degradation

#### 5.3 Evaluation Pipeline (`src/preprocessing/pipelines/evaluation_pipeline.py`)

**Purpose**: Model evaluation and results analysis.

**Key Components**:
- `ModelEvaluationPipeline`: Comprehensive model evaluation
- `ResultsAnalysisPipeline`: Results analysis and interpretation
- `VisualizationPipeline`: Results visualization
- `ReportGenerationPipeline`: Report generation

#### 5.4 Optimization Pipeline (`src/preprocessing/pipelines/optimization_pipeline.py`)

**Purpose**: Hyperparameter optimization pipeline.

**Key Components**:
- `OptunaOptimizationPipeline`: Optuna-based optimization
- `SearchSpaceManager`: Search space management
- `ObjectiveFunctionManager`: Objective function management
- `StudyManager`: Study management

### 6. Evaluation

#### 6.1 Metrics (`src/preprocessing/evaluation/metrics.py`)

**Purpose**: Custom evaluation metrics.

**Key Components**:
- `MASE`: Mean Absolute Scaled Error
- `RMSSE`: Root Mean Squared Scaled Error
- `MRAE`: Mean Relative Absolute Error
- `F1Zero`: F1 score for zero predictions
- `NonZeroMAE`: Non-zero Mean Absolute Error

#### 6.2 Evaluator (`src/preprocessing/evaluation/evaluator.py`)

**Purpose**: Main evaluation class.

**Key Components**:
- `Evaluator`: Comprehensive evaluation class
- `GroupEvaluator`: Group-based evaluation
- `TemporalEvaluator`: Temporal evaluation
- `StatisticalEvaluator`: Statistical evaluation

#### 6.3 Feature Importance (`src/preprocessing/evaluation/feature_importance.py`)

**Purpose**: Feature importance analysis.

**Key Components**:
- `CVFeatureImportance`: Cross-validation feature importance
- `FeatureImportanceAnalyzer`: Feature importance analysis
- `ImportanceVisualizer`: Feature importance visualization

### 7. Transformers

#### 7.1 Fourier (`src/preprocessing/transformers/fourier.py`)

**Purpose**: Fourier mode transformers.

**Key Components**:
- `FourierModes`: Fourier mode feature generation
- `MultiSeasonalFourier`: Multiple seasonality support
- `AdaptiveFourier`: Adaptive seasonality detection

#### 7.2 Encoding (`src/preprocessing/transformers/encoding.py`)

**Purpose**: Encoding transformers.

**Key Components**:
- `OHEwithBaseline`: OneHotEncoder with baseline category
- `BaselineEncoder`: Generic baseline encoding
- `TemporalEncoder`: Temporal feature encoding

#### 7.3 Scaling (`src/preprocessing/transformers/scaling.py`)

**Purpose**: Scaling transformers.

**Key Components**:
- `StandardScalerwithThreshold`: StandardScaler with threshold filtering
- `RobustScaler`: Robust scaling for outliers
- `MinMaxScaler`: Min-max scaling

#### 7.4 Temporal (`src/preprocessing/transformers/temporal.py`)

**Purpose**: Temporal feature transformers.

**Key Components**:
- `LagTransformer`: Lag feature creation
- `RollingTransformer`: Rolling window features
- `SeasonalTransformer`: Seasonal feature extraction
- `ZeroPredTransformer`: Zero prediction transformer

### 8. Optimization

#### 8.1 Optuna Configuration (`src/preprocessing/optimization/optuna_config.py`)

**Purpose**: Optuna configuration management.

**Key Components**:
- `OptunaConfig`: Optuna configuration class
- `StudyConfig`: Study configuration
- `SamplerConfig`: Sampler configuration

#### 8.2 Objective Functions (`src/preprocessing/optimization/objective_functions.py`)

**Purpose**: Objective functions for optimization.

**Key Components**:
- `RidgeObjective`: Ridge regression objective function
- `LGBMObjective`: LightGBM objective function
- `MultiObjective`: Multi-objective optimization

#### 8.3 Search Spaces (`src/preprocessing/optimization/search_spaces.py`)

**Purpose**: Hyperparameter search spaces.

**Key Components**:
- `RidgeSearchSpace`: Ridge regression search space
- `LGBMSearchSpace`: LightGBM search space
- `FeatureSearchSpace`: Feature engineering search space

### 9. Visualization

#### 9.1 Results Plots (`src/preprocessing/visualization/results_plots.py`)

**Purpose**: Results visualization.

**Key Components**:
- `ResultsPlotter`: Results plotting utilities
- `MetricsPlotter`: Metrics visualization
- `ComparisonPlotter`: Model comparison plots

#### 9.2 Series Plots (`src/preprocessing/visualization/series_plots.py`)

**Purpose**: Time series visualization.

**Key Components**:
- `SeriesPlotter`: Time series plotting utilities
- `AggregationPlotter`: Aggregation plots
- `ForecastPlotter`: Forecast visualization

#### 9.3 Diagnostic Plots (`src/preprocessing/visualization/diagnostic_plots.py`)

**Purpose**: Diagnostic plots.

**Key Components**:
- `DiagnosticPlotter`: Diagnostic plotting utilities
- `ResidualPlotter`: Residual analysis plots
- `PerformancePlotter`: Performance diagnostic plots

#### 9.4 Feature Plots (`src/preprocessing/visualization/feature_plots.py`)

**Purpose**: Feature analysis plots.

**Key Components**:
- `FeaturePlotter`: Feature plotting utilities
- `ImportancePlotter`: Feature importance plots
- `CorrelationPlotter`: Feature correlation plots

## Implementation Plan

### Phase 1: Foundation (Week 1-2)
1. Set up project structure
2. Implement configuration management
3. Create base classes and interfaces
4. Set up logging and error handling
5. Implement data loading utilities

### Phase 2: Feature Engineering (Week 3-4)
1. Implement Fourier feature engineering
2. Implement calendar feature engineering
3. Implement encoding features
4. Implement scaling features
5. Create feature preprocessing pipeline
6. Add comprehensive testing

### Phase 3: Models & Configuration (Week 5-6)
1. Implement base model classes
2. Implement MLForecast configuration
3. Implement model factory and registry
4. Create model configuration validation
5. Add model compatibility checks

### Phase 4: Pipelines (Week 7-8)
1. Implement feature pipeline
2. Implement modeling pipeline
3. Implement evaluation pipeline
4. Implement optimization pipeline
5. Create pipeline orchestration

### Phase 5: Evaluation & Optimization (Week 9-10)
1. Implement custom metrics
2. Implement evaluator class
3. Implement feature importance analysis
4. Implement hyperparameter optimization
5. Add optimization monitoring

### Phase 6: Visualization & Testing (Week 11-12)
1. Implement visualization utilities
2. Comprehensive unit testing
3. Integration testing
4. Performance testing
5. Documentation completion

## Key Considerations

### Error Handling & Validation
- Comprehensive input validation
- Model convergence monitoring
- Memory usage tracking
- Graceful failure handling
- Data integrity checks
- Configuration validation

### Logging
- Structured logging with different levels
- Performance monitoring
- Error tracking and reporting
- Progress tracking for long-running operations
- Optimization progress tracking

### Configuration Management
- Environment-specific configurations
- Parameter validation
- Default value management
- Configuration schema validation
- Model parameter management

### Testing
- Unit tests for all components
- Integration tests for pipelines
- Performance benchmarks
- Error scenario testing
- Model validation tests
- Optimization testing

### Caching
- Model caching with TTL
- Feature caching for expensive operations
- Results caching for reproducibility
- Cache invalidation strategies
- Optimization result caching

### Transformations
- Reusable transformer components
- Pipeline composition
- Memory-efficient transformations
- Validation and error handling
- Sklearn-compatible interfaces

### Performance Optimization
- Memory-efficient data processing
- Parallel processing where applicable
- Caching strategies
- Model optimization techniques
- Hyperparameter optimization

## Success Metrics

1. **Code Quality**: 90%+ test coverage, PEP 8 compliance
2. **Performance**: 50%+ faster than notebook execution
3. **Reliability**: 99%+ success rate in pipeline execution
4. **Maintainability**: Clear documentation, modular design
5. **Scalability**: Support for larger datasets and more models
6. **Reproducibility**: Deterministic results with proper seeding
7. **Optimization**: Effective hyperparameter optimization
8. **Visualization**: Comprehensive results visualization

## Risk Mitigation

1. **Memory Issues**: Implement chunked processing and memory monitoring
2. **Model Convergence**: Add convergence checks and fallback strategies
3. **Data Quality**: Comprehensive validation and error handling
4. **Performance**: Profiling and optimization throughout development
5. **Compatibility**: Version pinning and dependency management
6. **Optimization**: Robust optimization with early stopping and fallbacks
7. **Complexity**: Modular design with clear interfaces
8. **Maintenance**: Comprehensive documentation and testing 
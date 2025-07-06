# Streamlit Interactive Data Analysis & Model Evaluation Tool

## Overview

The Streamlit tool provides an interactive web interface for comprehensive data analysis, model evaluation, and forecasting experimentation for the M5 forecasting competition. It integrates the dataset processing capabilities from `src/dataset` and the preprocessing/modeling pipeline from `src/preprocessing` to create a unified analysis platform focused on intermittent time series analysis.

## Architecture

### Core Components

1. **Dataset Processing Module** (`src/dataset` integration)
   - Data loading and preprocessing using existing pipelines
   - Intermittent time series identification and analysis
   - Data quality assessment and filtering

2. **Data Analysis Module** (Intermittent Time Series Focus)
   - Zero-demand pattern analysis
   - Demand frequency and interval analysis
   - Price-demand correlation analysis
   - Event impact analysis

3. **Model Evaluation Module** (`src/preprocessing` integration)
   - Automated model training with proper baseline comparison
   - Performance monitoring using evaluator.py metrics
   - Experiment management and tracking

4. **Visualization Dashboard** (`src/preprocessing/visualization` integration)
   - Interactive charts using existing visualization modules
   - Real-time data exploration
   - Comparative analysis views

## Detailed Specification

### 1. Dataset Processing Module

#### 1.1 Data Loading and Processing Interface

**Location**: `src/streamlit/data_processing.py`

**Features**:
- **Dataset Creation**: Use `src/dataset` pipelines to create processed datasets
- **Data Loading**: Load existing processed datasets or create new ones
- **Quality Assessment**: Run data quality checks using existing validation modules
- **Intermittent Series Identification**: Identify and flag intermittent time series

**Implementation**:
```python
def create_dataset_interface():
    """
    Creates the dataset processing interface:
    - Dataset creation using src/dataset pipelines
    - Data quality assessment
    - Intermittent series identification
    - Data preview and statistics
    """
    from src.dataset.main import main as dataset_main
    from src.dataset.config import Config
    
    # Load configuration
    config = Config.from_file("config/default.yaml")
    
    # Create dataset
    dataset_main(config)
    
    # Load processed data
    features_df, targets_df = load_processed_data()
    
    return features_df, targets_df
```

#### 1.2 Intermittent Time Series Analysis

**Features**:
- **Zero Analysis**:
  - Percentage of zero sales by item/store
  - Zero patterns over time
  - Intermittent demand classification
- **Demand Patterns**:
  - Demand frequency analysis
  - Demand size distribution
  - Demand interval analysis
- **Correlation Analysis**:
  - Price-demand correlation
  - Cross-item correlations
  - Store-item correlations

**Statistics**:
- Average demand interval
- Coefficient of variation
- Zero percentage
- Demand size statistics

### 2. Data Analysis Module

#### 2.1 Time Series Analysis

**Features**:
- **Individual Time Series View**:
  - Select specific store-item combinations
  - Interactive time series plot with zoom/pan
  - Zero-demand highlighting
  - Trend analysis overlay
- **Aggregated Analysis**:
  - Store-level aggregation
  - Department-level aggregation
  - Item-level aggregation
  - Custom grouping options
- **Seasonality Analysis**:
  - Weekly patterns
  - Monthly patterns
  - Holiday effects
  - Event impact analysis

**Visualizations** (using `src/preprocessing/visualization`):
- Line charts with multiple series
- Heatmaps for seasonal patterns
- Box plots for distribution analysis
- Correlation matrices

#### 2.2 Event Impact Analysis

**Features**:
- **Event Selection**: Dropdown for available events (SuperBowl, LaborDay, etc.)
- **Before/After Analysis**:
  - Sales comparison before/after events
  - Statistical significance testing
  - Effect size calculation
- **Event Correlation**:
  - Correlation between events and sales
  - Lag analysis (how long effects last)
  - Cross-event interactions

**Implementation**:
```python
def analyze_event_impact(data, event_name, pre_days=30, post_days=30):
    """
    Analyzes the impact of specific events on sales:
    - Calculates average sales before/after event
    - Performs statistical tests
    - Visualizes impact patterns
    """
```

#### 2.3 Price Analysis

**Features**:
- **Price Change Detection**:
  - Identify items with significant price changes
  - Price volatility analysis
  - Price-sales elasticity calculation
- **Price-Sales Relationship**:
  - Scatter plots of price vs sales
  - Correlation analysis
  - Price sensitivity by item category
- **Price Forecasting**: Simple price trend analysis

**Visualizations**:
- Price change heatmaps
- Price-sales scatter plots
- Price volatility time series
- Elasticity coefficient plots

### 3. Model Evaluation Module

#### 3.1 Experiment Configuration

**Location**: `src/streamlit/model_evaluation.py`

**Features**:
- **Model Selection**:
  - Ridge Regression
  - LightGBM
  - Baseline models (Naive, Seasonal Naive, Zero)
  - Custom model parameters
- **Feature Engineering**:
  - Enable/disable specific feature types
  - Custom feature selection
  - Feature importance analysis
- **Cross-Validation Setup**:
  - Number of folds
  - Validation strategy
  - Test period selection

**Interface**:
```python
def create_experiment_config():
    """
    Creates experiment configuration interface:
    - Model selection checkboxes
    - Parameter tuning sliders
    - Feature engineering options
    - Validation settings
    """
```

#### 3.2 Model Training and Evaluation

**Features**:
- **Automated Training**:
  - Progress bars for training
  - Real-time logging
  - Error handling and recovery
- **Performance Metrics** (using `src/preprocessing/evaluation/evaluator.py`):
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MASE (Mean Absolute Scaled Error) - using SeasonalNaive baseline
  - RMSSE (Root Mean Squared Scaled Error) - using SeasonalNaive baseline
  - MRAE (Mean Relative Absolute Error) - using ZeroModel baseline
  - F1-Zero (F1 score for zero predictions)
  - Non-Zero MAE (MAE for non-zero values only)
- **Model Comparison**:
  - Side-by-side performance comparison
  - Statistical significance testing
  - Best model selection

**Implementation**:
```python
def run_model_experiment(config, data):
    """
    Runs complete model experiment:
    - Trains selected models
    - Performs cross-validation
    - Calculates metrics using evaluator.py
    - Returns results for visualization
    """
    from src.preprocessing.evaluation.evaluator import Evaluator
    from src.preprocessing.pipelines.modeling_pipeline import ModelTrainingPipeline
    
    # Run modeling pipeline
    pipeline = ModelTrainingPipeline(config)
    results = pipeline.run(data)
    
    # Evaluate using proper baselines
    evaluator = Evaluator(
        target_col="y",
        seasonal_baseline_col="SeasonalNaive",
        zero_baseline_col="ZeroModel"
    )
    
    metrics = evaluator.evaluate_by_group(results, ["unique_id"])
    
    return metrics
```

#### 3.3 Feature Importance Analysis

**Features**:
- **Global Feature Importance**:
  - Overall feature rankings
  - Feature importance plots
  - Feature correlation analysis
- **Local Feature Importance**:
  - Item-specific importance
  - Store-specific importance
  - Time-varying importance
- **Feature Selection**:
  - Automatic feature selection
  - Manual feature selection
  - Feature ablation studies

#### 3.4 Baseline Comparison

**Features**:
- **Baseline Models**:
  - Naive forecasting
  - Seasonal Naive
  - Zero model
  - Moving average
- **Performance Comparison**:
  - Relative performance metrics
  - Statistical significance
  - Improvement analysis
- **Baseline Customization**:
  - Custom baseline parameters
  - Ensemble baselines

### 4. Visualization Dashboard

#### 4.1 Interactive Charts

**Technology**: Integration with `src/preprocessing/visualization`

**Chart Types**:
- **Time Series Plots** (using `time_series_plots.py`):
  - Multi-series line charts
  - Zoom and pan capabilities
  - Hover information
  - Legend toggling
- **Distribution Plots** (using `feature_analysis.py`):
  - Histograms
  - Box plots
  - Violin plots
  - Density plots
- **Correlation Plots** (using `feature_analysis.py`):
  - Heatmaps
  - Scatter plots
  - Correlation matrices
- **Performance Plots** (using `evaluation_plots.py`):
  - Model comparison charts
  - Error analysis plots
  - Feature importance charts

#### 4.2 Dashboard Layout

**Sidebar**:
- Data filtering controls
- Model configuration
- Analysis type selection
- Export options

**Main Area**:
- **Tab 1**: Dataset Overview
  - Summary statistics
  - Data quality metrics
  - Intermittent series identification
  - Sample data table
- **Tab 2**: Time Series Analysis
  - Individual series plots
  - Aggregated analysis
  - Seasonal patterns
  - Zero-demand patterns
- **Tab 3**: Model Evaluation
  - Training progress
  - Performance metrics (MAE, RMSE, MASE, RMSSE, MRAE, F1-Zero, Non-Zero MAE)
  - Model comparison
- **Tab 4**: Feature Analysis
  - Feature importance
  - Correlation analysis
  - Feature selection
- **Tab 5**: Insights & Reports
  - Key findings
  - Recommendations
  - Export capabilities

### 5. Monitoring and Tracking

#### 5.1 Experiment Tracking

**Features**:
- **Experiment History**:
  - Save/load experiments
  - Version control
  - Experiment comparison
- **Performance Monitoring**:
  - Real-time metrics tracking
  - Performance alerts
  - Degradation detection
- **Resource Monitoring**:
  - Memory usage
  - CPU utilization
  - Training time tracking

#### 5.2 Data Quality Monitoring

**Features**:
- **Data Validation** (using `src/preprocessing/utils/validation.py`):
  - Missing value detection
  - Outlier detection
  - Data type validation
- **Quality Metrics**:
  - Completeness scores
  - Consistency checks
  - Accuracy metrics
- **Alert System**:
  - Quality threshold alerts
  - Automated notifications
  - Quality reports

### 6. File Structure

```
src/streamlit/
├── __init__.py
├── main.py                 # Main Streamlit app entry point
├── data_processing.py      # Dataset processing using src/dataset
├── data_analysis.py        # Intermittent time series analysis
├── model_evaluation.py     # Model evaluation using evaluator.py
├── visualizations.py       # Integration with visualization modules
├── monitoring.py           # Monitoring and tracking
├── utils/
│   ├── __init__.py
│   ├── data_utils.py       # Data processing utilities
│   ├── plot_utils.py       # Plotting utilities
│   └── config_utils.py     # Configuration utilities
└── components/
    ├── __init__.py
    ├── filters.py          # Filtering components
    ├── charts.py           # Chart components
    └── metrics.py          # Metrics display components
```

### 7. Configuration

#### 7.1 Streamlit Configuration

**File**: `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

#### 7.2 App Configuration

**File**: `config/streamlit_config.yaml`

```yaml
app:
  title: "M5 Forecasting Analysis Tool"
  description: "Interactive data analysis and model evaluation for M5 forecasting"
  
data:
  default_path: "data/processed"
  supported_formats: ["csv", "parquet"]
  max_file_size: 100MB
  
models:
  default_models: ["Ridge", "LGBMRegressor"]
  baseline_models: ["Naive", "SeasonalNaive", "Zero"]
  
evaluation:
  metrics: ["mae", "rmse", "mase", "rmsse", "mrae", "f1_zero", "non_zero_mae"]
  seasonal_baseline: "SeasonalNaive"
  zero_baseline: "ZeroModel"
  
visualization:
  default_theme: "plotly_white"
  max_points: 10000
  chart_height: 500
  
monitoring:
  enable_tracking: true
  save_experiments: true
  alert_threshold: 0.1
```

### 8. Usage Examples

#### 8.1 Dataset Processing and Analysis

```python
# Start the Streamlit app
streamlit run src/streamlit/main.py

# Navigate to Dataset Overview tab
# Create dataset using src/dataset pipeline
# View intermittent series identification
# Analyze zero-demand patterns
```

#### 8.2 Model Experimentation

```python
# Navigate to Model Evaluation tab
# Select models: Ridge, LGBMRegressor
# Enable feature engineering: calendar, fourier, lag features
# Set cross-validation: 5 folds
# Run experiment and view results with proper metrics
```

#### 8.3 Feature Analysis

```python
# Navigate to Feature Analysis tab
# View global feature importance
# Filter by top 20 features
# Analyze item-specific importance
# Export feature selection results
```

### 9. Performance Considerations

#### 9.1 Data Loading Optimization

- **Caching**: Use Streamlit's caching for expensive operations
- **Lazy Loading**: Load data only when needed
- **Data Sampling**: Use sampling for large datasets in preview mode

#### 9.2 Visualization Optimization

- **Downsampling**: Reduce data points for large time series
- **Progressive Loading**: Load charts progressively
- **Memory Management**: Clear unused data from memory

#### 9.3 Model Training Optimization

- **Background Processing**: Run training in background threads
- **Progress Updates**: Real-time progress updates
- **Error Handling**: Graceful error handling and recovery

### 10. Security and Privacy

#### 10.1 Data Security

- **Local Processing**: All data processing happens locally
- **No Data Upload**: No data is sent to external servers
- **Secure Storage**: Encrypted storage for sensitive data

#### 10.2 Access Control

- **User Authentication**: Optional user authentication
- **Role-based Access**: Different access levels for different users
- **Audit Logging**: Track user actions and data access

### 11. Deployment

#### 11.1 Local Development

```bash
# Install dependencies
uv sync --extra streamlit

# Run development server
streamlit run src/streamlit/main.py --server.port 8501
```

#### 11.2 Production Deployment

```bash
# Build Docker image
docker build -t m5-streamlit .

# Run container
docker run -p 8501:8501 m5-streamlit
```

### 12. Testing

#### 12.1 Unit Tests

```python
# Test data processing functions
pytest tests/test_streamlit_data_processing.py

# Test model evaluation functions
pytest tests/test_streamlit_model_evaluation.py

# Test visualization functions
pytest tests/test_streamlit_visualizations.py
```

#### 12.2 Integration Tests

```python
# Test complete workflow
pytest tests/test_streamlit_integration.py

# Test UI components
pytest tests/test_streamlit_ui.py
```

### 13. Future Enhancements

#### 13.1 Advanced Features

- **Real-time Data**: Connect to live data sources
- **Advanced Models**: Support for deep learning models
- **Automated Insights**: AI-powered insights generation
- **Collaboration**: Multi-user collaboration features

#### 13.2 Integration

- **MLflow**: Integration with MLflow for experiment tracking
- **Weights & Biases**: Integration with W&B for model monitoring
- **DVC**: Integration with DVC for data version control
- **Kubernetes**: Kubernetes deployment support

This specification provides a comprehensive framework for building an interactive Streamlit tool that leverages the existing `src/dataset` and `src/preprocessing` modules to create a powerful data analysis and model evaluation platform focused on intermittent time series analysis. 
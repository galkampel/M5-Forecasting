# M5 Forecasting Streamlit App

A comprehensive interactive web application for data analysis, model evaluation, and forecasting experimentation for the M5 forecasting competition.

## Features

### 📊 Dataset Processing
- **Dataset Creation**: Use existing `src/dataset` pipelines to create processed datasets
- **Data Loading**: Load existing processed datasets or create new ones
- **File Upload**: Upload custom CSV files for analysis
- **Quality Assessment**: Run data quality checks using existing validation modules
- **Intermittent Series Identification**: Identify and flag intermittent time series

### 📈 Time Series Analysis
- **Individual Series View**: Select specific store-item combinations with interactive plots
- **Aggregated Analysis**: Store-level, department-level, and item-level aggregation
- **Seasonality Analysis**: Weekly, monthly, and yearly patterns
- **Event Impact Analysis**: Analyze the impact of events on sales
- **Price Analysis**: Price-demand correlation and volatility analysis
- **Demand Statistics**: Comprehensive demand analysis and statistics

### 🤖 Model Evaluation
- **Automated Training**: Ridge Regression, LightGBM, and baseline models (Naive, SeasonalNaive, ZeroModel)
- **Performance Metrics**: MAE, RMSE, MASE, RMSSE, MRAE, F1-Zero, Non-Zero MAE
- **Cross-Validation**: Configurable cross-validation with proper baseline comparison
- **Feature Importance**: Global and local feature importance analysis
- **Hyperparameter Optimization**: Optuna integration for model tuning
- **Experiment Tracking**: Save and compare experiment results

### 🔍 Feature Analysis
- **Feature Distributions**: Interactive histograms and box plots
- **Feature Correlations**: Correlation matrices and heatmaps
- **Feature Trends**: Time-based feature analysis
- **Missing Values Analysis**: Data quality assessment

### 📋 Monitoring & Tracking
- **Experiment Tracking**: Save and compare experiments
- **Data Quality Monitoring**: Real-time quality metrics and alerts
- **Performance Monitoring**: Model performance tracking
- **Resource Monitoring**: CPU, memory, and disk usage
- **Alert System**: Configurable alerts for quality and performance issues

## Installation

### Prerequisites
- Python 3.12+
- UV package manager
- M5 forecasting project setup

### Setup
1. **Install Dependencies**:
   ```bash
   # Install using UV (recommended)
   uv sync
   
   # Or install specific packages
   uv add streamlit plotly
   ```

2. **Verify Installation**:
   ```bash
   python test_streamlit_app.py
   ```

3. **Run the App**:
   ```bash
   streamlit run src/streamlit/main.py
   ```

## Usage

### Starting the App
```bash
# Navigate to project directory
cd /path/to/m5_forecasting

# Run the Streamlit app
streamlit run src/streamlit/main.py
```

The app will open in your default web browser at `http://localhost:8501`.

### Navigation

The app has three main sections accessible from the sidebar:

1. **Dataset Overview**: Create/load datasets, upload files, and view data quality metrics
2. **Time Series Analysis**: Interactive time series exploration and analysis
3. **Model Evaluation**: Configure and run model experiments

### Workflow

#### 1. Dataset Processing
1. Navigate to "Dataset Overview"
2. Choose one of the following options:
   - Click "Create New Dataset" to run the dataset pipeline
   - Click "Load Existing Dataset" to load processed data
   - Upload custom CSV files using the file uploaders
3. View data quality metrics and intermittent series analysis

#### 2. Time Series Analysis
1. Navigate to "Time Series Analysis"
2. Select analysis type:
   - **Individual Series**: Analyze specific store-item combinations
   - **Aggregated Analysis**: Store-level, department-level, and item-level views
   - **Seasonality Analysis**: Weekly, monthly, and yearly patterns
   - **Price Analysis**: Price-demand correlation and volatility
   - **Demand Statistics**: Comprehensive demand analysis
3. Use interactive controls to explore the data
4. View patterns, trends, and seasonal effects

#### 3. Model Evaluation
1. Navigate to "Model Evaluation"
2. Configure experiment settings:
   - Select models (Ridge, LightGBM, Baselines)
   - Enable/disable feature engineering
   - Set cross-validation parameters
   - Enable hyperparameter optimization
3. Click "Run Model Training & Evaluation" to start training
4. View results and performance metrics
5. Analyze feature importance and model comparisons

## Configuration

### App Configuration
The app uses configuration files for customization:

- **`config/streamlit_config.yaml`**: App-specific configuration

### Key Configuration Options

#### Models
```yaml
models:
  default_models: ["Ridge", "LGBMRegressor"]
  baseline_models: ["Naive", "SeasonalNaive", "Zero"]
```

#### Evaluation Metrics
```yaml
evaluation:
  metrics: ["mae", "rmse", "mase", "rmsse", "mrae", "f1_zero", "non_zero_mae"]
  seasonal_baseline: "SeasonalNaive"
  zero_baseline: "ZeroModel"
```

#### Visualization
```yaml
visualization:
  default_theme: "plotly_white"
  max_points: 10000
  chart_height: 500
```

#### Monitoring
```yaml
monitoring:
  enable_tracking: true
  save_experiments: true
  alert_threshold: 0.1
```

## Architecture

### File Structure
```
src/streamlit/
├── main.py                 # Main Streamlit app entry point
├── data_processing.py      # Dataset processing and file upload
├── data_analysis.py        # Time series analysis and visualization
├── model_evaluation.py     # Model evaluation and training
├── visualizations.py       # Interactive visualization components
├── monitoring.py           # Monitoring and tracking
├── utils/
│   └── config_utils.py     # Configuration utilities
└── components/             # Reusable UI components
```

### Integration Points

#### Dataset Pipeline (`src/dataset`)
- Uses existing dataset creation pipeline
- Loads processed data from `data/processed/`
- Integrates with data validation modules

#### Preprocessing Pipeline (`src/preprocessing`)
- Uses existing modeling pipeline with AutoMLForecast
- Integrates with evaluation modules
- Uses feature importance analysis
- Supports cross-validation and hyperparameter optimization

#### Visualization Modules (`src/preprocessing/visualization`)
- Integrates with existing visualization modules
- Uses Plotly for interactive charts
- Supports custom chart creation

## Performance Considerations

### Data Loading
- **Caching**: Uses Streamlit's caching for expensive operations
- **Lazy Loading**: Loads data only when needed
- **Data Sampling**: Uses sampling for large datasets in preview mode

### Visualization
- **Downsampling**: Reduces data points for large time series
- **Progressive Loading**: Loads charts progressively
- **Memory Management**: Clears unused data from memory

### Model Training
- **Background Processing**: Runs training in background threads
- **Progress Updates**: Real-time progress updates
- **Error Handling**: Graceful error handling and recovery
- **Resource Management**: Monitors CPU and memory usage

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure you're in the project directory
cd /path/to/m5_forecasting

# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall dependencies
uv sync
```

#### Data Loading Issues
```bash
# Check if processed data exists
ls -la data/processed/

# Run dataset pipeline first
python src/dataset/main.py
```

#### Memory Issues
- Reduce `max_points` in visualization settings
- Use data sampling for large datasets
- Close unused browser tabs

#### Model Training Issues
- Check if all required packages are installed
- Verify data format and quality
- Check available system resources

### Debug Mode
```bash
# Run with debug information
streamlit run src/streamlit/main.py --logger.level=debug
```

## Development

### Adding New Features

1. **Create Module**:
   ```python
   # src/streamlit/new_feature.py
   import streamlit as st
   
   def create_new_feature():
       st.subheader("New Feature")
       # Implementation here
   ```

2. **Update Main App**:
   ```python
   # src/streamlit/main.py
   from src.streamlit.new_feature import create_new_feature
   
   # Add to navigation
   if page == "New Feature":
       create_new_feature()
   ```

3. **Add Configuration**:
   ```yaml
   # config/streamlit_config.yaml
   new_feature:
     enabled: true
     settings: {}
   ```

### Testing
```bash
# Run tests
python test_streamlit_app.py

# Run specific test
python -c "from src.streamlit.main import main; main()"
```

## Deployment

### Local Development
```bash
# Development server
streamlit run src/streamlit/main.py --server.port 8501
```

### Production Deployment
```bash
# Build Docker image
docker build -t m5-streamlit .

# Run container
docker run -p 8501:8501 m5-streamlit
```

### Docker Configuration
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "src/streamlit/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub
4. Contact the development team

## Changelog

### Version 2.0.0 (Current)
- **Enhanced Model Evaluation**: AutoMLForecast integration with Ridge and LightGBM
- **Improved Cross-Validation**: Proper baseline comparison and metrics calculation
- **File Upload Support**: Upload custom CSV files for analysis
- **Advanced Time Series Analysis**: Comprehensive demand statistics and price analysis
- **Better Error Handling**: Graceful error handling and recovery
- **Performance Optimizations**: Improved caching and memory management

### Version 1.0.0
- Initial release
- Basic dataset processing and analysis
- Model evaluation with cross-validation
- Interactive visualizations
- Monitoring and tracking features 
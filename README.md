# M5 Forecasting Project

A comprehensive Python package for Walmart sales forecasting with advanced monitoring, automation, and data quality management.

## 🚀 Features

- **Advanced ML Pipeline**: Complete preprocessing, modeling, and evaluation pipeline
- **Performance Monitoring**: Real-time system metrics and memory profiling
- **Data Quality Management**: Automated validation and quality monitoring
- **CI/CD Integration**: GitHub Actions with comprehensive testing
- **Containerization**: Docker support for development and production
- **Automation Tools**: Makefile and scripts for common tasks
- **Comprehensive Testing**: Unit, integration, and end-to-end tests
- **Code Quality**: Automated linting, formatting, and type checking

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Monitoring & Automation](#monitoring--automation)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## 🛠 Installation

### Prerequisites

- Python 3.12+
- UV package manager
- Git

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd m5_forecasting

# Install dependencies
uv sync
```

### Development Installation

```bash
# Install development dependencies
uv sync --extra dev --extra monitoring --extra validation

# Install pre-commit hooks
pre-commit install
```

### Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build specific services
docker-compose build jupyter monitoring
```

## 🚀 Quick Start

### Command Line Usage

```bash
# Run complete pipeline
python scripts/run_pipeline.py

# Run with custom configuration
python scripts/run_pipeline.py --config config/custom.yaml

# Run with monitoring
python scripts/run_pipeline.py --output-dir outputs/monitored_run
```

### Programmatic Usage

```python
from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.pipelines import ModelTrainingPipeline
from src.preprocessing.utils.monitoring import PipelineMonitor

# Load configuration
config = PreprocessingConfig()

# Initialize monitoring
monitor = PipelineMonitor()

# Run pipeline with monitoring
with monitor.monitor_pipeline_stage("model_training"):
    pipeline = ModelTrainingPipeline(config)
    results = pipeline.run(features_df, targets_df)
```

### Jupyter Development

```bash
# Start Jupyter Lab
uv run jupyter lab

# Or use Docker
docker-compose up jupyter
```

Access Jupyter at: http://localhost:8888 (token: m5forecast)

## 📊 Monitoring & Automation

### Performance Monitoring

The project includes comprehensive performance monitoring:

```python
from src.preprocessing.utils.monitoring import PipelineMonitor

# Initialize monitor
monitor = PipelineMonitor("logs/monitoring")

# Monitor pipeline stages
with monitor.monitor_pipeline_stage("data_loading"):
    # Your data loading code
    pass

# Generate performance report
report_path = monitor.generate_report()
```

### Data Quality Monitoring

Automated data validation and quality tracking:

```python
from src.preprocessing.utils.validation import DataQualityMonitor

# Initialize quality monitor
quality_monitor = DataQualityMonitor("outputs/data_quality")

# Monitor data quality
result = quality_monitor.monitor_dataframe(df, "dataset_name")

# Generate quality dashboard
dashboard = quality_monitor.generate_quality_dashboard()
```

### Automation Scripts

Use the Makefile for common tasks:

```bash
# Development setup
make dev-setup

# Run tests
make test
make test-coverage

# Code quality checks
make lint
make format
make type-check

# Run pipeline
make run-pipeline
make run-monitoring

# Data validation
make validate-data
make quality-report
```

### CI/CD Pipeline

The project includes GitHub Actions for automated testing and deployment:

- **Automated Testing**: Unit, integration, and end-to-end tests
- **Code Quality**: Linting, formatting, type checking, security scans
- **Dependency Management**: Automated dependency updates and security reviews
- **Deployment**: Automated deployment to production environments

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test types
pytest tests/ -m "unit" -v
pytest tests/ -m "integration" -v
pytest tests/ -m "e2e" -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Pipeline component integration
- **End-to-End Tests**: Complete pipeline validation
- **Performance Tests**: Benchmarking and profiling

## 🐳 Docker Support

### Development Environment

```bash
# Start all services
docker-compose up -d

# Start specific services
docker-compose up jupyter monitoring

# Build and run specific target
docker build --target development -t m5-forecast-dev .
docker run -p 8888:8888 m5-forecast-dev
```

### Production Deployment

```bash
# Build production image
docker build --target production -t m5-forecast-prod .

# Run production container
docker run -d \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -p 8000:8000 \
  m5-forecast-prod
```

## 📁 Project Structure

```
m5_forecasting/
├── src/                          # Source code
│   ├── dataset/                  # Dataset processing
│   │   ├── pipelines/           # Data pipelines
│   │   ├── transformers/        # Data transformers
│   │   └── utils/               # Utilities
│   └── preprocessing/           # ML preprocessing
│       ├── pipelines/           # ML pipelines
│       ├── models/              # Model configurations
│       ├── evaluation/          # Evaluation utilities
│       ├── visualization/       # Visualization tools
│       └── utils/               # Utilities
├── config/                      # Configuration files
├── data/                        # Data files
├── outputs/                     # Output files
├── logs/                        # Log files
├── tests/                       # Test files
├── scripts/                     # Automation scripts
├── docs/                        # Documentation
├── .github/                     # GitHub Actions
├── Dockerfile                   # Docker configuration
├── docker-compose.yml          # Docker Compose
├── Makefile                    # Automation tasks
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## ⚙️ Configuration

### Environment Variables

```bash
# Logging
LOG_LEVEL=INFO
ENABLE_MONITORING=true

# Data paths
DATA_DIR=./data
OUTPUT_DIR=./outputs
CACHE_DIR=./cache

# Model settings
ENABLE_OPTIMIZATION=true
N_TRIALS=100
```

### Configuration Files

The project uses YAML configuration files:

```yaml
# config/default.yaml
data:
  calendar_path: "data/calendar.csv"
  sales_path: "data/sales_train_validation.csv"
  output_dir: "data/processed"

models:
  ridge:
    enabled: true
    alpha: 1.0
  lgbm:
    enabled: true
    n_estimators: 100

evaluation:
  metrics: ["mae", "rmse", "mape"]
  save_predictions: true
```

## 📚 API Reference

### Core Classes

#### PipelineMonitor
Performance monitoring utility for tracking system resources and execution times.

```python
monitor = PipelineMonitor(output_dir="logs/monitoring")
with monitor.monitor_pipeline_stage("stage_name"):
    # Your code here
    pass
```

#### DataQualityMonitor
Data quality monitoring and validation utility.

```python
quality_monitor = DataQualityMonitor("outputs/data_quality")
result = quality_monitor.monitor_dataframe(df, "dataset_name")
```

#### ModelTrainingPipeline
Complete model training pipeline with AutoMLForecast.

```python
pipeline = ModelTrainingPipeline(config)
results = pipeline.run(features_df, targets_df)
```

### Command Line Tools

```bash
# Main pipeline
m5-forecast --config config/default.yaml

# Preprocessing only
m5-preprocess --output-dir outputs/preprocessing

# Model training
m5-train --config config/models.yaml

# Evaluation
m5-evaluate --output-dir outputs/evaluation

# Optimization
m5-optimize --n-trials 100
```

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `make dev-setup`
4. Make your changes
5. Run tests: `make test`
6. Submit a pull request

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Security checks
make security-check
```

### Pre-commit Hooks

The project uses pre-commit hooks for automated code quality:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 📊 Monitoring Dashboard

Access monitoring dashboards:

- **Performance Monitoring**: `logs/monitoring/`
- **Data Quality**: `outputs/data_quality/`
- **MLflow Tracking**: http://localhost:5000
- **Jupyter Lab**: http://localhost:8888

## 🔧 Troubleshooting

### Common Issues

1. **Memory Issues**: Use monitoring tools to identify bottlenecks
2. **Data Quality**: Check validation reports in `outputs/data_quality/`
3. **Performance**: Review monitoring reports in `logs/monitoring/`

### Getting Help

- Check the logs in `logs/` directory
- Review monitoring reports
- Run `make check-system` for system diagnostics
- Check Docker logs: `docker-compose logs`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- M5 Forecasting competition organizers
- MLForecast and AutoMLForecast teams
- Open source community contributors


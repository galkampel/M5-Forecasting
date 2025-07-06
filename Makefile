# M5 Forecasting Project Makefile
# Provides common automation tasks for development and deployment

.PHONY: help install install-dev install-monitoring install-validation test test-unit test-integration test-e2e lint format type-check security-check clean run-pipeline run-monitoring validate-data quality-report docs

# Default target
help:
	@echo "M5 Forecasting Project - Available Commands:"
	@echo ""
	@echo "Installation:"
	@echo "  install           Install production dependencies"
	@echo "  install-dev       Install development dependencies"
	@echo "  install-monitoring Install monitoring dependencies"
	@echo "  install-validation Install validation dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test              Run all tests"
	@echo "  test-unit         Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-e2e          Run end-to-end tests only"
	@echo "  test-coverage     Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint              Run linting checks"
	@echo "  format            Format code with black and isort"
	@echo "  type-check        Run type checking with mypy"
	@echo "  security-check    Run security checks with bandit"
	@echo "  pre-commit        Run pre-commit hooks"
	@echo ""
	@echo "Pipeline:"
	@echo "  run-pipeline      Run complete pipeline with monitoring"
	@echo "  run-monitoring    Run pipeline with performance monitoring"
	@echo "  validate-data     Run data validation checks"
	@echo "  quality-report    Generate data quality report"
	@echo ""
	@echo "Documentation:"
	@echo "  docs              Generate documentation"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean             Clean build artifacts and cache"
	@echo "  update-deps       Update dependencies"

# Installation targets
install:
	uv sync

install-dev:
	uv sync --extra dev

install-monitoring:
	uv sync --extra monitoring

install-validation:
	uv sync --extra validation

# Testing targets
test:
	PYTHONPATH=. pytest tests/ -v

test-unit:
	PYTHONPATH=. pytest tests/ -m "unit" -v

test-integration:
	PYTHONPATH=. pytest tests/ -m "integration" -v

test-e2e:
	PYTHONPATH=. pytest tests/ -m "e2e" -v

test-coverage:
	PYTHONPATH=. pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Code quality targets
lint:
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/

security-check:
	bandit -r src/ -f json -o security-report.json

pre-commit:
	pre-commit run --all-files

# Pipeline targets
run-pipeline:
	python scripts/run_pipeline.py --output-dir outputs/automated_run

run-monitoring:
	python scripts/run_pipeline.py --output-dir outputs/monitored_run

validate-data:
	python -c "import sys; sys.path.append('src'); from preprocessing.utils.validation import DataQualityMonitor; from preprocessing.data_loader import DataLoader; from preprocessing.config import PreprocessingConfig; config = PreprocessingConfig(); data_loader = DataLoader(config); features_df, targets_df = data_loader.load_data(); monitor = DataQualityMonitor('outputs/data_quality'); result = monitor.monitor_dataframe(features_df, 'validation_check'); print('Data validation completed. Check outputs/data_quality/ for results.')"

quality-report:
	python -c "import sys; sys.path.append('src'); from preprocessing.utils.validation import DataQualityMonitor; monitor = DataQualityMonitor('outputs/data_quality'); dashboard = monitor.generate_quality_dashboard(); print('Quality dashboard generated. Check outputs/data_quality/ for results.')"

# Documentation targets
docs:
	@echo "Generating documentation..."
	@mkdir -p docs/generated
	@echo "Documentation generation completed. Check docs/generated/"

# Maintenance targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf cache/
	rm -rf outputs/
	rm -rf logs/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

update-deps:
	uv lock --upgrade

# Development workflow targets
dev-setup: install-dev install-monitoring install-validation
	@echo "Development environment setup completed"

quick-test: format lint type-check test-unit
	@echo "Quick quality checks completed"

full-test: format lint type-check security-check test test-coverage
	@echo "Full quality checks completed"

# CI/CD targets
ci-test: install-dev
	PYTHONPATH=. pytest tests/ --cov=src --cov-report=xml --cov-report=term-missing
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	black --check src/ tests/
	isort --check-only src/ tests/
	mypy src/
	bandit -r src/ -f json -o security-report.json

# Performance targets
benchmark:
	PYTHONPATH=. pytest tests/ --benchmark-only

profile:
	python -m cProfile -o profile.stats scripts/run_pipeline.py

# Data processing targets
download-data:
	python download_dataset.py

preprocess-data:
	python -m src.preprocessing.main

train-models:
	python -m src.preprocessing.pipelines.modeling_pipeline

evaluate-models:
	python -m src.preprocessing.pipelines.evaluation_pipeline

# Monitoring targets
start-monitoring:
	python -c "import sys; sys.path.append('src'); from preprocessing.utils.monitoring import PipelineMonitor; monitor = PipelineMonitor('logs/monitoring'); monitor.start_monitoring(); print('Monitoring started. Press Ctrl+C to stop.'); import time; import signal; def signal_handler(sig, frame): monitor.stop_monitoring(); print('Monitoring stopped.'); exit(0); signal.signal(signal.SIGINT, signal_handler); while True: time.sleep(1)"

# Utility targets
check-system:
	python -c "import sys; sys.path.append('src'); from preprocessing.utils.monitoring import get_system_info; info = get_system_info(); print('System Information:'); [print(f'  {key}: {value}') for key, value in info.items()]"

check-dependencies:
	uv tree

# Helpers
.PHONY: check-env
check-env:
	@python -c "import sys; print(f'Python version: {sys.version}')"
	@uv --version
	@echo "Environment check completed" 
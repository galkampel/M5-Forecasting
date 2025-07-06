"""
Configuration utilities for Streamlit app.

This module handles loading and managing configuration for the Streamlit app.
"""

import streamlit as st
import yaml
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


def load_streamlit_config():
    """
    Load Streamlit app configuration.
    
    Returns:
        dict: Configuration dictionary
    """
    # Default configuration
    default_config = {
        'app': {
            'title': 'M5 Forecasting Analysis Tool',
            'description': 'Interactive data analysis and model evaluation for M5 forecasting'
        },
        'data': {
            'default_path': 'data/processed',
            'supported_formats': ['csv', 'parquet'],
            'max_file_size': '100MB'
        },
        'models': {
            'default_models': ['Ridge', 'LGBMRegressor'],
            'baseline_models': ['Naive', 'SeasonalNaive', 'Zero']
        },
        'evaluation': {
            'metrics': ['mae', 'rmse', 'mase', 'rmsse', 'mrae', 'f1_zero', 'non_zero_mae'],
            'seasonal_baseline': 'SeasonalNaive',
            'zero_baseline': 'ZeroModel'
        },
        'visualization': {
            'default_theme': 'plotly_white',
            'max_points': 10000,
            'chart_height': 500
        },
        'monitoring': {
            'enable_tracking': True,
            'save_experiments': True,
            'alert_threshold': 0.1
        }
    }
    
    # Try to load from config file
    config_file = Path("config/streamlit_config.yaml")
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                # Merge with default config
                default_config.update(file_config)
        except Exception as e:
            st.warning(f"Error loading config file: {str(e)}. Using default configuration.")
    
    return default_config


def get_config_value(config, key_path, default=None):
    """
    Get configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'app.title')
        default: Default value if key not found
        
    Returns:
        Value from configuration
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def save_streamlit_config(config):
    """
    Save Streamlit app configuration to file.
    
    Args:
        config: Configuration dictionary
    """
    try:
        # Create config directory if it doesn't exist
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        # Save to file
        config_file = config_dir / "streamlit_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        st.success("Configuration saved successfully!")
        
    except Exception as e:
        st.error(f"Error saving configuration: {str(e)}")


def create_config_interface():
    """
    Create configuration interface for Streamlit app.
    
    Returns:
        dict: Updated configuration
    """
    st.subheader("Configuration Settings")
    
    # Load current configuration
    config = load_streamlit_config()
    
    # App settings
    st.write("**App Settings**")
    col1, col2 = st.columns(2)
    
    with col1:
        config['app']['title'] = st.text_input(
            "App Title",
            value=config['app']['title']
        )
    
    with col2:
        config['app']['description'] = st.text_input(
            "App Description",
            value=config['app']['description']
        )
    
    # Data settings
    st.write("**Data Settings**")
    col1, col2 = st.columns(2)
    
    with col1:
        config['data']['default_path'] = st.text_input(
            "Default Data Path",
            value=config['data']['default_path']
        )
    
    with col2:
        config['data']['max_file_size'] = st.text_input(
            "Max File Size",
            value=config['data']['max_file_size']
        )
    
    # Model settings
    st.write("**Model Settings**")
    col1, col2 = st.columns(2)
    
    with col1:
        default_models = st.multiselect(
            "Default Models",
            options=['Ridge', 'LGBMRegressor', 'XGBoost', 'RandomForest'],
            default=config['models']['default_models']
        )
        config['models']['default_models'] = default_models
    
    with col2:
        baseline_models = st.multiselect(
            "Baseline Models",
            options=['Naive', 'SeasonalNaive', 'Zero', 'MovingAverage'],
            default=config['models']['baseline_models']
        )
        config['models']['baseline_models'] = baseline_models
    
    # Evaluation settings
    st.write("**Evaluation Settings**")
    col1, col2 = st.columns(2)
    
    with col1:
        metrics = st.multiselect(
            "Evaluation Metrics",
            options=['mae', 'rmse', 'mase', 'rmsse', 'mrae', 'f1_zero', 'non_zero_mae'],
            default=config['evaluation']['metrics']
        )
        config['evaluation']['metrics'] = metrics
    
    with col2:
        config['evaluation']['seasonal_baseline'] = st.selectbox(
            "Seasonal Baseline",
            options=['SeasonalNaive', 'Naive', 'Zero'],
            index=['SeasonalNaive', 'Naive', 'Zero'].index(config['evaluation']['seasonal_baseline'])
        )
    
    # Visualization settings
    st.write("**Visualization Settings**")
    col1, col2 = st.columns(2)
    
    with col1:
        config['visualization']['default_theme'] = st.selectbox(
            "Default Theme",
            options=['plotly_white', 'plotly_dark', 'ggplot2', 'seaborn'],
            index=['plotly_white', 'plotly_dark', 'ggplot2', 'seaborn'].index(config['visualization']['default_theme'])
        )
    
    with col2:
        config['visualization']['max_points'] = st.number_input(
            "Max Points",
            min_value=1000,
            max_value=50000,
            value=config['visualization']['max_points']
        )
    
    # Monitoring settings
    st.write("**Monitoring Settings**")
    col1, col2 = st.columns(2)
    
    with col1:
        config['monitoring']['enable_tracking'] = st.checkbox(
            "Enable Tracking",
            value=config['monitoring']['enable_tracking']
        )
    
    with col2:
        config['monitoring']['save_experiments'] = st.checkbox(
            "Save Experiments",
            value=config['monitoring']['save_experiments']
        )
    
    # Save configuration
    if st.button("Save Configuration"):
        save_streamlit_config(config)
    
    return config

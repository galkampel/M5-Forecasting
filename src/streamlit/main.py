"""
Main Streamlit app for M5 Forecasting Analysis Tool.

This module provides the main entry point for the interactive data analysis
and model evaluation tool.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import data processing functions
try:
    from src.streamlit.data_processing import (
        create_dataset_interface, 
        load_processed_data,
        identify_intermittent_series,
        calculate_demand_statistics,
        analyze_price_correlation,
        create_data_quality_report
    )
except ImportError as e:
    st.error(f"Error importing data processing modules: {e}")

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="M5 Forecasting Analysis Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Sidebar
    st.sidebar.title("M5 Forecasting Analysis")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Select Analysis",
        [
            "Dataset Overview",
            "Time Series Analysis", 
            "Model Evaluation",
        ],
    )
    
    # Initialize session state
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "features_df" not in st.session_state:
        st.session_state.features_df = None
    if "targets_df" not in st.session_state:
        st.session_state.targets_df = None
    if "experiment_results" not in st.session_state:
        st.session_state.experiment_results = None
    
    # Main content area
    if page == "Dataset Overview":
        show_dataset_overview()
    elif page == "Time Series Analysis":
        show_time_series_analysis()
    elif page == "Model Evaluation":
        show_model_evaluation()


def show_dataset_overview():
    """Display dataset overview tab."""
    st.title("📊 Dataset Overview")
    st.markdown("---")
    
    # Dataset processing section
    st.header("Dataset Processing")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Create New Dataset")
        if st.button("🔄 Create New Dataset", type="primary"):
            with st.spinner("Creating dataset..."):
                try:
                    features_df, targets_df = create_dataset_interface()
                    if features_df is not None and targets_df is not None:
                        st.session_state.features_df = features_df
                        st.session_state.targets_df = targets_df
                        st.session_state.data_loaded = True
                        st.success("Dataset created successfully!")
                    else:
                        st.error("Failed to create dataset. Please check the logs.")
                except Exception as e:
                    st.error(f"Error creating dataset: {str(e)}")
    
    with col2:
        st.subheader("Load Existing Dataset")
        if st.button("📁 Load Existing Dataset"):
            with st.spinner("Loading dataset..."):
                try:
                    features_df, targets_df = load_processed_data()
                    if features_df is not None and targets_df is not None:
                        st.session_state.features_df = features_df
                        st.session_state.targets_df = targets_df
                        st.session_state.data_loaded = True
                        st.success("Dataset loaded successfully!")
                    else:
                        st.error("Failed to load dataset. Please check if processed data exists.")
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
    
    # File upload section
    st.header("Upload Data Files")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Features File")
        uploaded_features = st.file_uploader(
            "Choose a features CSV file",
            type=['csv'],
            key="features_uploader"
        )
        
        if uploaded_features is not None:
            try:
                features_df = pd.read_csv(uploaded_features)
                st.session_state.features_df = features_df
                st.success(f"Features file uploaded: {features_df.shape}")
            except Exception as e:
                st.error(f"Error reading features file: {str(e)}")
    
    with col2:
        st.subheader("Upload Targets File")
        uploaded_targets = st.file_uploader(
            "Choose a targets CSV file",
            type=['csv'],
            key="targets_uploader"
        )
        
        if uploaded_targets is not None:
            try:
                targets_df = pd.read_csv(uploaded_targets)
                st.session_state.targets_df = targets_df
                st.success(f"Targets file uploaded: {targets_df.shape}")
            except Exception as e:
                st.error(f"Error reading targets file: {str(e)}")
    
    # Check if both files are uploaded
    if (st.session_state.features_df is not None and 
        st.session_state.targets_df is not None):
        st.session_state.data_loaded = True
        st.success("✅ Both files uploaded successfully! Data is ready for analysis.")
    
    # Data preview section
    if st.session_state.data_loaded:
        st.header("Data Preview")
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Features Data")
            if st.session_state.features_df is not None:
                st.write(f"Shape: {st.session_state.features_df.shape}")
                st.dataframe(st.session_state.features_df.head())
        
        with col2:
            st.subheader("Targets Data")
            if st.session_state.targets_df is not None:
                st.write(f"Shape: {st.session_state.targets_df.shape}")
                st.dataframe(st.session_state.targets_df.head())
        
        # Data quality report
        st.header("Data Quality Report")
        st.markdown("---")
        
        quality_report = create_data_quality_report(
            st.session_state.features_df, 
            st.session_state.targets_df
        )
        
        if quality_report:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if 'features' in quality_report:
                    st.subheader("Features Quality")
                    features_quality = quality_report['features']
                    st.write(f"Shape: {features_quality['shape']}")
                    st.write(f"Missing values: {features_quality['missing_values']} ({features_quality['missing_percentage']:.2f}%)")
                    st.write(f"Duplicate rows: {features_quality['duplicate_rows']}")
                    st.write(f"Numeric columns: {features_quality['numeric_columns']}")
                    st.write(f"Categorical columns: {features_quality['categorical_columns']}")
            
            with col2:
                if 'targets' in quality_report:
                    st.subheader("Targets Quality")
                    targets_quality = quality_report['targets']
                    st.write(f"Shape: {targets_quality['shape']}")
                    st.write(f"Missing values: {targets_quality['missing_values']} ({targets_quality['missing_percentage']:.2f}%)")
                    st.write(f"Duplicate rows: {targets_quality['duplicate_rows']}")
                    st.write(f"Unique series: {targets_quality['unique_series']}")
                    if targets_quality['date_range']['start'] and targets_quality['date_range']['end']:
                        st.write(f"Date range: {targets_quality['date_range']['start']} to {targets_quality['date_range']['end']}")
    
    else:
        st.info("Please create, load, or upload a dataset to begin analysis.")


def show_time_series_analysis():
    """Display time series analysis tab."""
    st.title("📈 Time Series Analysis")
    st.markdown("---")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data from the Dataset Overview tab first.")
        return
    
    # Import analysis functions
    try:
        from src.streamlit.data_analysis import (
            create_time_series_analysis,
            create_event_impact_analysis,
            create_price_analysis
        )
        from src.streamlit.visualizations import (
            create_individual_series_plot,
            create_aggregated_series_plot,
            create_seasonal_patterns_plot,
            create_zero_demand_patterns_plot,
            create_multiple_series_comparison
        )
    except ImportError as e:
        st.error(f"Error importing analysis modules: {e}")
        return
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        [
            "Overview",
            "Individual Series Analysis",
            "Aggregated Analysis",
            "Seasonal Patterns",
            "Zero Demand Analysis",
            "Multiple Series Comparison",
            "Event Impact Analysis",
            "Price Analysis"
        ]
    )
    
    # Get data from session state
    features_df = st.session_state.features_df
    targets_df = st.session_state.targets_df
    
    # Data overview
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Overview")
    if targets_df is not None:
        st.sidebar.write(f"**Targets:** {targets_df.shape}")
        if 'unique_id' in targets_df.columns:
            unique_series = targets_df['unique_id'].nunique()
            st.sidebar.write(f"**Unique Series:** {unique_series:,}")
        if 'ds' in targets_df.columns:
            # Create a copy for datetime conversion to avoid modifying original data
            targets_df_copy = targets_df.copy()
            if not pd.api.types.is_datetime64_any_dtype(targets_df_copy['ds']):
                targets_df_copy['ds'] = pd.to_datetime(targets_df_copy['ds'])
            start_date = targets_df_copy['ds'].min().strftime('%Y-%m-%d')
            end_date = targets_df_copy['ds'].max().strftime('%Y-%m-%d')
            st.sidebar.write(f"**Date Range:**")
            st.sidebar.write(f"{start_date} to {end_date}")
    
    if features_df is not None:
        st.sidebar.write(f"**Features:** {features_df.shape}")
    
    # Main analysis content
    if analysis_type == "Overview":
        st.header("📊 Quick Insights")
        st.markdown("---")
        
        if targets_df is not None and not targets_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_sales = targets_df['y'].sum()
                st.metric("Total Sales", f"{total_sales:,.0f}")
            
            with col2:
                avg_sales = targets_df['y'].mean()
                st.metric("Average Sales", f"{avg_sales:.2f}")
            
            with col3:
                zero_count = (targets_df['y'] == 0).sum()
                total_count = len(targets_df)
                zero_pct = zero_count / total_count * 100
                st.metric("Zero Sales", f"{zero_pct:.1f}%")
            
            with col4:
                if 'unique_id' in targets_df.columns:
                    unique_series = targets_df['unique_id'].nunique()
                    st.metric("Unique Series", f"{unique_series:,}")
                else:
                    st.metric("Unique Series", "N/A")
        
        # Data quality summary
        if targets_df is not None:
            st.subheader("Data Quality Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Missing Values:**")
                missing_count = targets_df.isnull().sum().sum()
                st.write(f"- Total missing: {missing_count}")
                
                if missing_count > 0:
                    missing_cols = targets_df.columns[targets_df.isnull().any()].tolist()
                    st.write(f"- Columns with missing values: {', '.join(missing_cols)}")
            
            with col2:
                st.write("**Data Types:**")
                for col, dtype in targets_df.dtypes.items():
                    st.write(f"- {col}: {dtype}")
        else:
            st.warning("No target data available for analysis.")
    
    elif analysis_type == "Individual Series Analysis":
        st.header("Individual Series Analysis")
        st.markdown("---")
        
        if targets_df is not None and not targets_df.empty:
            create_time_series_analysis(targets_df)
        else:
            st.warning("No target data available for analysis.")
    
    elif analysis_type == "Aggregated Analysis":
        st.header("Aggregated Time Series Analysis")
        st.markdown("---")
        
        if targets_df is not None and not targets_df.empty:
            create_aggregated_series_plot(targets_df)
        else:
            st.warning("No target data available for analysis.")
    
    elif analysis_type == "Seasonal Patterns":
        st.header("Seasonal Patterns Analysis")
        st.markdown("---")
        
        if targets_df is not None and not targets_df.empty:
            create_seasonal_patterns_plot(targets_df)
        else:
            st.warning("No target data available for analysis.")
    
    elif analysis_type == "Zero Demand Analysis":
        st.header("Zero Demand Patterns Analysis")
        st.markdown("---")
        
        if targets_df is not None and not targets_df.empty:
            create_zero_demand_patterns_plot(targets_df)
        else:
            st.warning("No target data available for analysis.")
    
    elif analysis_type == "Multiple Series Comparison":
        st.header("Multiple Series Comparison")
        st.markdown("---")
        
        if targets_df is not None and not targets_df.empty:
            create_multiple_series_comparison(targets_df)
        else:
            st.warning("No target data available for analysis.")
    
    elif analysis_type == "Event Impact Analysis":
        st.header("Event Impact Analysis")
        st.markdown("---")
        
        if features_df is not None and targets_df is not None:
            create_event_impact_analysis(features_df, targets_df)
        else:
            st.warning("Both features and targets data required for event impact analysis.")
    
    elif analysis_type == "Price Analysis":
        st.header("Price Analysis")
        st.markdown("---")
        
        if features_df is not None and targets_df is not None:
            create_price_analysis(features_df, targets_df)
        else:
            st.warning("Both features and targets data required for price analysis.")


def show_model_evaluation():
    """Display model evaluation tab."""
    st.title("🤖 Model Evaluation")
    st.markdown("---")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data from the Dataset Overview tab first.")
        return
    
    # Import evaluation module
    try:
        from src.streamlit.model_evaluation import create_evaluation_interface
        create_evaluation_interface()
    except ImportError as e:
        st.error(f"Error importing evaluation modules: {e}")
        st.write("Model evaluation will be implemented here.")


if __name__ == "__main__":
    main()

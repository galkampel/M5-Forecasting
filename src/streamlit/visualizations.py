"""
Visualization module for Streamlit app.

This module integrates with src/preprocessing/visualization to create
interactive charts and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

def create_visualization_dashboard():
    """
    Create the main visualization dashboard.
    """
    st.title("📊 Visualization Dashboard")
    st.markdown("---")
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Select Visualization Type",
        [
            "Time Series Analysis",
            "Feature Analysis", 
            "Model Performance",
            "Evaluation Results",
            "Custom Charts"
        ]
    )
    
    if viz_type == "Time Series Analysis":
        create_time_series_visualizations()
    elif viz_type == "Feature Analysis":
        create_feature_analysis_visualizations()
    elif viz_type == "Model Performance":
        create_model_performance_visualizations()
    elif viz_type == "Evaluation Results":
        create_evaluation_visualizations()
    elif viz_type == "Custom Charts":
        create_custom_charts()


def create_time_series_visualizations():
    """
    Create time series visualizations.
    """
    st.subheader("Time Series Visualizations")
    
    # Check if data is available
    if 'targets_df' not in st.session_state or st.session_state.targets_df is None:
        st.warning("No time series data available. Please load data first.")
        return
    
    targets_df = st.session_state.targets_df
    
    # Visualization options
    viz_option = st.selectbox(
        "Select Time Series Visualization",
        [
            "Individual Series",
            "Aggregated Series",
            "Seasonal Patterns",
            "Zero Demand Patterns",
            "Multiple Series Comparison"
        ]
    )
    
    if viz_option == "Individual Series":
        create_individual_series_plot(targets_df)
    elif viz_option == "Aggregated Series":
        create_aggregated_series_plot(targets_df)
    elif viz_option == "Seasonal Patterns":
        create_seasonal_patterns_plot(targets_df)
    elif viz_option == "Zero Demand Patterns":
        create_zero_demand_patterns_plot(targets_df)
    elif viz_option == "Multiple Series Comparison":
        create_multiple_series_comparison(targets_df)


def create_feature_analysis_visualizations():
    """
    Create feature analysis visualizations.
    """
    st.subheader("Feature Analysis Visualizations")
    
    # Check if data is available
    if 'features_df' not in st.session_state or st.session_state.features_df is None:
        st.warning("No feature data available. Please load data first.")
        return
    
    features_df = st.session_state.features_df
    
    # Visualization options
    viz_option = st.selectbox(
        "Select Feature Analysis Visualization",
        [
            "Feature Distributions",
            "Feature Correlations",
            "Feature Importance",
            "Feature Trends",
            "Missing Values Analysis"
        ]
    )
    
    if viz_option == "Feature Distributions":
        create_feature_distributions_plot(features_df)
    elif viz_option == "Feature Correlations":
        create_feature_correlations_plot(features_df)
    elif viz_option == "Feature Importance":
        create_feature_importance_plot(features_df)
    elif viz_option == "Feature Trends":
        create_feature_trends_plot(features_df)
    elif viz_option == "Missing Values Analysis":
        create_missing_values_plot(features_df)


def create_model_performance_visualizations():
    """
    Create model performance visualizations.
    """
    st.subheader("Model Performance Visualizations")
    
    # Check if results are available
    if 'experiment_results' not in st.session_state or st.session_state.experiment_results is None:
        st.warning("No experiment results available. Please run an experiment first.")
        return
    
    results = st.session_state.experiment_results
    
    # Visualization options
    viz_option = st.selectbox(
        "Select Model Performance Visualization",
        [
            "Performance Comparison",
            "Error Analysis",
            "Prediction vs Actual",
            "Model Rankings",
            "Performance Trends"
        ]
    )
    
    if viz_option == "Performance Comparison":
        create_performance_comparison_plot(results)
    elif viz_option == "Error Analysis":
        create_error_analysis_plot(results)
    elif viz_option == "Prediction vs Actual":
        create_prediction_vs_actual_plot(results)
    elif viz_option == "Model Rankings":
        create_model_rankings_plot(results)
    elif viz_option == "Performance Trends":
        create_performance_trends_plot(results)


def create_evaluation_visualizations():
    """
    Create evaluation result visualizations.
    """
    st.subheader("Evaluation Result Visualizations")
    
    # Check if results are available
    if 'experiment_results' not in st.session_state or st.session_state.experiment_results is None:
        st.warning("No experiment results available. Please run an experiment first.")
        return
    
    results = st.session_state.experiment_results
    
    # Visualization options
    viz_option = st.selectbox(
        "Select Evaluation Visualization",
        [
            "Metric Distributions",
            "Baseline Comparison",
            "Error Distributions",
            "Performance by Category",
            "Cross-Validation Results"
        ]
    )
    
    if viz_option == "Metric Distributions":
        create_metric_distributions_plot(results)
    elif viz_option == "Baseline Comparison":
        create_baseline_comparison_plot(results)
    elif viz_option == "Error Distributions":
        create_error_distributions_plot(results)
    elif viz_option == "Performance by Category":
        create_performance_by_category_plot(results)
    elif viz_option == "Cross-Validation Results":
        create_cv_results_plot(results)


def create_custom_charts():
    """
    Create custom charts based on user input.
    """
    st.subheader("Custom Charts")
    
    # Chart type selection
    chart_type = st.selectbox(
        "Select Chart Type",
        [
            "Line Chart",
            "Bar Chart", 
            "Scatter Plot",
            "Heatmap",
            "Box Plot",
            "Histogram"
        ]
    )
    
    # Data selection
    if 'targets_df' in st.session_state and st.session_state.targets_df is not None:
        data_source = st.selectbox("Select Data Source", ["Targets", "Features"])
        
        if data_source == "Targets":
            df = st.session_state.targets_df
        else:
            df = st.session_state.features_df if 'features_df' in st.session_state else None
        
        if df is not None:
            create_custom_chart(df, chart_type)
        else:
            st.warning("No data available for custom charts.")
    else:
        st.warning("No data available. Please load data first.")


def create_individual_series_plot(targets_df):
    """
    Create individual time series plot.
    
    Args:
        targets_df: DataFrame with target values
    """
    if 'unique_id' not in targets_df.columns:
        st.warning("No unique_id column found in data.")
        return
    
    # Series selection
    unique_series = targets_df['unique_id'].unique()
    selected_series = st.selectbox("Select Series", unique_series)
    
    if selected_series:
        # Filter data
        series_data = targets_df[targets_df['unique_id'] == selected_series].copy()
        
        # Convert ds to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(series_data['ds']):
            series_data['ds'] = pd.to_datetime(series_data['ds'])
        
        # Sort by date
        series_data = series_data.sort_values('ds')
        
        # Create plot
        fig = go.Figure()
        
        # Add main series
        fig.add_trace(go.Scatter(
            x=series_data['ds'],
            y=series_data['y'],
            mode='lines+markers',
            name='Sales',
            line=dict(color='blue', width=2)
        ))
        
        # Highlight zero values
        zero_data = series_data[series_data['y'] == 0]
        if not zero_data.empty:
            fig.add_trace(go.Scatter(
                x=zero_data['ds'],
                y=zero_data['y'],
                mode='markers',
                name='Zero Sales',
                marker=dict(color='red', size=8, symbol='x')
            ))
        
        fig.update_layout(
            title=f"Time Series: {selected_series}",
            xaxis_title="Date",
            yaxis_title="Sales",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)


def create_aggregated_series_plot(targets_df):
    """
    Create aggregated time series plot.
    
    Args:
        targets_df: DataFrame with target values
    """
    # Create a copy to avoid modifying the original DataFrame
    targets_df_copy = targets_df.copy()
    
    # Aggregation level - only state, store, and department
    agg_level = st.selectbox("Aggregation Level", ["State", "Store", "Department"])
    
    # Extract aggregation key based on unique_id format
    # Format: "CA_1__FOODS_3_001" -> state="CA", store="CA_1", department="FOODS_3"
    if agg_level == "State":
        # Extract state: first string when splitting by "_"
        targets_df_copy['agg_key'] = targets_df_copy['unique_id'].str.split('_').str[0]
    elif agg_level == "Store":
        # Extract store: first string when splitting by "__"
        targets_df_copy['agg_key'] = targets_df_copy['unique_id'].str.split('__').str[0]
    elif agg_level == "Department":
        # Extract department: string between store and item id
        # Split by "__" first, then take the first part of the second split
        targets_df_copy['agg_key'] = targets_df_copy['unique_id'].str.split('__').str[1].str.split('_').str[0] + '_' + targets_df_copy['unique_id'].str.split('__').str[1].str.split('_').str[1]
    
    # Aggregate by key and date
    if not pd.api.types.is_datetime64_any_dtype(targets_df_copy['ds']):
        targets_df_copy['ds'] = pd.to_datetime(targets_df_copy['ds'])
    
    agg_data = targets_df_copy.groupby(['agg_key', 'ds'])['y'].sum().reset_index()
    
    # Create line plot
    fig = px.line(
        agg_data,
        x='ds',
        y='y',
        color='agg_key',
        title=f"Aggregated Sales by {agg_level}"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display aggregation statistics
    st.subheader(f"{agg_level} Statistics")
    
    # Calculate statistics for each aggregation key
    agg_stats = targets_df_copy.groupby('agg_key').agg({
        'y': ['sum', 'mean', 'std', 'count']
    }).round(2)
    
    # Flatten column names
    agg_stats.columns = ['Total_Sales', 'Mean_Sales', 'Std_Sales', 'Observations']
    agg_stats = agg_stats.reset_index()
    
    # Display statistics table
    st.dataframe(agg_stats, use_container_width=True)
    
    # Show summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = targets_df_copy['y'].sum()
        st.metric("Total Sales", f"{total_sales:,.0f}")
    
    with col2:
        unique_groups = targets_df_copy['agg_key'].nunique()
        st.metric(f"Unique {agg_level}s", f"{unique_groups:,}")
    
    with col3:
        avg_sales_per_group = total_sales / unique_groups
        st.metric(f"Avg Sales per {agg_level}", f"{avg_sales_per_group:,.0f}")
    
    with col4:
        start_date = targets_df_copy['ds'].min().strftime('%Y-%m-%d')
        end_date = targets_df_copy['ds'].max().strftime('%Y-%m-%d')
        st.write("**Date Range:**")
        st.write(f"{start_date} to {end_date}")


def create_seasonal_patterns_plot(targets_df):
    """
    Create seasonal patterns plot.
    
    Args:
        targets_df: DataFrame with target values
    """
    # Create a copy to avoid modifying the original DataFrame
    targets_df_copy = targets_df.copy()
    
    # Convert ds to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(targets_df_copy['ds']):
        targets_df_copy['ds'] = pd.to_datetime(targets_df_copy['ds'])
    
    # Extract time components
    targets_df_copy['weekday'] = targets_df_copy['ds'].dt.day_name()
    targets_df_copy['month'] = targets_df_copy['ds'].dt.month
    targets_df_copy['day_of_year'] = targets_df_copy['ds'].dt.dayofyear
    
    # Seasonal pattern type
    pattern_type = st.selectbox("Seasonal Pattern", ["Weekly", "Monthly", "Yearly"])
    
    if pattern_type == "Weekly":
        # Weekly patterns
        weekly_avg = targets_df_copy.groupby('weekday')['y'].mean().reset_index()
        
        fig = px.bar(
            weekly_avg,
            x='weekday',
            y='y',
            title="Average Sales by Day of Week"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif pattern_type == "Monthly":
        # Monthly patterns
        monthly_avg = targets_df_copy.groupby('month')['y'].mean().reset_index()
        
        fig = px.bar(
            monthly_avg,
            x='month',
            y='y',
            title="Average Sales by Month"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif pattern_type == "Yearly":
        # Yearly patterns
        yearly_avg = targets_df_copy.groupby('day_of_year')['y'].mean().reset_index()
        
        fig = px.line(
            yearly_avg,
            x='day_of_year',
            y='y',
            title="Average Sales by Day of Year"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def create_zero_demand_patterns_plot(targets_df):
    """
    Create zero demand patterns plot.
    
    Args:
        targets_df: DataFrame with target values
    """
    # Calculate zero percentage by series
    zero_stats = targets_df.groupby('unique_id').agg({
        'y': lambda x: (x == 0).sum() / len(x) * 100
    }).rename(columns={'y': 'zero_percentage'})
    
    # Create histogram
    fig = px.histogram(
        zero_stats,
        x='zero_percentage',
        nbins=20,
        title="Distribution of Zero Percentages",
        labels={'zero_percentage': 'Zero Percentage (%)'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Zero %", f"{zero_stats['zero_percentage'].mean():.1f}%")
    
    with col2:
        st.metric("Median Zero %", f"{zero_stats['zero_percentage'].median():.1f}%")
    
    with col3:
        intermittent_count = (zero_stats['zero_percentage'] > 50).sum()
        total_count = len(zero_stats)
        st.metric("Intermittent Series", f"{intermittent_count}/{total_count}")


def create_multiple_series_comparison(targets_df):
    """
    Create multiple series comparison plot.
    
    Args:
        targets_df: DataFrame with target values
    """
    # Series selection
    unique_series = targets_df['unique_id'].unique()
    selected_series = st.multiselect(
        "Select Series (max 5)",
        unique_series,
        default=unique_series[:3] if len(unique_series) >= 3 else unique_series
    )
    
    if selected_series:
        # Filter data
        series_data = targets_df[targets_df['unique_id'].isin(selected_series)].copy()
        
        # Convert ds to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(series_data['ds']):
            series_data['ds'] = pd.to_datetime(series_data['ds'])
        
        # Create plot
        fig = px.line(
            series_data,
            x='ds',
            y='y',
            color='unique_id',
            title="Multiple Series Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def create_feature_distributions_plot(features_df):
    """
    Create feature distributions plot.
    
    Args:
        features_df: DataFrame with features
    """
    # Select numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found in features.")
        return
    
    # Feature selection
    selected_feature = st.selectbox("Select Feature", numeric_cols)
    
    if selected_feature:
        # Create histogram
        fig = px.histogram(
            features_df,
            x=selected_feature,
            nbins=30,
            title=f"Distribution of {selected_feature}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{features_df[selected_feature].mean():.3f}")
        
        with col2:
            st.metric("Std", f"{features_df[selected_feature].std():.3f}")
        
        with col3:
            st.metric("Min", f"{features_df[selected_feature].min():.3f}")
        
        with col4:
            st.metric("Max", f"{features_df[selected_feature].max():.3f}")


def create_feature_correlations_plot(features_df):
    """
    Create feature correlations plot.
    
    Args:
        features_df: DataFrame with features
    """
    # Select numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found in features.")
        return
    
    # Limit to top 20 features for performance
    if len(numeric_cols) > 20:
        st.warning("Too many features for correlation matrix. Showing top 20.")
        numeric_cols = numeric_cols[:20]
    
    # Calculate correlation matrix
    corr_matrix = features_df[numeric_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_feature_importance_plot(features_df):
    """
    Create feature importance plot.
    
    Args:
        features_df: DataFrame with features
    """
    st.info("Feature importance analysis requires model training results.")
    st.write("Please run a model experiment first to see feature importance.")


def create_feature_trends_plot(features_df):
    """
    Create feature trends plot.
    
    Args:
        features_df: DataFrame with features
    """
    if 'ds' not in features_df.columns:
        st.warning("No date column found in features.")
        return
    
    # Convert ds to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(features_df['ds']):
        features_df['ds'] = pd.to_datetime(features_df['ds'])
    
    # Select numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'ds']
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found in features.")
        return
    
    # Feature selection
    selected_feature = st.selectbox("Select Feature", numeric_cols)
    
    if selected_feature:
        # Aggregate by date
        trend_data = features_df.groupby('ds')[selected_feature].mean().reset_index()
        
        # Create line plot
        fig = px.line(
            trend_data,
            x='ds',
            y=selected_feature,
            title=f"Trend of {selected_feature} Over Time"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def create_missing_values_plot(features_df):
    """
    Create missing values analysis plot.
    
    Args:
        features_df: DataFrame with features
    """
    # Calculate missing values
    missing_data = features_df.isnull().sum()
    missing_pct = (missing_data / len(features_df)) * 100
    
    missing_df = pd.DataFrame({
        'column': missing_data.index,
        'missing_count': missing_data.values,
        'missing_percentage': missing_pct.values
    }).sort_values('missing_percentage', ascending=False)
    
    # Filter columns with missing values
    missing_df = missing_df[missing_df['missing_count'] > 0]
    
    if len(missing_df) == 0:
        st.success("No missing values found in the dataset!")
        return
    
    # Create bar plot
    fig = px.bar(
        missing_df.head(20),
        x='column',
        y='missing_percentage',
        title="Missing Values by Column (Top 20)",
        labels={'missing_percentage': 'Missing Percentage (%)'}
    )
    
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


def create_performance_comparison_plot(results):
    """
    Create performance comparison plot.
    
    Args:
        results: Experiment results
    """
    st.info("Performance comparison visualization will be implemented here.")


def create_error_analysis_plot(results):
    """
    Create error analysis plot.
    
    Args:
        results: Experiment results
    """
    st.info("Error analysis visualization will be implemented here.")


def create_prediction_vs_actual_plot(results):
    """
    Create prediction vs actual plot.
    
    Args:
        results: Experiment results
    """
    st.info("Prediction vs actual visualization will be implemented here.")


def create_model_rankings_plot(results):
    """
    Create model rankings plot.
    
    Args:
        results: Experiment results
    """
    st.info("Model rankings visualization will be implemented here.")


def create_performance_trends_plot(results):
    """
    Create performance trends plot.
    
    Args:
        results: Experiment results
    """
    st.info("Performance trends visualization will be implemented here.")


def create_metric_distributions_plot(results):
    """
    Create metric distributions plot.
    
    Args:
        results: Experiment results
    """
    st.info("Metric distributions visualization will be implemented here.")


def create_baseline_comparison_plot(results):
    """
    Create baseline comparison plot.
    
    Args:
        results: Experiment results
    """
    st.info("Baseline comparison visualization will be implemented here.")


def create_error_distributions_plot(results):
    """
    Create error distributions plot.
    
    Args:
        results: Experiment results
    """
    st.info("Error distributions visualization will be implemented here.")


def create_performance_by_category_plot(results):
    """
    Create performance by category plot.
    
    Args:
        results: Experiment results
    """
    st.info("Performance by category visualization will be implemented here.")


def create_cv_results_plot(results):
    """
    Create cross-validation results plot.
    
    Args:
        results: Experiment results
    """
    st.info("Cross-validation results visualization will be implemented here.")


def create_custom_chart(df, chart_type):
    """
    Create custom chart based on user input.
    
    Args:
        df: DataFrame with data
        chart_type: Type of chart to create
    """
    # Column selection
    if chart_type in ["Line Chart", "Bar Chart", "Scatter Plot"]:
        x_col = st.selectbox("Select X-axis column", df.columns)
        y_col = st.selectbox("Select Y-axis column", df.columns)
        
        if chart_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col, title=f"{chart_type}: {x_col} vs {y_col}")
        elif chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, title=f"{chart_type}: {x_col} vs {y_col}")
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{chart_type}: {x_col} vs {y_col}")
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Heatmap":
        # Select numeric columns for correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for heatmap.")
            return
        
        # Limit columns for performance
        if len(numeric_cols) > 10:
            st.warning("Too many columns for heatmap. Showing top 10.")
            numeric_cols = numeric_cols[:10]
        
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Box Plot":
        # Select columns for box plot
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for box plot.")
            return
        
        selected_col = st.selectbox("Select column for box plot", numeric_cols)
        fig = px.box(df, y=selected_col, title=f"Box Plot: {selected_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Histogram":
        # Select column for histogram
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for histogram.")
            return
        
        selected_col = st.selectbox("Select column for histogram", numeric_cols)
        nbins = st.slider("Number of bins", min_value=5, max_value=50, value=20)
        
        fig = px.histogram(df, x=selected_col, nbins=nbins, title=f"Histogram: {selected_col}")
        st.plotly_chart(fig, use_container_width=True)

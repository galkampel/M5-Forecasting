"""
Monitoring module for Streamlit app.

This module handles experiment tracking, data quality monitoring,
and performance monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


def create_monitoring_dashboard():
    """
    Create the main monitoring dashboard.
    """
    st.title("📊 Monitoring Dashboard")
    st.markdown("---")
    
    # Monitoring type selection
    monitor_type = st.selectbox(
        "Select Monitoring Type",
        [
            "Experiment Tracking",
            "Data Quality Monitoring",
            "Performance Monitoring",
            "Resource Monitoring",
            "Alert System"
        ]
    )
    
    if monitor_type == "Experiment Tracking":
        create_experiment_tracking()
    elif monitor_type == "Data Quality Monitoring":
        create_data_quality_monitoring()
    elif monitor_type == "Performance Monitoring":
        create_performance_monitoring()
    elif monitor_type == "Resource Monitoring":
        create_resource_monitoring()
    elif monitor_type == "Alert System":
        create_alert_system()


def create_experiment_tracking():
    """
    Create experiment tracking interface.
    """
    st.subheader("Experiment Tracking")
    
    # Load experiment history
    experiments = load_experiment_history()
    
    if experiments:
        # Display experiment history
        st.write("**Experiment History**")
        
        # Convert to DataFrame for display
        exp_df = pd.DataFrame(experiments)
        st.dataframe(exp_df)
        
        # Experiment comparison
        if len(experiments) > 1:
            st.write("**Experiment Comparison**")
            create_experiment_comparison(experiments)
    
    # Save current experiment
    st.write("**Save Current Experiment**")
    
    if 'experiment_results' in st.session_state and st.session_state.experiment_results is not None:
        experiment_name = st.text_input("Experiment Name", value=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        experiment_description = st.text_area("Experiment Description")
        
        if st.button("Save Experiment"):
            save_experiment(experiment_name, experiment_description, st.session_state.experiment_results)
            st.success("Experiment saved successfully!")
    else:
        st.info("No experiment results to save. Run an experiment first.")


def create_data_quality_monitoring():
    """
    Create data quality monitoring interface.
    """
    st.subheader("Data Quality Monitoring")
    
    if 'features_df' not in st.session_state or 'targets_df' not in st.session_state:
        st.warning("No data available for quality monitoring.")
        return
    
    features_df = st.session_state.features_df
    targets_df = st.session_state.targets_df
    
    # Data quality metrics
    quality_metrics = calculate_data_quality_metrics(features_df, targets_df)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Completeness Score", f"{quality_metrics['completeness']:.1f}%")
    
    with col2:
        st.metric("Consistency Score", f"{quality_metrics['consistency']:.1f}%")
    
    with col3:
        st.metric("Accuracy Score", f"{quality_metrics['accuracy']:.1f}%")
    
    with col4:
        st.metric("Overall Quality", f"{quality_metrics['overall']:.1f}%")
    
    # Quality alerts
    st.write("**Quality Alerts**")
    alerts = generate_quality_alerts(quality_metrics)
    
    if alerts:
        for alert in alerts:
            if alert['severity'] == 'high':
                st.error(f"🔴 {alert['message']}")
            elif alert['severity'] == 'medium':
                st.warning(f"🟡 {alert['message']}")
            else:
                st.info(f"🔵 {alert['message']}")
    else:
        st.success("✅ No quality issues detected!")
    
    # Quality trends
    st.write("**Quality Trends**")
    create_quality_trends_plot()


def create_performance_monitoring():
    """
    Create performance monitoring interface.
    """
    st.subheader("Performance Monitoring")
    
    if 'experiment_results' not in st.session_state or st.session_state.experiment_results is None:
        st.warning("No experiment results available for performance monitoring.")
        return
    
    results = st.session_state.experiment_results
    
    # Performance metrics
    performance_metrics = calculate_performance_metrics(results)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best MAE", f"{performance_metrics['best_mae']:.3f}")
    
    with col2:
        st.metric("Best RMSE", f"{performance_metrics['best_rmse']:.3f}")
    
    with col3:
        st.metric("Best MASE", f"{performance_metrics['best_mase']:.3f}")
    
    with col4:
        st.metric("Best Model", performance_metrics['best_model'])
    
    # Performance trends
    st.write("**Performance Trends**")
    create_performance_trends_plot()
    
    # Performance alerts
    st.write("**Performance Alerts**")
    alerts = generate_performance_alerts(performance_metrics)
    
    if alerts:
        for alert in alerts:
            if alert['severity'] == 'high':
                st.error(f"🔴 {alert['message']}")
            elif alert['severity'] == 'medium':
                st.warning(f"🟡 {alert['message']}")
            else:
                st.info(f"🔵 {alert['message']}")
    else:
        st.success("✅ Performance is within acceptable ranges!")


def create_resource_monitoring():
    """
    Create resource monitoring interface.
    """
    st.subheader("Resource Monitoring")
    
    # Simulate resource metrics
    try:
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CPU Usage", f"{cpu_percent:.1f}%")
        
        with col2:
            st.metric("Memory Usage", f"{memory_percent:.1f}%")
        
        with col3:
            st.metric("Disk Usage", f"{disk_percent:.1f}%")
        
        # Resource alerts
        st.write("**Resource Alerts**")
        alerts = generate_resource_alerts(cpu_percent, memory_percent, disk_percent)
        
        if alerts:
            for alert in alerts:
                if alert['severity'] == 'high':
                    st.error(f"🔴 {alert['message']}")
                elif alert['severity'] == 'medium':
                    st.warning(f"🟡 {alert['message']}")
                else:
                    st.info(f"🔵 {alert['message']}")
        else:
            st.success("✅ Resource usage is normal!")
        
        # Resource trends
        st.write("**Resource Trends**")
        create_resource_trends_plot()
        
    except ImportError:
        st.warning("psutil not available. Install with: pip install psutil")


def create_alert_system():
    """
    Create alert system interface.
    """
    st.subheader("Alert System")
    
    # Alert configuration
    st.write("**Alert Configuration**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_quality_threshold = st.slider("Data Quality Threshold (%)", min_value=50, max_value=100, value=80)
        performance_threshold = st.slider("Performance Threshold (MAE)", min_value=0.1, max_value=10.0, value=1.0)
    
    with col2:
        resource_threshold = st.slider("Resource Usage Threshold (%)", min_value=50, max_value=100, value=80)
        enable_notifications = st.checkbox("Enable Notifications", value=True)
    
    # Alert history
    st.write("**Alert History**")
    alerts = load_alert_history()
    
    if alerts:
        alert_df = pd.DataFrame(alerts)
        st.dataframe(alert_df)
    else:
        st.info("No alerts in history.")
    
    # Test alerts
    st.write("**Test Alerts**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Test Data Quality Alert"):
            create_test_alert("data_quality", "Test data quality alert", "medium")
    
    with col2:
        if st.button("Test Performance Alert"):
            create_test_alert("performance", "Test performance alert", "high")
    
    with col3:
        if st.button("Test Resource Alert"):
            create_test_alert("resource", "Test resource alert", "low")


def load_experiment_history():
    """
    Load experiment history from file.
    
    Returns:
        list: List of experiment records
    """
    try:
        history_file = Path("logs/experiment_history.json")
        if history_file.exists():
            with open(history_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading experiment history: {str(e)}")
    
    return []


def save_experiment(name, description, results):
    """
    Save experiment to history.
    
    Args:
        name: Experiment name
        description: Experiment description
        results: Experiment results
    """
    try:
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Load existing history
        history = load_experiment_history()
        
        # Create new experiment record
        experiment = {
            'name': name,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'status': 'completed'
        }
        
        # Add to history
        history.append(experiment)
        
        # Save to file
        history_file = logs_dir / "experiment_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
    except Exception as e:
        st.error(f"Error saving experiment: {str(e)}")


def create_experiment_comparison(experiments):
    """
    Create experiment comparison visualization.
    
    Args:
        experiments: List of experiment records
    """
    if len(experiments) < 2:
        st.warning("Need at least 2 experiments for comparison.")
        return
    
    # Create comparison chart
    exp_names = [exp['name'] for exp in experiments]
    exp_timestamps = [exp['timestamp'] for exp in experiments]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=exp_timestamps,
        y=exp_names,
        mode='markers',
        name='Experiments',
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title="Experiment Timeline",
        xaxis_title="Timestamp",
        yaxis_title="Experiment Name"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def calculate_data_quality_metrics(features_df, targets_df):
    """
    Calculate data quality metrics.
    
    Args:
        features_df: Features DataFrame
        targets_df: Targets DataFrame
        
    Returns:
        dict: Quality metrics
    """
    metrics = {}
    
    try:
        # Completeness (no missing values)
        features_missing = features_df.isnull().sum().sum()
        targets_missing = targets_df.isnull().sum().sum()
        total_cells = features_df.shape[0] * features_df.shape[1] + targets_df.shape[0] * targets_df.shape[1]
        
        metrics['completeness'] = ((total_cells - features_missing - targets_missing) / total_cells) * 100
        
        # Consistency (no duplicates)
        features_duplicates = features_df.duplicated().sum()
        targets_duplicates = targets_df.duplicated().sum()
        total_rows = len(features_df) + len(targets_df)
        
        metrics['consistency'] = ((total_rows - features_duplicates - targets_duplicates) / total_rows) * 100
        
        # Accuracy (reasonable value ranges)
        # For targets, check if values are non-negative
        if 'y' in targets_df.columns:
            negative_values = (targets_df['y'] < 0).sum()
            total_targets = len(targets_df)
            metrics['accuracy'] = ((total_targets - negative_values) / total_targets) * 100
        else:
            metrics['accuracy'] = 100.0
        
        # Overall quality (average of all metrics)
        metrics['overall'] = (metrics['completeness'] + metrics['consistency'] + metrics['accuracy']) / 3
        
    except Exception as e:
        st.error(f"Error calculating quality metrics: {str(e)}")
        metrics = {'completeness': 0, 'consistency': 0, 'accuracy': 0, 'overall': 0}
    
    return metrics


def generate_quality_alerts(metrics):
    """
    Generate quality alerts based on metrics.
    
    Args:
        metrics: Quality metrics dictionary
        
    Returns:
        list: List of alerts
    """
    alerts = []
    
    # Completeness alerts
    if metrics['completeness'] < 90:
        alerts.append({
            'severity': 'high' if metrics['completeness'] < 80 else 'medium',
            'message': f"Low data completeness: {metrics['completeness']:.1f}%"
        })
    
    # Consistency alerts
    if metrics['consistency'] < 95:
        alerts.append({
            'severity': 'medium',
            'message': f"Data consistency issues: {metrics['consistency']:.1f}%"
        })
    
    # Accuracy alerts
    if metrics['accuracy'] < 100:
        alerts.append({
            'severity': 'high' if metrics['accuracy'] < 95 else 'medium',
            'message': f"Data accuracy issues: {metrics['accuracy']:.1f}%"
        })
    
    # Overall quality alerts
    if metrics['overall'] < 85:
        alerts.append({
            'severity': 'high' if metrics['overall'] < 75 else 'medium',
            'message': f"Overall data quality is low: {metrics['overall']:.1f}%"
        })
    
    return alerts


def create_quality_trends_plot():
    """
    Create quality trends plot.
    """
    # Placeholder for quality trends
    st.info("Quality trends visualization will be implemented here.")


def calculate_performance_metrics(results):
    """
    Calculate performance metrics from results.
    
    Args:
        results: Experiment results
        
    Returns:
        dict: Performance metrics
    """
    metrics = {
        'best_mae': float('inf'),
        'best_rmse': float('inf'),
        'best_mase': float('inf'),
        'best_model': 'None'
    }
    
    try:
        if results and 'metrics' in results:
            # Find best metrics
            for col in results['metrics'].columns:
                if col.endswith('_mae'):
                    model_name = col.replace('_mae', '')
                    mae_value = results['metrics'][col].mean()
                    if mae_value < metrics['best_mae']:
                        metrics['best_mae'] = mae_value
                        metrics['best_model'] = model_name
                
                elif col.endswith('_rmse'):
                    rmse_value = results['metrics'][col].mean()
                    if rmse_value < metrics['best_rmse']:
                        metrics['best_rmse'] = rmse_value
                
                elif col.endswith('_mase'):
                    mase_value = results['metrics'][col].mean()
                    if mase_value < metrics['best_mase']:
                        metrics['best_mase'] = mase_value
    
    except Exception as e:
        st.error(f"Error calculating performance metrics: {str(e)}")
    
    return metrics


def generate_performance_alerts(metrics):
    """
    Generate performance alerts based on metrics.
    
    Args:
        metrics: Performance metrics dictionary
        
    Returns:
        list: List of alerts
    """
    alerts = []
    
    # MAE alerts
    if metrics['best_mae'] > 2.0:
        alerts.append({
            'severity': 'high' if metrics['best_mae'] > 5.0 else 'medium',
            'message': f"High MAE: {metrics['best_mae']:.3f}"
        })
    
    # RMSE alerts
    if metrics['best_rmse'] > 3.0:
        alerts.append({
            'severity': 'high' if metrics['best_rmse'] > 7.0 else 'medium',
            'message': f"High RMSE: {metrics['best_rmse']:.3f}"
        })
    
    # MASE alerts
    if metrics['best_mase'] > 1.5:
        alerts.append({
            'severity': 'high' if metrics['best_mase'] > 2.0 else 'medium',
            'message': f"High MASE: {metrics['best_mase']:.3f}"
        })
    
    return alerts


def create_performance_trends_plot():
    """
    Create performance trends plot.
    """
    # Placeholder for performance trends
    st.info("Performance trends visualization will be implemented here.")


def generate_resource_alerts(cpu_percent, memory_percent, disk_percent):
    """
    Generate resource alerts based on usage.
    
    Args:
        cpu_percent: CPU usage percentage
        memory_percent: Memory usage percentage
        disk_percent: Disk usage percentage
        
    Returns:
        list: List of alerts
    """
    alerts = []
    
    # CPU alerts
    if cpu_percent > 80:
        alerts.append({
            'severity': 'high' if cpu_percent > 90 else 'medium',
            'message': f"High CPU usage: {cpu_percent:.1f}%"
        })
    
    # Memory alerts
    if memory_percent > 80:
        alerts.append({
            'severity': 'high' if memory_percent > 90 else 'medium',
            'message': f"High memory usage: {memory_percent:.1f}%"
        })
    
    # Disk alerts
    if disk_percent > 80:
        alerts.append({
            'severity': 'high' if disk_percent > 90 else 'medium',
            'message': f"High disk usage: {disk_percent:.1f}%"
        })
    
    return alerts


def create_resource_trends_plot():
    """
    Create resource trends plot.
    """
    # Placeholder for resource trends
    st.info("Resource trends visualization will be implemented here.")


def load_alert_history():
    """
    Load alert history from file.
    
    Returns:
        list: List of alert records
    """
    try:
        alert_file = Path("logs/alert_history.json")
        if alert_file.exists():
            with open(alert_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading alert history: {str(e)}")
    
    return []


def create_test_alert(alert_type, message, severity):
    """
    Create a test alert.
    
    Args:
        alert_type: Type of alert
        message: Alert message
        severity: Alert severity
    """
    try:
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Load existing alerts
        alerts = load_alert_history()
        
        # Create new alert
        alert = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to history
        alerts.append(alert)
        
        # Save to file
        alert_file = logs_dir / "alert_history.json"
        with open(alert_file, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        st.success(f"Test alert created: {message}")
        
    except Exception as e:
        st.error(f"Error creating test alert: {str(e)}")

"""
Model evaluation module for Streamlit app.

This module provides comprehensive model evaluation functionality integrating
with the src/preprocessing pipeline and using proper evaluation metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.preprocessing.evaluation.evaluator import Evaluator
    from src.preprocessing.pipelines.modeling_pipeline import ModelTrainingPipeline
    from src.preprocessing.models.mlforecast_params import MLForecastParams
    from src.preprocessing.utils.logging import get_logger
    from src.preprocessing.config import PreprocessingConfig
except ImportError as e:
    st.error(f"Error importing preprocessing modules: {e}")


def create_evaluation_interface():
    """Create the evaluation interface."""
    st.title("🔍 Model Evaluation")
    
    # Evaluation type selection
    evaluation_type = st.sidebar.selectbox(
        "Select Evaluation Type",
        [
            "Experiment Configuration",
            "Model Training & Evaluation", 
            "Evaluation Metrics",
            "Feature Importance Analysis",
            "Visualizations"
        ]
    )
    
    # Display selected evaluation type
    if evaluation_type == "Experiment Configuration":
        show_experiment_configuration()
    elif evaluation_type == "Model Training & Evaluation":
        show_model_training_evaluation()
    elif evaluation_type == "Evaluation Metrics":
        show_model_evaluation()
    elif evaluation_type == "Feature Importance Analysis":
        show_feature_importance_analysis()
    elif evaluation_type == "Visualizations":
        show_visualizations()


def show_experiment_configuration():
    """Display experiment configuration interface."""
    st.subheader("Experiment Configuration")
    
    # Get data from session state
    features_df = st.session_state.get('features_df')
    targets_df = st.session_state.get('targets_df')
    
    if features_df is None or targets_df is None:
        st.warning("Please load data from the Dataset Overview tab first.")
        return
    
    # Model selection
    st.write("**Model Selection**")
    col1, col2 = st.columns(2)
    
    with col1:
        use_ridge = st.checkbox("Ridge Regression", value=True)
        use_lgbm = st.checkbox("LightGBM", value=True)
    
    with col2:
        use_baselines = st.checkbox("Baseline Models", value=True)
        enable_optimization = st.checkbox("Enable Hyperparameter Optimization", value=False)
    
    # Feature engineering
    st.write("**Feature Engineering**")
    col1, col2 = st.columns(2)
    
    with col1:
        use_calendar = st.checkbox("Calendar Features", value=True)
        use_fourier = st.checkbox("Fourier Features", value=True)
        use_lags = st.checkbox("Lag Features", value=True)
    
    with col2:
        use_rolling = st.checkbox("Rolling Features", value=True)
        use_encoding = st.checkbox("Encoding Features", value=True)
        use_scaling = st.checkbox("Scaling Features", value=True)
    
    # Cross-validation settings
    st.write("**Cross-Validation Settings**")
    col1, col2 = st.columns(2)
    
    with col1:
        n_folds = st.slider("Number of Folds", min_value=2, max_value=10, value=6)
        test_size = st.slider("Test Size (days)", min_value=7, max_value=90, value=30)
    
    with col2:
        seasonal_period = st.number_input("Seasonal Period", min_value=1, value=7)
        enable_feature_importance = st.checkbox("Enable Feature Importance", value=True)
    
    # Optimization settings
    if enable_optimization:
        st.write("**Optimization Settings**")
        col1, col2 = st.columns(2)
        
        with col1:
            n_trials = st.slider("Number of Trials", min_value=10, max_value=100, value=20)
            timeout = st.number_input("Timeout (seconds)", min_value=60, value=300)
        
        with col2:
            metric = st.selectbox("Optimization Metric", ["rmse", "mae", "mase", "rmsse"])
    
    # Save configuration to session state
    if st.button("Save Configuration", type="primary"):
        config = {
            'models': {
                'ridge': use_ridge,
                'lgbm': use_lgbm,
                'baselines': use_baselines
            },
            'features': {
                'calendar': use_calendar,
                'fourier': use_fourier,
                'lags': use_lags,
                'rolling': use_rolling,
                'encoding': use_encoding,
                'scaling': use_scaling
            },
            'cv': {
                'n_folds': n_folds,
                'test_size': test_size,
                'seasonal_period': seasonal_period
            },
            'optimization': {
                'enabled': enable_optimization,
                'n_trials': n_trials if enable_optimization else 0,
                'timeout': timeout if enable_optimization else 0,
                'metric': metric if enable_optimization else 'rmse'
            },
            'feature_importance': enable_feature_importance
        }
        
        st.session_state.experiment_config = config
        st.success("Configuration saved successfully!")
        
        # Display configuration summary
        st.subheader("Configuration Summary")
        st.json(config)


def show_model_training_evaluation():
    """Display model training and evaluation interface."""
    st.subheader("Model Training & Evaluation")
    
    # Check if configuration exists
    if 'experiment_config' not in st.session_state:
        st.warning("Please configure the experiment first.")
        return
    
    # Get data from session state
    features_df = st.session_state.get('features_df')
    targets_df = st.session_state.get('targets_df')
    
    if features_df is None or targets_df is None:
        st.warning("Please load data from the Dataset Overview tab first.")
        return
    
    config = st.session_state.experiment_config
    
    if st.button("Run Model Training & Evaluation", type="primary"):
        with st.spinner("Training models and evaluating performance..."):
            try:
                # Ensure consistent data types for 'ds' column before merging
                features_df_copy = features_df.copy()
                targets_df_copy = targets_df.copy()
                
                if 'ds' in features_df_copy.columns:
                    features_df_copy['ds'] = pd.to_datetime(features_df_copy['ds'])
                if 'ds' in targets_df_copy.columns:
                    targets_df_copy['ds'] = pd.to_datetime(targets_df_copy['ds'])
                
                # Prepare data
                merged_df = features_df_copy.merge(targets_df_copy, on=['unique_id', 'ds'], how='inner')
                
                # Run modeling pipeline
                results = run_modeling_pipeline(merged_df, config)
                
                if results:
                    st.session_state.experiment_results = results
                    st.success("Model training and evaluation completed successfully!")
                    
                    # Display quick results
                    display_quick_results(results)
                else:
                    st.error("Model training failed. Please check the configuration.")
                    
            except Exception as e:
                st.error(f"Error in model training: {str(e)}")


def show_model_evaluation():
    """Display comprehensive model evaluation interface."""
    st.subheader("Model Evaluation")
    
    if 'experiment_results' not in st.session_state:
        st.warning("Please run model training first.")
        return
    
    results = st.session_state.experiment_results
    
    # Create tabs for different evaluation views
    tab1, tab2 = st.tabs(["Performance Metrics", "Combined Analysis"])
    
    with tab1:
        show_performance_metrics_section(results)
    
    with tab2:
        show_combined_analysis_section(results)


def show_performance_metrics_section(results):
    """Display detailed performance metrics section."""
    st.subheader("Performance Metrics")
    
    try:
        # Check if results is a dictionary (from modeling pipeline)
        if isinstance(results, dict):
            # Display metrics from the modeling pipeline
            if 'cross_validation' in results:
                cv_results = results['cross_validation']
                
                # Collect all metrics for comparison
                all_metrics = {}
                
                for model_name, model_result in cv_results.items():
                    if model_name == 'baselines':
                        continue
                        
                    if isinstance(model_result, dict) and 'cv_metrics' in model_result:
                        cv_metrics = model_result['cv_metrics']
                        if isinstance(cv_metrics, dict) and 'error' not in cv_metrics:
                            # Filter out metrics that contain "_vs_" in their names
                            filtered_metrics = {k: v for k, v in cv_metrics.items() if '_vs_' not in k}
                            all_metrics[model_name] = filtered_metrics
                
                if all_metrics:
                    # Create comparison dataframe
                    metrics_df = pd.DataFrame(all_metrics).T
                    
                    # Display metrics table
                    st.write("**Model Performance Metrics:**")
                    st.dataframe(metrics_df.round(4))
                    
                    # Create metrics visualization
                    create_metrics_visualization(metrics_df)
                    
                    # Display individual model details
                    st.subheader("Individual Model Details")
                    for model_name, model_result in cv_results.items():
                        if model_name == 'baselines':
                            continue
                            
                        st.write(f"**{model_name} Model:**")
                        
                        if isinstance(model_result, dict):
                            if 'cv_metrics' in model_result:
                                cv_metrics = model_result['cv_metrics']
                                if isinstance(cv_metrics, dict):
                                    if 'error' in cv_metrics:
                                        st.error(f"Error: {cv_metrics['error']}")
                                    else:
                                        # Filter out metrics that contain "_vs_" in their names
                                        filtered_metrics = {k: v for k, v in cv_metrics.items() if '_vs_' not in k}
                                        # Display metrics as a table
                                        model_metrics_df = pd.DataFrame([filtered_metrics])
                                        st.dataframe(model_metrics_df.round(4))
                                else:
                                    st.write("CV Metrics:", cv_metrics)
                            else:
                                st.write("No CV metrics available")
                                
                            if 'status' in model_result and model_result['status'] == 'error':
                                st.error(f"Model Error: {model_result.get('error', 'Unknown error')}")
                        else:
                            st.write("Model Result:", model_result)
                            
                        st.write("---")
                else:
                    st.warning("No valid model metrics found")
            else:
                st.write("No cross-validation results found")
                st.json(results)
        else:
            # Handle case where results might be a DataFrame (fallback)
            st.warning("Unexpected results format. Expected dictionary from modeling pipeline.")
            st.write("Results type:", type(results))
            if hasattr(results, 'columns'):
                st.write("Results columns:", list(results.columns))
        
    except Exception as e:
        st.error(f"Error calculating performance metrics: {str(e)}")


def show_combined_analysis_section(results):
    """Display combined analysis of all models and baselines."""
    st.subheader("Combined Analysis")
    
    try:
        # Check if results is a dictionary (from modeling pipeline)
        if isinstance(results, dict):
            if 'cross_validation' in results:
                cv_results = results['cross_validation']
                
                # Collect all metrics including baselines
                all_metrics = {}
                
                # First, add baseline metrics if available
                if 'baselines' in cv_results:
                    baseline_info = cv_results['baselines']
                    if isinstance(baseline_info, dict):
                        baseline_path = baseline_info.get('cv_predictions_path')
                        if baseline_path and Path(baseline_path).exists():
                            try:
                                baseline_df = pd.read_csv(baseline_path)
                                
                                # Filter for actual baseline model columns only
                                baseline_models = ['Naive', 'SeasonalNaive', 'ZeroModel']
                                available_baselines = [col for col in baseline_models if col in baseline_df.columns]
                                
                                if available_baselines:
                                    st.write(f"**Baseline models included:** {available_baselines}")
                                    
                                    # Calculate baseline metrics including all required metrics
                                    for baseline_col in available_baselines:
                                        y_true = baseline_df['y'].astype(float)
                                        y_pred = baseline_df[baseline_col].astype(float)
                                        
                                        # Remove NaN values
                                        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                                        y_true_clean = y_true[mask]
                                        y_pred_clean = y_pred[mask]
                                        
                                        if len(y_true_clean) > 0:
                                            from src.preprocessing.evaluation.evaluator import f1_zero, non_zero_mae
                                            
                                            # Calculate all metrics including the missing ones
                                            mae = np.mean(np.abs(y_true_clean - y_pred_clean))
                                            rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
                                            f1_zero_score = f1_zero(y_true_clean, y_pred_clean)
                                            non_zero_mae_score = non_zero_mae(y_true_clean, y_pred_clean)
                                            
                                            # Calculate MASE (Mean Absolute Scaled Error)
                                            naive_errors = np.abs(np.diff(y_true_clean))
                                            if len(naive_errors) > 0:
                                                mase = mae / np.mean(naive_errors) if np.mean(naive_errors) > 0 else float('inf')
                                            else:
                                                mase = float('inf')
                                            
                                            # Calculate RMSSE (Root Mean Square Scaled Error)
                                            if len(naive_errors) > 0:
                                                rmse_naive = np.sqrt(np.mean(naive_errors ** 2))
                                                rmsse = rmse / rmse_naive if rmse_naive > 0 else float('inf')
                                            else:
                                                rmsse = float('inf')
                                            
                                            # Calculate MRAE (Mean Relative Absolute Error)
                                            mean_actual = np.mean(y_true_clean)
                                            mrae = mae / mean_actual if mean_actual > 0 else float('inf')
                                            
                                            all_metrics[f"{baseline_col} (Baseline)"] = {
                                                'mae': mae,
                                                'rmse': rmse,
                                                'mase': mase,
                                                'rmsse': rmsse,
                                                'mrae': mrae,
                                                'f1_zero': f1_zero_score,
                                                'non_zero_mae': non_zero_mae_score
                                            }
                                    
                            except Exception as e:
                                st.error(f"Could not load baseline results: {str(e)}")
                
                # Add main model metrics
                for model_name, model_result in cv_results.items():
                    if model_name == 'baselines':
                        continue
                        
                    if isinstance(model_result, dict) and 'cv_metrics' in model_result:
                        cv_metrics = model_result['cv_metrics']
                        if isinstance(cv_metrics, dict) and 'error' not in cv_metrics:
                            # Filter out metrics that contain "_vs_" in their names
                            filtered_metrics = {k: v for k, v in cv_metrics.items() if '_vs_' not in k}
                            all_metrics[model_name] = filtered_metrics
                
                if all_metrics:
                    # Create comparison dataframe
                    metrics_df = pd.DataFrame(all_metrics).T
                    
                    # Display complete metrics table
                    st.write("**Complete Model Comparison (All Models + Baselines):**")
                    st.dataframe(metrics_df.round(4))
                    
                    # Create comprehensive comparison plots
                    available_metrics = ['mae', 'rmse', 'mase', 'rmsse', 'mrae', 'f1_zero', 'non_zero_mae']
                    plot_metrics = [m for m in available_metrics if m in metrics_df.columns]
                    
                    if plot_metrics:
                        # Create subplots for metrics comparison
                        n_metrics = len(plot_metrics)
                        cols = min(3, n_metrics)  # Max 3 columns
                        rows = (n_metrics + cols - 1) // cols
                        
                        fig = make_subplots(
                            rows=rows, cols=cols,
                            subplot_titles=[f'{m.upper()} Comparison' for m in plot_metrics]
                        )
                        
                        for i, metric in enumerate(plot_metrics):
                            row = (i // cols) + 1
                            col = (i % cols) + 1
                            
                            # Color code: baselines in red, main models in blue
                            colors = ['red' if '(Baseline)' in model else 'blue' 
                                    for model in metrics_df.index]
                            
                            fig.add_trace(
                                go.Bar(
                                    x=list(metrics_df.index),
                                    y=metrics_df[metric],
                                    name=metric.upper(),
                                    marker_color=colors,
                                    showlegend=False
                                ),
                                row=row, col=col
                            )
                        
                        fig.update_layout(
                            height=300 * rows, 
                            showlegend=False,
                            title_text="Complete Model Performance Comparison"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add legend
                        st.write("**Legend:**")
                        st.write("🔴 Red bars: Baseline models")
                        st.write("🔵 Blue bars: Main models")
                        
                        # Show best model for each metric
                        st.write("**Best Model by Metric:**")
                        for metric in plot_metrics:
                            if metric in metrics_df.columns:
                                # For error metrics, lower is better
                                if metric in ['mae', 'rmse', 'mase', 'rmsse', 'mrae']:
                                    best_model = metrics_df[metric].idxmin()
                                    best_value = metrics_df[metric].min()
                                    st.write(f"- **{metric.upper()}**: {best_model} ({best_value:.4f})")
                                # For other metrics, higher is better
                                else:
                                    best_model = metrics_df[metric].idxmax()
                                    best_value = metrics_df[metric].max()
                                    st.write(f"- **{metric.upper()}**: {best_model} ({best_value:.4f})")
                        
                        # Add metric interpretation
                        st.write("**Metric Interpretation:**")
                        st.write("- **MAE/RMSE**: Lower values indicate better performance")
                        st.write("- **MASE**: Mean Absolute Scaled Error - values < 1 indicate better than naive forecast")
                        st.write("- **RMSSE**: Root Mean Square Scaled Error - values < 1 indicate better than naive forecast")
                        st.write("- **MRAE**: Mean Relative Absolute Error - lower values indicate better performance")
                        st.write("- **F1-Zero**: Higher values indicate better zero-prediction accuracy")
                        st.write("- **Non-Zero MAE**: Lower values indicate better non-zero prediction accuracy")
                    else:
                        st.warning("No comparable metrics available for plotting")
                else:
                    st.warning("No valid model metrics found for comparison")
            else:
                st.warning("No cross-validation results found")
        else:
            st.warning("No results available for combined analysis")
            
    except Exception as e:
        st.error(f"Error in combined analysis: {str(e)}")


def show_feature_importance_analysis():
    """Display feature importance analysis."""
    st.subheader("Feature Importance Analysis")
    
    if 'experiment_results' not in st.session_state:
        st.warning("Please run model training first.")
        return
    
    results = st.session_state.experiment_results
    
    # Feature importance analysis
    try:
        create_feature_importance_analysis(results)
    except Exception as e:
        st.error(f"Error in feature importance analysis: {str(e)}")


def run_modeling_pipeline(data, config):
    """
    Run the modeling pipeline using src/preprocessing.
    
    Args:
        data: Merged features and targets DataFrame
        config: Experiment configuration
        
    Returns:
        DataFrame: Model results
    """
    import streamlit as st  # Add streamlit import for this function
    
    try:
        # Import preprocessing config
        from src.preprocessing.config import PreprocessingConfig
        
        # Create preprocessing configuration
        preprocessing_config = PreprocessingConfig()
        
        # Deep data isolation: Create a completely clean copy
        # st.info("Creating isolated data copy for modeling pipeline...")
        
        # Create a deep copy of the data to prevent any contamination
        isolated_data = data.copy(deep=True)
        
        # Data validation: Check for string columns that should be numeric
        # string_columns = isolated_data.select_dtypes(include=['object']).columns.tolist()
        # problematic_columns = []
        
        # for col in string_columns:
        #     if col in ['unique_id', 'ds']:  # These are expected to be strings
        #         continue
        #     # Check if column contains weekday names or other categorical strings
        #     unique_vals = isolated_data[col].dropna().unique()
        #     if len(unique_vals) > 0:
        #         sample_val = str(unique_vals[0])
        #         if any(day in sample_val.lower() for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
        #             problematic_columns.append(col)
        
        # if problematic_columns:
        #     st.warning(f"Found potentially problematic string columns: {problematic_columns}")
        #     st.info("These columns may cause issues with the modeling pipeline. Removing them to ensure clean data.")
            
        #     # Remove problematic columns from the isolated copy
        #     isolated_data = isolated_data.drop(columns=problematic_columns)
        #     st.info(f"Removed problematic columns: {problematic_columns}")
        
        # Ensure datetime column is properly formatted
        if 'ds' in isolated_data.columns:
            isolated_data['ds'] = pd.to_datetime(isolated_data['ds'])
        
        # st.success(f"Data isolation complete. Clean data shape: {isolated_data.shape}")
        # st.info(f"Clean columns: {list(isolated_data.columns)}")
        
        # Configure models based on config (use correct model names)
        if config['models'].get('Ridge', config['models'].get('ridge', False)):
            preprocessing_config.models.ridge.enabled = True
        else:
            preprocessing_config.models.ridge.enabled = False
        
        if config['models'].get('LGBMRegressor', config['models'].get('lgbm', False)):
            preprocessing_config.models.lgbm.enabled = True
        else:
            preprocessing_config.models.lgbm.enabled = False
        
        # Configure cross-validation
        preprocessing_config.cv.n_windows = config['cv']['n_folds']
        preprocessing_config.cv.forecast_horizon = config['cv']['test_size']
        
        # Configure optimization
        preprocessing_config.optimization.enabled = config['optimization']['enabled']
        if config['optimization']['enabled']:
            preprocessing_config.optimization.n_trials = config['optimization']['n_trials']
            preprocessing_config.optimization.timeout = config['optimization']['timeout']
        
        # Apply the same column filtering logic as _prepare_mlforecast_data
        # Drop irrelevant columns as per notebook
        features_to_drop = [
            "wm_yr_wk",
            "wday",
            "year",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
        ]
        isolated_data = isolated_data.drop(columns=features_to_drop, errors="ignore")
        
        # Split features and targets
        features_df = isolated_data.drop(columns=['y'])
        targets_df = isolated_data[['unique_id', 'ds', 'y']]
        
        # Run the modeling pipeline
        from src.preprocessing.pipelines.modeling_pipeline import ModelTrainingPipeline
        pipeline = ModelTrainingPipeline(preprocessing_config)
        results = pipeline.run(features_df, targets_df)
        return results
        
    except Exception as e:
        import streamlit as st
        st.error(f"Error in modeling pipeline: {str(e)}")
        return None


def display_quick_results(results):
    """Display quick results summary."""
    st.subheader("Quick Results Summary")
    
    # Check if results is a dictionary (from modeling pipeline)
    if isinstance(results, dict):
        # Display pipeline results structure
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Count models trained
            models_trained = len([k for k in results.keys() if k != 'cross_validation'])
            st.metric("Models Trained", models_trained)
        
        with col2:
            # Count cross-validation results
            if 'cross_validation' in results:
                cv_results = results['cross_validation']
                cv_models = len([k for k in cv_results.keys() if k != 'baselines'])
                st.metric("CV Models", cv_models)
            else:
                st.metric("CV Models", 0)
        
        with col3:
            # Check if any models have errors
            error_count = 0
            if 'cross_validation' in results:
                for model_result in results['cross_validation'].values():
                    if isinstance(model_result, dict) and model_result.get('status') == 'error':
                        error_count += 1
            st.metric("Models with Errors", error_count)
        
        with col4:
            # Show overall status
            if error_count == 0:
                st.metric("Status", "✅ Success")
            else:
                st.metric("Status", "⚠️ Errors")
        
        # Show detailed results structure
        st.subheader("Results Structure")
        st.json(results)
        
    else:
        # Handle case where results might be a DataFrame (fallback)
        st.warning("Unexpected results format. Expected dictionary from modeling pipeline.")
        st.write("Results type:", type(results))
        if hasattr(results, 'columns'):
            st.write("Results columns:", list(results.columns))


def create_metrics_visualization(metrics):
    """Create metrics visualization."""
    st.subheader("Metrics Visualization")
    
    # Filter out metrics that contain "_vs_" in their names
    numeric_cols = metrics.select_dtypes(include=[np.number]).columns
    filtered_cols = [col for col in numeric_cols if '_vs_' not in col]
    
    # Select metrics to plot
    selected_metrics = st.multiselect(
        "Select Metrics to Visualize",
        filtered_cols,
        default=filtered_cols[:4] if len(filtered_cols) >= 4 else filtered_cols
    )
    
    if selected_metrics:
        # Create box plots for selected metrics
        fig = go.Figure()
        
        for metric in selected_metrics:
            fig.add_trace(go.Box(
                y=metrics[metric],
                name=metric,
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title="Distribution of Performance Metrics",
            yaxis_title="Metric Value",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)


def create_feature_importance_analysis(results):
    """Create feature importance analysis."""
    # Check if results is a dictionary (from modeling pipeline)
    if isinstance(results, dict):
        if 'cross_validation' in results:
            cv_results = results['cross_validation']
            
            # Collect feature importance from all models
            feature_importance_data = {}
            
            for model_name, model_result in cv_results.items():
                if model_name == 'baselines':
                    continue
                    
                if isinstance(model_result, dict) and 'feature_importance' in model_result:
                    fi_result = model_result['feature_importance']
                    if isinstance(fi_result, dict) and 'error' not in fi_result:
                        feature_importance_data[model_name] = fi_result
            
            if feature_importance_data:
                # Display feature importance for each model
                for model_name, fi_scores in feature_importance_data.items():
                    st.write(f"**{model_name} Feature Importance:**")
                    
                    # Convert to DataFrame for better display
                    fi_df = pd.DataFrame(list(fi_scores.items()), 
                                       columns=['Feature', 'Importance'])
                    fi_df = fi_df.sort_values('Importance', ascending=False)
                    
                    # Display top features
                    st.dataframe(fi_df.head(10).round(4))
                    
                    # Create feature importance plot
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=fi_df['Feature'][:10],
                        y=fi_df['Importance'][:10],
                        name=model_name
                    ))
                    
                    fig.update_layout(
                        title=f"{model_name} - Top 10 Feature Importance",
                        xaxis_title="Features",
                        yaxis_title="Importance Score",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("---")
            else:
                st.warning("No feature importance data available. This may be because:")
                st.write("- Cross-validation failed to complete")
                st.write("- Feature importance analysis encountered errors")
                st.write("- Models don't support feature importance analysis")
                
                # Show error messages if available
                for model_name, model_result in cv_results.items():
                    if model_name == 'baselines':
                        continue
                        
                    if isinstance(model_result, dict) and 'feature_importance' in model_result:
                        fi_result = model_result['feature_importance']
                        if isinstance(fi_result, dict) and 'error' in fi_result:
                            st.error(f"{model_name}: {fi_result['error']}")
        else:
            st.warning("No cross-validation results found for feature importance analysis")
    else:
        st.warning("Unexpected results format. Expected dictionary from modeling pipeline.")


def show_visualizations():
    """Display visualizations interface."""
    st.subheader("Visualizations")
    
    if 'experiment_results' not in st.session_state:
        st.warning("Please run model training first.")
        return
    
    results = st.session_state.experiment_results
    
    # Visualization type selection
    visualization_type = st.sidebar.selectbox(
        "Select Visualization Type",
        [
            "Series Analysis",
            "Aggregation Plots"
        ]
    )
    
    if visualization_type == "Series Analysis":
        show_series_analysis(results)
    elif visualization_type == "Aggregation Plots":
        show_aggregation_plots(results)


def show_series_analysis(results):
    """Display series analysis plots."""
    st.subheader("Series Analysis")
    
    if 'cross_validation' not in results:
        st.warning("No cross-validation results found.")
        return
    
    cv_results = results['cross_validation']
    
    # Model selection
    available_models = [k for k in cv_results.keys() if k != 'baselines']
    if not available_models:
        st.warning("No models found in cross-validation results.")
        return
    
    model_name = st.selectbox("Select Model", available_models)
    
    if model_name not in cv_results:
        st.warning(f"Model '{model_name}' not found in cross-validation results.")
        return
    
    model_result = cv_results[model_name]
    
    # Get data from session state for full time series
    features_df = st.session_state.get('features_df')
    targets_df = st.session_state.get('targets_df')
    
    if features_df is None or targets_df is None:
        st.warning("Please load data from the Dataset Overview tab first.")
        return
    
    # Ensure consistent data types for 'ds' column before merging
    features_df_copy = features_df.copy()
    targets_df_copy = targets_df.copy()
    
    if 'ds' in features_df_copy.columns:
        features_df_copy['ds'] = pd.to_datetime(features_df_copy['ds'])
    if 'ds' in targets_df_copy.columns:
        targets_df_copy['ds'] = pd.to_datetime(targets_df_copy['ds'])
    
    # Merge features and targets to get full time series
    full_data = features_df_copy.merge(targets_df_copy, on=['unique_id', 'ds'], how='inner')
    
    # Series selection
    available_series = sorted(full_data['unique_id'].unique())
    selected_series = st.selectbox("Select Series", available_series)
    
    # Load predictions
    if 'cv_predictions_path' not in model_result:
        st.warning("Predictions path not found in model results.")
        return
    
    predictions_path = model_result['cv_predictions_path']
    if not Path(predictions_path).exists():
        st.warning(f"Predictions file not found: {predictions_path}")
        return
    
    try:
        predictions_df = pd.read_csv(predictions_path)
        if 'ds' in predictions_df.columns:
            predictions_df['ds'] = pd.to_datetime(predictions_df['ds'])
        
        # Filter predictions for selected series
        series_predictions = predictions_df[predictions_df['unique_id'] == selected_series].copy()
        series_full_data = full_data[full_data['unique_id'] == selected_series].copy()
        
        if series_predictions.empty:
            st.warning(f"No predictions found for series {selected_series}")
            return
        
        # Create tabs for the three plots
        tab1, tab2, tab3 = st.tabs(["Time Series", "Actual vs Predicted", "Residual Plot"])
        
        with tab1:
            st.write("**Time Series: Full Series**")
            
            # Create time series plot of the entire series
            fig = go.Figure()
            
            # Plot full time series (training data)
            fig.add_trace(go.Scatter(
                x=series_full_data['ds'],
                y=series_full_data['y'],
                mode='lines',
                name='Full Time Series',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title=f"Time Series: {selected_series}",
                xaxis_title="Date",
                yaxis_title="Values",
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display series statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Observations", len(series_full_data))
            with col2:
                start_date = series_full_data['ds'].min().strftime('%Y-%m-%d')
                end_date = series_full_data['ds'].max().strftime('%Y-%m-%d')
                st.write("**Date Range:**")
                st.write(f"{start_date} to {end_date}")
            with col3:
                st.metric("Mean Value", f"{series_full_data['y'].mean():.2f}")
            with col4:
                st.metric("Std Dev", f"{series_full_data['y'].std():.2f}")
        
        with tab2:
            st.write("**Actual vs Predicted: Test Time**")
            
            # Create actual vs predicted plot for test time
            fig = go.Figure()
            
            # Plot actual values
            fig.add_trace(go.Scatter(
                x=series_predictions['ds'],
                y=series_predictions['y'],
                mode='lines',
                name='Actual',
                line=dict(color='blue', width=2)
            ))
            
            # Plot predicted values
            fig.add_trace(go.Scatter(
                x=series_predictions['ds'],
                y=series_predictions[model_name],
                mode='lines',
                name='Predicted',
                line=dict(color='orange', width=2)
            ))
            
            fig.update_layout(
                title=f"Actual vs Predicted: {selected_series} - {model_name}",
                xaxis_title="Date",
                yaxis_title="Values",
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display prediction statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                mae = abs(series_predictions['y'] - series_predictions[model_name]).mean()
                st.metric("MAE", f"{mae:.4f}")
            with col2:
                rmse = np.sqrt(((series_predictions['y'] - series_predictions[model_name]) ** 2).mean())
                st.metric("RMSE", f"{rmse:.4f}")
            with col3:
                correlation = series_predictions['y'].corr(series_predictions[model_name])
                st.metric("Correlation", f"{correlation:.4f}")
            with col4:
                st.metric("Test Points", len(series_predictions))
        
        with tab3:
            st.write("**Residual Plot: Test Time**")
            
            # Calculate residuals
            series_predictions['residuals'] = series_predictions['y'] - series_predictions[model_name]
            
            # Create residual plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=series_predictions['ds'],
                y=series_predictions['residuals'],
                mode='lines',
                name='Residuals',
                line=dict(color='salmon', width=2)
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="red", 
                         annotation_text="Zero Residual", annotation_position="top right")
            
            fig.update_layout(
                title=f"Residuals: {selected_series} - {model_name}",
                xaxis_title="Date",
                yaxis_title="Residuals (Actual - Predicted)",
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display residual statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                mean_residual = series_predictions['residuals'].mean()
                st.metric("Mean Residual", f"{mean_residual:.4f}")
            with col2:
                std_residual = series_predictions['residuals'].std()
                st.metric("Std Residual", f"{std_residual:.4f}")
            with col3:
                positive_residuals = (series_predictions['residuals'] > 0).sum()
                total_residuals = len(series_predictions)
                positive_pct = (positive_residuals / total_residuals) * 100
                st.metric("Positive Residuals", f"{positive_pct:.1f}%")
            with col4:
                max_residual = series_predictions['residuals'].abs().max()
                st.metric("Max Abs Residual", f"{max_residual:.4f}")
            
            # Residual distribution
            st.subheader("Residual Distribution")
            fig_hist = px.histogram(
                series_predictions,
                x='residuals',
                nbins=30,
                title=f"Residual Distribution: {selected_series} - {model_name}",
                labels={'residuals': 'Residuals'}
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_hist, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating series analysis plots: {str(e)}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")


def show_aggregation_plots(results):
    """Display aggregation plots for predictions vs actual values."""
    st.subheader("Aggregation Plots")
    
    # Add information about what data is being used
    # st.info("""
    # **Data Source**: These plots show aggregated predictions from **all cross-validation folds** 
    # (typically 5 time windows), providing a comprehensive view of model performance across 
    # different time periods. This is more robust than using only the last fold.
    # """)
    
    if 'cross_validation' not in results:
        st.warning("No cross-validation results found.")
        return
    
    cv_results = results['cross_validation']
    
    # Model selection
    available_models = [k for k in cv_results.keys() if k != 'baselines']
    if not available_models:
        st.warning("No models found in cross-validation results.")
        return
    
    model_name = st.selectbox("Select Model", available_models)
    
    if model_name not in cv_results:
        st.warning(f"Model '{model_name}' not found in cross-validation results.")
        return
    
    model_result = cv_results[model_name]
    
    # Aggregation options
    col1, col2 = st.columns(2)
    
    with col1:
        aggregation_freq = st.selectbox(
            "Aggregation Frequency",
            ["D", "W", "M"],
            format_func=lambda x: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[x]
        )
    
    with col2:
        aggregation_type = st.selectbox(
            "Aggregation Type",
            ["sum", "mean", "median"]
        )
    
    if 'cv_predictions_path' in model_result:
        predictions_path = model_result['cv_predictions_path']
        if Path(predictions_path).exists():
            try:
                predictions_df = pd.read_csv(predictions_path)
                
                # Check if this is an error file
                if 'error' in predictions_df.columns:
                    st.error(f"CV predictions extraction failed: {predictions_df['error'].iloc[0]}")
                    return
                
                # Convert ds to datetime if it's not already
                if 'ds' in predictions_df.columns:
                    predictions_df['ds'] = pd.to_datetime(predictions_df['ds'])
                    
                    # Show data information
                    total_points = len(predictions_df)
                    unique_dates = predictions_df['ds'].nunique()
                    start_date = predictions_df['ds'].min().strftime('%Y-%m-%d')
                    end_date = predictions_df['ds'].max().strftime('%Y-%m-%d')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predictions", f"{total_points:,}")
                    with col2:
                        st.metric("Unique Dates", f"{unique_dates:,}")
                    with col3:
                        st.write("**Date Range:**")
                        st.write(f"{start_date} to {end_date}")
                    
                    # Create aggregated plots
                    create_aggregated_plot(
                        predictions_df, 
                        model_name, 
                        aggregation_freq, 
                        aggregation_type
                    )
                else:
                    st.warning("No date column found in predictions data.")
                    
            except Exception as e:
                st.error(f"Error loading predictions: {str(e)}")
        else:
            st.warning(f"Predictions file not found: {predictions_path}")
    else:
        st.warning("Predictions path not found in model results.")


def create_aggregated_plot(
    df: pd.DataFrame, 
    model_name: str, 
    agg_freq: str, 
    agg_type: str
):
    """
    Create aggregated plot for predictions vs actual values.
    
    Args:
        df: DataFrame with predictions and actual values
        model_name: Name of the model
        agg_freq: Aggregation frequency ('D', 'W', 'M')
        agg_type: Aggregation type ('sum', 'mean', 'median')
    """
    st.write(f"**Aggregated {agg_type.title()} by {agg_freq}**")
    
    # Ensure we have the required columns
    required_cols = ['ds', 'y', model_name]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return
    
    # Remove rows with NaN values
    df_clean = df[required_cols].dropna()
    
    if len(df_clean) == 0:
        st.warning("No valid data for aggregation.")
        return
    
    # Group by date frequency and aggregate
    df_agg = df_clean.groupby(pd.Grouper(key='ds', freq=agg_freq)).agg({
        'y': agg_type,
        model_name: agg_type
    }).dropna()
    
    if len(df_agg) == 0:
        st.warning("No data after aggregation.")
        return
    
    # Create the aggregated plot
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=df_agg.index,
        y=df_agg['y'],
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Add predicted values
    fig.add_trace(go.Scatter(
        x=df_agg.index,
        y=df_agg[model_name],
        mode='lines+markers',
        name=f'{model_name} (Predicted)',
        line=dict(color='orange', width=2),
        marker=dict(size=6)
    ))
    
    # Update layout
    freq_name = {"D": "Daily", "W": "Weekly", "M": "Monthly"}[agg_freq]
    fig.update_layout(
        title=f"{model_name} - Aggregated {agg_type.title()} by {freq_name}",
        xaxis_title="Date",
        yaxis_title=f"{agg_type.title()} Values",
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display aggregated statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mae = np.mean(np.abs(df_agg['y'] - df_agg[model_name]))
        st.metric("MAE", f"{mae:.4f}")
    
    with col2:
        rmse = np.sqrt(np.mean((df_agg['y'] - df_agg[model_name]) ** 2))
        st.metric("RMSE", f"{rmse:.4f}")
    
    with col3:
        correlation = df_agg['y'].corr(df_agg[model_name])
        st.metric("Correlation", f"{correlation:.4f}")
    
    with col4:
        total_actual = df_agg['y'].sum()
        total_predicted = df_agg[model_name].sum()
        bias = (total_predicted - total_actual) / total_actual * 100
        st.metric("Bias (%)", f"{bias:.2f}%")
    
    # Display aggregated data table
    st.write("**Aggregated Data Table**")
    agg_display = df_agg.reset_index()
    agg_display['ds'] = agg_display['ds'].dt.strftime('%Y-%m-%d')
    agg_display.columns = ['Date', 'Actual', 'Predicted']
    agg_display['Error'] = agg_display['Actual'] - agg_display['Predicted']
    agg_display['Abs_Error'] = np.abs(agg_display['Error'])
    
    st.dataframe(agg_display.round(4), use_container_width=True)

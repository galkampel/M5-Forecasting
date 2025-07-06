"""
Data analysis module for Streamlit app.

This module handles time series analysis, event impact analysis,
price analysis, and intermittent time series analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


def create_time_series_analysis(targets_df):
    """
    Create individual time series analysis with ADI-CV² clustering.
    
    Args:
        targets_df: DataFrame with target values
    """
    st.subheader("Individual Time Series Analysis")
    
    if targets_df is None or targets_df.empty:
        st.warning("No data available for analysis.")
        return
    
    # Create a copy to avoid modifying the original DataFrame
    targets_df_copy = targets_df.copy()
    
    # Calculate ADI-CV² scores for all series
    adi_cv_scores = calculate_adi_cv_scores(targets_df_copy)
    
    if adi_cv_scores.empty:
        st.warning("Could not calculate ADI-CV² scores.")
        return
    
    # Display clustering results
    st.write("### ADI-CV² Clustering Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        smooth_count = (adi_cv_scores['cluster'] == 'Smooth').sum()
        total_count = len(adi_cv_scores)
        st.metric("Smooth Series", f"{smooth_count:,} ({smooth_count/total_count*100:.1f}%)")
    
    with col2:
        erratic_count = (adi_cv_scores['cluster'] == 'Erratic').sum()
        st.metric("Erratic Series", f"{erratic_count:,} ({erratic_count/total_count*100:.1f}%)")
    
    with col3:
        lumpy_count = (adi_cv_scores['cluster'] == 'Lumpy').sum()
        st.metric("Lumpy Series", f"{lumpy_count:,} ({lumpy_count/total_count*100:.1f}%)")
    
    with col4:
        intermittent_count = (adi_cv_scores['cluster'] == 'Intermittent').sum()
        st.metric("Intermittent Series", f"{intermittent_count:,} ({intermittent_count/total_count*100:.1f}%)")
    
    # ADI-CV² scatter plot
    st.write("### ADI-CV² Clustering Visualization")
    
    fig = px.scatter(
        adi_cv_scores,
        x='ADI',
        y='CV2',
        color='cluster',
        title="ADI-CV² Clustering of Time Series",
        labels={'ADI': 'Average Demand Interval', 'CV2': 'Coefficient of Variation²'},
        color_discrete_map={
            'Smooth': 'green',
            'Erratic': 'orange', 
            'Lumpy': 'red',
            'Intermittent': 'purple'
        }
    )
    
    # Add cluster boundaries
    fig.add_hline(y=0.49, line_dash="dash", line_color="gray", 
                  annotation_text="CV² = 0.49")
    fig.add_vline(x=1.32, line_dash="dash", line_color="gray", 
                  annotation_text="ADI = 1.32")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Series selection with cluster filter
    st.write("### Series Selection")
    
    # Cluster filter
    cluster_filter = st.selectbox(
        "Filter by Cluster",
        ["All Clusters"] + adi_cv_scores['cluster'].unique().tolist()
    )
    
    if cluster_filter == "All Clusters":
        filtered_series = adi_cv_scores['unique_id'].tolist()
    else:
        filtered_series = adi_cv_scores[adi_cv_scores['cluster'] == cluster_filter]['unique_id'].tolist()
    
    if 'unique_id' in targets_df_copy.columns and filtered_series:
        selected_series = st.selectbox(
            "Select Series",
            filtered_series,
            index=0 if len(filtered_series) > 0 else None
        )
        
        if selected_series:
            # Get cluster info for selected series
            series_cluster_info = adi_cv_scores[adi_cv_scores['unique_id'] == selected_series].iloc[0]
            
            # Display cluster information
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Cluster", series_cluster_info['cluster'])
            
            with col2:
                st.metric("ADI", f"{series_cluster_info['ADI']:.3f}")
            
            with col3:
                st.metric("CV²", f"{series_cluster_info['CV2']:.3f}")
            
            with col4:
                st.metric("Zero %", f"{series_cluster_info['zero_percentage']:.1f}%")
            
            # Filter data for selected series
            series_data = targets_df_copy[targets_df_copy['unique_id'] == selected_series].copy()
            
            # Convert ds to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(series_data['ds']):
                series_data['ds'] = pd.to_datetime(series_data['ds'])
            
            # Sort by date
            series_data = series_data.sort_values('ds')
            
            # Create time series plot
            fig = go.Figure()
            
            # Add main series
            fig.add_trace(go.Scatter(
                x=series_data['ds'],
                y=series_data['y'],
                mode='lines+markers',
                name='Sales',
                line=dict(color='blue')
            ))
            
            # Highlight zero values
            zero_data = series_data[series_data['y'] == 0]
            if not zero_data.empty:
                fig.add_trace(go.Scatter(
                    x=zero_data['ds'],
                    y=zero_data['y'],
                    mode='markers',
                    name='Zero Sales',
                    marker=dict(color='red', size=8)
                ))
            
            fig.update_layout(
                title=f"Time Series: {selected_series} ({series_cluster_info['cluster']} Cluster)",
                xaxis_title="Date",
                yaxis_title="Sales",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Sales", f"{series_data['y'].mean():.2f}")
            
            with col2:
                st.metric("Std Sales", f"{series_data['y'].std():.2f}")
            
            with col3:
                zero_pct = (series_data['y'] == 0).sum() / len(series_data) * 100
                st.metric("Zero %", f"{zero_pct:.1f}%")
            
            with col4:
                st.metric("Total Observations", len(series_data))
    else:
        st.warning("No series available for the selected cluster.")


def calculate_adi_cv_scores(targets_df):
    """
    Calculate ADI (Average Demand Interval) and CV² (Coefficient of Variation squared) scores.
    
    Args:
        targets_df: DataFrame with target values
        
    Returns:
        DataFrame: Series with ADI, CV², and cluster classification
    """
    try:
        if 'unique_id' not in targets_df.columns or 'y' not in targets_df.columns:
            return pd.DataFrame()
        
        results = []
        
        for unique_id in targets_df['unique_id'].unique():
            series_data = targets_df[targets_df['unique_id'] == unique_id]['y'].values
            
            # Calculate ADI (Average Demand Interval)
            non_zero_demands = series_data[series_data > 0]
            if len(non_zero_demands) > 1:
                # Calculate intervals between non-zero demands
                intervals = []
                last_non_zero_idx = -1
                
                for i, demand in enumerate(series_data):
                    if demand > 0:
                        if last_non_zero_idx >= 0:
                            intervals.append(i - last_non_zero_idx)
                        last_non_zero_idx = i
                
                if intervals:
                    adi = np.mean(intervals)
                else:
                    adi = 1.0
            else:
                adi = 1.0
            
            # Calculate CV² (Coefficient of Variation squared)
            if len(non_zero_demands) > 1:
                mean_demand = np.mean(non_zero_demands)
                std_demand = np.std(non_zero_demands)
                cv2 = (std_demand / mean_demand) ** 2 if mean_demand > 0 else 0
            else:
                cv2 = 0.0
            
            # Calculate zero percentage
            zero_percentage = (series_data == 0).sum() / len(series_data) * 100
            
            # Classify cluster based on ADI-CV² framework
            if adi < 1.32 and cv2 < 0.49:
                cluster = 'Smooth'
            elif adi >= 1.32 and cv2 < 0.49:
                cluster = 'Intermittent'
            elif adi < 1.32 and cv2 >= 0.49:
                cluster = 'Erratic'
            else:  # adi >= 1.32 and cv2 >= 0.49
                cluster = 'Lumpy'
            
            results.append({
                'unique_id': unique_id,
                'ADI': adi,
                'CV2': cv2,
                'zero_percentage': zero_percentage,
                'cluster': cluster,
                'mean_demand': np.mean(non_zero_demands) if len(non_zero_demands) > 0 else 0,
                'std_demand': np.std(non_zero_demands) if len(non_zero_demands) > 1 else 0
            })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        st.error(f"Error calculating ADI-CV² scores: {str(e)}")
        return pd.DataFrame()


def create_event_impact_analysis(features_df, targets_df):
    """
    Create comprehensive event impact analysis.
    
    Args:
        features_df: DataFrame with features including events
        targets_df: DataFrame with target values
    """
    st.subheader("Event Impact Analysis")
    
    if features_df is None or targets_df is None:
        st.warning("No data available for analysis.")
        return
    
    try:
        # Ensure consistent datetime types for 'ds' column before merging
        features_copy = features_df.copy()
        targets_copy = targets_df.copy()
        
        # Convert ds to datetime if needed
        if 'ds' in features_copy.columns:
            if not pd.api.types.is_datetime64_any_dtype(features_copy['ds']):
                features_copy['ds'] = pd.to_datetime(features_copy['ds'])
        
        if 'ds' in targets_copy.columns:
            if not pd.api.types.is_datetime64_any_dtype(targets_copy['ds']):
                targets_copy['ds'] = pd.to_datetime(targets_copy['ds'])
        
        # Merge features and targets
        merged_df = features_copy.merge(targets_copy, on=['unique_id', 'ds'], 
                                     how='inner')
        
        # Find binary event columns (excluding 'is_item_exists')
        binary_cols = []
        for col in merged_df.columns:
            if col not in ['unique_id', 'ds', 'y', 'is_item_exists']:
                unique_vals = merged_df[col].dropna().unique()
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, True, False}):
                    binary_cols.append(col)
        
        if not binary_cols:
            st.info("No binary event columns found in the data.")
            return
        
        # Separate SNAP events from other events
        snap_events = [col for col in binary_cols if col.lower().startswith('snap')]
        other_events = [col for col in binary_cols if not col.lower().startswith('snap')]
        
        st.write(f"**Found {len(binary_cols)} binary event columns:**")
        st.write(f"- SNAP events: {len(snap_events)}")
        st.write(f"- Other events: {len(other_events)}")
        
        # Event type selection
        event_type = st.selectbox(
            "Select Event Type to Analyze",
            ["All Events", "SNAP Events", "Other Events"]
        )
        
        if event_type == "SNAP Events":
            events_to_analyze = snap_events
            event_category = "SNAP"
        elif event_type == "Other Events":
            events_to_analyze = other_events
            event_category = "Other"
        else:
            events_to_analyze = binary_cols
            event_category = "All"
        
        if not events_to_analyze:
            st.warning(f"No {event_type.lower()} found in the data.")
            return
        
        # Overall event impact summary
        st.write(f"### {event_category} Events Impact Summary")
        
        # Calculate overall impact metrics
        impact_summary = []
        for event_col in events_to_analyze:
            event_sales = merged_df[merged_df[event_col] == 1]['y'].mean()
            non_event_sales = merged_df[merged_df[event_col] == 0]['y'].mean()
            event_count = (merged_df[event_col] == 1).sum()
            total_count = len(merged_df)
            event_frequency = event_count / total_count * 100
            
            if non_event_sales > 0:
                impact_pct = ((event_sales - non_event_sales) / non_event_sales) * 100
            else:
                impact_pct = 0
            
            impact_summary.append({
                'Event': event_col,
                'Event_Sales': event_sales,
                'Non_Event_Sales': non_event_sales,
                'Impact_%': impact_pct,
                'Event_Frequency_%': event_frequency,
                'Event_Count': event_count
            })
        
        impact_df = pd.DataFrame(impact_summary)
        
        # Display impact summary table
        st.write("**Event Impact Summary Table**")
        st.dataframe(
            impact_df.round(2),
            use_container_width=True
        )
        
        # Top impactful events
        st.write("### Top Impactful Events")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Most positive impact
            positive_impact = impact_df[impact_df['Impact_%'] > 0].nlargest(5, 'Impact_%')
            if not positive_impact.empty:
                st.write("**Events with Highest Positive Impact:**")
                for _, row in positive_impact.iterrows():
                    st.write(f"- {row['Event']}: +{row['Impact_%']:.1f}%")
            else:
                st.write("**No events with positive impact found.**")
        
        with col2:
            # Most negative impact
            negative_impact = impact_df[impact_df['Impact_%'] < 0].nsmallest(5, 'Impact_%')
            if not negative_impact.empty:
                st.write("**Events with Highest Negative Impact:**")
                for _, row in negative_impact.iterrows():
                    st.write(f"- {row['Event']}: {row['Impact_%']:.1f}%")
            else:
                st.write("**No events with negative impact found.**")
        
        # Detailed analysis for selected event
        st.write("### Detailed Event Analysis")
        
        selected_event = st.selectbox(
            "Select Event for Detailed Analysis",
            events_to_analyze
        )
        
        if selected_event:
            # Event impact analysis for selected event
            st.write(f"#### {selected_event} Detailed Analysis")
            
            # Get event data
            event_data = merged_df[merged_df[selected_event] == 1]
            non_event_data = merged_df[merged_df[selected_event] == 0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Sales During Event", f"{event_data['y'].mean():.2f}")
            
            with col2:
                st.metric("Sales Without Event", f"{non_event_data['y'].mean():.2f}")
            
            with col3:
                impact = ((event_data['y'].mean() - non_event_data['y'].mean()) / 
                         non_event_data['y'].mean()) * 100
                st.metric("Impact (%)", f"{impact:+.1f}%")
            
            with col4:
                event_freq = (merged_df[selected_event] == 1).sum() / len(merged_df) * 100
                st.metric("Event Frequency", f"{event_freq:.1f}%")
            
            # Time series plot showing event periods
            if 'ds' in merged_df.columns:
                st.write("#### Event Timeline Analysis")
                
                # Aggregate by date
                daily_data = merged_df.groupby('ds').agg({
                    'y': 'sum',
                    selected_event: 'max'
                }).reset_index()
                
                # Create plot
                fig = go.Figure()
                
                # Add sales line
                fig.add_trace(go.Scatter(
                    x=daily_data['ds'],
                    y=daily_data['y'],
                    mode='lines',
                    name='Daily Sales',
                    line=dict(color='blue')
                ))
                
                # Highlight event periods
                event_dates = daily_data[daily_data[selected_event] == 1]
                if not event_dates.empty:
                    fig.add_trace(go.Scatter(
                        x=event_dates['ds'],
                        y=event_dates['y'],
                        mode='markers',
                        name=f'{selected_event} Periods',
                        marker=dict(color='red', size=8)
                    ))
                
                fig.update_layout(
                    title=f"Sales and {selected_event} Events Over Time",
                    xaxis_title="Date",
                    yaxis_title="Daily Sales",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical significance test
            st.write("#### Statistical Significance")
            
            from scipy import stats
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(event_data['y'], non_event_data['y'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("T-Statistic", f"{t_stat:.3f}")
            
            with col2:
                st.metric("P-Value", f"{p_value:.4f}")
            
            with col3:
                if p_value < 0.05:
                    significance = "Significant"
                else:
                    significance = "Not Significant"
                st.metric("Significance", significance)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(event_data) - 1) * event_data['y'].var() + 
                                 (len(non_event_data) - 1) * non_event_data['y'].var()) / 
                                (len(event_data) + len(non_event_data) - 2))
            cohens_d = (event_data['y'].mean() - non_event_data['y'].mean()) / pooled_std
            
            st.write(f"**Effect Size (Cohen's d):** {cohens_d:.3f}")
            
            if abs(cohens_d) < 0.2:
                effect_size = "Small"
            elif abs(cohens_d) < 0.5:
                effect_size = "Medium"
            else:
                effect_size = "Large"
            
            st.write(f"**Effect Size Interpretation:** {effect_size}")
        
        # Event correlation analysis
        if len(events_to_analyze) > 1:
            st.write("### Event Correlation Analysis")
            
            # Calculate correlation matrix for events
            event_corr = merged_df[events_to_analyze].corr()
            
            # Create correlation heatmap
            fig = px.imshow(
                event_corr,
                title="Event Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Find highly correlated events
            high_corr_pairs = []
            for i in range(len(event_corr.columns)):
                for j in range(i+1, len(event_corr.columns)):
                    corr_val = event_corr.iloc[i, j]
                    if abs(corr_val) > 0.7:  # High correlation threshold
                        high_corr_pairs.append({
                            'Event 1': event_corr.columns[i],
                            'Event 2': event_corr.columns[j],
                            'Correlation': corr_val
                        })
            
            if high_corr_pairs:
                st.write("**Highly Correlated Event Pairs (|r| > 0.7):**")
                high_corr_df = pd.DataFrame(high_corr_pairs)
                st.dataframe(high_corr_df.round(3), use_container_width=True)
            else:
                st.write("**No highly correlated event pairs found.**")
        
    except Exception as e:
        st.error(f"Error in event impact analysis: {str(e)}")


def create_price_analysis(features_df, targets_df):
    """
    Create comprehensive price analysis.
    
    Args:
        features_df: DataFrame with features including prices
        targets_df: DataFrame with target values
    """
    st.subheader("Price Analysis")
    
    if features_df is None or targets_df is None:
        st.warning("No data available for analysis.")
        return
    
    try:
        # Ensure consistent datetime types for 'ds' column before merging
        features_copy = features_df.copy()
        targets_copy = targets_df.copy()
        
        # Convert ds to datetime if needed
        if 'ds' in features_copy.columns:
            if not pd.api.types.is_datetime64_any_dtype(features_copy['ds']):
                features_copy['ds'] = pd.to_datetime(features_copy['ds'])
        
        if 'ds' in targets_copy.columns:
            if not pd.api.types.is_datetime64_any_dtype(targets_copy['ds']):
                targets_copy['ds'] = pd.to_datetime(targets_copy['ds'])
        
        # Merge features and targets
        merged_df = features_copy.merge(targets_copy, on=['unique_id', 'ds'], 
                                     how='inner')
        
        # Find price columns
        price_cols = [col for col in merged_df.columns 
                     if 'price' in col.lower()]
        
        if not price_cols:
            st.info("No price columns found in the data.")
            return
        
        st.write(f"**Found price columns:** {', '.join(price_cols)}")
        
        # Price analysis for each price column
        for price_col in price_cols:
            st.write(f"### {price_col} Analysis")
            
            # Filter out zero prices (item not available)
            available_data = merged_df[merged_df[price_col] > 0].copy()
            unavailable_data = merged_df[merged_df[price_col] == 0].copy()
            
            st.write(f"**Data Availability:**")
            st.write(f"- Available items: {len(available_data):,} ({len(available_data)/len(merged_df)*100:.1f}%)")
            st.write(f"- Unavailable items: {len(unavailable_data):,} ({len(unavailable_data)/len(merged_df)*100:.1f}%)")
            
            if len(available_data) == 0:
                st.warning("No available items found (all prices are zero).")
                continue
            
            # Basic statistics for available items
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Price", f"{available_data[price_col].mean():.2f}")
            
            with col2:
                st.metric("Median Price", f"{available_data[price_col].median():.2f}")
            
            with col3:
                st.metric("Min Price", f"{available_data[price_col].min():.2f}")
            
            with col4:
                st.metric("Max Price", f"{available_data[price_col].max():.2f}")
            
            # Price-demand correlation for available items
            correlation = available_data[price_col].corr(available_data['y'])
            st.write(f"**Price-Demand Correlation (Available Items):** {correlation:.3f}")
            
            # Price elasticity analysis
            st.write("### Price Elasticity Analysis")
            
            # Create price bins with proper handling of duplicates
            try:
                # Use qcut with duplicates='drop' to handle duplicate bin edges
                # Adjust number of bins based on unique values
                unique_prices = available_data[price_col].nunique()
                if unique_prices >= 5:
                    price_bins = pd.qcut(available_data[price_col], q=5, 
                                        labels=['Very Low', 'Low', 'Medium', 'High', 
                                               'Very High'], duplicates='drop')
                    available_data['price_bin'] = price_bins
                    
                    # Calculate average demand by price bin
                    price_demand = available_data.groupby('price_bin')['y'].mean().reset_index()
                    
                    # Plot price-demand relationship
                    fig = px.bar(
                        price_demand,
                        x='price_bin',
                        y='y',
                        title=f"Average Demand by {price_col} Level (Available Items)",
                        labels={'y': 'Average Demand', 'price_bin': 'Price Level'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Not enough unique price values ({unique_prices}) for binning")
                    
            except ValueError as e:
                st.warning(f"Could not create price bins: {str(e)}")
                st.write("**Price Distribution:**")
                
                # Show price distribution instead
                fig = px.histogram(
                    available_data,
                    x=price_col,
                    nbins=20,
                    title=f"{price_col} Distribution (Available Items)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot of price vs demand (sample for performance)
            sample_size = min(1000, len(available_data))
            fig = px.scatter(
                available_data.sample(sample_size),
                x=price_col,
                y='y',
                title="Price vs Demand Scatter Plot (Available Items)",
                labels={'y': 'Demand', price_col: 'Price'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Price trends over time
            if 'ds' in available_data.columns:
                st.write("### Price Trends Over Time")
                
                # Aggregate by date
                daily_price = available_data.groupby('ds').agg({
                    price_col: 'mean',
                    'y': 'sum'
                }).reset_index()
                
                # Create subplot
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Average Price Over Time (Available Items)', 
                                   'Daily Sales Over Time'),
                    vertical_spacing=0.1
                )
                
                # Price plot
                fig.add_trace(
                    go.Scatter(x=daily_price['ds'], y=daily_price[price_col], 
                              mode='lines', name='Average Price'),
                    row=1, col=1
                )
                
                # Sales plot
                fig.add_trace(
                    go.Scatter(x=daily_price['ds'], y=daily_price['y'], 
                              mode='lines', name='Daily Sales'),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Availability analysis
            st.write("### Item Availability Analysis")
            
            if len(unavailable_data) > 0:
                # Analyze when items are unavailable
                if 'ds' in unavailable_data.columns:
                    # Daily availability
                    daily_availability = merged_df.groupby('ds').agg({
                        price_col: lambda x: (x > 0).sum() / len(x) * 100
                    }).reset_index()
                    
                    fig = px.line(
                        daily_availability,
                        x='ds',
                        y=price_col,
                        title="Item Availability Over Time (%)",
                        labels={price_col: 'Availability %', 'ds': 'Date'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Availability by unique_id
                availability_by_item = merged_df.groupby('unique_id').agg({
                    price_col: lambda x: (x > 0).sum() / len(x) * 100
                }).reset_index()
                
                fig = px.histogram(
                    availability_by_item,
                    x=price_col,
                    nbins=20,
                    title="Distribution of Item Availability by Product",
                    labels={price_col: 'Availability %'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_availability = availability_by_item[price_col].mean()
                    st.metric("Average Availability", f"{avg_availability:.1f}%")
                
                with col2:
                    min_availability = availability_by_item[price_col].min()
                    st.metric("Minimum Availability", f"{min_availability:.1f}%")
                
                with col3:
                    max_availability = availability_by_item[price_col].max()
                    st.metric("Maximum Availability", f"{max_availability:.1f}%")
            
            st.write("---")
        
    except Exception as e:
        st.error(f"Error in price analysis: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")

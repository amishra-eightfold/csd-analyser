"""Visualization functions for the support ticket analysis dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta
import logging
import traceback
from config.logging_config import get_logger

# Initialize logger
logger = get_logger('visualization')

def debug(message, data=None, category="visualization"):
    """Log debug information to the logger."""
    if hasattr(st.session_state, 'debug_logger'):
        st.session_state.debug_logger.log(message, data, category)
    
    # Log to file logger
    logger = get_logger(category)
    if data is not None:
        # Convert data to string if needed for logging
        if isinstance(data, dict):
            try:
                import json
                logger.info(f"{message} - {json.dumps(data)}")
            except:
                logger.info(f"{message} - {str(data)}")
        else:
            logger.info(f"{message} - {str(data)}")
    else:
        logger.info(message)

def create_time_series_chart(df: pd.DataFrame, chart_settings: Dict) -> Optional[go.Figure]:
    """Create a time series chart for ticket trends.
    
    Args:
        df: DataFrame containing ticket data
        chart_settings: Dictionary with chart configuration options
        
    Returns:
        Plotly figure object or None if creation fails
    """
    try:
        if df.empty:
            debug("Cannot create time series: DataFrame is empty")
            return None
            
        # Extract settings
        metric = chart_settings.get('metric', 'count')
        group_by = chart_settings.get('group_by', None)
        time_unit = chart_settings.get('time_unit', 'month')
        chart_title = chart_settings.get('title', 'Ticket Trends')
        
        # Prepare time column based on time unit
        if time_unit == 'week':
            # Group by week number
            time_col = 'Week_Number'
            if 'Week_Number' not in df.columns:
                df['Week_Number'] = df['Created Date'].dt.isocalendar().week
        elif time_unit == 'day':
            # Daily data
            time_col = 'Created Date'
            df['Day'] = df['Created Date'].dt.date
        else:
            # Default monthly
            time_col = 'Month_Year'
            if 'Month_Year' not in df.columns:
                df['Month_Year'] = df['Created Date'].dt.to_period('M').astype(str)
        
        debug(f"Creating time series chart", {
            'metric': metric, 
            'group_by': group_by, 
            'time_unit': time_unit,
            'rows': len(df)
        })
        
        # Prepare data for plotting
        if metric == 'count':
            # Just count tickets
            if group_by:
                # Group by time and the grouping variable
                pivot_df = df.pivot_table(
                    index=time_col, 
                    columns=group_by, 
                    values='Case Number', 
                    aggfunc='count'
                ).fillna(0)
                
                # Reset index for plotting
                pivot_df = pivot_df.reset_index()
                
                # Create figure
                fig = go.Figure()
                
                # Add each category as a separate line
                for category in pivot_df.columns[1:]:  # Skip first column (time)
                    fig.add_trace(go.Scatter(
                        x=pivot_df[time_col],
                        y=pivot_df[category],
                        mode='lines+markers',
                        name=f"{category}"
                    ))
                
                # Update layout
                fig.update_layout(
                    title=chart_title,
                    xaxis_title="Time Period",
                    yaxis_title="Number of Tickets",
                    legend_title=group_by,
                    template="plotly_white"
                )
                
            else:
                # Group by time only
                counts = df.groupby(time_col).size().reset_index(name='count')
                
                # Create figure with a single line
                fig = px.line(
                    counts, 
                    x=time_col, 
                    y='count',
                    title=chart_title,
                    labels={'count': 'Number of Tickets', time_col: 'Time Period'},
                    markers=True,
                    template="plotly_white"
                )
                
        elif metric == 'resolution_time':
            # Average resolution time
            if 'Resolution Time (Days)' in df.columns:
                if group_by:
                    # Group by time and the grouping variable
                    pivot_df = df.pivot_table(
                        index=time_col, 
                        columns=group_by, 
                        values='Resolution Time (Days)', 
                        aggfunc='mean'
                    ).fillna(0)
                    
                    # Reset index for plotting
                    pivot_df = pivot_df.reset_index()
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Add each category as a separate line
                    for category in pivot_df.columns[1:]:  # Skip first column (time)
                        fig.add_trace(go.Scatter(
                            x=pivot_df[time_col],
                            y=pivot_df[category],
                            mode='lines+markers',
                            name=f"{category}"
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title=chart_title,
                        xaxis_title="Time Period",
                        yaxis_title="Average Resolution Time (Days)",
                        legend_title=group_by,
                        template="plotly_white"
                    )
                    
                else:
                    # Group by time only
                    avg_times = df.groupby(time_col)['Resolution Time (Days)'].mean().reset_index()
                    
                    # Create figure with a single line
                    fig = px.line(
                        avg_times, 
                        x=time_col, 
                        y='Resolution Time (Days)',
                        title=chart_title,
                        labels={'Resolution Time (Days)': 'Avg. Resolution Time (Days)', time_col: 'Time Period'},
                        markers=True,
                        template="plotly_white"
                    )
            else:
                debug("Cannot create resolution time chart: 'Resolution Time (Days)' column not found")
                return None
                
        elif metric == 'csat':
            # Average CSAT scores
            if 'CSAT' in df.columns:
                if group_by:
                    # Group by time and the grouping variable
                    pivot_df = df.pivot_table(
                        index=time_col, 
                        columns=group_by, 
                        values='CSAT', 
                        aggfunc='mean'
                    ).fillna(0)
                    
                    # Reset index for plotting
                    pivot_df = pivot_df.reset_index()
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Add each category as a separate line
                    for category in pivot_df.columns[1:]:  # Skip first column (time)
                        fig.add_trace(go.Scatter(
                            x=pivot_df[time_col],
                            y=pivot_df[category],
                            mode='lines+markers',
                            name=f"{category}"
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title=chart_title,
                        xaxis_title="Time Period",
                        yaxis_title="Average CSAT Score",
                        legend_title=group_by,
                        template="plotly_white"
                    )
                    
                else:
                    # Group by time only
                    avg_csat = df.groupby(time_col)['CSAT'].mean().reset_index()
                    
                    # Create figure with a single line
                    fig = px.line(
                        avg_csat, 
                        x=time_col, 
                        y='CSAT',
                        title=chart_title,
                        labels={'CSAT': 'Avg. CSAT Score', time_col: 'Time Period'},
                        markers=True,
                        template="plotly_white"
                    )
            else:
                debug("Cannot create CSAT chart: 'CSAT' column not found")
                return None
        else:
            debug(f"Unknown metric for time series: {metric}")
            return None
            
        # Common figure updates
        fig.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=50, b=40),
        )
        
        return fig
        
    except Exception as e:
        error_msg = f"Error creating time series chart: {str(e)}"
        logger.error(error_msg, exc_info=True)
        debug(error_msg, {'traceback': traceback.format_exc()}, category="error")
        return None

def create_distribution_chart(df: pd.DataFrame, chart_settings: Dict) -> Optional[go.Figure]:
    """Create a distribution chart for ticket analysis.
    
    Args:
        df: DataFrame containing ticket data
        chart_settings: Dictionary with chart configuration options
        
    Returns:
        Plotly figure object or None if creation fails
    """
    try:
        if df.empty:
            debug("Cannot create distribution chart: DataFrame is empty")
            return None
            
        # Extract settings
        dimension = chart_settings.get('dimension', 'Status')
        chart_type = chart_settings.get('chart_type', 'bar')
        chart_title = chart_settings.get('title', f'Tickets by {dimension}')
        
        debug(f"Creating distribution chart", {
            'dimension': dimension, 
            'chart_type': chart_type,
            'rows': len(df)
        })
        
        # Check if dimension exists in dataframe
        if dimension not in df.columns:
            debug(f"Cannot create distribution chart: '{dimension}' column not found")
            return None
            
        # Count values for the selected dimension
        counts = df[dimension].value_counts().reset_index()
        counts.columns = [dimension, 'Count']
        
        # Sort values if appropriate
        if dimension in ['Priority', 'Status']:
            # Custom ordering for common fields
            if dimension == 'Priority':
                priority_order = ['Critical', 'High', 'Medium', 'Low']
                counts[dimension] = pd.Categorical(
                    counts[dimension], 
                    categories=priority_order,
                    ordered=True
                )
                counts = counts.sort_values(dimension)
            elif dimension == 'Status':
                status_order = ['Open', 'In Progress', 'Pending', 'Closed']
                counts[dimension] = pd.Categorical(
                    counts[dimension], 
                    categories=status_order,
                    ordered=True
                )
                counts = counts.sort_values(dimension)
        else:
            # Sort by count for other dimensions
            counts = counts.sort_values('Count', ascending=False)
            
        # Create charts based on type
        if chart_type == 'bar':
            fig = px.bar(
                counts, 
                x=dimension, 
                y='Count',
                title=chart_title,
                color=dimension,
                labels={'Count': 'Number of Tickets'},
                template="plotly_white"
            )
        elif chart_type == 'pie':
            fig = px.pie(
                counts, 
                names=dimension, 
                values='Count',
                title=chart_title,
                template="plotly_white"
            )
        else:
            debug(f"Unknown chart type: {chart_type}")
            return None
            
        # Update layout
        fig.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=50, b=40),
        )
        
        return fig
        
    except Exception as e:
        error_msg = f"Error creating distribution chart: {str(e)}"
        logger.error(error_msg, exc_info=True)
        debug(error_msg, {'traceback': traceback.format_exc()}, category="error")
        return None

def create_correlation_chart(df: pd.DataFrame, chart_settings: Dict) -> Optional[go.Figure]:
    """Create a correlation or cross-dimension chart.
    
    Args:
        df: DataFrame containing ticket data
        chart_settings: Dictionary with chart configuration options
        
    Returns:
        Plotly figure object or None if creation fails
    """
    try:
        if df.empty:
            debug("Cannot create correlation chart: DataFrame is empty")
            return None
            
        # Extract settings
        x_dimension = chart_settings.get('x_dimension', 'Product Area')
        y_dimension = chart_settings.get('y_dimension', 'Status')
        chart_title = chart_settings.get('title', f'{x_dimension} vs {y_dimension}')
        
        debug(f"Creating correlation chart", {
            'x_dimension': x_dimension, 
            'y_dimension': y_dimension,
            'rows': len(df)
        })
        
        # Check if dimensions exist in dataframe
        if x_dimension not in df.columns or y_dimension not in df.columns:
            debug(f"Cannot create correlation chart: Dimensions not found")
            return None
            
        # Create cross-tabulation
        crosstab = pd.crosstab(df[x_dimension], df[y_dimension])
        
        # Create heatmap
        fig = px.imshow(
            crosstab,
            labels=dict(x=y_dimension, y=x_dimension, color="Count"),
            x=crosstab.columns,
            y=crosstab.index,
            color_continuous_scale="Viridis",
            title=chart_title,
            text_auto=True,
            aspect="auto",
            template="plotly_white"
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            margin=dict(l=40, r=40, t=50, b=40),
        )
        
        return fig
        
    except Exception as e:
        error_msg = f"Error creating correlation chart: {str(e)}"
        logger.error(error_msg, exc_info=True)
        debug(error_msg, {'traceback': traceback.format_exc()}, category="error")
        return None

def create_metrics_indicators(df: pd.DataFrame) -> Dict[str, Dict]:
    """Create metrics and indicators for the dashboard header.
    
    Args:
        df: DataFrame containing ticket data
        
    Returns:
        Dictionary of metrics with their values and deltas
    """
    try:
        if df.empty:
            debug("Cannot create metrics: DataFrame is empty")
            return {}
            
        metrics = {}
        
        # Calculate ticket count
        total_tickets = len(df)
        metrics['total_tickets'] = {
            'value': total_tickets,
            'delta': None,
            'label': 'Total Tickets'
        }
        
        # Open tickets
        if 'Status' in df.columns:
            open_tickets = df[df['Status'] != 'Closed'].shape[0]
            metrics['open_tickets'] = {
                'value': open_tickets,
                'delta': None,
                'label': 'Open Tickets',
                'percentage': round((open_tickets / total_tickets) * 100) if total_tickets > 0 else 0
            }
        
        # Calculate average resolution time
        if 'Resolution Time (Days)' in df.columns:
            resolution_values = df['Resolution Time (Days)'].dropna()
            if len(resolution_values) > 0:
                avg_resolution = round(resolution_values.mean(), 1)
                metrics['avg_resolution_time'] = {
                    'value': avg_resolution,
                    'delta': None,
                    'label': 'Avg. Resolution Time (Days)'
                }
        
        # Calculate average CSAT
        if 'CSAT' in df.columns:
            csat_values = df['CSAT'].dropna()
            if len(csat_values) > 0:
                avg_csat = round(csat_values.mean(), 2)
                metrics['avg_csat'] = {
                    'value': avg_csat,
                    'delta': None,
                    'label': 'Avg. CSAT Score',
                    'max_value': 5
                }
        
        # Add critical tickets if Priority exists
        if 'Priority' in df.columns:
            critical_tickets = df[df['Priority'] == 'Critical'].shape[0]
            metrics['critical_tickets'] = {
                'value': critical_tickets,
                'delta': None,
                'label': 'Critical Tickets',
                'percentage': round((critical_tickets / total_tickets) * 100) if total_tickets > 0 else 0
            }
        
        debug("Metrics created successfully", {'metrics': list(metrics.keys())})
        return metrics
        
    except Exception as e:
        error_msg = f"Error creating metrics: {str(e)}"
        logger.error(error_msg, exc_info=True)
        debug(error_msg, {'traceback': traceback.format_exc()}, category="error")
        return {}

def display_charts(df: pd.DataFrame, chart_configs: List[Dict]) -> None:
    """Display multiple charts based on configuration.
    
    Args:
        df: DataFrame containing ticket data
        chart_configs: List of chart configuration dictionaries
        
    Returns:
        None
    """
    try:
        if df.empty:
            st.warning("No data available for visualization")
            return
            
        debug(f"Displaying charts", {'chart_count': len(chart_configs)})
        
        # Process each chart configuration
        for config in chart_configs:
            chart_type = config.get('type')
            
            if chart_type == 'time_series':
                fig = create_time_series_chart(df, config)
            elif chart_type == 'distribution':
                fig = create_distribution_chart(df, config)
            elif chart_type == 'correlation':
                fig = create_correlation_chart(df, config)
            else:
                debug(f"Unknown chart type: {chart_type}")
                continue
                
            # Display the chart if it was created successfully
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Could not create {config.get('title', 'chart')}")
                
    except Exception as e:
        error_msg = f"Error displaying charts: {str(e)}"
        logger.error(error_msg, exc_info=True)
        debug(error_msg, {'traceback': traceback.format_exc()}, category="error")
        st.error("An error occurred while displaying charts. Please check the logs for details.") 
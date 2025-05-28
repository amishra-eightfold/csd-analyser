"""Visualization functions for the support ticket analysis application.

This module contains functions for creating visualizations for the support ticket dashboard.
It handles generation of charts for ticket metrics, trends, and patterns.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging
from plotly.subplots import make_subplots

from config.logging_config import get_logger
from utils.text_processing import clean_text, get_technical_stopwords
from utils.visualization_helpers import truncate_string

# Initialize logger
logger = get_logger('visualization')

# Color palettes for different visualizations
VIRIDIS_PALETTE = ["#440154", "#3B528B", "#21918C", "#5EC962", "#FDE725"]  # Viridis colors
AQUA_PALETTE = ["#E0F7FA", "#80DEEA", "#26C6DA", "#00ACC1", "#00838F", "#006064"]  # Material Cyan/Aqua
PRIORITY_COLORS = {
    'P1': VIRIDIS_PALETTE[0],
    'P2': VIRIDIS_PALETTE[1],
    'P3': VIRIDIS_PALETTE[2],
    'P4': VIRIDIS_PALETTE[3]
}

# Define custom color palettes for each visualization type
VOLUME_PALETTE = [AQUA_PALETTE[2], AQUA_PALETTE[4]]  # Two distinct colors for Created/Closed
PRIORITY_PALETTE = VIRIDIS_PALETTE  # Viridis for priority levels
CSAT_PALETTE = AQUA_PALETTE  # Aqua palette for CSAT
HEATMAP_PALETTE = "viridis"  # Viridis colorscale for heatmaps

def create_ticket_volume_chart(df: pd.DataFrame, time_unit: str = 'month') -> go.Figure:
    """Create a chart showing ticket volume over time.
    
    Args:
        df: DataFrame containing case data
        time_unit: Time unit for aggregation ('day', 'week', 'month')
        
    Returns:
        Plotly figure object
    """
    if df.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="Ticket Volume (No Data Available)",
            xaxis_title="Date",
            yaxis_title="Number of Tickets"
        )
        return fig
    
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Ensure created date is datetime
    if 'Created Date' in data.columns:
        data['Created Date'] = pd.to_datetime(data['Created Date'])
    else:
        # Return empty figure if no date column
        fig = go.Figure()
        fig.update_layout(
            title="Ticket Volume (Date Information Missing)",
            xaxis_title="Date",
            yaxis_title="Number of Tickets"
        )
        return fig
    
    # Set datetime index
    data.set_index('Created Date', inplace=True)
    
    # Create an aligned closed date series
    closed_data = df.copy()
    if 'Closed Date' in closed_data.columns:
        closed_data = closed_data.dropna(subset=['Closed Date'])
        closed_data['Closed Date'] = pd.to_datetime(closed_data['Closed Date'])
        closed_data.set_index('Closed Date', inplace=True)
    
    # Prepare resampling parameters based on time unit
    if time_unit == 'day':
        resample_rule = 'D'
        date_format = '%Y-%m-%d'
        title_unit = "Daily"
    elif time_unit == 'week':
        resample_rule = 'W'
        date_format = '%Y-%m-%d'
        title_unit = "Weekly"
    else:  # Default to month
        resample_rule = 'M'  # Use 'M' for month-end frequency
        date_format = '%b %Y'
        title_unit = "Monthly"
    
    # Resample data
    created_tickets = data.resample(resample_rule).size()
    
    # For closed tickets, handle empty case
    if 'Closed Date' in df.columns and not closed_data.empty:
        closed_tickets = closed_data.resample(resample_rule).size()
        
        # Align indexes for created and closed tickets
        idx = created_tickets.index.union(closed_tickets.index)
        created_tickets = created_tickets.reindex(idx, fill_value=0)
        closed_tickets = closed_tickets.reindex(idx, fill_value=0)
        
        # Calculate running net (created - closed)
        net_tickets = created_tickets - closed_tickets
        running_backlog = net_tickets.cumsum()
        
        # Convert to DataFrame
        volume_df = pd.DataFrame({
            'Created': created_tickets,
            'Closed': closed_tickets,
            'Net': net_tickets,
            'Backlog': running_backlog
        })
    else:
        # Only created tickets available
        volume_df = pd.DataFrame({
            'Created': created_tickets,
            'Backlog': created_tickets.cumsum()
        })
    
    # Reset index for plotting
    volume_df = volume_df.reset_index()
    volume_df.rename(columns={'index': 'Date'}, inplace=True)
    
    # Format dates for display
    if 'Date' in volume_df.columns and pd.api.types.is_datetime64_any_dtype(volume_df['Date']):
        volume_df['Date'] = volume_df['Date'].dt.strftime(date_format)
    
    # Calculate month-over-month percentage changes
    if len(volume_df) > 1:
        try:
            volume_df['Created_MoM_%'] = volume_df['Created'].pct_change() * 100
            if 'Closed' in volume_df.columns:
                volume_df['Closed_MoM_%'] = volume_df['Closed'].pct_change() * 100
        except Exception as e:
            logger.warning(f"Error calculating MoM percentages: {str(e)}")
            # Proceed without MoM percentages
    
    # Enhanced color palette
    created_color = "#00B4D8"  # Bright cyan
    closed_color = "#0077B6"   # Dark blue
    net_color = "#03045E"      # Navy
    backlog_color = "#EF476F"  # Bright pink
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Check if Date column exists in the DataFrame
    if 'Date' not in volume_df.columns:
        # Return empty figure if Date column is missing
        fig.update_layout(
            title="Ticket Volume (Date Format Error)",
            xaxis_title="Date",
            yaxis_title="Number of Tickets"
        )
        return fig
    
    # Add created tickets bar
    fig.add_trace(
        go.Bar(
            x=volume_df['Date'],
            y=volume_df['Created'],
            name='Created Tickets',
            marker_color=created_color,
            opacity=0.9,
            text=volume_df['Created'],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Created: %{y}' + ('<br>%{customdata}' if 'Created_MoM_%' in volume_df.columns else ''),
            customdata=volume_df['Created_MoM_%'].round(1).apply(lambda x: f'MoM Change: {x}%' if not pd.isna(x) else '') if 'Created_MoM_%' in volume_df.columns else None
        ),
        secondary_y=False
    )
    
    # Add closed tickets bar if available
    if 'Closed' in volume_df.columns:
        fig.add_trace(
            go.Bar(
                x=volume_df['Date'],
                y=volume_df['Closed'],
                name='Closed Tickets',
                marker_color=closed_color,
                opacity=0.9,
                text=volume_df['Closed'],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Closed: %{y}' + ('<br>%{customdata}' if 'Closed_MoM_%' in volume_df.columns else ''),
                customdata=volume_df['Closed_MoM_%'].round(1).apply(lambda x: f'MoM Change: {x}%' if not pd.isna(x) else '') if 'Closed_MoM_%' in volume_df.columns else None
            ),
            secondary_y=False
        )
    
    # Add backlog line on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=volume_df['Date'],
            y=volume_df['Backlog'],
            mode='lines+markers',
            name='Ticket Backlog',
            line=dict(color=backlog_color, width=3),
            marker=dict(size=8),
            hovertemplate='<b>%{x}</b><br>Backlog: %{y}'
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"{title_unit} Ticket Volume Analysis",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18)
        },
        xaxis_title="Date",
        yaxis_title="Number of Tickets",
        yaxis2_title="Ticket Backlog",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        height=500,
        margin=dict(l=50, r=50, t=80, b=80),
        hovermode="x unified",
        barmode='group',
        bargap=0.15,        # Gap between bars of adjacent location coordinates
        bargroupgap=0.1,    # Gap between bars of the same location coordinates
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)',
            tickangle=45  # Angle the labels
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)',
            zeroline=True,
            zerolinecolor='rgba(0, 0, 0, 0.2)',
            zerolinewidth=1
        ),
        yaxis2=dict(
            showgrid=False,
            zeroline=False
        )
    )
    
    # Add annotations for latest values
    if len(volume_df) > 0 and 'Date' in volume_df.columns:
        latest = volume_df.iloc[-1]
        
        # Add annotation for latest created tickets
        fig.add_annotation(
            x=latest['Date'],
            y=latest['Created'],
            text=f"{latest['Created']}",
            yshift=10,
            showarrow=False,
            font=dict(color=created_color, size=14)
        )
        
        # Add annotation for latest closed tickets if available
        if 'Closed' in volume_df.columns:
            fig.add_annotation(
                x=latest['Date'],
                y=latest['Closed'],
                text=f"{latest['Closed']}",
                yshift=10,
                showarrow=False,
                font=dict(color=closed_color, size=14)
            )
        
        # Add annotation for latest backlog
        fig.add_annotation(
            x=latest['Date'],
            y=latest['Backlog'],
            text=f"Backlog: {latest['Backlog']}",
            yshift=10,
            showarrow=True,
            arrowhead=4,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=backlog_color,
            font=dict(color=backlog_color, size=14)
        )
    
    return fig

def create_resolution_time_chart(df: pd.DataFrame) -> go.Figure:
    """Create a chart showing average resolution time by priority.
    
    Args:
        df: DataFrame containing case data
        
    Returns:
        Plotly figure object
    """
    if df.empty or 'Resolution Time (Days)' not in df.columns:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="Resolution Time by Priority (No Data Available)",
            xaxis_title="Priority",
            yaxis_title="Average Resolution Time (Days)"
        )
        return fig
    
    # Filter for closed tickets with resolution time
    closed_tickets = df[
        (df['Status'] == 'Closed') & 
        (~df['Resolution Time (Days)'].isna())
    ].copy()
    
    if closed_tickets.empty:
        # Return empty figure if no closed tickets
        fig = go.Figure()
        fig.update_layout(
            title="Resolution Time by Priority (No Closed Tickets Available)",
            xaxis_title="Priority",
            yaxis_title="Average Resolution Time (Days)"
        )
        return fig
    
    # Determine which priority field to use, prioritizing Internal_Priority__c
    priority_field = None
    for field in ['Internal_Priority__c', 'Highest Priority', 'Priority']:
        if field in closed_tickets.columns:
            priority_field = field
            logger.info(f"Using {priority_field} for resolution time analysis")
            break
    
    if priority_field is None:
        # Return empty figure if no priority field is available
        fig = go.Figure()
        fig.update_layout(
            title="Resolution Time by Priority (No Priority Data Available)",
            xaxis_title="Priority",
            yaxis_title="Average Resolution Time (Days)"
        )
        return fig
            
    # Group by priority and calculate mean resolution time
    priority_resolution = closed_tickets.groupby(priority_field)['Resolution Time (Days)'].agg(
        ['mean', 'count', 'median', 'min', 'max']
    ).reset_index()
    
    # Sort by priority
    priority_order = ['P1', 'P2', 'P3', 'P4']
    priority_resolution[priority_field] = pd.Categorical(
        priority_resolution[priority_field], 
        categories=priority_order, 
        ordered=True
    )
    priority_resolution = priority_resolution.sort_values(priority_field)
    
    # Create plot
    fig = go.Figure()
    
    # Add bar chart for average resolution time
    fig.add_trace(go.Bar(
        x=priority_resolution[priority_field],
        y=priority_resolution['mean'],
        text=priority_resolution['mean'].round(1),
        textposition='auto',
        marker_color=[PRIORITY_COLORS.get(p, '#CCCCCC') for p in priority_resolution[priority_field]],
        name='Average Resolution Time',
        hovertemplate='<b>%{x}</b><br>Average: %{y:.1f} days<br>Median: %{customdata[0]:.1f} days<br>Min: %{customdata[1]:.1f} days<br>Max: %{customdata[2]:.1f} days<br>Count: %{customdata[3]} tickets',
        customdata=np.column_stack((
            priority_resolution['median'],
            priority_resolution['min'],
            priority_resolution['max'],
            priority_resolution['count']
        ))
    ))
    
    # Add error bars
    for i, row in priority_resolution.iterrows():
        fig.add_trace(go.Scatter(
            x=[row[priority_field], row[priority_field]],
            y=[row['min'], row['max']],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.5)', width=1),
            showlegend=False,
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter(
            x=[row[priority_field]],
            y=[row['median']],
            mode='markers',
            marker=dict(
                color='rgba(0,0,0,0.8)',
                symbol='line-ns',
                size=10,
                line=dict(width=2)
            ),
            name='Median' if i == 0 else None,
            showlegend=i == 0,
            hoverinfo='none'
        ))
    
    # Update layout
    fig.update_layout(
        title="Resolution Time by Priority",
        xaxis_title="Priority",
        yaxis_title="Resolution Time (Days)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def create_first_response_time_chart(df: pd.DataFrame, response_hours: pd.Series = None) -> go.Figure:
    """Create a chart showing first response time distribution by priority.
    
    Args:
        df: DataFrame containing case data
        response_hours: Optional pre-calculated response hours Series
        
    Returns:
        Plotly figure object
    """
    if df.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="First Response Time Analysis (No Data Available)",
            xaxis_title="Priority",
            yaxis_title="Response Time (Hours)"
        )
        return fig
    
    # Calculate response times if not provided
    if response_hours is None:
        try:
            from utils.time_analysis import calculate_first_response_time
            response_hours, stats = calculate_first_response_time(df, allow_synthetic=False)
            if stats['valid_records'] == 0:
                # No valid response time data
                fig = go.Figure()
                fig.update_layout(
                    title="First Response Time Analysis (No Valid Data)",
                    xaxis_title="Priority",
                    yaxis_title="Response Time (Hours)"
                )
                return fig
        except Exception as e:
            logger.error(f"Error calculating first response time: {str(e)}")
            # Return empty figure if calculation fails
            fig = go.Figure()
            fig.update_layout(
                title="First Response Time Analysis (Calculation Error)",
                xaxis_title="Priority",
                yaxis_title="Response Time (Hours)"
            )
            return fig
    
    # Create figure
    fig = go.Figure()
    
    # Get priority column - try multiple possible names with Internal_Priority__c first
    priority_col = None
    for col in ['Internal_Priority__c', 'Highest Priority', 'Highest_Priority', 'Priority']:
        if col in df.columns:
            priority_col = col
            break
    
    if priority_col is None:
        # Can't proceed without priority information
        fig.update_layout(
            title="First Response Time Analysis (Priority Data Missing)",
            xaxis_title="Priority",
            yaxis_title="Response Time (Hours)"
        )
        return fig
    
    # Sort priorities in the correct order (P0/P1 to P4)
    priorities = sorted(df[priority_col].unique())
    
    # Add a box plot for each priority level
    for priority in priorities:
        if pd.isna(priority) or priority in ['Not Set', 'Unknown', '', None]:
            continue
            
        priority_mask = df[priority_col] == priority
        priority_data = response_hours[priority_mask].dropna()
        
        if len(priority_data) > 0:
            fig.add_trace(go.Box(
                y=priority_data,
                name=f'Priority {priority}',
                marker_color=PRIORITY_COLORS.get(priority, VIRIDIS_PALETTE[0]),
                boxpoints='outliers'
            ))
    
    # Add SLA threshold lines if applicable
    sla_thresholds = {
        'P0': 1,     # 1 hour
        'P1': 24,    # 24 hours
        'P2': 48,    # 48 hours
        # P3/P4 have no SLA
    }
    
    for priority, threshold in sla_thresholds.items():
        if priority in priorities:
            fig.add_shape(
                type="line",
                x0=priorities.index(priority) - 0.4,
                y0=threshold,
                x1=priorities.index(priority) + 0.4,
                y1=threshold,
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",
                ),
                name=f"SLA: {threshold}h"
            )
    
    # Update layout
    fig.update_layout(
        title='First Response Time Distribution by Priority',
        yaxis_title='Response Time (Hours)',
        xaxis_title='Priority',
        showlegend=True,
        boxmode='group',
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def create_wordcloud(df: pd.DataFrame, column: str = 'Subject') -> plt.Figure:
    """Create a wordcloud visualization from text data.
    
    Args:
        df: DataFrame containing case data
        column: Column name containing text data
        
    Returns:
        Matplotlib figure object
    """
    if df.empty or column not in df.columns:
        # Create empty figure with message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"No data available for {column}", 
                horizontalalignment='center', verticalalignment='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Combine all text in the column
    text = ' '.join([str(text) for text in df[column].fillna('')])
    
    if not text.strip():
        # Create empty figure with message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"No text data available in {column}", 
                horizontalalignment='center', verticalalignment='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Clean text
    cleaned_text = clean_text(text)
    
    # Get stopwords
    stopwords = set(STOPWORDS)
    technical_stopwords = get_technical_stopwords()
    stopwords.update(technical_stopwords)
    
    # Create wordcloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100,
        stopwords=stopwords,
        contour_width=3,
        contour_color='steelblue'
    ).generate(cleaned_text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud: {column}', fontsize=15)
    
    return fig

def create_priority_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create a chart showing ticket distribution by priority.
    
    Args:
        df: DataFrame containing case data
        
    Returns:
        Plotly figure object
    """
    # Determine which priority field to use, prioritizing Internal_Priority__c
    priority_field = None
    for field in ['Internal_Priority__c', 'Highest Priority', 'Priority']:
        if field in df.columns and not df[field].isna().all():
            priority_field = field
            logger.info(f"Using {priority_field} for priority distribution chart")
            break
    
    if df.empty or priority_field is None:
        # Return empty figure if no valid priority data
        fig = go.Figure()
        fig.update_layout(
            title="Ticket Distribution by Priority (No Data Available)",
            xaxis_title="Priority",
            yaxis_title="Number of Tickets"
        )
        return fig
    
    # Count tickets by priority
    priority_counts = df[priority_field].value_counts().reset_index()
    priority_counts.columns = ['Priority', 'Count']
    
    # Calculate percentages
    total = priority_counts['Count'].sum()
    priority_counts['Percentage'] = (priority_counts['Count'] / total * 100).round(1)
    
    # Sort by priority
    priority_order = ['P0', 'P1', 'P2', 'P3', 'P4']
    priority_counts['Priority'] = pd.Categorical(
        priority_counts['Priority'], 
        categories=priority_order, 
        ordered=True
    )
    priority_counts = priority_counts.sort_values('Priority')
    
    # Create plot
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=priority_counts['Priority'],
        y=priority_counts['Count'],
        text=[f"{count} ({pct}%)" for count, pct in 
              zip(priority_counts['Count'], priority_counts['Percentage'])],
        textposition='auto',
        marker_color=[PRIORITY_COLORS.get(p, '#CCCCCC') for p in priority_counts['Priority']],
        hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{customdata:.1f}%',
        customdata=priority_counts['Percentage']
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Ticket Distribution by {priority_field.replace('_c', '').replace('__', ' ')}",
        xaxis_title="Priority",
        yaxis_title="Number of Tickets",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def create_product_area_chart(df: pd.DataFrame) -> go.Figure:
    """Create a chart showing ticket distribution by product area.
    
    Args:
        df: DataFrame containing case data
        
    Returns:
        Plotly figure object
    """
    if df.empty or 'Product_Area__c' not in df.columns:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="Ticket Distribution by Product Area (No Data Available)",
            xaxis_title="Product Area",
            yaxis_title="Number of Tickets"
        )
        return fig
    
    # Count tickets by product area
    product_area_counts = df['Product_Area__c'].fillna('Unspecified').value_counts().reset_index()
    product_area_counts.columns = ['Product Area', 'Count']
    
    # Calculate percentages
    total = product_area_counts['Count'].sum()
    product_area_counts['Percentage'] = (product_area_counts['Count'] / total * 100).round(1)
    
    # Sort by count (descending)
    product_area_counts = product_area_counts.sort_values('Count', ascending=False)
    
    # Limit to top 10 for readability
    if len(product_area_counts) > 10:
        top_areas = product_area_counts.iloc[:9].copy()
        others = pd.DataFrame({
            'Product Area': ['Other Areas'],
            'Count': [product_area_counts.iloc[9:]['Count'].sum()],
            'Percentage': [product_area_counts.iloc[9:]['Percentage'].sum()]
        })
        product_area_counts = pd.concat([top_areas, others])
    
    # Create color scale
    colors = px.colors.sequential.Viridis[:len(product_area_counts)]
    
    # Create plot
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=product_area_counts['Product Area'],
        y=product_area_counts['Count'],
        text=[f"{count} ({pct}%)" for count, pct in 
              zip(product_area_counts['Count'], product_area_counts['Percentage'])],
        textposition='auto',
        marker_color=colors,
        hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{customdata:.1f}%',
        customdata=product_area_counts['Percentage']
    ))
    
    # Update layout
    fig.update_layout(
        title="Ticket Distribution by Product Area",
        xaxis_title="Product Area",
        yaxis_title="Number of Tickets",
        height=600,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Update x-axis for better readability
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_csat_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create a chart showing CSAT distribution.
    
    Args:
        df: DataFrame containing case data with CSAT information
        
    Returns:
        Plotly figure object
    """
    if df.empty or 'CSAT' not in df.columns:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="CSAT Distribution (No Data Available)",
            xaxis_title="CSAT Score",
            yaxis_title="Number of Tickets"
        )
        return fig
    
    # Filter for tickets with CSAT scores
    csat_data = df.dropna(subset=['CSAT']).copy()
    
    if csat_data.empty:
        # Return empty figure if no CSAT data
        fig = go.Figure()
        fig.update_layout(
            title="CSAT Distribution (No CSAT Data Available)",
            xaxis_title="CSAT Score",
            yaxis_title="Number of Tickets"
        )
        return fig
    
    # Count tickets by CSAT score
    csat_counts = csat_data['CSAT'].value_counts().reset_index()
    csat_counts.columns = ['CSAT', 'Count']
    
    # Calculate percentages
    total = csat_counts['Count'].sum()
    csat_counts['Percentage'] = (csat_counts['Count'] / total * 100).round(1)
    
    # Sort by CSAT score
    csat_counts = csat_counts.sort_values('CSAT')
    
    # Define colors (from worst to best)
    colors = [
        '#d73027',  # Red (1)
        '#fc8d59',  # Orange (2)
        '#fee090',  # Yellow (3)
        '#91bfdb',  # Light Blue (4)
        '#4575b4'   # Dark Blue (5)
    ]
    
    # Create plot
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=csat_counts['CSAT'],
        y=csat_counts['Count'],
        text=[f"{count} ({pct}%)" for count, pct in 
              zip(csat_counts['Count'], csat_counts['Percentage'])],
        textposition='auto',
        marker_color=[colors[int(score)-1] if 1 <= score <= 5 else '#CCCCCC' 
                     for score in csat_counts['CSAT']],
        hovertemplate='<b>CSAT %{x}</b><br>Count: %{y}<br>Percentage: %{customdata:.1f}%',
        customdata=csat_counts['Percentage']
    ))
    
    # Calculate average CSAT
    avg_csat = csat_data['CSAT'].mean()
    
    # Update layout
    fig.update_layout(
        title=f"CSAT Distribution (Average: {avg_csat:.2f})",
        xaxis_title="CSAT Score",
        yaxis_title="Number of Tickets",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Add vertical line for average CSAT
    fig.add_shape(
        type="line",
        x0=avg_csat,
        x1=avg_csat,
        y0=0,
        y1=csat_counts['Count'].max(),
        line=dict(color="black", width=2, dash="dash")
    )
    
    # Add annotation for average
    fig.add_annotation(
        x=avg_csat,
        y=csat_counts['Count'].max(),
        text=f"Average: {avg_csat:.2f}",
        showarrow=True,
        arrowhead=1
    )
    
    return fig

def create_root_cause_chart(df: pd.DataFrame) -> go.Figure:
    """Create a chart showing ticket distribution by root cause.
    
    Args:
        df: DataFrame containing case data with RCA__c information
        
    Returns:
        Plotly figure object
    """
    if df.empty or 'RCA__c' not in df.columns:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="Ticket Distribution by Root Cause (No Data Available)",
            xaxis_title="Root Cause",
            yaxis_title="Number of Tickets"
        )
        return fig
    
    # Count tickets by root cause
    root_cause_counts = df['RCA__c'].fillna('Not Specified').value_counts().reset_index()
    root_cause_counts.columns = ['Root Cause', 'Count']
    
    # Calculate percentages
    total = root_cause_counts['Count'].sum()
    root_cause_counts['Percentage'] = (root_cause_counts['Count'] / total * 100).round(1)
    
    # Sort by count (descending)
    root_cause_counts = root_cause_counts.sort_values('Count', ascending=False)
    
    # Limit to top 10 for readability
    if len(root_cause_counts) > 10:
        top_causes = root_cause_counts.iloc[:9].copy()
        others = pd.DataFrame({
            'Root Cause': ['Other Causes'],
            'Count': [root_cause_counts.iloc[9:]['Count'].sum()],
            'Percentage': [root_cause_counts.iloc[9:]['Percentage'].sum()]
        })
        root_cause_counts = pd.concat([top_causes, others])
    
    # Create color scale
    colors = px.colors.sequential.Plasma[:len(root_cause_counts)]
    
    # Create plot
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=root_cause_counts['Root Cause'],
        y=root_cause_counts['Count'],
        text=[f"{count} ({pct}%)" for count, pct in 
              zip(root_cause_counts['Count'], root_cause_counts['Percentage'])],
        textposition='auto',
        marker_color=colors,
        hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{customdata:.1f}%',
        customdata=root_cause_counts['Percentage']
    ))
    
    # Update layout
    fig.update_layout(
        title="Ticket Distribution by Root Cause",
        xaxis_title="Root Cause",
        yaxis_title="Number of Tickets",
        height=600,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Update x-axis for better readability
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_root_cause_product_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create a heatmap showing the relationship between root causes and product areas.
    
    Args:
        df: DataFrame containing case data with RCA__c and Product_Area__c information
        
    Returns:
        Plotly figure object
    """
    if df.empty or 'RCA__c' not in df.columns or 'Product_Area__c' not in df.columns:
        # Return empty figure if required columns are missing
        fig = go.Figure()
        fig.update_layout(
            title="Root Cause by Product Area (Data Missing)",
            xaxis_title="Product Area",
            yaxis_title="Root Cause"
        )
        return fig
    
    # Fill NA values
    data = df.copy()
    data['RCA__c'] = data['RCA__c'].fillna('Not Specified')
    data['Product_Area__c'] = data['Product_Area__c'].fillna('Not Specified')
    
    # Create cross-tabulation
    cross_tab = pd.crosstab(
        data['RCA__c'], 
        data['Product_Area__c'],
        normalize='columns'  # Normalize by column (product area)
    ).round(2)
    
    # Filter for top root causes and product areas to keep the chart readable
    top_root_causes = data['RCA__c'].value_counts().nlargest(8).index
    top_product_areas = data['Product_Area__c'].value_counts().nlargest(8).index
    
    # Filter crosstab
    filtered_cross_tab = cross_tab.loc[
        cross_tab.index.isin(top_root_causes), 
        cross_tab.columns.isin(top_product_areas)
    ]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=filtered_cross_tab.values,
        x=filtered_cross_tab.columns,
        y=filtered_cross_tab.index,
        colorscale='Viridis',
        colorbar=dict(title='Proportion'),
        hovertemplate='Root Cause: %{y}<br>Product Area: %{x}<br>Proportion: %{z:.0%}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title="Root Cause Distribution by Product Area",
        xaxis_title="Product Area",
        yaxis_title="Root Cause",
        height=600,
        margin=dict(l=60, r=40, t=60, b=120)
    )
    
    # Update axes for better readability
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_priority_time_chart(df: pd.DataFrame, history_df: pd.DataFrame) -> go.Figure:
    """Create a chart showing average time spent in each priority level for tickets that had priority changes.
    
    Args:
        df: DataFrame containing case data
        history_df: DataFrame containing case history data
        
    Returns:
        Plotly figure object
    """
    # Filter for closed tickets only
    closed_tickets = df[df['Status'] == 'Closed'].copy()
    
    if closed_tickets.empty or history_df.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="Average Time in Each Priority (No Data Available)",
            xaxis_title="Priority",
            yaxis_title="Average Time (Days)"
        )
        return fig
    
    # Filter history for priority changes
    priority_changes = history_df[history_df['Field'] == 'Priority'].copy()
    
    if priority_changes.empty:
        # Return empty figure if no priority changes
        fig = go.Figure()
        fig.update_layout(
            title="Average Time in Each Priority (No Priority Changes Found)",
            xaxis_title="Priority",
            yaxis_title="Average Time (Days)"
        )
        return fig
    
    # Get list of case IDs with priority changes that are also closed
    closed_ticket_ids = set(closed_tickets['Id'])
    cases_with_changes = priority_changes[priority_changes['CaseId'].isin(closed_ticket_ids)]
    
    if cases_with_changes.empty:
        # Return empty figure if no closed tickets with priority changes
        fig = go.Figure()
        fig.update_layout(
            title="Average Time in Each Priority (No Closed Tickets with Priority Changes)",
            xaxis_title="Priority",
            yaxis_title="Average Time (Days)"
        )
        return fig
    
    # Calculate time spent in each priority
    time_in_priority = {}
    
    # Process each case
    for case_id in cases_with_changes['CaseId'].unique():
        # Get all priority changes for this case
        case_changes = cases_with_changes[cases_with_changes['CaseId'] == case_id].sort_values('Created Date')
        
        # Get case creation and closure dates
        case_created = closed_tickets.loc[closed_tickets['Id'] == case_id, 'Created Date'].iloc[0]
        case_closed = closed_tickets.loc[closed_tickets['Id'] == case_id, 'Closed Date'].iloc[0]
        
        # Convert both timestamps to tz-naive if either has timezone info
        if case_created.tzinfo is not None or case_closed.tzinfo is not None:
            case_created = case_created.tz_localize(None) if case_created.tzinfo is not None else case_created
            case_closed = case_closed.tz_localize(None) if case_closed.tzinfo is not None else case_closed
        
        # Get initial priority
        initial_priority = closed_tickets.loc[closed_tickets['Id'] == case_id, 'Priority'].iloc[0]
        
        # Initialize time tracking
        last_change_date = case_created
        current_priority = initial_priority
        
        # Track time in initial priority (before first change)
        if case_changes.empty:
            # No changes, all time spent in initial priority
            days_in_priority = (case_closed - case_created).total_seconds() / (24 * 3600)
            if current_priority not in time_in_priority:
                time_in_priority[current_priority] = []
            time_in_priority[current_priority].append(days_in_priority)
        else:
            # Process each priority change
            for _, change in case_changes.iterrows():
                change_date = change['Created Date']
                # Convert change_date to tz-naive if needed
                if change_date.tzinfo is not None:
                    change_date = change_date.tz_localize(None)
                
                # Calculate time spent in current priority
                days_in_priority = (change_date - last_change_date).total_seconds() / (24 * 3600)
                
                # Add to tracking
                if current_priority not in time_in_priority:
                    time_in_priority[current_priority] = []
                time_in_priority[current_priority].append(days_in_priority)
                
                # Update for next iteration
                last_change_date = change_date
                current_priority = change['NewValue']
            
            # Add time from last change to closure
            days_in_priority = (case_closed - last_change_date).total_seconds() / (24 * 3600)
            if current_priority not in time_in_priority:
                time_in_priority[current_priority] = []
            time_in_priority[current_priority].append(days_in_priority)
    
    # Calculate averages
    avg_time_in_priority = {
        priority: sum(days) / len(days)
        for priority, days in time_in_priority.items()
        if days  # Ensure we have data
    }
    
    # Define priority order for sorting
    priority_order = {'P0': 0, 'P1': 1, 'P2': 2, 'P3': 3, 'P4': 4, 'P5': 5}
    
    # Sort priorities
    sorted_priorities = sorted(
        avg_time_in_priority.keys(),
        key=lambda x: priority_order.get(x, 999)
    )
    sorted_times = [avg_time_in_priority[p] for p in sorted_priorities]
    
    # Create color scale based on priority level
    colors = px.colors.sequential.Viridis[::-1]  # Reverse for highest priority to be darkest
    color_scale = {
        priority: colors[i % len(colors)]
        for i, priority in enumerate(sorted_priorities)
    }
    bar_colors = [color_scale.get(p, '#636EFA') for p in sorted_priorities]
    
    # Create figure
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(
        go.Bar(
            x=sorted_priorities,
            y=sorted_times,
            marker_color=bar_colors,
            text=[f"{t:.1f} days" for t in sorted_times],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Average time: %{y:.1f} days<extra></extra>'
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Average Time in Each Priority Level (Closed Tickets with Priority Changes)",
        xaxis_title="Priority",
        yaxis_title="Average Time (Days)",
        xaxis=dict(categoryorder='array', categoryarray=sorted_priorities),
        hoverlabel=dict(bgcolor="white", font_size=14),
        plot_bgcolor='white',
        bargap=0.2
    )
    
    # Add gridlines
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211,211,211,0.3)'
    )
    
    return fig

def create_status_time_chart(df: pd.DataFrame, history_df: pd.DataFrame) -> go.Figure:
    """Create a chart showing average time spent in each status for tickets.
    
    Args:
        df: DataFrame containing case data
        history_df: DataFrame containing case history data
        
    Returns:
        Plotly figure object
    """
    if df.empty or history_df.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="Average Time in Each Status (No Data Available)",
            xaxis_title="Status",
            yaxis_title="Average Time (Days)"
        )
        return fig
    
    # Filter history for status changes
    status_changes = history_df[history_df['Field'] == 'Status'].copy()
    
    if status_changes.empty:
        # Return empty figure if no status changes
        fig = go.Figure()
        fig.update_layout(
            title="Average Time in Each Status (No Status Changes Found)",
            xaxis_title="Status",
            yaxis_title="Average Time (Days)"
        )
        return fig
    
    # Calculate time spent in each status
    time_in_status = {}
    
    # Process each case
    for case_id in df['Id'].unique():
        # Get all status changes for this case
        case_changes = status_changes[status_changes['CaseId'] == case_id].sort_values('Created Date')
        
        # Get case creation date
        case_created = df.loc[df['Id'] == case_id, 'Created Date'].iloc[0]
        
        # Get case closure date if closed
        is_closed = df.loc[df['Id'] == case_id, 'Status'].iloc[0] == 'Closed'
        if is_closed:
            case_closed = df.loc[df['Id'] == case_id, 'Closed Date'].iloc[0] 
        else:
            case_closed = pd.Timestamp.now()
            # Ensure now() is tz-naive if case_created is tz-naive
            if case_created.tzinfo is None:
                case_closed = case_closed.tz_localize(None)
            elif case_created.tzinfo is not None:
                # Make sure now() has the same timezone as case_created
                case_closed = case_closed.tz_localize('UTC').tz_convert(case_created.tzinfo)
        
        # Ensure both datetimes have consistent timezone info
        if case_created.tzinfo is not None and case_closed.tzinfo is None:
            case_closed = case_closed.tz_localize(case_created.tzinfo)
        elif case_created.tzinfo is None and case_closed.tzinfo is not None:
            case_created = case_created.tz_localize(case_closed.tzinfo)
        
        # Get initial status
        initial_status = df.loc[df['Id'] == case_id, 'Status'].iloc[0]
        if not is_closed and case_changes.empty:
            # Skip open tickets with no changes - would just inflate current status
            continue
        
        # Initialize time tracking
        last_change_date = case_created
        current_status = initial_status
        
        # Track time in initial status (before first change)
        if case_changes.empty:
            # No changes, all time spent in initial status
            days_in_status = (case_closed - case_created).total_seconds() / (24 * 3600)
            if current_status not in time_in_status:
                time_in_status[current_status] = []
            time_in_status[current_status].append(days_in_status)
        else:
            # Process each status change
            for _, change in case_changes.iterrows():
                change_date = change['Created Date']
                
                # Ensure timezone consistency
                if last_change_date.tzinfo is not None and change_date.tzinfo is None:
                    change_date = change_date.tz_localize(last_change_date.tzinfo)
                elif last_change_date.tzinfo is None and change_date.tzinfo is not None:
                    last_change_date = last_change_date.tz_localize(change_date.tzinfo)
                elif last_change_date.tzinfo is not None and change_date.tzinfo is not None and last_change_date.tzinfo != change_date.tzinfo:
                    change_date = change_date.tz_convert(last_change_date.tzinfo)
                
                # Calculate time spent in current status
                days_in_status = (change_date - last_change_date).total_seconds() / (24 * 3600)
                
                # Add to tracking
                if current_status not in time_in_status:
                    time_in_status[current_status] = []
                time_in_status[current_status].append(days_in_status)
                
                # Update for next iteration
                last_change_date = change_date
                current_status = change['NewValue']
            
            # Add time from last change to closure (or now for open tickets)
            # Ensure timezone consistency
            if last_change_date.tzinfo is not None and case_closed.tzinfo is None:
                case_closed = case_closed.tz_localize(last_change_date.tzinfo)
            elif last_change_date.tzinfo is None and case_closed.tzinfo is not None:
                last_change_date = last_change_date.tz_localize(case_closed.tzinfo)
            elif last_change_date.tzinfo is not None and case_closed.tzinfo is not None and last_change_date.tzinfo != case_closed.tzinfo:
                case_closed = case_closed.tz_convert(last_change_date.tzinfo)
                
            days_in_status = (case_closed - last_change_date).total_seconds() / (24 * 3600)
            if current_status not in time_in_status:
                time_in_status[current_status] = []
            time_in_status[current_status].append(days_in_status)
    
    # Calculate averages
    avg_time_in_status = {
        status: sum(days) / len(days)
        for status, days in time_in_status.items()
        if days  # Ensure we have data
    }
    
    # Define status order based on common workflow
    common_order = [
        'New', 'Open', 'In Progress', 'Pending', 'On Hold', 
        'Waiting on Customer', 'Waiting on Engineering',
        'Resolved', 'Closed'
    ]
    
    # Sort statuses by common workflow order then alphabetically for any not in the list
    def status_sort_key(status):
        if status in common_order:
            return (0, common_order.index(status))
        return (1, status)
    
    sorted_statuses = sorted(avg_time_in_status.keys(), key=status_sort_key)
    sorted_times = [avg_time_in_status[s] for s in sorted_statuses]
    
    # Create color mapping for statuses
    status_colors = {
        'New': '#FFA15A',  # Orange
        'Open': '#FF6692',  # Pink
        'In Progress': '#B6E880',  # Light green
        'Pending': '#FF97FF',  # Light purple
        'On Hold': '#FECB52',  # Yellow
        'Waiting on Customer': '#636EFA',  # Blue
        'Waiting on Engineering': '#EF553B',  # Red
        'Resolved': '#00CC96',  # Green
        'Closed': '#AB63FA'  # Purple
    }
    
    # Create color list for bars
    bar_colors = [status_colors.get(s, '#636EFA') for s in sorted_statuses]
    
    # Create figure
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(
        go.Bar(
            x=sorted_statuses,
            y=sorted_times,
            marker_color=bar_colors,
            text=[f"{t:.1f} days" for t in sorted_times],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Average time: %{y:.1f} days<extra></extra>'
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Average Time in Each Status",
        xaxis_title="Status",
        yaxis_title="Average Time (Days)",
        hoverlabel=dict(bgcolor="white", font_size=14),
        plot_bgcolor='white',
        bargap=0.2
    )
    
    # Add gridlines
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211,211,211,0.3)'
    )
    
    return fig 
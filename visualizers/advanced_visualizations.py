"""Advanced visualization components for the Support Ticket Analysis application."""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple

__all__ = [
    'create_csat_analysis',
    'create_word_clouds',
    'create_root_cause_analysis',
    'create_first_response_analysis'
]

def create_first_response_analysis(df: pd.DataFrame) -> Tuple[go.Figure, Dict]:
    """Create first response time analysis with box plots by priority.
    
    Args:
        df: DataFrame containing ticket data
        
    Returns:
        Tuple of (plotly figure, statistics dictionary)
    """
    try:
        df = df.copy()
        
        # Calculate first response time in hours, handling potential errors
        df['first_response_hours'] = None  # Initialize column
        
        # Only calculate for rows where both dates are present
        mask = df['First Response Time'].notna() & df['Created Date'].notna()
        if mask.any():
            df.loc[mask, 'first_response_hours'] = (
                pd.to_datetime(df.loc[mask, 'First Response Time']) - 
                pd.to_datetime(df.loc[mask, 'Created Date'])
            ).dt.total_seconds() / 3600
        
        # Filter out invalid response times (negative or extremely large values)
        valid_mask = (
            df['first_response_hours'].notna() & 
            (df['first_response_hours'] > 0) & 
            (df['first_response_hours'] < 720)  # Cap at 30 days
        )
        
        if not valid_mask.any():
            raise ValueError("No valid response time data found after filtering")
            
        analysis_df = df[valid_mask].copy()
        
        # Create box plot
        fig = go.Figure()
        
        # Add box plot for each priority level
        for priority in sorted(analysis_df['Priority'].unique()):
            priority_data = analysis_df[analysis_df['Priority'] == priority]['first_response_hours']
            if len(priority_data) > 0:
                fig.add_trace(go.Box(
                    y=priority_data,
                    name=f'Priority {priority}',
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8
                ))
        
        # Update layout
        fig.update_layout(
            title='First Response Time Distribution by Priority',
            yaxis_title='Response Time (Hours)',
            showlegend=True,
            boxmode='group',
            height=600,
            yaxis=dict(
                type='log',  # Use log scale for better visualization
                title='Response Time (Hours) - Log Scale'
            )
        )
        
        # Calculate statistics
        stats_dict = {
            'overall_mean': analysis_df['first_response_hours'].mean(),
            'overall_median': analysis_df['first_response_hours'].median(),
            'by_priority': analysis_df.groupby('Priority').agg({
                'first_response_hours': ['count', 'mean', 'median', 'std']
            }).round(2).to_dict(),
            'sla_compliance': {
                'within_24h': (analysis_df['first_response_hours'] <= 24).mean() * 100,
                'within_48h': (analysis_df['first_response_hours'] <= 48).mean() * 100,
                'within_72h': (analysis_df['first_response_hours'] <= 72).mean() * 100
            }
        }
        
        return fig, stats_dict
        
    except Exception as e:
        raise Exception(f"Error in first response time analysis: {str(e)}")

def create_csat_analysis(df: pd.DataFrame) -> Tuple[go.Figure, Dict]:
    """Create CSAT trend analysis with confidence intervals.
    
    Args:
        df: DataFrame containing CSAT data
        
    Returns:
        Tuple of (plotly figure, statistics dictionary)
    """
    # Ensure CSAT and date columns are properly formatted
    df = df.copy()
    df['Created Date'] = pd.to_datetime(df['Created Date'])
    df['Month'] = df['Created Date'].dt.strftime('%Y-%m')
    df['CSAT'] = pd.to_numeric(df['CSAT'], errors='coerce')
    
    # Calculate monthly CSAT statistics
    monthly_stats = df.groupby('Month').agg({
        'CSAT': ['count', 'mean', 'std']
    }).reset_index()
    monthly_stats.columns = ['Month', 'Count', 'Mean', 'Std']
    
    # Calculate confidence intervals (95%)
    z_score = 1.96  # 95% confidence interval
    monthly_stats['CI_lower'] = monthly_stats['Mean'] - z_score * (monthly_stats['Std'] / np.sqrt(monthly_stats['Count']))
    monthly_stats['CI_upper'] = monthly_stats['Mean'] + z_score * (monthly_stats['Std'] / np.sqrt(monthly_stats['Count']))
    
    # Create visualization with subplots to show both bars and confidence intervals
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for average CSAT
    fig.add_trace(
        go.Bar(
            x=monthly_stats['Month'],
            y=monthly_stats['Mean'],
            name='Average CSAT',
            marker_color='rgb(26, 118, 255)',
            text=monthly_stats['Mean'].round(2),
            textposition='auto',
        ),
        secondary_y=False
    )
    
    # Add response count as a line on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=monthly_stats['Month'],
            y=monthly_stats['Count'],
            name='Response Count',
            line=dict(color='gray', dash='dot'),
            mode='lines+markers',
        ),
        secondary_y=True
    )
    
    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=monthly_stats['Month'],
            y=monthly_stats['CI_upper'],
            mode='lines',
            name='Upper CI',
            line=dict(width=1, color='rgba(0,0,255,0.3)'),
            showlegend=False
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_stats['Month'],
            y=monthly_stats['CI_lower'],
            mode='lines',
            name='95% Confidence Interval',
            fill='tonexty',
            fillcolor='rgba(0,0,255,0.1)',
            line=dict(width=1, color='rgba(0,0,255,0.3)')
        ),
        secondary_y=False
    )
    
    # Update layout
    fig.update_layout(
        title='Monthly CSAT Trends with Response Volume',
        hovermode='x unified',
        showlegend=True,
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1
    )
    
    # Update axes titles
    fig.update_yaxes(title_text="CSAT Score", secondary_y=False)
    fig.update_yaxes(title_text="Number of Responses", secondary_y=True)
    fig.update_xaxes(title_text="Month")
    
    # Calculate summary statistics
    stats_dict = {
        'overall_mean': df['CSAT'].mean(),
        'overall_median': df['CSAT'].median(),
        'trend': 'Improving' if monthly_stats['Mean'].iloc[-1] > monthly_stats['Mean'].iloc[0] else 'Declining',
        'response_rate': (df['CSAT'].notna().sum() / len(df)) * 100,
        'monthly_stats': monthly_stats.to_dict('records')
    }
    
    return fig, stats_dict

def create_word_clouds(df: pd.DataFrame) -> Dict[str, plt.Figure]:
    """Create word clouds for Subject and Description fields.
    
    Args:
        df: DataFrame containing ticket data
        
    Returns:
        Dictionary containing word cloud figures
    """
    word_clouds = {}
    
    # Configure word cloud parameters
    wordcloud_params = {
        'width': 800,
        'height': 400,
        'background_color': 'white',
        'max_words': 100,
        'collocations': False
    }
    
    # Create word clouds for Subject and Description
    for field in ['Subject', 'Description']:
        if field in df.columns:
            # Combine all text
            text = ' '.join([str(x) for x in df[field].dropna()])
            
            # Generate word cloud
            wordcloud = WordCloud(**wordcloud_params).generate(text)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'{field} Word Cloud')
            
            word_clouds[field] = fig
    
    return word_clouds

def create_root_cause_analysis(df: pd.DataFrame) -> Tuple[Dict[str, go.Figure], Dict]:
    """Create comprehensive root cause analysis visualizations.
    
    Args:
        df: DataFrame containing ticket data
        
    Returns:
        Tuple of (dictionary of figures, statistics dictionary)
    """
    df = df.copy()
    df['Created Date'] = pd.to_datetime(df['Created Date'])
    df['Month'] = df['Created Date'].dt.strftime('%Y-%m')
    
    figures = {}
    
    # 1. Root cause trends over time
    monthly_rca = df.groupby(['Month', 'Root Cause']).size().reset_index(name='Count')
    fig_trends = px.line(
        monthly_rca,
        x='Month',
        y='Count',
        color='Root Cause',
        title='Root Cause Trends Over Time'
    )
    figures['trends'] = fig_trends
    
    # 2. Average resolution time by root cause
    df['resolution_time_days'] = (pd.to_datetime(df['Closed Date']) - pd.to_datetime(df['Created Date'])).dt.total_seconds() / (24 * 3600)
    
    fig_resolution = go.Figure()
    
    for rca in df['Root Cause'].unique():
        rca_data = df[df['Root Cause'] == rca]['resolution_time_days']
        if len(rca_data) > 0:
            fig_resolution.add_trace(go.Box(
                y=rca_data,
                name=rca,
                boxpoints='outliers'
            ))
    
    fig_resolution.update_layout(
        title='Resolution Time Distribution by Root Cause',
        yaxis_title='Resolution Time (Days)',
        showlegend=True,
        boxmode='group'
    )
    figures['resolution_time'] = fig_resolution
    
    # Calculate statistics
    stats_dict = {
        'rca_distribution': df['Root Cause'].value_counts().to_dict(),
        'avg_resolution_by_rca': df.groupby('Root Cause')['resolution_time_days'].mean().to_dict(),
        'monthly_trends': monthly_rca.to_dict('records')
    }
    
    return figures, stats_dict 
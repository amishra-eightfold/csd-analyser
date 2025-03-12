"""
Advanced Pattern Evolution Visualizations Module

This module provides sophisticated visualization capabilities for analyzing pattern evolution
in support ticket data. It includes trend identification, forecasting, and interactive
visualizations.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List, Any
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, r2_score

# Custom color palettes
VIRIDIS_PALETTE = ["#440154", "#3B528B", "#21918C", "#5EC962", "#FDE725"]
PRIORITY_COLORS = {
    'P1': VIRIDIS_PALETTE[0],
    'P2': VIRIDIS_PALETTE[1],
    'P3': VIRIDIS_PALETTE[2],
    'P4': VIRIDIS_PALETTE[3]
}

def create_trend_forecast(data, column, forecast_periods=3):
    """Create trend forecast for a given column."""
    try:
        # Check if we have enough data for seasonal decomposition (at least 2 full cycles)
        min_periods_required = 24  # 2 years of monthly data
        
        if len(data) < min_periods_required:
            # Fallback to simple exponential smoothing without seasonal component
            model = ExponentialSmoothing(
                data[column],
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            )
        else:
            # Use full model with seasonal component
            model = ExponentialSmoothing(
                data[column],
                trend='add',
                seasonal='add',
                seasonal_periods=12,
                initialization_method='estimated'
            )
        
        # Fit the model
        fit = model.fit()
        
        # Generate forecast
        forecast = fit.forecast(forecast_periods)
        
        # Create confidence intervals
        conf_int = pd.DataFrame(
            index=forecast.index,
            columns=['lower', 'upper'],
            data=np.column_stack([
                forecast - 1.96 * fit.mse ** 0.5,
                forecast + 1.96 * fit.mse ** 0.5
            ])
        )
        
        # Create visualization
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[column],
            name='Historical',
            mode='lines+markers'
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast,
            name='Forecast',
            mode='lines+markers',
            line=dict(dash='dash')
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=conf_int.index,
            y=conf_int['upper'],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=conf_int.index,
            y=conf_int['lower'],
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            name='95% Confidence Interval',
            fillcolor='rgba(68, 68, 68, 0.2)'
        ))
        
        fig.update_layout(
            title=f'{column} Forecast',
            xaxis_title='Time',
            yaxis_title=column,
            hovermode='x unified'
        )
        
        # Prepare statistics
        stats = {
            'model_metrics': {
                'mse': fit.mse,
                'aic': fit.aic
            },
            'forecast_values': forecast.tolist(),
            'confidence_intervals': {
                'lower': conf_int['lower'].tolist(),
                'upper': conf_int['upper'].tolist()
            }
        }
        
        return fig, stats
        
    except Exception as e:
        # Create a simple trend line using linear regression as fallback
        X = np.arange(len(data)).reshape(-1, 1)
        y = data[column].values
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate forecast points
        future_X = np.arange(len(data), len(data) + forecast_periods).reshape(-1, 1)
        forecast = model.predict(future_X)
        
        # Create visualization
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[column],
            name='Historical',
            mode='lines+markers'
        ))
        
        # Add linear trend forecast
        fig.add_trace(go.Scatter(
            x=pd.date_range(start=data.index[-1], periods=forecast_periods + 1, freq='M')[1:],
            y=forecast,
            name='Linear Trend',
            mode='lines+markers',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title=f'{column} Linear Trend (Limited Data)',
            xaxis_title='Time',
            yaxis_title=column,
            hovermode='x unified'
        )
        
        # Prepare basic statistics
        stats = {
            'model_metrics': {
                'mse': mean_squared_error(y, model.predict(X)),
                'r2': r2_score(y, model.predict(X))
            },
            'forecast_values': forecast.tolist(),
            'confidence_intervals': {
                'lower': (forecast - model.predict(X).std()).tolist(),
                'upper': (forecast + model.predict(X).std()).tolist()
            }
        }
        
        return fig, stats

def create_pattern_correlation_matrix(
    df: pd.DataFrame,
    time_column: str,
    pattern_columns: List[str]
) -> Tuple[go.Figure, Dict[str, float]]:
    """
    Create a correlation matrix visualization for pattern relationships.
    
    Args:
        df: DataFrame containing pattern data
        time_column: Name of the column containing timestamps
        pattern_columns: List of columns to analyze for correlations
        
    Returns:
        Tuple containing:
        - Plotly figure with correlation matrix
        - Dictionary with correlation statistics
    """
    # Calculate correlations
    corr_matrix = df[pattern_columns].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=pattern_columns,
        y=pattern_columns,
        colorscale='Viridis',
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    # Update layout
    fig.update_layout(
        title='Pattern Correlation Matrix',
        width=800,
        height=800
    )
    
    # Calculate correlation statistics
    stats = {
        'mean_correlation': np.mean(np.abs(corr_matrix.values)),
        'max_correlation': np.max(np.abs(corr_matrix.values[~np.eye(corr_matrix.shape[0], dtype=bool)])),
        'min_correlation': np.min(np.abs(corr_matrix.values[~np.eye(corr_matrix.shape[0], dtype=bool)]))
    }
    
    return fig, stats

def create_seasonal_decomposition(
    df: pd.DataFrame,
    time_column: str,
    value_column: str
) -> Tuple[go.Figure, Dict[str, Any]]:
    """
    Create a seasonal decomposition visualization.
    
    Args:
        df: DataFrame containing time series data
        time_column: Name of the column containing timestamps
        value_column: Name of the column containing values to analyze
        
    Returns:
        Tuple containing:
        - Plotly figure with seasonal decomposition
        - Dictionary with decomposition statistics
    """
    # Ensure data is sorted by time
    df = df.sort_values(time_column)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
        vertical_spacing=0.1
    )
    
    # Calculate rolling statistics
    window = min(12, len(df) // 2)  # Use 12 months or half the data length
    trend = df[value_column].rolling(window=window, center=True).mean()
    seasonal = df[value_column] - trend
    residual = seasonal - seasonal.rolling(window=window, center=True).mean()
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=df[time_column], y=df[value_column], name='Observed'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df[time_column], y=trend, name='Trend'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df[time_column], y=seasonal, name='Seasonal'),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df[time_column], y=residual, name='Residual'),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=1000,
        title_text='Seasonal Decomposition',
        showlegend=False
    )
    
    # Calculate statistics
    stats = {
        'trend': {
            'mean': trend.mean(),
            'std': trend.std(),
            'slope': np.polyfit(np.arange(len(trend)), trend.dropna(), 1)[0]
        },
        'seasonal': {
            'amplitude': seasonal.std(),
            'max_effect': seasonal.max(),
            'min_effect': seasonal.min()
        },
        'residual': {
            'std': residual.std(),
            'skewness': residual.skew()
        }
    }
    
    return fig, stats

def analyze_pattern_evolution(
    df: pd.DataFrame,
    time_column: str,
    pattern_columns: List[str],
    forecast_periods: int = 3
) -> Tuple[Dict[str, go.Figure], Dict[str, Any]]:
    """
    Comprehensive pattern evolution analysis with visualizations.
    
    Args:
        df: DataFrame containing pattern data
        time_column: Name of the column containing timestamps
        pattern_columns: List of columns to analyze for patterns
        forecast_periods: Number of periods to forecast
        
    Returns:
        Tuple containing:
        - Dictionary of Plotly figures
        - Dictionary with analysis statistics
    """
    figures = {}
    stats = {}
    
    # Create trend forecast for each pattern
    for column in pattern_columns:
        forecast_fig, forecast_stats = create_trend_forecast(
            df, column, forecast_periods
        )
        figures[f'{column}_forecast'] = forecast_fig
        stats[f'{column}_forecast'] = forecast_stats
    
    # Create correlation matrix
    corr_fig, corr_stats = create_pattern_correlation_matrix(
        df, time_column, pattern_columns
    )
    figures['correlation_matrix'] = corr_fig
    stats['correlations'] = corr_stats
    
    # Create seasonal decomposition for each pattern
    for column in pattern_columns:
        decomp_fig, decomp_stats = create_seasonal_decomposition(
            df, time_column, column
        )
        figures[f'{column}_decomposition'] = decomp_fig
        stats[f'{column}_decomposition'] = decomp_stats
    
    return figures, stats 
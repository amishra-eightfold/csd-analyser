"""Base visualization module for common functionality."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

class BaseVisualizer:
    """Base class for visualization functionality."""

    # Common color palettes
    VIRIDIS_PALETTE = ["#440154", "#3B528B", "#21918C", "#5EC962", "#FDE725"]
    AQUA_PALETTE = ["#E0F7FA", "#80DEEA", "#26C6DA", "#00ACC1", "#00838F", "#006064"]
    PRIORITY_COLORS = {
        'P0': VIRIDIS_PALETTE[0],
        'P1': VIRIDIS_PALETTE[1],
        'P2': VIRIDIS_PALETTE[2],
        'P3': VIRIDIS_PALETTE[3]
    }

    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        """Initialize base visualizer with common settings."""
        self.style = style
        self.figure_size = (12, 6)
        self._setup_style()

    def _setup_style(self):
        """Set up common style settings."""
        plt.style.use(self.style)
        plt.rcParams['figure.figsize'] = self.figure_size
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10

    def validate_dataframe(self, df: pd.DataFrame, required_columns: List[str] = None) -> None:
        """Validate DataFrame and its required columns."""
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
        if required_columns and not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")

    def prepare_time_data(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Prepare time-series data with proper date handling."""
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df['Month'] = df[date_column].dt.to_period('M')
        return df

    def calculate_response_time(
        self,
        df: pd.DataFrame,
        response_time_col: str,
        created_date_col: str,
        unit: str = 'hours'
    ) -> pd.Series:
        """Calculate response time in specified unit."""
        multiplier = 3600 if unit == 'hours' else 86400  # seconds per hour/day
        return (
            pd.to_datetime(df[response_time_col]) -
            pd.to_datetime(df[created_date_col])
        ).dt.total_seconds() / multiplier

    def create_monthly_aggregation(
        self,
        df: pd.DataFrame,
        value_column: str,
        agg_func: str = 'mean'
    ) -> pd.DataFrame:
        """Create monthly aggregation of data."""
        monthly_data = df.groupby('Month')[value_column].agg(agg_func).reset_index()
        monthly_data['Month'] = monthly_data['Month'].astype(str)
        return monthly_data

    def setup_plotly_figure(
        self,
        title: str,
        xaxis_title: str,
        yaxis_title: str,
        **kwargs
    ) -> go.Figure:
        """Set up a plotly figure with common settings."""
        fig = go.Figure()
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            **kwargs
        )
        return fig

    def setup_matplotlib_figure(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
        figsize: Tuple[int, int] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Set up a matplotlib figure with common settings."""
        fig, ax = plt.subplots(figsize=figsize or self.figure_size)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig, ax 
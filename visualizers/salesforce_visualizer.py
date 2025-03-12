"""Salesforce data visualizer for CSD Analyzer."""
from typing import List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps

def handle_plot_errors(func):
    """Decorator for handling plot errors."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            plt.close()
            raise ValueError(f"Error creating plot: {str(e)}")
    return wrapper

class SalesforceVisualizer:
    """Create visualizations for Salesforce data."""

    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        """Initialize the visualizer with style settings.
        
        Args:
            style: The matplotlib style to use
        """
        self.style = style
        plt.style.use(self.style)
        self.colors = sns.color_palette('husl', n_colors=15)
        self.fig_size = (12, 6)

    @handle_plot_errors
    def plot_case_volume(self, df: pd.DataFrame, title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot case volume over time."""
        if df is None or df.empty:
            raise ValueError("Invalid or empty DataFrame")

        fig, ax = plt.subplots(figsize=self.fig_size)
        
        daily_created = df.groupby(df['CreatedDate'].dt.date).size()
        daily_closed = df[df['ClosedDate'].notna()].groupby(df['ClosedDate'].dt.date).size()

        daily_created.plot(label='Created', ax=ax, color=self.colors[0])
        daily_closed.plot(label='Closed', ax=ax, color=self.colors[1])

        ax.set_title(title or 'Case Volume Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Cases')
        ax.legend()
        plt.tight_layout()

        return fig, ax

    @handle_plot_errors
    def plot_priority_distribution(self, df: pd.DataFrame, title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot case priority distribution."""
        if df is None or df.empty:
            raise ValueError("Invalid or empty DataFrame")

        fig, ax = plt.subplots(figsize=self.fig_size)
        
        priority_counts = df['Internal_Priority__c'].value_counts()
        sns.barplot(x=priority_counts.index, 
                   y=priority_counts.values,
                   palette=self.colors[:len(priority_counts)],
                   ax=ax)

        ax.set_title(title or 'Case Priority Distribution')
        ax.set_xlabel('Priority')
        ax.set_ylabel('Number of Cases')
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig, ax

    @handle_plot_errors
    def plot_csat_distribution(self, df: pd.DataFrame, title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot CSAT score distribution."""
        if df is None or df.empty:
            raise ValueError("Invalid or empty DataFrame")

        fig, ax = plt.subplots(figsize=self.fig_size)
        
        csat_scores = df['CSAT__c'].dropna()
        sns.histplot(data=csat_scores, bins=5, color=self.colors[2], ax=ax)

        ax.set_title(title or 'CSAT Score Distribution')
        ax.set_xlabel('CSAT Score')
        ax.set_ylabel('Number of Cases')
        plt.tight_layout()

        return fig, ax

    @handle_plot_errors
    def plot_response_times(self, df: pd.DataFrame, title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot first response time distribution."""
        if df is None or df.empty:
            raise ValueError("Invalid or empty DataFrame")

        fig, ax = plt.subplots(figsize=self.fig_size)
        
        response_times = df['First_Response_Time__c'].dropna()
        sns.boxplot(y=response_times, color=self.colors[3], ax=ax)

        ax.set_title(title or 'First Response Time Distribution')
        ax.set_ylabel('Hours')
        plt.tight_layout()

        return fig, ax

    @handle_plot_errors
    def plot_product_areas(self, df: pd.DataFrame, title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot cases by product area."""
        if df is None or df.empty:
            raise ValueError("Invalid or empty DataFrame")

        fig, ax = plt.subplots(figsize=self.fig_size)
        
        area_counts = df['Product_Area__c'].value_counts()
        sns.barplot(x=area_counts.values,
                   y=area_counts.index,
                   palette=self.colors[:len(area_counts)],
                   ax=ax)

        ax.set_title(title or 'Cases by Product Area')
        ax.set_xlabel('Number of Cases')
        ax.set_ylabel('Product Area')
        plt.tight_layout()

        return fig, ax

    @handle_plot_errors
    def plot_correlation_matrix(self, df: pd.DataFrame, 
                              numeric_columns: Optional[List[str]] = None,
                              title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot correlation matrix for numeric columns."""
        if df is None or df.empty:
            raise ValueError("Invalid or empty DataFrame")

        if numeric_columns:
            df_numeric = df[numeric_columns]
        else:
            df_numeric = df.select_dtypes(include=['float64', 'int64'])

        if df_numeric.empty:
            raise ValueError("No numeric columns available for correlation analysis")

        corr_matrix = df_numeric.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix,
                   annot=True,
                   cmap='coolwarm',
                   center=0,
                   ax=ax)

        ax.set_title(title or 'Correlation Matrix')
        plt.tight_layout()

        return fig, ax 
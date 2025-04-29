"""Salesforce data visualizer for CSD Analyzer."""
from typing import List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps
import numpy as np

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
    def plot_case_volume(self, df: pd.DataFrame, title: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot case volume over time."""
        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create daily case volume
        daily_created = df.groupby(df['CreatedDate'].dt.date).size()
        daily_closed = df[df['ClosedDate'].notna()].groupby(df['ClosedDate'].dt.date).size()

        # Plot the data
        daily_created.plot(label='Created', ax=ax)
        daily_closed.plot(label='Closed', ax=ax)

        # Customize the plot
        ax.set_title(title or 'Case Volume Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Cases')
        ax.legend()
        ax.grid(True)

        return fig, ax

    @handle_plot_errors
    def plot_priority_distribution(self, df: pd.DataFrame, title: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot case priority distribution."""
        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create priority distribution
        priority_counts = df['Internal_Priority__c'].value_counts()

        # Plot the data
        sns.barplot(x=priority_counts.index, y=priority_counts.values, palette='viridis', ax=ax)

        # Customize the plot
        ax.set_title(title or 'Case Priority Distribution')
        ax.set_xlabel('Priority')
        ax.set_ylabel('Number of Cases')
        ax.grid(True)

        return fig, ax

    @handle_plot_errors
    def plot_csat_distribution(self, df: pd.DataFrame, title: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot CSAT score distribution."""
        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Filter out null CSAT scores
        csat_data = df[df['CSAT__c'].notna()]

        # Plot the data
        sns.histplot(data=csat_data, x='CSAT__c', bins=5, ax=ax)

        # Customize the plot
        ax.set_title(title or 'CSAT Score Distribution')
        ax.set_xlabel('CSAT Score')
        ax.set_ylabel('Number of Cases')
        ax.grid(True)

        return fig, ax

    @handle_plot_errors
    def plot_response_times(self, df: pd.DataFrame, title: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot first response time distribution."""
        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Filter out null response times
        response_data = df[df['First_Response_Time__c'].notna()]

        # Plot the data
        sns.boxplot(y=response_data['First_Response_Time__c'], ax=ax)

        # Customize the plot
        ax.set_title(title or 'First Response Time Distribution')
        ax.set_ylabel('Hours')
        ax.grid(True)

        return fig, ax

    @handle_plot_errors
    def plot_product_areas(self, df: pd.DataFrame, title: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot case distribution by product area."""
        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create product area distribution
        area_counts = df['Product_Area__c'].value_counts()

        # Plot the data
        sns.barplot(y=area_counts.index, x=area_counts.values, palette='viridis', ax=ax)

        # Customize the plot
        ax.set_title(title or 'Cases by Product Area')
        ax.set_xlabel('Number of Cases')
        ax.set_ylabel('Product Area')
        ax.grid(True)

        return fig, ax

    @handle_plot_errors
    def plot_correlation_matrix(self, df: pd.DataFrame, numeric_columns: List[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot correlation matrix for numeric columns."""
        if df is None or df.empty:
            raise ValueError("Invalid or empty DataFrame")

        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns

        if not numeric_columns:
            raise ValueError("No numeric columns available for correlation analysis")

        df_numeric = df[numeric_columns]
        corr_matrix = df_numeric.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix,
                   annot=True,
                   cmap='coolwarm',
                   center=0,
                   ax=ax)

        ax.set_title("Correlation Matrix")
        plt.tight_layout()

        return fig, ax 
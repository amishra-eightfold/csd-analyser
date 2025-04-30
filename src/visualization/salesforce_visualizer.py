"""
Salesforce data visualization module.

This module provides visualization capabilities for Salesforce data using Plotly.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Callable
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .base_visualizer import BaseVisualizer

def handle_plot_errors(func: Callable) -> Callable:
    """
    Decorator for handling visualization errors gracefully.
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    def wrapper(self, *args, **kwargs) -> Any:
        """
        Wrapper function that catches and handles exceptions.
        
        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Any: The result of the function or an empty figure in case of error
        """
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
            # Return empty figure to prevent app crash
            return plt.figure() if 'plt' in func.__name__ else go.Figure()
    return wrapper

class SalesforceVisualizer(BaseVisualizer):
    """
    Creates visualizations for Salesforce case data analysis.
    Handles various chart types including ticket volume, response times,
    CSAT scores, and product area distributions.
    """

    def __init__(self, style: str = "seaborn-v0_8-whitegrid") -> None:
        """
        Initialize the visualizer with style settings.
        
        Args:
            style: The style to use for plots
        """
        super().__init__(style)
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        
        # Define color palettes
        self.VIRIDIS_PALETTE = ["#440154", "#3B528B", "#21918C", "#5EC962", "#FDE725"]
        self.AQUA_PALETTE = ["#E0F7FA", "#80DEEA", "#26C6DA", "#00ACC1", "#00838F", "#006064"]
        self.PRIORITY_COLORS = {
            'P0': self.VIRIDIS_PALETTE[0],
            'P1': self.VIRIDIS_PALETTE[1],
            'P2': self.VIRIDIS_PALETTE[2],
            'P3': self.VIRIDIS_PALETTE[3]
        }

    @handle_plot_errors
    def plot_case_volume(self, df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a case volume over time plot.
        
        Args:
            df: DataFrame containing case data with CreatedDate column
        
        Returns:
            Tuple[plt.Figure, plt.Axes]: Matplotlib figure and axes objects
            
        Raises:
            ValueError: If DataFrame is None or empty
        """
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
            
        fig, ax = plt.subplots(figsize=(12, 6))
        df['CreatedDate'] = pd.to_datetime(df['CreatedDate'])
        daily_cases = df.groupby(df['CreatedDate'].dt.date).size()
        
        ax.plot(daily_cases.index, daily_cases.values, marker='o')
        ax.set_title('Case Volume Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Cases')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig, ax
        
    @handle_plot_errors
    def plot_priority_distribution(self, df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a priority distribution plot.
        
        Args:
            df: DataFrame containing case data with Priority column
        
        Returns:
            Tuple[plt.Figure, plt.Axes]: Matplotlib figure and axes objects
            
        Raises:
            ValueError: If DataFrame is None or empty
        """
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
            
        fig, ax = plt.subplots(figsize=(10, 6))
        priority_counts = df['Priority'].value_counts()
        
        sns.barplot(x=priority_counts.index, y=priority_counts.values, ax=ax)
        ax.set_title('Case Priority Distribution')
        ax.set_xlabel('Priority')
        ax.set_ylabel('Number of Cases')
        plt.tight_layout()
        
        return fig, ax
        
    @handle_plot_errors
    def plot_csat_distribution(self, df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a CSAT score distribution plot.
        
        Args:
            df: DataFrame containing case data with CSAT__c column
        
        Returns:
            Tuple[plt.Figure, plt.Axes]: Matplotlib figure and axes objects
            
        Raises:
            ValueError: If DataFrame is None or empty
        """
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
            
        fig, ax = plt.subplots(figsize=(10, 6))
        csat_data = df['CSAT__c'].dropna()
        
        sns.histplot(data=csat_data, bins=5, ax=ax)
        ax.set_title('CSAT Score Distribution')
        ax.set_xlabel('CSAT Score')
        ax.set_ylabel('Number of Cases')
        plt.tight_layout()
        
        return fig, ax
        
    @handle_plot_errors
    def plot_response_times(self, df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a response time distribution plot.
        
        Args:
            df: DataFrame containing case data with First_Response_Time__c column
        
        Returns:
            Tuple[plt.Figure, plt.Axes]: Matplotlib figure and axes objects
            
        Raises:
            ValueError: If DataFrame is None or empty
        """
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
            
        fig, ax = plt.subplots(figsize=(12, 6))
        response_times = df['First_Response_Time__c'].dropna()
        
        sns.boxplot(y=response_times, ax=ax)
        ax.set_title('First Response Time Distribution')
        ax.set_ylabel('Hours')
        plt.tight_layout()
        
        return fig, ax
        
    @handle_plot_errors
    def plot_product_areas(self, df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a product area distribution plot.
        
        Args:
            df: DataFrame containing case data with Product_Area__c column
        
        Returns:
            Tuple[plt.Figure, plt.Axes]: Matplotlib figure and axes objects
            
        Raises:
            ValueError: If DataFrame is None or empty
        """
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
            
        fig, ax = plt.subplots(figsize=(12, 6))
        product_counts = df['Product_Area__c'].value_counts()
        
        sns.barplot(x=product_counts.index, y=product_counts.values, ax=ax)
        ax.set_title('Cases by Product Area')
        ax.set_xlabel('Product Area')
        ax.set_ylabel('Number of Cases')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig, ax
        
    @handle_plot_errors
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a correlation matrix plot for numeric columns.
        
        Args:
            df: DataFrame containing case data
            numeric_columns: List of numeric column names to include in correlation analysis
        
        Returns:
            Tuple[plt.Figure, plt.Axes]: Matplotlib figure and axes objects
            
        Raises:
            ValueError: If DataFrame is None or empty
        """
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
            
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
        corr_matrix = df[numeric_columns].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')
        plt.tight_layout()
        
        return fig, ax
        
    @handle_plot_errors
    def create_ticket_volume_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a ticket volume chart showing created vs closed tickets.
        
        Args:
            df: DataFrame containing ticket data with 'Created Date' and 'Closed Date' columns
            
        Returns:
            go.Figure: Plotly figure object with volume chart
        """
        self.validate_dataframe(df, ['Created Date', 'Closed Date'])
        
        # Prepare data
        df = self.prepare_time_data(df, 'Created Date')
        df = self.prepare_time_data(df, 'Closed Date')
        
        # Calculate monthly counts
        monthly_created = self.create_monthly_aggregation(df, 'Created Date', 'count')
        monthly_closed = self.create_monthly_aggregation(df, 'Closed Date', 'count')
        
        # Create figure
        fig = self.setup_plotly_figure(
            title='Monthly Ticket Volume',
            xaxis_title='Month',
            yaxis_title='Number of Tickets',
            barmode='group'
        )
        
        # Add traces
        fig.add_trace(go.Bar(
            name='Created',
            x=monthly_created['Month'],
            y=monthly_created[monthly_created.columns[-1]],
            marker_color=self.PRIORITY_COLORS.get('P2', '#33BB33')
        ))
        fig.add_trace(go.Bar(
            name='Closed',
            x=monthly_closed['Month'],
            y=monthly_closed[monthly_closed.columns[-1]],
            marker_color=self.PRIORITY_COLORS.get('P3', '#3388FF')
        ))
        
        return fig

    @handle_plot_errors
    def create_response_time_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a response time analysis chart.
        
        Args:
            df: DataFrame containing ticket data with response time information
            
        Returns:
            go.Figure: Plotly figure object with response time chart
        """
        self.validate_dataframe(df, ['First Response Time', 'Created Date', 'Priority'])
        
        # Calculate response time
        df = df.copy()
        df['Response Time (Hours)'] = self.calculate_response_time(
            df, 'First Response Time', 'Created Date', 'hours'
        )
        
        # Create figure
        fig = self.setup_plotly_figure(
            title='Response Time by Priority',
            xaxis_title='Priority',
            yaxis_title='Response Time (Hours)',
            showlegend=True
        )
        
        # Create boxplot for each priority level
        priorities = sorted(df['Priority'].unique())
        
        for priority in priorities:
            priority_data = df[df['Priority'] == priority]['Response Time (Hours)'].dropna()
            
            if len(priority_data) > 0:
                fig.add_trace(go.Box(
                    y=priority_data,
                    name=priority,
                    marker_color=self.PRIORITY_COLORS.get(priority, self.VIRIDIS_PALETTE[0])
                ))
        
        return fig
    
    @handle_plot_errors
    def create_csat_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a CSAT analysis chart.
        
        Args:
            df: DataFrame containing ticket data with CSAT scores
            
        Returns:
            go.Figure: Plotly figure object with CSAT chart
        """
        self.validate_dataframe(df, ['CSAT'])
        
        # Create figure
        fig = self.setup_plotly_figure(
            title='CSAT Distribution',
            xaxis_title='CSAT Score',
            yaxis_title='Number of Cases',
            showlegend=False
        )
        
        # Get valid CSAT data
        csat_data = df['CSAT'].dropna()
        
        if len(csat_data) > 0:
            # Create histogram
            fig.add_trace(go.Histogram(
                x=csat_data,
                marker_color=self.VIRIDIS_PALETTE[2],
                nbinsx=5
            ))
            
            # Add average line
            avg_csat = csat_data.mean()
            fig.add_vline(x=avg_csat, line_dash="dash", line_color="red",
                         annotation_text=f"Average: {avg_csat:.2f}", 
                         annotation_position="top right")
        
        return fig
        
    @handle_plot_errors
    def create_product_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a product area analysis heatmap.
        
        Args:
            df: DataFrame containing ticket data with product area and priority information
            
        Returns:
            go.Figure: Plotly figure object with product heatmap
        """
        self.validate_dataframe(df, ['Product_Area__c', 'Priority'])
        
        # Create pivot table
        pivot_data = pd.crosstab(df['Product_Area__c'], df['Priority'])
        
        # Create figure
        fig = self.setup_plotly_figure(
            title='Cases by Product Area and Priority',
            showlegend=False
        )
        
        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale=px.colors.sequential.Viridis,
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            xaxis_title='Priority',
            yaxis_title='Product Area'
        )
        
        return fig
        
    @handle_plot_errors
    def create_correlation_matrix(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create a correlation matrix visualization.
        
        Args:
            df: DataFrame containing ticket data
            numeric_columns: List of numeric columns to include in correlation analysis
            
        Returns:
            go.Figure: Plotly figure object with correlation matrix
        """
        # Identify numeric columns if not provided
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
        # Calculate correlation matrix
        corr_matrix = df[numeric_columns].corr()
        
        # Create figure
        fig = self.setup_plotly_figure(
            title='Correlation Matrix',
            showlegend=False
        )
        
        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate='%{text}',
            hoverongaps=False
        ))
        
        return fig 
"""
Salesforce data visualization module.

This module provides visualization capabilities for Salesforce data using Plotly.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .base_visualizer import BaseVisualizer

class SalesforceVisualizer(BaseVisualizer):
    """
    Creates visualizations for Salesforce case data analysis.
    Handles various chart types including ticket volume, response times,
    CSAT scores, and product area distributions.
    """

    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
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

    def plot_case_volume(self, df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
        """Create a case volume over time plot."""
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
        
    def plot_priority_distribution(self, df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
        """Create a priority distribution plot."""
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
        
    def plot_csat_distribution(self, df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
        """Create a CSAT score distribution plot."""
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
        
    def plot_response_times(self, df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
        """Create a response time distribution plot."""
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
            
        fig, ax = plt.subplots(figsize=(12, 6))
        response_times = df['First_Response_Time__c'].dropna()
        
        sns.boxplot(y=response_times, ax=ax)
        ax.set_title('First Response Time Distribution')
        ax.set_ylabel('Hours')
        plt.tight_layout()
        
        return fig, ax
        
    def plot_product_areas(self, df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
        """Create a product area distribution plot."""
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
        
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create a correlation matrix plot for numeric columns."""
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
        
    def create_ticket_volume_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create a ticket volume chart showing created vs closed tickets."""
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

    def create_response_time_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create a response time analysis chart."""
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
        
        # Add box plots for each priority
        for priority in sorted(df['Priority'].unique()):
            priority_data = df[df['Priority'] == priority]
            fig.add_trace(go.Box(
                y=priority_data['Response Time (Hours)'],
                name=f'Priority {priority}',
                marker_color=self.PRIORITY_COLORS.get(priority, '#FF4444')
            ))
        
        return fig

    def create_csat_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create a CSAT trend chart."""
        self.validate_dataframe(df, ['Created Date', 'CSAT'])
        
        # Prepare data
        df = self.prepare_time_data(df, 'Created Date')
        monthly_csat = self.create_monthly_aggregation(df, 'CSAT')
        
        # Create figure
        fig = self.setup_plotly_figure(
            title='CSAT Trend',
            xaxis_title='Month',
            yaxis_title='Average CSAT Score',
            showlegend=False
        )
        
        # Add trace
        fig.add_trace(go.Scatter(
            x=monthly_csat['Month'],
            y=monthly_csat['CSAT'],
            mode='lines+markers',
            line=dict(color=self.PRIORITY_COLORS.get('P2', '#33BB33'), width=2),
            marker=dict(size=8)
        ))
        
        return fig

    def create_product_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create a product area/feature heatmap."""
        self.validate_dataframe(df, ['Product Area', 'Product Feature'])
        
        # Create pivot table
        pivot = pd.crosstab(df['Product Area'], df['Product Feature'])
        
        # Create figure
        fig = self.setup_plotly_figure(
            title='Ticket Distribution by Product',
            xaxis_title='Product Feature',
            yaxis_title='Product Area'
        )
        
        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Viridis',
            colorbar=dict(title='Count')
        ))
        
        return fig

    def create_correlation_matrix(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None
    ) -> go.Figure:
        """Create a correlation matrix visualization."""
        # Select numeric columns if not specified
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.validate_dataframe(df, numeric_columns)
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_columns].corr()
        
        # Create figure
        fig = self.setup_plotly_figure(
            title='Correlation Matrix',
            xaxis_title='',
            yaxis_title=''
        )
        
        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title='Correlation')
        ))
        
        return fig 
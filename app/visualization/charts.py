"""Charts module for CSD Analyzer."""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from ..core.config import config

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generates interactive visualizations for support data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize chart generator.
        
        Args:
            df (pd.DataFrame): DataFrame containing support ticket data
        """
        self.df = df
        colors = config.get_visualization_colors()
        self.color_palette = colors['viridis_palette']
        self.priority_colors = colors['priority_colors']
        
    def create_ticket_volume_chart(self, 
                                 grouping: str = 'ME',
                                 title: Optional[str] = None) -> go.Figure:
        """
        Create ticket volume trend chart.
        
        Args:
            grouping (str): Time grouping ('D' for daily, 'W' for weekly, 'ME' for monthly)
            title (Optional[str]): Chart title
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            # Group data by time period
            ticket_counts = self.df.resample(grouping, on='Created Date').size()
            
            # Create figure
            fig = go.Figure()
            
            # Add volume line
            fig.add_trace(
                go.Scatter(
                    x=ticket_counts.index,
                    y=ticket_counts.values,
                    mode='lines+markers',
                    name='Ticket Volume',
                    line=dict(color=self.color_palette[0])
                )
            )
            
            # Update layout
            fig.update_layout(
                title=title or 'Ticket Volume Over Time',
                xaxis_title='Date',
                yaxis_title='Number of Tickets',
                showlegend=True,
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create volume chart: {str(e)}")
            return go.Figure()
            
    def create_resolution_time_chart(self,
                                   by_priority: bool = True,
                                   title: Optional[str] = None) -> go.Figure:
        """
        Create resolution time analysis chart.
        
        Args:
            by_priority (bool): Whether to break down by priority
            title (Optional[str]): Chart title
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            if by_priority:
                # Calculate average resolution time by priority
                resolution_by_priority = self.df.groupby('Priority')['Resolution Time (Days)'].mean()
                
                # Create figure
                fig = go.Figure()
                
                # Add bar chart
                fig.add_trace(
                    go.Bar(
                        x=resolution_by_priority.index,
                        y=resolution_by_priority.values,
                        marker_color=[self.priority_colors.get(p, self.color_palette[0]) for p in resolution_by_priority.index],
                        name='Resolution Time by Priority'
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=title or 'Average Resolution Time by Priority',
                    xaxis_title='Priority',
                    yaxis_title='Average Resolution Time (Days)',
                    showlegend=False,
                    template='plotly_white'
                )
                
            else:
                # Create box plot of resolution times
                fig = go.Figure()
                
                fig.add_trace(
                    go.Box(
                        y=self.df['Resolution Time (Days)'],
                        name='Resolution Time Distribution',
                        marker_color=self.color_palette[0]
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=title or 'Resolution Time Distribution',
                    yaxis_title='Resolution Time (Days)',
                    showlegend=False,
                    template='plotly_white'
                )
                
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create resolution time chart: {str(e)}")
            return go.Figure()
            
    def create_priority_distribution_chart(self,
                                         title: Optional[str] = None) -> go.Figure:
        """
        Create priority distribution chart.
        
        Args:
            title (Optional[str]): Chart title
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            # Get priority distribution
            priority_counts = self.df['Priority'].value_counts()
            
            # Create figure
            fig = go.Figure()
            
            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=priority_counts.index,
                    y=priority_counts.values,
                    marker_color=[self.priority_colors.get(p, self.color_palette[0]) for p in priority_counts.index],
                    name='Priority Distribution'
                )
            )
            
            # Update layout
            fig.update_layout(
                title=title or 'Ticket Priority Distribution',
                xaxis_title='Priority',
                yaxis_title='Number of Tickets',
                showlegend=False,
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create priority chart: {str(e)}")
            return go.Figure()
            
    def create_csat_trend_chart(self,
                               grouping: str = 'ME',
                               title: Optional[str] = None) -> go.Figure:
        """
        Create CSAT trend chart.
        
        Args:
            grouping (str): Time grouping ('D' for daily, 'W' for weekly, 'ME' for monthly)
            title (Optional[str]): Chart title
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            # Calculate CSAT trends
            csat_trend = self.df.resample(grouping, on='Created Date')['CSAT'].mean()
            ticket_counts = self.df.resample(grouping, on='Created Date').size()
            
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add CSAT line
            fig.add_trace(
                go.Scatter(
                    x=csat_trend.index,
                    y=csat_trend.values,
                    mode='lines+markers',
                    name='CSAT Score',
                    line=dict(color=self.color_palette[0]),
                    hovertemplate='%{x}<br>CSAT: %{y:.2f}<extra></extra>'
                ),
                secondary_y=False
            )
            
            # Add ticket volume bars
            fig.add_trace(
                go.Bar(
                    x=ticket_counts.index,
                    y=ticket_counts.values,
                    name='Ticket Volume',
                    marker_color=self.color_palette[1],
                    opacity=0.3,
                    hovertemplate='%{x}<br>Tickets: %{y}<extra></extra>'
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title=title or 'CSAT Score Trend',
                showlegend=True,
                hovermode='x unified',
                template='plotly_white'
            )
            
            # Update axes
            fig.update_yaxes(title_text="CSAT Score", secondary_y=False)
            fig.update_yaxes(title_text="Number of Tickets", secondary_y=True)
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create CSAT trend chart: {str(e)}")
            return go.Figure()
            
    def create_product_area_chart(self,
                                metric: str = 'count',
                                title: Optional[str] = None) -> go.Figure:
        """
        Create product area analysis chart.
        
        Args:
            metric (str): Metric to analyze ('count', 'resolution_time', 'csat')
            title (Optional[str]): Chart title
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            if metric == 'count':
                # Calculate ticket counts by product area
                data = self.df['Product Area'].value_counts()
                y_title = 'Number of Tickets'
                hover_template = '%{x}<br>Tickets: %{y}<extra></extra>'
                
            elif metric == 'resolution_time':
                # Calculate average resolution time by product area
                data = self.df.groupby('Product Area')['Resolution Time (Days)'].mean()
                y_title = 'Average Resolution Time (Days)'
                hover_template = '%{x}<br>Avg Resolution: %{y:.1f} days<extra></extra>'
                
            else:  # csat
                # Calculate average CSAT by product area
                data = self.df.groupby('Product Area')['CSAT'].mean()
                y_title = 'Average CSAT Score'
                hover_template = '%{x}<br>Avg CSAT: %{y:.2f}<extra></extra>'
            
            # Sort data
            data = data.sort_values(ascending=True)
            
            # Create figure
            fig = go.Figure()
            
            # Add horizontal bar trace
            fig.add_trace(
                go.Bar(
                    x=data.values,
                    y=data.index,
                    orientation='h',
                    marker_color=self.color_palette[0],
                    hovertemplate=hover_template
                )
            )
            
            # Update layout
            fig.update_layout(
                title=title or f'Product Area Analysis - {metric.title()}',
                xaxis_title=y_title,
                yaxis_title='Product Area',
                showlegend=False,
                template='plotly_white',
                height=max(400, len(data) * 30)  # Dynamic height based on number of areas
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create product area chart: {str(e)}")
            return go.Figure()
            
    def create_correlation_matrix(self,
                                title: Optional[str] = None) -> go.Figure:
        """
        Create correlation matrix heatmap.
        
        Args:
            title (Optional[str]): Chart title
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            # Select numeric columns
            numeric_cols = [
                'Resolution Time (Days)',
                'CSAT',
                'First Response Time'
            ]
            
            # Calculate correlation matrix
            corr_matrix = self.df[numeric_cols].corr()
            
            # Create figure
            fig = go.Figure()
            
            # Add heatmap trace
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values.round(2),
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10},
                    hoverongaps=False,
                    hovertemplate='%{x}<br>%{y}<br>Correlation: %{z:.2f}<extra></extra>'
                )
            )
            
            # Update layout
            fig.update_layout(
                title=title or 'Metric Correlation Matrix',
                template='plotly_white',
                width=600,
                height=600
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create correlation matrix: {str(e)}")
            return go.Figure()

"""
Advanced visualization module for complex data analysis.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List
from .base_visualizer import BaseVisualizer

class AdvancedVisualizer(BaseVisualizer):
    """Advanced visualization class for complex analysis."""

    def create_csat_analysis(self, df: pd.DataFrame) -> Tuple[go.Figure, Dict[str, Any]]:
        """
        Create CSAT analysis visualization and statistics.
        
        Args:
            df: DataFrame containing CSAT data
            
        Returns:
            Tuple containing the figure and statistics dictionary
        """
        self.validate_dataframe(df, ['Created Date', 'CSAT'])
        
        # Calculate CSAT statistics
        csat_stats = {
            'overall_mean': df['CSAT'].mean(),
            'response_rate': (df['CSAT'].notna().sum() / len(df)) * 100
        }
        
        # Prepare data
        df = self.prepare_time_data(df, 'Created Date')
        monthly_csat = df.groupby('Month')['CSAT'].agg(['mean', 'count']).reset_index()
        
        # Calculate trend
        if len(monthly_csat) >= 2:
            first_csat = monthly_csat['mean'].iloc[0]
            last_csat = monthly_csat['mean'].iloc[-1]
            csat_stats['trend'] = 'Improving' if last_csat > first_csat else 'Declining'
        else:
            csat_stats['trend'] = 'Insufficient data'
        
        # Create visualization
        fig = self.setup_plotly_figure(
            title='CSAT Trend Analysis',
            xaxis_title='Month',
            yaxis_title='CSAT Score',
            showlegend=True
        )
        
        fig.add_trace(go.Scatter(
            x=monthly_csat['Month'].astype(str),
            y=monthly_csat['mean'],
            mode='lines+markers',
            name='Average CSAT',
            line=dict(color=self.AQUA_PALETTE[2], width=2),
            marker=dict(size=8)
        ))
        
        return fig, csat_stats

    def create_word_clouds(self, df: pd.DataFrame) -> Dict[str, plt.Figure]:
        """
        Create word clouds from text fields.
        
        Args:
            df: DataFrame containing text data
            
        Returns:
            Dictionary of matplotlib figures for each text field
        """
        self.validate_dataframe(df)
        word_clouds = {}
        text_fields = ['Subject', 'Description']
        
        for field in text_fields:
            if field in df.columns:
                # Combine all text
                text = ' '.join(df[field].dropna().astype(str))
                
                # Create word cloud
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    max_words=100
                ).generate(text)
                
                # Create matplotlib figure
                fig, ax = self.setup_matplotlib_figure(
                    title=f'{field} Word Cloud',
                    xlabel='',
                    ylabel='',
                    figsize=(10, 5)
                )
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                
                word_clouds[field] = fig
                plt.close(fig)
        
        return word_clouds

    def create_root_cause_analysis(self, df: pd.DataFrame) -> Tuple[go.Figure, Dict[str, Any]]:
        """
        Create root cause analysis visualization and statistics.
        
        Args:
            df: DataFrame containing root cause data
            
        Returns:
            Tuple containing the figure and statistics dictionary
        """
        self.validate_dataframe(df, ['Root Cause'])
        
        # Calculate root cause statistics
        root_cause_counts = df['Root Cause'].value_counts()
        root_cause_stats = {
            'total_cases': len(df),
            'unique_causes': len(root_cause_counts),
            'top_causes': root_cause_counts.head(5).to_dict()
        }
        
        # Create visualization
        fig = self.setup_plotly_figure(
            title='Root Cause Distribution',
            xaxis_title='Root Cause',
            yaxis_title='Number of Cases',
            showlegend=False
        )
        
        fig.add_trace(go.Bar(
            x=root_cause_counts.index,
            y=root_cause_counts.values,
            marker_color=self.AQUA_PALETTE[2]
        ))
        
        return fig, root_cause_stats

    def create_first_response_analysis(self, df: pd.DataFrame) -> Tuple[go.Figure, Dict[str, Any]]:
        """
        Create first response time analysis visualization and statistics.
        
        Args:
            df: DataFrame containing response time data
            
        Returns:
            Tuple containing the figure and statistics dictionary
        """
        self.validate_dataframe(df, ['First Response Time', 'Created Date', 'Priority'])
        
        # Calculate response time
        df = df.copy()
        df['Response Hours'] = self.calculate_response_time(
            df, 'First Response Time', 'Created Date', 'hours'
        )
        
        # Calculate statistics
        response_stats = {
            'mean_response_time': df['Response Hours'].mean(),
            'median_response_time': df['Response Hours'].median(),
            'within_sla': (df['Response Hours'] <= 24).mean() * 100  # Assuming 24-hour SLA
        }
        
        # Create visualization
        fig = self.setup_plotly_figure(
            title='First Response Time Distribution by Priority',
            xaxis_title='Priority',
            yaxis_title='Response Time (Hours)',
            showlegend=True
        )
        
        for priority in sorted(df['Priority'].unique()):
            priority_data = df[df['Priority'] == priority]['Response Hours']
            
            fig.add_trace(go.Box(
                y=priority_data,
                name=f'Priority {priority}',
                boxpoints='outliers',
                marker_color=self.PRIORITY_COLORS.get(priority, self.VIRIDIS_PALETTE[0])
            ))
        
        return fig, response_stats 
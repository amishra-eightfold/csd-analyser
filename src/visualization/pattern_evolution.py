"""
Pattern evolution analysis module.

This module provides functionality for analyzing patterns and trends in support ticket data.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from .base_visualizer import BaseVisualizer

class PatternEvolutionVisualizer(BaseVisualizer):
    """Visualizer for pattern evolution analysis."""

    def analyze_pattern_evolution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the evolution of patterns in support ticket data.
        
        Args:
            df: DataFrame containing support ticket data
            
        Returns:
            Dictionary containing analysis results and visualizations
        """
        try:
            # Validate input data
            if df.empty:
                return {}

            # Validate required columns
            required_columns = ['Created Date', 'Closed Date', 'Root Cause',
                              'Priority', 'CSAT', 'Resolution Time (Days)']
            if not all(col in df.columns for col in required_columns):
                return {}

            results = {}
            
            # Prepare data
            df = self.prepare_time_data(df.copy(), 'Created Date')
            
            # 1. Root Cause Evolution
            try:
                root_cause_monthly = pd.crosstab(
                    df['Month'],
                    df['Root Cause'],
                    normalize='index'
                ) * 100
                
                fig_root_cause = self.setup_plotly_figure(
                    title='Root Cause Distribution Trend',
                    xaxis_title='Month',
                    yaxis_title='Percentage of Tickets',
                    hovermode='x unified'
                )
                
                for column in root_cause_monthly.columns:
                    fig_root_cause.add_trace(go.Scatter(
                        name=column,
                        x=root_cause_monthly.index.astype(str),
                        y=root_cause_monthly[column],
                        mode='lines+markers',
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))
                
                fig_root_cause.update_layout(yaxis=dict(ticksuffix='%'))
                results['root_cause_evolution'] = fig_root_cause
            except Exception as e:
                self.logger.error(f"Error in root cause evolution analysis: {str(e)}")
            
            # 2. Resolution Time Patterns
            try:
                resolution_monthly = df.groupby(['Month', 'Priority'])['Resolution Time (Days)'].mean().unstack()
                
                if not resolution_monthly.empty:
                    fig_resolution = self.setup_plotly_figure(
                        title='Resolution Time Trend by Priority',
                        xaxis_title='Month',
                        yaxis_title='Average Resolution Time (Days)',
                        hovermode='x unified'
                    )
                    
                    for priority in resolution_monthly.columns:
                        fig_resolution.add_trace(go.Scatter(
                            name=f'Priority {priority}',
                            x=resolution_monthly.index.astype(str),
                            y=resolution_monthly[priority],
                            mode='lines+markers',
                            line=dict(width=2),
                            marker=dict(size=6),
                            line_color=self.PRIORITY_COLORS.get(priority, self.VIRIDIS_PALETTE[0])
                        ))
                    
                    results['resolution_time_evolution'] = fig_resolution
            except Exception as e:
                self.logger.error(f"Error in resolution time evolution analysis: {str(e)}")
            
            # 3. CSAT Evolution
            try:
                csat_monthly = df.groupby('Month')['CSAT'].agg(['mean', 'std']).reset_index()
                
                if not csat_monthly.empty and not csat_monthly['mean'].isna().all():
                    fig_csat = self.setup_plotly_figure(
                        title='CSAT Evolution with Confidence Interval',
                        xaxis_title='Month',
                        yaxis_title='CSAT Score',
                        hovermode='x unified'
                    )
                    
                    fig_csat.add_trace(go.Scatter(
                        name='CSAT Score',
                        x=csat_monthly['Month'].astype(str),
                        y=csat_monthly['mean'],
                        mode='lines+markers',
                        line=dict(color=self.AQUA_PALETTE[2], width=2),
                        marker=dict(size=8)
                    ))
                    
                    # Add confidence interval if std is available
                    if not csat_monthly['std'].isna().all():
                        fig_csat.add_trace(go.Scatter(
                            name='Upper Bound',
                            x=csat_monthly['Month'].astype(str),
                            y=csat_monthly['mean'] + csat_monthly['std'],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False
                        ))
                        
                        fig_csat.add_trace(go.Scatter(
                            name='Lower Bound',
                            x=csat_monthly['Month'].astype(str),
                            y=csat_monthly['mean'] - csat_monthly['std'],
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(38, 198, 218, 0.2)',
                            showlegend=False
                        ))
                    
                    results['csat_evolution'] = fig_csat
            except Exception as e:
                self.logger.error(f"Error in CSAT evolution analysis: {str(e)}")
            
            # 4. Calculate pattern statistics
            try:
                pattern_stats = {
                    'total_tickets': len(df),
                    'unique_root_causes': len(df['Root Cause'].unique()),
                    'avg_resolution_time': df['Resolution Time (Days)'].mean(),
                    'avg_csat': df['CSAT'].mean(),
                    'trend_summary': {}
                }
                
                # Calculate resolution time trend
                if 'Resolution Time (Days)' in df.columns:
                    first_month = df.groupby('Month')['Resolution Time (Days)'].mean().iloc[0]
                    last_month = df.groupby('Month')['Resolution Time (Days)'].mean().iloc[-1]
                    pattern_stats['trend_summary']['resolution_time'] = (
                        'Improving' if last_month < first_month else 'Declining'
                    )
                
                # Calculate CSAT trend
                if 'CSAT' in df.columns:
                    first_month_csat = df.groupby('Month')['CSAT'].mean().iloc[0]
                    last_month_csat = df.groupby('Month')['CSAT'].mean().iloc[-1]
                    pattern_stats['trend_summary']['csat'] = (
                        'Improving' if last_month_csat > first_month_csat else 'Declining'
                    )
                
                results['pattern_stats'] = pattern_stats
            except Exception as e:
                self.logger.error(f"Error calculating pattern statistics: {str(e)}")
                results['pattern_stats'] = {
                    'total_tickets': len(df),
                    'unique_root_causes': len(df['Root Cause'].unique()) if 'Root Cause' in df.columns else 0,
                    'avg_resolution_time': None,
                    'avg_csat': None,
                    'trend_summary': {}
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in pattern evolution analysis: {str(e)}")
            return {} 
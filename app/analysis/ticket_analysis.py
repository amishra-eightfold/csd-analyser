"""Support ticket analysis module for CSD Analyzer."""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TicketAnalyzer:
    """Analyzes support ticket data to generate insights and metrics."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer with ticket data.
        
        Args:
            df (pd.DataFrame): DataFrame containing support ticket data
        """
        self.df = df
        
    def get_basic_metrics(self) -> Dict[str, Any]:
        """
        Calculate basic ticket metrics.
        
        Returns:
            Dict[str, Any]: Dictionary containing basic metrics
        """
        try:
            metrics = {
                'total_tickets': len(self.df),
                'avg_resolution_time': self.df['Resolution Time (Days)'].mean(),
                'avg_csat': self.df['CSAT'].mean(),
                'escalation_rate': (self.df['IsEscalated'].sum() / len(self.df)) if len(self.df) > 0 else 0
            }
            
            # Status breakdown
            status_counts = self.df['Status'].value_counts()
            metrics['status_breakdown'] = {
                status: count for status, count in status_counts.items()
            }
            
            # Priority breakdown
            priority_counts = self.df['Priority'].value_counts()
            metrics['priority_breakdown'] = {
                priority: count for priority, count in priority_counts.items()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate basic metrics: {str(e)}")
            return {}
            
    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze ticket trends."""
        try:
            # Calculate ticket volume trends
            ticket_counts = self.df.resample('M', on='Created Date').size()
            
            # Calculate resolution time trends
            resolution_times = self.df.resample('M', on='Created Date')['Resolution Time (Days)'].mean()
            
            # Calculate CSAT trends
            csat_scores = self.df.resample('M', on='Created Date')['CSAT'].mean()
            
            return {
                'volume_trend': ticket_counts.to_dict(),
                'resolution_trend': resolution_times.to_dict(),
                'csat_trend': csat_scores.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze trends: {str(e)}")
            return {}
            
    def identify_patterns(self) -> Dict[str, Any]:
        """
        Identify patterns in support tickets.
        
        Returns:
            Dict[str, Any]: Dictionary containing identified patterns
        """
        try:
            patterns = {}
            
            # Product area analysis
            product_area_counts = self.df['Product Area'].value_counts()
            patterns['product_areas'] = {
                area: count for area, count in product_area_counts.items()
            }
            
            # Feature analysis
            feature_counts = self.df['Product Feature'].value_counts()
            patterns['features'] = {
                feature: count for feature, count in feature_counts.items()
            }
            
            # Root cause analysis
            root_cause_counts = self.df['Root Cause'].value_counts()
            patterns['root_causes'] = {
                cause: count for cause, count in root_cause_counts.items()
            }
            
            # Identify correlations
            correlations = {}
            numeric_columns = ['Resolution Time (Days)', 'CSAT']
            categorical_columns = ['Priority', 'Status', 'Product Area', 'Root Cause']
            
            for num_col in numeric_columns:
                for cat_col in categorical_columns:
                    if num_col in self.df.columns and cat_col in self.df.columns:
                        agg_data = self.df.groupby(cat_col)[num_col].mean()
                        correlations[f"{cat_col}_vs_{num_col}"] = agg_data.to_dict()
            
            patterns['correlations'] = correlations
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to identify patterns: {str(e)}")
            return {}
            
    def get_critical_issues(self) -> List[Dict[str, Any]]:
        """
        Identify critical support issues.
        
        Returns:
            List[Dict[str, Any]]: List of critical issues
        """
        try:
            critical_issues = []
            
            # High priority escalated tickets
            escalated_high_priority = self.df[
                (self.df['IsEscalated'] == True) & 
                (self.df['Priority'].isin(['P1', 'P2', 'Critical', 'High']))
            ]
            
            for _, ticket in escalated_high_priority.iterrows():
                critical_issues.append({
                    'case_number': ticket['CaseNumber'],
                    'subject': ticket['Subject'],
                    'priority': ticket['Priority'],
                    'status': ticket['Status'],
                    'created_date': ticket['Created Date'],
                    'resolution_time': ticket.get('Resolution Time (Days)', None),
                    'product_area': ticket['Product Area']
                })
            
            return critical_issues
            
        except Exception as e:
            logger.error(f"Failed to identify critical issues: {str(e)}")
            return []
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """
        Calculate trend direction and magnitude.
        
        Args:
            series (pd.Series): Time series data
            
        Returns:
            str: Trend description
        """
        if len(series) < 2:
            return "Insufficient data"
            
        try:
            # Calculate percentage change
            pct_change = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
            
            # Determine trend direction
            if abs(pct_change) < 0.05:
                return "Stable"
            elif pct_change > 0:
                return "Increasing" if pct_change > 0.2 else "Slightly increasing"
            else:
                return "Decreasing" if pct_change < -0.2 else "Slightly decreasing"
                
        except Exception:
            return "Unable to calculate trend"

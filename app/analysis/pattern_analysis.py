"""Pattern analysis module for CSD Analyzer."""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from .text_analysis import TextAnalyzer

logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """Analyzes patterns and trends in support ticket data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize pattern analyzer.
        
        Args:
            df (pd.DataFrame): DataFrame containing support ticket data
        """
        self.df = df
        self.text_analyzer = TextAnalyzer()
        
    def analyze_trends(self) -> Dict[str, Any]:
        """
        Analyze trends in support tickets.
        
        Returns:
            Dict[str, Any]: Dictionary containing trend analysis results
        """
        try:
            trends = {}
            
            # Volume trends
            volume_trends = self.df.resample('M', on='Created Date').size()
            trends['volume_trend'] = {
                'values': volume_trends.to_dict(),
                'trend': 'increasing' if volume_trends.is_monotonic_increasing else
                        'decreasing' if volume_trends.is_monotonic_decreasing else 'fluctuating'
            }
            
            # Priority trends
            priority_trends = self.df.groupby([pd.Grouper(key='Created Date', freq='M'), 'Priority']).size().unstack(fill_value=0)
            trends['priority_trends'] = {
                priority: {
                    'values': priority_trends[priority].to_dict(),
                    'trend': 'increasing' if priority_trends[priority].is_monotonic_increasing else
                            'decreasing' if priority_trends[priority].is_monotonic_decreasing else 'fluctuating'
                }
                for priority in priority_trends.columns
            }
            
            # Resolution time trends
            resolution_trends = self.df.resample('M', on='Created Date')['Resolution Time (Days)'].mean()
            trends['resolution_trends'] = {
                'values': resolution_trends.to_dict(),
                'trend': 'increasing' if resolution_trends.is_monotonic_increasing else
                        'decreasing' if resolution_trends.is_monotonic_decreasing else 'fluctuating'
            }
            
            # CSAT trends
            csat_trends = self.df.resample('M', on='Created Date')['CSAT'].mean()
            trends['csat_trends'] = {
                'values': csat_trends.to_dict(),
                'trend': 'increasing' if csat_trends.is_monotonic_increasing else
                        'decreasing' if csat_trends.is_monotonic_decreasing else 'fluctuating'
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {str(e)}")
            return {}
            
    def analyze_time_patterns(self) -> Dict[str, Any]:
        """
        Analyze temporal patterns in support tickets.
        
        Returns:
            Dict[str, Any]: Dictionary containing time-based patterns
        """
        try:
            patterns = {}
            
            # Daily patterns
            patterns['daily'] = self._analyze_daily_patterns()
            
            # Weekly patterns
            patterns['weekly'] = self._analyze_weekly_patterns()
            
            # Monthly patterns
            patterns['monthly'] = self._analyze_monthly_patterns()
            
            # Seasonal patterns
            patterns['seasonal'] = self._analyze_seasonal_patterns()
            
            return patterns
            
        except Exception as e:
            logger.error(f"Time pattern analysis failed: {str(e)}")
            return {}
            
    def analyze_correlations(self) -> Dict[str, float]:
        """
        Analyze correlations between different metrics.
        
        Returns:
            Dict[str, float]: Dictionary containing correlation coefficients
        """
        try:
            correlations = {}
            
            # Numeric columns to analyze
            numeric_cols = [
                'Resolution Time (Days)',
                'CSAT',
                'First Response Time'
            ]
            
            # Calculate correlations between numeric columns
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    if col1 in self.df.columns and col2 in self.df.columns:
                        correlation = self.df[col1].corr(self.df[col2])
                        correlations[f"{col1}_vs_{col2}"] = correlation
            
            return correlations
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {str(e)}")
            return {}
            
    def identify_anomalies(self, 
                          threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Identify anomalies in support ticket patterns.
        
        Args:
            threshold (float): Z-score threshold for anomaly detection
            
        Returns:
            List[Dict[str, Any]]: List of identified anomalies
        """
        try:
            anomalies = []
            
            # Check for resolution time anomalies
            if 'Resolution Time (Days)' in self.df.columns:
                z_scores = np.abs(stats.zscore(self.df['Resolution Time (Days)'].fillna(0)))
                anomalous_tickets = self.df[z_scores > threshold]
                
                for _, ticket in anomalous_tickets.iterrows():
                    anomalies.append({
                        'type': 'resolution_time',
                        'case_number': ticket['CaseNumber'],
                        'value': ticket['Resolution Time (Days)'],
                        'z_score': float(z_scores[ticket.name]),
                        'created_date': ticket['Created Date']
                    })
            
            # Check for unusual patterns in ticket volume
            daily_counts = self.df.resample('D', on='Created Date').size()
            z_scores = np.abs(stats.zscore(daily_counts))
            
            for date, count in daily_counts[z_scores > threshold].items():
                anomalies.append({
                    'type': 'volume',
                    'date': date,
                    'count': int(count),
                    'z_score': float(z_scores[date]),
                    'expected': float(daily_counts.mean())
                })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            return []
            
    def identify_issue_clusters(self, 
                              min_cluster_size: int = 3) -> List[Dict[str, Any]]:
        """
        Identify clusters of related issues.
        
        Args:
            min_cluster_size (int): Minimum size for a cluster
            
        Returns:
            List[Dict[str, Any]]: List of issue clusters
        """
        try:
            # Extract text features
            texts = self.df['Subject'] + ' ' + self.df['Description']
            keywords = self.text_analyzer.extract_keywords(texts, top_n=50)
            
            # Create feature matrix
            feature_matrix = np.zeros((len(self.df), len(keywords)))
            for i, text in enumerate(texts):
                for j, kw in enumerate(keywords):
                    if kw['keyword'] in text.lower():
                        feature_matrix[i, j] = 1
            
            # Normalize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
            
            # Perform clustering
            clustering = DBSCAN(
                eps=0.5,
                min_samples=min_cluster_size,
                metric='cosine'
            )
            clusters = clustering.fit_predict(scaled_features)
            
            # Analyze clusters
            cluster_info = []
            for cluster_id in set(clusters):
                if cluster_id == -1:  # Skip noise points
                    continue
                    
                cluster_tickets = self.df[clusters == cluster_id]
                
                # Get common keywords for cluster
                cluster_texts = cluster_tickets['Subject'] + ' ' + cluster_tickets['Description']
                cluster_keywords = self.text_analyzer.extract_keywords(cluster_texts, top_n=5)
                
                cluster_info.append({
                    'cluster_id': int(cluster_id),
                    'size': len(cluster_tickets),
                    'keywords': [kw['keyword'] for kw in cluster_keywords],
                    'avg_resolution_time': float(cluster_tickets['Resolution Time (Days)'].mean()),
                    'avg_csat': float(cluster_tickets['CSAT'].mean()),
                    'tickets': cluster_tickets['CaseNumber'].tolist()
                })
            
            return cluster_info
            
        except Exception as e:
            logger.error(f"Issue clustering failed: {str(e)}")
            return []
            
    def _analyze_daily_patterns(self) -> Dict[str, Any]:
        """Analyze patterns within days."""
        try:
            # Convert to local timezone for accurate daily patterns
            self.df['hour'] = self.df['Created Date'].dt.hour
            
            # Analyze hourly distribution
            hourly_dist = self.df['hour'].value_counts().sort_index()
            
            # Identify peak hours
            peak_hours = hourly_dist[hourly_dist > hourly_dist.mean()].index.tolist()
            
            return {
                'distribution': hourly_dist.to_dict(),
                'peak_hours': peak_hours,
                'avg_tickets_per_hour': float(hourly_dist.mean())
            }
            
        except Exception as e:
            logger.error(f"Daily pattern analysis failed: {str(e)}")
            return {}
            
    def _analyze_weekly_patterns(self) -> Dict[str, Any]:
        """Analyze patterns within weeks."""
        try:
            # Get day of week (0 = Monday, 6 = Sunday)
            self.df['weekday'] = self.df['Created Date'].dt.dayofweek
            
            # Analyze daily distribution
            daily_dist = self.df['weekday'].value_counts().sort_index()
            
            # Map numeric days to names
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            distribution = {day_names[i]: int(count) for i, count in daily_dist.items()}
            
            # Identify busiest days
            busy_days = [day_names[i] for i in daily_dist[daily_dist > daily_dist.mean()].index]
            
            return {
                'distribution': distribution,
                'busy_days': busy_days,
                'avg_tickets_per_day': float(daily_dist.mean())
            }
            
        except Exception as e:
            logger.error(f"Weekly pattern analysis failed: {str(e)}")
            return {}
            
    def _analyze_monthly_patterns(self) -> Dict[str, Any]:
        """Analyze patterns within months."""
        try:
            # Get month
            self.df['month'] = self.df['Created Date'].dt.month
            
            # Analyze monthly distribution
            monthly_dist = self.df['month'].value_counts().sort_index()
            
            # Map numeric months to names
            month_names = [
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ]
            distribution = {month_names[i-1]: int(count) for i, count in monthly_dist.items()}
            
            # Identify peak months
            peak_months = [month_names[i-1] for i in monthly_dist[monthly_dist > monthly_dist.mean()].index]
            
            return {
                'distribution': distribution,
                'peak_months': peak_months,
                'avg_tickets_per_month': float(monthly_dist.mean())
            }
            
        except Exception as e:
            logger.error(f"Monthly pattern analysis failed: {str(e)}")
            return {}
            
    def _analyze_seasonal_patterns(self) -> Dict[str, Any]:
        """Analyze seasonal patterns."""
        try:
            # Define seasons
            season_months = {
                'Winter': [12, 1, 2],
                'Spring': [3, 4, 5],
                'Summer': [6, 7, 8],
                'Fall': [9, 10, 11]
            }
            
            # Map months to seasons
            self.df['season'] = self.df['Created Date'].dt.month.map(
                {m: s for s, months in season_months.items() for m in months}
            )
            
            # Analyze seasonal distribution
            seasonal_dist = self.df['season'].value_counts()
            
            # Calculate seasonal metrics
            seasonal_metrics = {}
            for season in season_months.keys():
                season_data = self.df[self.df['season'] == season]
                if len(season_data) > 0:
                    seasonal_metrics[season] = {
                        'ticket_count': len(season_data),
                        'avg_resolution_time': float(season_data['Resolution Time (Days)'].mean()),
                        'avg_csat': float(season_data['CSAT'].mean())
                    }
            
            return {
                'distribution': {s: int(c) for s, c in seasonal_dist.items()},
                'metrics': seasonal_metrics,
                'peak_season': seasonal_dist.idxmax()
            }
            
        except Exception as e:
            logger.error(f"Seasonal pattern analysis failed: {str(e)}")
            return {}

    def identify_patterns(self) -> Dict[str, Any]:
        """
        Identify key patterns and insights from the data.
        
        Returns:
            Dict[str, Any]: Dictionary containing identified patterns and recommendations
        """
        try:
            patterns = {}
            
            # Get trends
            trends = self.analyze_trends()
            
            # Volume patterns
            if 'volume_trends' in trends:
                volume_trend = trends['volume_trends']
                patterns['volume'] = {
                    'trend': volume_trend['trend'],
                    'peak_period': volume_trend.get('peak_period', 'N/A'),
                    'growth_rate': volume_trend.get('growth_rate', 0),
                    'recommendations': []
                }
                
                # Add volume-based recommendations
                if volume_trend['trend'] == 'increasing':
                    patterns['volume']['recommendations'].extend([
                        "Consider increasing support team capacity",
                        "Review and optimize ticket routing",
                        "Identify common issues for self-service solutions"
                    ])
                elif volume_trend['trend'] == 'decreasing':
                    patterns['volume']['recommendations'].extend([
                        "Analyze successful reduction strategies",
                        "Document best practices",
                        "Consider resource reallocation"
                    ])
            
            # Resolution time patterns
            if 'resolution_trends' in trends:
                resolution_trend = trends['resolution_trends']
                patterns['resolution'] = {
                    'trend': resolution_trend['trend'],
                    'improvement_rate': resolution_trend.get('improvement_rate', 0),
                    'recommendations': []
                }
                
                # Add resolution time recommendations
                if resolution_trend['trend'] == 'increasing':
                    patterns['resolution']['recommendations'].extend([
                        "Review ticket complexity trends",
                        "Identify bottlenecks in resolution process",
                        "Consider additional training or resources"
                    ])
                elif resolution_trend['trend'] == 'decreasing':
                    patterns['resolution']['recommendations'].extend([
                        "Document successful resolution strategies",
                        "Share best practices across team",
                        "Monitor quality alongside speed"
                    ])
            
            # CSAT patterns
            if 'csat_trends' in trends:
                csat_trend = trends['csat_trends']
                patterns['satisfaction'] = {
                    'trend': csat_trend['trend'],
                    'change': csat_trend.get('change', 0),
                    'recommendations': []
                }
                
                # Add CSAT recommendations
                if csat_trend['trend'] == 'decreasing':
                    patterns['satisfaction']['recommendations'].extend([
                        "Review customer feedback in detail",
                        "Identify pain points in support process",
                        "Consider customer communication improvements"
                    ])
                elif csat_trend['trend'] == 'increasing':
                    patterns['satisfaction']['recommendations'].extend([
                        "Document successful customer interactions",
                        "Share positive feedback with team",
                        "Build on successful strategies"
                    ])
            
            # Time-based patterns
            time_patterns = self.analyze_time_patterns()
            patterns['timing'] = {
                'daily_peak': time_patterns.get('daily', {}).get('peak_hour'),
                'weekly_peak': time_patterns.get('weekly', {}).get('peak_day'),
                'recommendations': [
                    "Align staffing with peak hours",
                    "Consider follow-the-sun support model",
                    "Optimize resource allocation"
                ]
            }
            
            # Priority patterns
            if 'priority_trends' in trends:
                patterns['priority'] = {
                    'trends': trends['priority_trends'],
                    'recommendations': [
                        "Review priority assignment criteria",
                        "Monitor escalation patterns",
                        "Ensure proper resource allocation"
                    ]
                }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to identify patterns: {str(e)}")
            return {}

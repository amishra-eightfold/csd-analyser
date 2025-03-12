"""Pattern recognition utilities for support ticket analysis."""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import networkx as nx
from datetime import datetime, timedelta
import logging
from collections import defaultdict

class PatternDetector:
    """Advanced pattern detection for support tickets."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2
        )
        self.similarity_threshold = 0.3
        
    def _preprocess_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess text data for analysis."""
        processed_df = df.copy()
        
        # Combine relevant text fields
        text_fields = ['Subject', 'Description']
        processed_df['combined_text'] = processed_df[text_fields].apply(
            lambda x: ' '.join(str(val) for val in x if pd.notna(val)), axis=1
        )
        
        # Add basic sentiment analysis
        processed_df['sentiment'] = processed_df['combined_text'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
        
        return processed_df
        
    def _extract_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract patterns based on temporal analysis."""
        temporal_patterns = {
            'daily_patterns': {},
            'weekly_patterns': {},
            'monthly_trends': {},
            'seasonal_effects': {}
        }
        
        # Convert to datetime if needed
        df['Created Date'] = pd.to_datetime(df['Created Date'])
        
        # Daily patterns
        df['hour'] = df['Created Date'].dt.hour
        hourly_dist = df.groupby('hour').size()
        peak_hours = hourly_dist[hourly_dist > hourly_dist.mean()].index.tolist()
        temporal_patterns['daily_patterns'] = {
            'peak_hours': peak_hours,
            'distribution': hourly_dist.to_dict()
        }
        
        # Weekly patterns
        df['weekday'] = df['Created Date'].dt.day_name()
        weekly_dist = df.groupby('weekday').size()
        temporal_patterns['weekly_patterns'] = {
            'distribution': weekly_dist.to_dict(),
            'busiest_days': weekly_dist.nlargest(2).index.tolist()
        }
        
        # Monthly trends
        monthly_data = df.set_index('Created Date').resample('M').size()
        temporal_patterns['monthly_trends'] = {
            'trend': 'increasing' if monthly_data.is_monotonic_increasing else 
                    'decreasing' if monthly_data.is_monotonic_decreasing else 'fluctuating',
            'peak_months': monthly_data.nlargest(3).index.strftime('%Y-%m').tolist()
        }
        
        return temporal_patterns
        
    def _detect_issue_clusters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect clusters of similar issues using DBSCAN."""
        # Vectorize text
        text_vectors = self.vectorizer.fit_transform(df['combined_text'])
        
        # Perform clustering
        clustering = DBSCAN(
            eps=0.3,
            min_samples=2,
            metric='cosine'
        ).fit(text_vectors)
        
        # Analyze clusters
        df['cluster'] = clustering.labels_
        clusters = {}
        
        for cluster_id in set(clustering.labels_):
            if cluster_id != -1:  # Ignore noise points
                cluster_tickets = df[df['cluster'] == cluster_id]
                
                # Get common terms
                cluster_text = ' '.join(cluster_tickets['combined_text'])
                feature_names = self.vectorizer.get_feature_names_out()
                tfidf_scores = text_vectors[clustering.labels_ == cluster_id].mean(axis=0).A1
                top_terms = [
                    feature_names[i] for i in tfidf_scores.argsort()[-5:][::-1]
                ]
                
                clusters[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_tickets),
                    'top_terms': top_terms,
                    'avg_priority': cluster_tickets['Priority'].mode().iloc[0],
                    'avg_resolution_time': cluster_tickets['Resolution Time (Days)'].mean(),
                    'product_areas': cluster_tickets['Product Area'].value_counts().to_dict()
                }
                
        return clusters
        
    def _analyze_priority_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in ticket priorities."""
        priority_patterns = {
            'distribution': {},
            'resolution_times': {},
            'escalation_patterns': {},
            'product_impact': {}
        }
        
        # Priority distribution over time
        priority_dist = df.groupby(['Priority', pd.Grouper(key='Created Date', freq='M')]).size()
        priority_patterns['distribution'] = {
            priority: dist.to_dict() 
            for priority, dist in priority_dist.groupby(level=0)
        }
        
        # Resolution time patterns
        resolution_by_priority = df.groupby('Priority')['Resolution Time (Days)'].agg(['mean', 'std']).to_dict()
        priority_patterns['resolution_times'] = resolution_by_priority
        
        # Product area impact
        product_priority = df.groupby(['Product Area', 'Priority']).size().unstack(fill_value=0)
        priority_patterns['product_impact'] = {
            area: priorities.to_dict()
            for area, priorities in product_priority.iterrows()
        }
        
        return priority_patterns
        
    def _detect_correlation_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect correlations between different ticket attributes."""
        correlations = {
            'priority_correlations': {},
            'resolution_correlations': {},
            'product_correlations': {}
        }
        
        # Priority vs Resolution Time
        priority_res_corr = df.pivot_table(
            values='Resolution Time (Days)',
            index='Priority',
            aggfunc=['mean', 'count']
        ).to_dict()
        correlations['priority_correlations']['resolution_time'] = priority_res_corr
        
        # Product Area vs Priority
        prod_priority_corr = pd.crosstab(df['Product Area'], df['Priority'])
        correlations['product_correlations']['priority'] = prod_priority_corr.to_dict()
        
        # Sentiment vs Resolution Time
        sentiment_bins = pd.qcut(df['sentiment'], q=3, labels=['negative', 'neutral', 'positive'])
        sentiment_res_corr = df.groupby(sentiment_bins)['Resolution Time (Days)'].mean().to_dict()
        correlations['resolution_correlations']['sentiment'] = sentiment_res_corr
        
        return correlations
        
    def detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Main method to detect patterns in support tickets."""
        try:
            # Preprocess data
            processed_df = self._preprocess_text(df)
            
            # Detect various types of patterns
            patterns = {
                'temporal_patterns': self._extract_temporal_patterns(processed_df),
                'issue_clusters': self._detect_issue_clusters(processed_df),
                'priority_patterns': self._analyze_priority_patterns(processed_df),
                'correlations': self._detect_correlation_patterns(processed_df)
            }
            
            # Add metadata
            patterns['metadata'] = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_tickets': len(df),
                'date_range': {
                    'start': df['Created Date'].min().strftime('%Y-%m-%d'),
                    'end': df['Created Date'].max().strftime('%Y-%m-%d')
                }
            }
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in pattern detection: {str(e)}")
            return {
                'error': str(e),
                'temporal_patterns': {},
                'issue_clusters': {},
                'priority_patterns': {},
                'correlations': {}
            }
            
class PatternAnalyzer:
    """Analyzes detected patterns to generate insights."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def _analyze_temporal_insights(self, temporal_patterns: Dict) -> List[Dict[str, str]]:
        """Generate insights from temporal patterns."""
        insights = []
        
        # Daily patterns
        if 'daily_patterns' in temporal_patterns:
            peak_hours = temporal_patterns['daily_patterns'].get('peak_hours', [])
            if peak_hours:
                insights.append({
                    'type': 'temporal',
                    'category': 'daily',
                    'finding': f"Peak ticket creation hours are {', '.join(map(str, peak_hours))}",
                    'impact': 'High',
                    'action': 'Consider adjusting support coverage for peak hours'
                })
                
        # Weekly patterns
        if 'weekly_patterns' in temporal_patterns:
            busiest_days = temporal_patterns['weekly_patterns'].get('busiest_days', [])
            if busiest_days:
                insights.append({
                    'type': 'temporal',
                    'category': 'weekly',
                    'finding': f"Highest ticket volume on {' and '.join(busiest_days)}",
                    'impact': 'Medium',
                    'action': 'Plan for increased support capacity on these days'
                })
                
        return insights
        
    def _analyze_cluster_insights(self, clusters: Dict) -> List[Dict[str, str]]:
        """Generate insights from issue clusters."""
        insights = []
        
        for cluster_id, cluster_data in clusters.items():
            if cluster_data['size'] >= 3:  # Significant clusters
                insights.append({
                    'type': 'cluster',
                    'category': 'issue_pattern',
                    'finding': f"Found cluster of {cluster_data['size']} related issues with terms: {', '.join(cluster_data['top_terms'])}",
                    'impact': 'High' if cluster_data['size'] > 5 else 'Medium',
                    'action': f"Investigate common root cause in {cluster_data['avg_priority']} priority tickets"
                })
                
        return insights
        
    def _analyze_priority_insights(self, priority_patterns: Dict) -> List[Dict[str, str]]:
        """Generate insights from priority patterns."""
        insights = []
        
        # Resolution time analysis
        if 'resolution_times' in priority_patterns:
            for priority, stats in priority_patterns['resolution_times'].items():
                if stats['mean'] > 5:  # High resolution time
                    insights.append({
                        'type': 'priority',
                        'category': 'resolution_time',
                        'finding': f"{priority} tickets take {stats['mean']:.1f} days on average to resolve",
                        'impact': 'High' if priority in ['P0', 'P1'] else 'Medium',
                        'action': 'Review resolution process for optimization opportunities'
                    })
                    
        return insights
        
    def _analyze_correlation_insights(self, correlations: Dict) -> List[Dict[str, str]]:
        """Generate insights from correlation patterns."""
        insights = []
        
        # Priority vs Resolution correlations
        if 'priority_correlations' in correlations:
            res_corr = correlations['priority_correlations'].get('resolution_time', {})
            for priority, data in res_corr.items():
                if isinstance(data, dict) and 'mean' in data and data['mean'] > 5:
                    insights.append({
                        'type': 'correlation',
                        'category': 'priority_resolution',
                        'finding': f"Strong correlation between {priority} priority and long resolution times",
                        'impact': 'High',
                        'action': 'Investigate bottlenecks in high-priority ticket resolution'
                    })
                    
        return insights
        
    def analyze_patterns(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Main method to analyze patterns and generate insights."""
        try:
            # Generate insights from different pattern types
            insights = {
                'temporal_insights': self._analyze_temporal_insights(patterns.get('temporal_patterns', {})),
                'cluster_insights': self._analyze_cluster_insights(patterns.get('issue_clusters', {})),
                'priority_insights': self._analyze_priority_insights(patterns.get('priority_patterns', {})),
                'correlation_insights': self._analyze_correlation_insights(patterns.get('correlations', {}))
            }
            
            # Generate summary
            total_insights = sum(len(v) for v in insights.values())
            high_priority_insights = sum(
                1 for insight_list in insights.values()
                for insight in insight_list
                if insight.get('impact') == 'High'
            )
            
            insights['summary'] = {
                'total_insights': total_insights,
                'high_priority_insights': high_priority_insights,
                'analysis_coverage': list(patterns.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error in pattern analysis: {str(e)}")
            return {
                'error': str(e),
                'temporal_insights': [],
                'cluster_insights': [],
                'priority_insights': [],
                'correlation_insights': [],
                'summary': {
                    'total_insights': 0,
                    'high_priority_insights': 0,
                    'analysis_coverage': [],
                    'timestamp': datetime.now().isoformat()
                }
            } 
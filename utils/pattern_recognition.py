"""Pattern recognition module for support ticket analysis."""

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from collections import Counter
import traceback
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from datetime import datetime, timedelta
from config.logging_config import get_logger
from .time_analysis import calculate_first_response_time, calculate_sla_breaches

class PatternRecognizer:
    """Recognizes patterns in support ticket data."""
    
    def __init__(self):
        """Initialize the pattern recognizer."""
        self.logger = get_logger('pattern_recognition')
    
    def analyze_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze patterns in the support ticket data.
        
        Args:
            df: DataFrame containing support ticket data
            
        Returns:
            Dictionary containing detected patterns and their analysis
        """
        try:
            if df.empty:
                self.logger.warning("Empty DataFrame provided for pattern analysis")
                return self._create_empty_pattern_result()
            
            patterns = []
            
            # Analyze text patterns in Subject field
            if 'Subject' in df.columns:
                text_patterns = self._analyze_text_patterns(df['Subject'])
                patterns.extend(text_patterns)
            
            # Analyze root cause patterns
            if 'Root Cause' in df.columns:
                root_cause_patterns = self._analyze_root_cause_patterns(df)
                patterns.extend(root_cause_patterns)
            
            # Analyze resolution time patterns
            if 'Resolution Time (Days)' in df.columns:
                resolution_patterns = self._analyze_resolution_patterns(df)
                patterns.extend(resolution_patterns)
                
            # Analyze first response time patterns
            if all(col in df.columns for col in ['First Response Time', 'Highest_Priority']):
                response_patterns = self._analyze_first_response_patterns(df)
                patterns.extend(response_patterns)
            
            # Calculate confidence distribution
            confidence_dist = {
                'high': len([p for p in patterns if p['confidence'] == 'high']),
                'medium': len([p for p in patterns if p['confidence'] == 'medium']),
                'low': len([p for p in patterns if p['confidence'] == 'low'])
            }
            
            return {
                'patterns': patterns,
                'confidence_distribution': confidence_dist,
                'total_patterns': len(patterns),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in pattern analysis: {str(e)}",
                            extra={'traceback': traceback.format_exc()})
            return self._create_empty_pattern_result()
    
    def _create_empty_pattern_result(self) -> Dict[str, Any]:
        """Create an empty pattern result structure."""
        return {
            'patterns': [],
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'total_patterns': 0,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _analyze_text_patterns(self, series: pd.Series) -> List[Dict[str, Any]]:
        """Analyze patterns in text data."""
        patterns = []
        try:
            # Get word frequencies
            words = ' '.join(series).lower().split()
            word_freq = pd.Series(words).value_counts()
            
            # Find common phrases
            for word, freq in word_freq.items():
                if freq >= 3:  # Minimum frequency threshold
                    confidence = 'high' if freq >= 10 else 'medium' if freq >= 5 else 'low'
                    patterns.append({
                        'pattern': f"Frequent term: {word}",
                        'frequency': int(freq),
                        'confidence': confidence
                    })
            
        except Exception as e:
            self.logger.error(f"Error in text pattern analysis: {str(e)}")
        
        return patterns
    
    def _analyze_root_cause_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze patterns in root causes."""
        patterns = []
        try:
            root_cause_counts = df['Root Cause'].value_counts()
            total_tickets = len(df)
            
            for cause, count in root_cause_counts.items():
                if pd.isna(cause) or cause in ['Not Specified', 'Unknown', None]:
                    continue
                    
                percentage = (count / total_tickets) * 100
                confidence = 'high' if percentage >= 20 else 'medium' if percentage >= 10 else 'low'
                
                patterns.append({
                    'pattern': f"Common root cause: {cause}",
                    'frequency': int(count),
                    'percentage': round(percentage, 2),
                    'confidence': confidence
                })
                
        except Exception as e:
            self.logger.error(f"Error in root cause pattern analysis: {str(e)}")
        
        return patterns
    
    def _analyze_resolution_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze patterns in resolution times."""
        patterns = []
        try:
            resolution_times = df['Resolution Time (Days)'].dropna()
            
            if not resolution_times.empty:
                avg_resolution = resolution_times.mean()
                median_resolution = resolution_times.median()
                
                # Check for long resolution time pattern
                long_resolutions = resolution_times[resolution_times > (median_resolution * 2)]
                if len(long_resolutions) > 0:
                    percentage = (len(long_resolutions) / len(resolution_times)) * 100
                    confidence = 'high' if percentage >= 15 else 'medium' if percentage >= 5 else 'low'
                    
                    patterns.append({
                        'pattern': "Extended resolution time pattern detected",
                        'frequency': len(long_resolutions),
                        'percentage': round(percentage, 2),
                        'confidence': confidence
                    })
                
        except Exception as e:
            self.logger.error(f"Error in resolution pattern analysis: {str(e)}")
        
        return patterns
    
    def _analyze_first_response_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze patterns in first response times by priority.
        
        Args:
            df: DataFrame containing ticket data with First Response Time and Highest_Priority
            
        Returns:
            List of dictionaries containing first response time patterns
        """
        patterns = []
        try:
            # Calculate response times using standardized function
            response_hours, stats = calculate_first_response_time(df)
            
            if stats['valid_records'] == 0:
                self.logger.warning("No valid response time data found")
                return patterns
            
            # Calculate breach statistics
            breach_df = calculate_sla_breaches(response_hours, df['Highest_Priority'])
            
            # Generate patterns from breach statistics
            for _, row in breach_df.iterrows():
                priority = row['Priority']
                
                # Basic response time pattern
                patterns.append({
                    'pattern': f"First response time pattern for {priority}",
                    'priority': priority,
                    'avg_response_hours': row['Mean Hours'],
                    'median_response_hours': row['Median Hours'],
                    'p90_response_hours': row['90th Percentile'],
                    'sample_size': row['Count'],
                    'confidence': 'high' if row['Count'] >= 30 else 'medium' if row['Count'] >= 10 else 'low',
                    'has_sla': not pd.isna(row['SLA Breach %'])
                })
                
                # Add pattern for concerning response times
                if not pd.isna(row['SLA Breach %']) and row['SLA Breach %'] > 20:
                    patterns.append({
                        'pattern': f"High SLA breach rate for {priority} tickets",
                        'priority': priority,
                        'breach_percentage': row['SLA Breach %'],
                        'confidence': 'high' if row['Count'] >= 30 else 'medium' if row['Count'] >= 10 else 'low',
                        'severity': 'high' if row['SLA Breach %'] > 50 else 'medium'
                    })
            
            # Add validation statistics to patterns if there were issues
            if stats['invalid_records'] > 0:
                patterns.append({
                    'pattern': 'Data quality issue in response times',
                    'description': f"{stats['invalid_records']} records ({(stats['invalid_records']/stats['total_records']*100):.1f}%) had invalid response times",
                    'confidence': 'high',
                    'severity': 'medium' if stats['invalid_records']/stats['total_records'] < 0.1 else 'high'
                })
            
        except Exception as e:
            self.logger.error(f"Error in first response time pattern analysis: {str(e)}",
                            extra={'traceback': traceback.format_exc()})
        
        return patterns

class PatternAnalyzer:
    """Advanced pattern analysis for support ticket trends."""
    
    def __init__(self):
        """Initialize the pattern analyzer."""
        self.logger = get_logger('pattern_recognition')
        self.recognizer = PatternRecognizer()
        
    def analyze_trend(self, tickets_df: pd.DataFrame, time_window: str = '7D') -> Dict[str, Any]:
        """
        Analyze pattern trends over time.
        
        Args:
            tickets_df: DataFrame containing support ticket data
            time_window: Time window for trend analysis (e.g., '7D', '30D')
            
        Returns:
            Dict[str, Any]: Dictionary containing trend analysis results with structure:
                {
                    'trends': List of trend data per time window,
                    'summary': String summarizing the trend analysis
                }
        """
        try:
            if tickets_df.empty:
                self.logger.warning("Empty DataFrame provided for trend analysis")
                return {'trends': [], 'summary': 'No data available for trend analysis'}
                
            # Ensure required columns exist
            required_cols = ['Subject', 'CreatedDate']
            if not all(col in tickets_df.columns for col in required_cols):
                missing_cols = set(required_cols) - set(tickets_df.columns)
                self.logger.error(f"Missing required columns: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert CreatedDate to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(tickets_df['CreatedDate']):
                tickets_df['CreatedDate'] = pd.to_datetime(tickets_df['CreatedDate'])
            
            # Group tickets by time window
            tickets_df = tickets_df.sort_values('CreatedDate')
            time_groups = self._create_time_groups(tickets_df, time_window)
            
            if not time_groups:
                self.logger.warning("No valid time groups created for trend analysis")
                return {'trends': [], 'summary': 'Could not create time groups for analysis'}
            
            trends = []
            for start_date, group_df in time_groups:
                # Get patterns for this time window
                window_patterns = self.recognizer.analyze_patterns(group_df)
                
                if window_patterns and window_patterns.get('patterns'):
                    trends.append({
                        'period_start': start_date.strftime('%Y-%m-%d'),
                        'patterns': window_patterns['patterns'],
                        'ticket_count': len(group_df),
                        'top_pattern': window_patterns['patterns'][0] if window_patterns['patterns'] else None
                    })
            
            # Generate trend summary
            summary = self._generate_trend_summary(trends)
            
            self.logger.info("Trend analysis completed successfully", extra={
                'time_windows': len(time_groups),
                'trends_found': len(trends)
            })
            
            return {
                'trends': trends,
                'summary': summary
            }
            
        except Exception as e:
            error_msg = f"Error in trend analysis: {str(e)}"
            self.logger.error(error_msg, extra={'traceback': traceback.format_exc()})
            return {'trends': [], 'summary': f'Error during trend analysis: {str(e)}'}
    
    def _create_time_groups(self, df: pd.DataFrame, window: str) -> List[Tuple[datetime, pd.DataFrame]]:
        """Create time-based groups of tickets."""
        try:
            # Parse window string to timedelta
            window_days = pd.Timedelta(window).days
            
            # Calculate date range
            start_date = df['CreatedDate'].min()
            end_date = df['CreatedDate'].max()
            
            # Create time windows
            groups = []
            current_start = start_date
            while current_start <= end_date:
                current_end = current_start + timedelta(days=window_days)
                mask = (df['CreatedDate'] >= current_start) & (df['CreatedDate'] < current_end)
                groups.append((current_start, df[mask]))
                current_start = current_end
                
            return groups
            
        except Exception as e:
            self.logger.error(f"Error creating time groups: {str(e)}")
            return []
    
    def _generate_trend_summary(self, trends: List[Dict[str, Any]]) -> str:
        """Generate a summary of pattern trends."""
        try:
            if not trends:
                return "No trends detected in the analyzed time period."
            
            # Track pattern evolution
            pattern_counts = Counter()
            pattern_confidence = {}
            
            for trend in trends:
                if trend.get('top_pattern'):
                    pattern = trend['top_pattern']['pattern']
                    pattern_counts[pattern] += 1
                    
                    # Track highest confidence for each pattern
                    current_confidence = trend['top_pattern']['confidence']
                    if pattern not in pattern_confidence or current_confidence > pattern_confidence[pattern]:
                        pattern_confidence[pattern] = current_confidence
            
            # Generate summary
            most_common = pattern_counts.most_common(3)
            if not most_common:
                return "No consistent patterns detected across time periods."
            
            summary_parts = ["Pattern trend analysis:"]
            
            for pattern, count in most_common:
                frequency = (count / len(trends)) * 100
                confidence = pattern_confidence[pattern]
                summary_parts.append(
                    f"- '{pattern}' appeared in {count} periods "
                    f"({frequency:.1f}% frequency, {confidence:.2f} confidence)"
                )
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating trend summary: {str(e)}")
            return "Error generating trend summary."
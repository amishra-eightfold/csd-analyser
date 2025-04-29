"""Tests for visualization functions."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

class TestFirstResponseVisualization(unittest.TestCase):
    """Test cases for first response time visualization."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        created_dates = [
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            datetime(2024, 1, 1, 12, 0),
            datetime(2024, 1, 1, 13, 0),
            datetime(2024, 1, 1, 14, 0),
        ]
        
        response_times = [
            datetime(2024, 1, 1, 10, 30),  # 30 min response
            datetime(2024, 1, 1, 13, 0),   # 2 hour response
            datetime(2024, 1, 1, 14, 0),   # 2 hour response
            datetime(2024, 1, 1, 15, 0),   # 2 hour response
            datetime(2024, 1, 1, 16, 0),   # 2 hour response
        ]
        
        priorities = ['P0', 'P1', 'P2', 'P3', 'P1']
        
        self.test_df = pd.DataFrame({
            'Created Date': created_dates,
            'First Response Time': response_times,
            'Highest_Priority': priorities
        })

    def test_response_time_calculation(self):
        """Test response time calculation in hours."""
        # Calculate response time
        self.test_df['Response Hours'] = (
            self.test_df['First Response Time'] - self.test_df['Created Date']
        ).dt.total_seconds() / 3600
        
        # Check P0 response time
        p0_response = self.test_df[self.test_df['Highest_Priority'] == 'P0']['Response Hours'].iloc[0]
        self.assertAlmostEqual(p0_response, 0.5, places=1)  # 30 minutes = 0.5 hours
        
        # Check P1 response times
        p1_responses = self.test_df[self.test_df['Highest_Priority'] == 'P1']['Response Hours']
        self.assertTrue(all(p1_responses <= 24))  # All P1 responses within SLA

    def test_sla_breach_calculation(self):
        """Test SLA breach percentage calculation."""
        # Calculate response time
        self.test_df['Response Hours'] = (
            self.test_df['First Response Time'] - self.test_df['Created Date']
        ).dt.total_seconds() / 3600
        
        # Define SLA thresholds
        sla_thresholds = {
            'P0': 1,     # 1 hour
            'P1': 24,    # 24 hours
            'P2': 48,    # 48 hours
            # P3 has no SLA
        }
        
        # Calculate breach percentages
        for priority in ['P0', 'P1', 'P2']:
            priority_data = self.test_df[self.test_df['Highest_Priority'] == priority]
            if len(priority_data) > 0:
                threshold = sla_thresholds[priority]
                breached = priority_data[priority_data['Response Hours'] > threshold]
                breach_percentage = (len(breached) / len(priority_data)) * 100
                
                # All our test data is within SLA
                self.assertEqual(breach_percentage, 0.0)

    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        # Calculate response time
        self.test_df['Response Hours'] = (
            self.test_df['First Response Time'] - self.test_df['Created Date']
        ).dt.total_seconds() / 3600
        
        # Calculate summary statistics
        summary_stats = self.test_df.groupby('Highest_Priority').agg({
            'Response Hours': ['count', 'mean', 'median', 
                             lambda x: x.quantile(0.90)]  # 90th percentile
        }).round(2)
        
        summary_stats.columns = ['Count', 'Mean Hours', 'Median Hours', '90th Percentile']
        
        # Check P0 statistics
        p0_stats = summary_stats.loc['P0']
        self.assertEqual(p0_stats['Count'], 1)
        self.assertAlmostEqual(p0_stats['Mean Hours'], 0.5, places=1)
        self.assertAlmostEqual(p0_stats['Median Hours'], 0.5, places=1)
        
        # Check P1 statistics (should have 2 tickets)
        p1_stats = summary_stats.loc['P1']
        self.assertEqual(p1_stats['Count'], 2)
        self.assertTrue(all(p1_stats[['Mean Hours', 'Median Hours', '90th Percentile']] <= 24))

    def test_edge_cases(self):
        """Test handling of edge cases."""
        # Test empty DataFrame
        empty_df = pd.DataFrame(columns=self.test_df.columns)
        summary_stats = empty_df.groupby('Highest_Priority').agg({
            'Response Hours': ['count', 'mean', 'median']
        }).round(2)
        self.assertTrue(summary_stats.empty)
        
        # Test missing response times
        df_with_nulls = self.test_df.copy()
        df_with_nulls.loc[0, 'First Response Time'] = None
        valid_responses = df_with_nulls[df_with_nulls['First Response Time'].notna()]
        self.assertEqual(len(valid_responses), len(self.test_df) - 1)
        
        # Test invalid priorities
        df_invalid = self.test_df.copy()
        df_invalid['Highest_Priority'] = 'Invalid'
        valid_priorities = df_invalid[
            df_invalid['Highest_Priority'].isin(['P0', 'P1', 'P2', 'P3'])
        ]
        self.assertEqual(len(valid_priorities), 0)

if __name__ == '__main__':
    unittest.main() 
"""Tests for pattern recognition module."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.pattern_recognition import PatternRecognizer

class TestPatternRecognizer(unittest.TestCase):
    """Test cases for PatternRecognizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.recognizer = PatternRecognizer()
        
        # Create sample data with different time formats
        created_dates = [
            datetime(2024, 1, 1, 10, 0),  # 10:00
            datetime(2024, 1, 1, 11, 0),  # 11:00
            datetime(2024, 1, 1, 12, 0),  # 12:00
            datetime(2024, 1, 1, 13, 0),  # 13:00
            datetime(2024, 1, 1, 14, 0),  # 14:00
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

    def test_first_response_time_calculation(self):
        """Test first response time calculation for different time formats."""
        # Test with datetime format
        patterns = self.recognizer._analyze_first_response_patterns(self.test_df)
        
        # Verify patterns exist
        self.assertTrue(len(patterns) > 0)
        
        # Find P0 pattern
        p0_pattern = next((p for p in patterns if p['priority'] == 'P0'), None)
        self.assertIsNotNone(p0_pattern)
        
        # Check P0 response time (should be 0.5 hours)
        self.assertAlmostEqual(p0_pattern['avg_response_hours'], 0.5, places=1)
        
        # Test with timedelta format
        self.test_df['First Response Time'] = (
            self.test_df['First Response Time'] - self.test_df['Created Date']
        )
        patterns_delta = self.recognizer._analyze_first_response_patterns(self.test_df)
        
        # Verify patterns are consistent between formats
        p0_pattern_delta = next((p for p in patterns_delta if p['priority'] == 'P0'), None)
        self.assertAlmostEqual(
            p0_pattern['avg_response_hours'],
            p0_pattern_delta['avg_response_hours'],
            places=1
        )

    def test_sla_thresholds(self):
        """Test SLA threshold calculations and breach percentages."""
        patterns = self.recognizer._analyze_first_response_patterns(self.test_df)
        
        # Test P0 SLA (1 hour)
        p0_pattern = next((p for p in patterns if p['priority'] == 'P0'), None)
        self.assertIsNotNone(p0_pattern)
        self.assertEqual(p0_pattern['has_sla'], True)
        self.assertEqual(p0_pattern['sla_breach_percentage'], 0.0)  # 30min response
        
        # Test P1 SLA (24 hours)
        p1_pattern = next((p for p in patterns if p['priority'] == 'P1'), None)
        self.assertIsNotNone(p1_pattern)
        self.assertEqual(p1_pattern['has_sla'], True)
        self.assertEqual(p1_pattern['sla_breach_percentage'], 0.0)  # 2 hour responses
        
        # Test P2 SLA (48 hours)
        p2_pattern = next((p for p in patterns if p['priority'] == 'P2'), None)
        self.assertIsNotNone(p2_pattern)
        self.assertEqual(p2_pattern['has_sla'], True)
        self.assertEqual(p2_pattern['sla_breach_percentage'], 0.0)  # 2 hour response
        
        # Test P3 (no SLA)
        p3_pattern = next((p for p in patterns if p['priority'] == 'P3'), None)
        self.assertIsNotNone(p3_pattern)
        self.assertEqual(p3_pattern['has_sla'], False)
        self.assertEqual(p3_pattern['sla_breach_percentage'], 0.0)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test empty DataFrame
        empty_df = pd.DataFrame(columns=self.test_df.columns)
        patterns = self.recognizer._analyze_first_response_patterns(empty_df)
        self.assertEqual(len(patterns), 0)
        
        # Test missing columns
        invalid_df = pd.DataFrame({'Invalid': [1, 2, 3]})
        patterns = self.recognizer._analyze_first_response_patterns(invalid_df)
        self.assertEqual(len(patterns), 0)
        
        # Test null values
        df_with_nulls = self.test_df.copy()
        df_with_nulls.loc[0, 'First Response Time'] = None
        patterns = self.recognizer._analyze_first_response_patterns(df_with_nulls)
        self.assertTrue(len(patterns) > 0)  # Should still process valid rows
        
        # Test invalid priorities
        df_invalid_priority = self.test_df.copy()
        df_invalid_priority['Highest_Priority'] = 'Invalid'
        patterns = self.recognizer._analyze_first_response_patterns(df_invalid_priority)
        self.assertEqual(len(patterns), 0)  # Should not process invalid priorities

    def test_confidence_levels(self):
        """Test confidence level calculations based on sample size."""
        # Create larger dataset
        large_df = pd.concat([self.test_df] * 10, ignore_index=True)  # 50 records
        patterns = self.recognizer._analyze_first_response_patterns(large_df)
        
        for pattern in patterns:
            if pattern['sample_size'] >= 30:
                self.assertEqual(pattern['confidence'], 'high')
            elif pattern['sample_size'] >= 10:
                self.assertEqual(pattern['confidence'], 'medium')
            else:
                self.assertEqual(pattern['confidence'], 'low')

    def test_breach_pattern_detection(self):
        """Test detection of concerning breach patterns."""
        # Create dataset with high breach rate
        created_dates = [datetime(2024, 1, 1, 10, 0)] * 5
        response_times = [
            datetime(2024, 1, 1, 10, 30),  # Within SLA
            datetime(2024, 1, 1, 12, 0),   # Breached
            datetime(2024, 1, 1, 13, 0),   # Breached
            datetime(2024, 1, 1, 14, 0),   # Breached
            datetime(2024, 1, 1, 15, 0),   # Breached
        ]
        priorities = ['P0'] * 5
        
        breach_df = pd.DataFrame({
            'Created Date': created_dates,
            'First Response Time': response_times,
            'Highest_Priority': priorities
        })
        
        patterns = self.recognizer._analyze_first_response_patterns(breach_df)
        
        # Find breach pattern
        breach_pattern = next(
            (p for p in patterns if 'High SLA breach rate' in p['pattern']),
            None
        )
        
        self.assertIsNotNone(breach_pattern)
        self.assertEqual(breach_pattern['priority'], 'P0')
        self.assertGreater(breach_pattern['breach_percentage'], 50)
        self.assertEqual(breach_pattern['severity'], 'high')

if __name__ == '__main__':
    unittest.main() 
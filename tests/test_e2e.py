"""End-to-end tests for CSD Analyzer."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from app.core.config import config
from app.core.salesforce import salesforce
from app.core.openai_client import openai
from app.analysis.ticket_analysis import TicketAnalyzer
from app.analysis.text_analysis import TextAnalyzer
from app.analysis.pattern_analysis import PatternAnalyzer
from app.visualization.charts import ChartGenerator
from app.export.excel import ExcelExporter
from app.export.powerpoint import PowerPointExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestE2E(unittest.TestCase):
    """End-to-end test cases for CSD Analyzer."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and configurations."""
        # Create sample test data
        cls.test_data = pd.DataFrame({
            'CaseNumber': [f'CASE-{i:03d}' for i in range(1, 11)],
            'Subject': [f'Test Case {i}' for i in range(1, 11)],
            'Description': [f'Description for case {i}' for i in range(1, 11)],
            'Created Date': pd.date_range(start='2024-01-01', periods=10),
            'Closed Date': pd.date_range(start='2024-01-02', periods=10),
            'Status': np.random.choice(['New', 'In Progress', 'Closed'], 10),
            'Priority': np.random.choice(['P1', 'P2', 'P3', 'P4'], 10),
            'Product Area': np.random.choice(['Area1', 'Area2', 'Area3'], 10),
            'Product Feature': np.random.choice(['Feature1', 'Feature2'], 10),
            'Root Cause': np.random.choice(['Cause1', 'Cause2', 'Not Specified'], 10),
            'First Response Time': pd.date_range(start='2024-01-01 01:00:00', periods=10),
            'CSAT': np.random.uniform(1, 5, 10),
            'IsEscalated': np.random.choice([True, False], 10),
            'Resolution Time (Days)': np.random.uniform(0.5, 5, 10)
        })
        
        # Ensure output directory exists
        os.makedirs('tests/output', exist_ok=True)
        
    def test_01_config(self):
        """Test configuration loading."""
        self.assertIsNotNone(config)
        self.assertIsInstance(config.is_production, bool)
        timeouts = config.get_api_timeouts()
        self.assertIsInstance(timeouts, dict)
        self.assertIn('query', timeouts)
        
    def test_02_salesforce_connection(self):
        """Test Salesforce connection."""
        if not all([
            config.salesforce_username,
            config.salesforce_password,
            config.salesforce_security_token
        ]):
            self.skipTest("Salesforce credentials not configured")
            
        connected = salesforce.connect()
        self.assertTrue(connected)
        self.assertTrue(salesforce.is_connected)
        
    def test_03_ticket_analysis(self):
        """Test ticket analysis functionality."""
        analyzer = TicketAnalyzer(self.test_data)
        
        # Test basic metrics
        metrics = analyzer.get_basic_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertEqual(metrics['total_tickets'], len(self.test_data))
        
        # Test trend analysis
        trends = analyzer.analyze_trends()
        self.assertIsInstance(trends, dict)
        self.assertIn('volume_trend', trends)
        
        # Test pattern identification
        patterns = analyzer.identify_patterns()
        self.assertIsInstance(patterns, dict)
        self.assertIn('product_areas', patterns)
        
    def test_04_text_analysis(self):
        """Test text analysis functionality."""
        analyzer = TextAnalyzer()
        
        # Test text preprocessing
        sample_text = "Test case with special chars: 123!@#"
        processed = analyzer.preprocess_text(sample_text)
        self.assertIsInstance(processed, str)
        
        # Test keyword extraction
        texts = self.test_data['Description'].tolist()
        keywords = analyzer.extract_keywords(texts, top_n=5)
        self.assertIsInstance(keywords, list)
        self.assertLessEqual(len(keywords), 5)
        
    def test_05_pattern_analysis(self):
        """Test pattern analysis functionality."""
        analyzer = PatternAnalyzer(self.test_data)
        
        # Test time patterns
        patterns = analyzer.analyze_time_patterns()
        self.assertIsInstance(patterns, dict)
        self.assertIn('daily', patterns)
        
        # Test anomaly detection
        anomalies = analyzer.identify_anomalies()
        self.assertIsInstance(anomalies, list)
        
    def test_06_chart_generation(self):
        """Test chart generation functionality."""
        generator = ChartGenerator(self.test_data)
        
        # Test volume chart
        volume_chart = generator.create_ticket_volume_chart()
        self.assertIsNotNone(volume_chart)
        
        # Test priority chart
        priority_chart = generator.create_priority_distribution_chart()
        self.assertIsNotNone(priority_chart)
        
    def test_07_excel_export(self):
        """Test Excel export functionality."""
        exporter = ExcelExporter(self.test_data)
        output_file = 'tests/output/test_analysis.xlsx'
        
        try:
            result = exporter.export_analysis(output_file)
            self.assertEqual(result, output_file)
            self.assertTrue(os.path.exists(output_file))
            self.assertGreater(os.path.getsize(output_file), 0)
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)
                
    def test_08_powerpoint_export(self):
        """Test PowerPoint export functionality."""
        exporter = PowerPointExporter(self.test_data)
        output_file = 'tests/output/test_analysis.pptx'
        
        try:
            result = exporter.export_analysis(output_file)
            self.assertEqual(result, output_file)
            self.assertTrue(os.path.exists(output_file))
            self.assertGreater(os.path.getsize(output_file), 0)
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)
                
    def test_09_openai_integration(self):
        """Test OpenAI integration."""
        if not config.openai_api_key:
            self.skipTest("OpenAI API key not configured")
            
        prompt = "Analyze this test case"
        try:
            insights = openai.generate_insights(prompt)
            self.assertIsInstance(insights, str)
            self.assertGreater(len(insights), 0)
        except Exception as e:
            self.fail(f"OpenAI integration failed: {str(e)}")
            
    def test_10_end_to_end_flow(self):
        """Test complete end-to-end flow."""
        try:
            # 1. Load and validate configuration
            self.assertIsNotNone(config)
            
            # 2. Initialize analyzers
            ticket_analyzer = TicketAnalyzer(self.test_data)
            pattern_analyzer = PatternAnalyzer(self.test_data)
            
            # 3. Generate analysis
            metrics = ticket_analyzer.get_basic_metrics()
            patterns = pattern_analyzer.analyze_time_patterns()
            
            # 4. Generate visualizations
            chart_gen = ChartGenerator(self.test_data)
            volume_chart = chart_gen.create_ticket_volume_chart()
            priority_chart = chart_gen.create_priority_distribution_chart()
            
            # 5. Export results
            excel_file = 'tests/output/e2e_test.xlsx'
            pptx_file = 'tests/output/e2e_test.pptx'
            
            try:
                # Export to Excel
                excel_exporter = ExcelExporter(self.test_data)
                excel_result = excel_exporter.export_analysis(excel_file)
                self.assertTrue(os.path.exists(excel_result))
                
                # Export to PowerPoint
                pptx_exporter = PowerPointExporter(self.test_data)
                pptx_result = pptx_exporter.export_analysis(pptx_file)
                self.assertTrue(os.path.exists(pptx_result))
                
            finally:
                # Clean up
                for file in [excel_file, pptx_file]:
                    if os.path.exists(file):
                        os.remove(file)
                        
        except Exception as e:
            self.fail(f"End-to-end test failed: {str(e)}")

if __name__ == '__main__':
    unittest.main() 
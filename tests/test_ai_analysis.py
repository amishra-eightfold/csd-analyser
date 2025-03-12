"""Tests for the AI analysis functionality."""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import json
from datetime import datetime, timedelta
from utils.ai_analysis import AIAnalyzer, ContextManager

class TestContextManager(unittest.TestCase):
    """Test cases for the ContextManager class."""
    
    def setUp(self):
        self.context_manager = ContextManager()
        
    def test_add_insight(self):
        """Test adding insights to context."""
        insight1 = {"test": "insight1"}
        insight2 = {"test": "insight2"}
        
        self.context_manager.add_insight(insight1)
        self.assertEqual(len(self.context_manager.previous_insights), 1)
        self.assertEqual(self.context_manager.previous_insights[0], insight1)
        
        # Test limit of 5 insights
        for i in range(5):
            self.context_manager.add_insight(insight2)
        self.assertEqual(len(self.context_manager.previous_insights), 5)
        self.assertEqual(self.context_manager.previous_insights[-1], insight2)
        
    def test_update_patterns(self):
        """Test updating pattern frequencies."""
        patterns = {"pattern1": 2, "pattern2": 1}
        self.context_manager.update_patterns(patterns)
        
        self.assertEqual(self.context_manager.global_patterns["pattern1"], 2)
        self.assertEqual(self.context_manager.global_patterns["pattern2"], 1)
        
        # Test accumulation
        self.context_manager.update_patterns({"pattern1": 1})
        self.assertEqual(self.context_manager.global_patterns["pattern1"], 3)
        
    def test_add_priority_context(self):
        """Test adding priority context."""
        priority_data = {"metric": "value"}
        self.context_manager.add_priority_context("P1", priority_data)
        
        self.assertEqual(len(self.context_manager.priority_context["P1"]), 1)
        self.assertEqual(self.context_manager.priority_context["P1"][0], priority_data)
        
    def test_get_summary_context(self):
        """Test getting consolidated context."""
        insight = {"test": "insight"}
        patterns = {"pattern1": 1}
        priority_data = {"metric": "value"}
        
        self.context_manager.add_insight(insight)
        self.context_manager.update_patterns(patterns)
        self.context_manager.add_priority_context("P1", priority_data)
        
        context = self.context_manager.get_summary_context()
        
        self.assertIn("previous_insights", context)
        self.assertIn("global_patterns", context)
        self.assertIn("priority_context", context)
        self.assertIn("temporal_context", context)
        
class TestAIAnalyzer(unittest.TestCase):
    """Test cases for the AIAnalyzer class."""
    
    def setUp(self):
        self.mock_client = Mock()
        self.analyzer = AIAnalyzer(self.mock_client)
        
        # Create sample test data
        self.test_data = pd.DataFrame({
            'Id': ['1', '2', '3'],
            'Subject': ['Test 1', 'Test 2', 'Test 3'],
            'Description': ['Description 1', 'Description 2', 'Description 3'],
            'Status': ['Open', 'Closed', 'Open'],
            'Priority': ['P1', 'P2', 'P1'],
            'Product Area': ['Area1', 'Area2', 'Area1'],
            'Created Date': [
                datetime.now() - timedelta(days=5),
                datetime.now() - timedelta(days=3),
                datetime.now() - timedelta(days=1)
            ],
            'Resolution Time (Days)': [2.5, 1.5, None]
        })
        
    def test_prepare_chunk_prompt(self):
        """Test prompt preparation for chunk analysis."""
        chunk_data = [{"id": "1", "subject": "test"}]
        context = {"previous_insights": []}
        
        prompt = self.analyzer._prepare_chunk_prompt(chunk_data, context)
        
        self.assertIsInstance(prompt, str)
        self.assertIn("Previous Context", prompt)
        self.assertIn("Current Tickets", prompt)
        
    @patch('utils.ai_analysis.json.loads')
    def test_analyze_chunk(self, mock_loads):
        """Test chunk analysis with mocked OpenAI response."""
        chunk_data = [{"id": "1", "subject": "test"}]
        context = {"previous_insights": []}
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"chunk_summary": {"main_findings": "test"}}'
        self.mock_client.chat.completions.create.return_value = mock_response
        
        # Mock json loads to return expected structure
        mock_loads.return_value = {
            "chunk_summary": {"main_findings": "test"},
            "patterns": [{"pattern": "test pattern", "frequency": "high"}]
        }
        
        result = self.analyzer._analyze_chunk(chunk_data, context)
        
        self.assertIsInstance(result, dict)
        self.assertIn("chunk_summary", result)
        self.mock_client.chat.completions.create.assert_called_once()
        
    def test_analyze_tickets(self):
        """Test full ticket analysis pipeline."""
        # Mock OpenAI responses
        mock_chunk_response = Mock()
        mock_chunk_response.choices = [Mock()]
        mock_chunk_response.choices[0].message.content = json.dumps({
            "chunk_summary": {"main_findings": "test"},
            "patterns": [{"pattern": "test pattern", "frequency": "high"}]
        })
        
        mock_final_response = Mock()
        mock_final_response.choices = [Mock()]
        mock_final_response.choices[0].message.content = json.dumps({
            "executive_summary": {
                "key_findings": ["finding1", "finding2"],
                "critical_patterns": ["pattern1"]
            },
            "recommendations": [
                {"title": "rec1", "description": "desc1", "priority": "High"}
            ]
        })
        
        self.mock_client.chat.completions.create.side_effect = [
            mock_chunk_response,
            mock_final_response
        ]
        
        result = self.analyzer.analyze_tickets(self.test_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn("executive_summary", result)
        self.assertIn("recommendations", result)
        self.assertIn("metadata", result)
        
    def test_error_handling(self):
        """Test error handling in analysis pipeline."""
        # Make OpenAI client raise an exception
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        result = self.analyzer.analyze_tickets(self.test_data)
        
        self.assertIn("error", result)
        self.assertIn("executive_summary", result)
        self.assertEqual(result["executive_summary"]["key_findings"], ["Analysis failed"])
        
if __name__ == '__main__':
    unittest.main() 
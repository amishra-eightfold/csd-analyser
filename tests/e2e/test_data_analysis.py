"""End-to-end tests for data analysis functionality."""

import pytest
import pandas as pd
import streamlit as st
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime, timedelta
import json
import os
from openai import OpenAI

class MockSessionState:
    """Mock Streamlit session state."""
    def __init__(self):
        self._dict = {}

    def __getattr__(self, key):
        if key not in self._dict:
            self._dict[key] = None
        return self._dict[key]

    def __setattr__(self, key, value):
        if key == '_dict':
            super().__setattr__(key, value)
        else:
            self._dict[key] = value

    def __iter__(self):
        return iter(self._dict)

    def __contains__(self, key):
        return key in self._dict

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        self._dict[key] = value

def test_full_analysis_flow(
    sample_case_data,
    sample_comment_data,
    sample_email_data,
    mock_openai_response,
    mock_salesforce_response,
    mock_salesforce
):
    """
    Test the complete data analysis flow from data fetching to visualization.
    
    This test covers:
    1. Data fetching from Salesforce
    2. Data preprocessing and cleaning
    3. Basic analysis generation
    4. AI-powered insights generation
    5. Visualization creation
    """
    # Mock Streamlit session state
    mock_state = MockSessionState()
    mock_state.sf_connection = mock_salesforce
    mock_state.selected_customers = ['Customer1']
    mock_state.date_range = (
        datetime.now() - timedelta(days=30),
        datetime.now()
    )
    mock_state.debug_mode = False
    
    # Mock Streamlit components
    with patch('streamlit.session_state', mock_state), \
         patch('streamlit.sidebar.selectbox') as mock_selectbox, \
         patch('streamlit.sidebar.date_input') as mock_date_input, \
         patch('streamlit.sidebar.checkbox') as mock_checkbox, \
         patch('streamlit.progress') as mock_progress, \
         patch('streamlit.error') as mock_error, \
         patch('streamlit.warning') as mock_warning:
        
        # Set up mock returns
        mock_selectbox.return_value = 'Customer1'
        mock_date_input.return_value = mock_state.date_range
        mock_checkbox.return_value = True
        mock_progress.return_value = MagicMock()
        
        # Mock Salesforce connection and queries
        with patch('salesforce_config.execute_soql_query') as mock_execute_query:
            mock_execute_query.return_value = mock_salesforce_response['records']
            
            # Mock OpenAI API calls
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
                # Create a proper mock OpenAI client
                mock_openai = Mock(spec=OpenAI)
                mock_openai.chat = Mock()
                mock_openai.chat.completions = Mock()
                mock_openai.chat.completions.create = Mock()
                
                # Set up the mock response
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message = Mock()
                mock_response.choices[0].message.content = json.dumps({
                    'executive_summary': {
                        'key_findings': ['Finding 1', 'Finding 2']
                    },
                    'pattern_insights': {
                        'recurring_issues': ['Issue 1', 'Issue 2'],
                        'confidence_levels': {
                            'high_confidence': ['Pattern 1', 'Pattern 2']
                        }
                    },
                    'trend_analysis': {
                        'pattern_evolution': ['Trend 1', 'Trend 2']
                    },
                    'recommendations': ['Rec 1', 'Rec 2'],
                    'customer_impact_analysis': 'Impact analysis',
                    'next_steps': ['Step 1', 'Step 2'],
                    'metadata': {'version': '1.0'}
                })
                mock_openai.chat.completions.create.return_value = mock_response
                
                # Import here to avoid circular imports
                from app import (
                    fetch_data,
                    display_detailed_analysis,
                    generate_ai_insights
                )
                
                # Test data fetching
                cases_df = fetch_data(
                    customers=['Customer1'],
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now()
                )['cases']
                assert not cases_df.empty
                assert len(cases_df) == 2  # We expect 2 cases for Customer1
                assert all(cases_df['Account_Name'] == 'Customer1')
                
                # Test AI insights generation
                insights = generate_ai_insights(cases_df)
                assert isinstance(insights, dict)
                assert insights['status'] == 'success'
                assert 'insights' in insights
                assert 'timestamp' in insights
                
                insights_data = insights['insights']
                assert 'key_findings' in insights_data
                assert 'patterns' in insights_data
                assert 'recommendations' in insights_data
                assert 'summary' in insights_data
                assert 'metadata' in insights_data
                
                # Test visualization
                display_detailed_analysis(insights, cases_df)
                
                # Verify no errors were shown
                mock_error.assert_not_called() 
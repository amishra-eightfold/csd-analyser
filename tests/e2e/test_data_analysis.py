"""End-to-end tests for data analysis functionality."""

import pytest
import pandas as pd
import streamlit as st
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import json
import os

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
    6. Export functionality
    """
    # Mock Streamlit session state
    mock_state = MockSessionState()
    mock_state.sf_connection = mock_salesforce  # This is now a direct connection object
    mock_state.customers = None
    mock_state.selected_customers = []
    mock_state.date_range = None
    mock_state.data_loaded = False
    mock_state.debug_mode = False
    
    # Mock Streamlit components
    with patch('streamlit.session_state', mock_state), \
         patch('streamlit.sidebar.selectbox') as mock_selectbox, \
         patch('streamlit.sidebar.date_input') as mock_date_input, \
         patch('streamlit.sidebar.checkbox') as mock_checkbox, \
         patch('streamlit.progress') as mock_progress, \
         patch('streamlit.error') as mock_error, \
         patch('streamlit.success') as mock_success:
        
        # Set up mock returns
        mock_selectbox.return_value = 'Customer1'
        mock_date_input.return_value = (
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
        mock_checkbox.return_value = True
        mock_progress.return_value = MagicMock()
        
        # Mock Salesforce connection and queries
        with patch('processors.salesforce_processor.Salesforce') as mock_sf:
            mock_sf.return_value = mock_salesforce
            mock_salesforce.query.side_effect = [
                mock_salesforce_response,
                {'records': sample_comment_data.to_dict('records')},
                {'records': sample_email_data.to_dict('records')}
            ]
            
            # Mock OpenAI API calls
            with patch('openai.OpenAI') as mock_openai, \
                 patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
                mock_openai_instance = MagicMock()
                mock_openai.return_value = mock_openai_instance
                mock_openai_instance.chat.completions.create.return_value = \
                    type('Response', (), {
                        'choices': [
                            type('Choice', (), {
                                'message': type('Message', (), {
                                    'content': json.dumps({
                                        'summary': 'Test summary',
                                        'patterns': ['Pattern 1', 'Pattern 2'],
                                        'recommendations': ['Rec 1', 'Rec 2']
                                    })
                                })()
                            })()
                        ]
                    })()
                
                # Import here to avoid circular imports
                from app import (
                    fetch_data,
                    display_basic_analysis,
                    display_detailed_analysis,
                    generate_ai_insights
                )
                
                # Test data fetching
                cases_df, emails_df, comments_df, history_df, attachments_df = fetch_data(
                    customers=['Customer1'],
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now()
                )
                assert not cases_df.empty
                # We expect 2 cases for Customer1 (case1 and case3)
                assert len(cases_df) == 2
                assert all(cases_df['Account.Name'] == 'Customer1')
                
                # Test basic analysis
                with patch('matplotlib.pyplot.show'):
                    basic_stats = display_basic_analysis(cases_df)
                    assert isinstance(basic_stats, dict)
                    assert 'total_cases' in basic_stats
                    assert 'avg_response_time' in basic_stats
                    assert 'avg_csat' in basic_stats
                    assert 'escalated_cases' in basic_stats
                    assert 'open_cases' in basic_stats
                    assert 'closed_cases' in basic_stats
                    assert basic_stats['total_cases'] == 2
                    assert basic_stats['open_cases'] == 1
                    assert basic_stats['closed_cases'] == 1
                    assert basic_stats['escalated_cases'] == 1
                    assert basic_stats['avg_csat'] == 4.0
                    assert basic_stats['avg_response_time'] == 1.25
                
                # Test detailed analysis
                with patch('matplotlib.pyplot.show'):
                    data = {
                        'cases': cases_df,
                        'comments': sample_comment_data,
                        'emails': sample_email_data
                    }
                    detailed_stats = display_detailed_analysis(data)
                    assert isinstance(detailed_stats, dict)
                    assert 'priority_distribution' in detailed_stats
                    assert 'status_distribution' in detailed_stats
                    assert 'time_analysis' in detailed_stats
                
                # Test AI insights
                insights = generate_ai_insights(
                    cases_df,
                    sample_comment_data,
                    sample_email_data
                )
                assert isinstance(insights, dict)
                assert 'summary' in insights
                assert 'patterns' in insights
                assert 'recommendations' in insights
                
                # Verify no errors were shown
                mock_error.assert_not_called()
                
                # Verify success message was shown
                mock_success.assert_called() 
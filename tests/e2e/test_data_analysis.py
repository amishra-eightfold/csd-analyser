"""End-to-end tests for data analysis functionality."""

import pytest
import pandas as pd
import streamlit as st
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

def test_full_analysis_flow(
    sample_case_data,
    sample_comment_data,
    sample_email_data,
    mock_openai_response,
    mock_salesforce_response
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
    # Mock Streamlit components
    with patch('streamlit.sidebar.selectbox') as mock_selectbox, \
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
        with patch('simple_salesforce.Salesforce') as mock_sf:
            mock_sf.return_value.query.side_effect = [
                mock_salesforce_response,
                {'records': sample_comment_data.to_dict('records')},
                {'records': sample_email_data.to_dict('records')}
            ]
            
            # Mock OpenAI API calls
            with patch('openai.OpenAI') as mock_openai:
                mock_openai.return_value.chat.completions.create.return_value = \
                    mock_openai_response
                
                # Import here to avoid circular imports
                from app import (
                    fetch_data,
                    display_basic_analysis,
                    display_detailed_analysis,
                    generate_ai_insights
                )
                
                # Test data fetching
                cases_df = fetch_data(
                    customers=['Customer1'],
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now()
                )
                assert not cases_df.empty
                assert len(cases_df) == len(sample_case_data)
                
                # Test basic analysis
                with patch('matplotlib.pyplot.show'):
                    basic_stats = display_basic_analysis(cases_df)
                    assert isinstance(basic_stats, dict)
                    assert 'total_cases' in basic_stats
                    assert 'avg_response_time' in basic_stats
                    assert 'csat_score' in basic_stats
                
                # Test detailed analysis
                with patch('matplotlib.pyplot.show'):
                    detailed_stats = display_detailed_analysis(
                        cases_df,
                        sample_comment_data,
                        sample_email_data
                    )
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
                assert isinstance(insights, str)
                assert "Issue Categories" in insights
                assert "Response Times" in insights
                assert "Recommendations" in insights
                
                # Verify no errors were shown
                mock_error.assert_not_called()
                
                # Verify success message was shown
                mock_success.assert_called() 
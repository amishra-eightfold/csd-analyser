"""Unit tests for the Salesforce data processor."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from processors.salesforce_processor import SalesforceDataProcessor

def test_init(mock_salesforce_auth):
    """Test processor initialization."""
    processor = SalesforceDataProcessor(mock_salesforce_auth)
    assert processor is not None
    assert processor.sf_connection == mock_salesforce_auth

def test_fetch_data(mock_salesforce_auth, mock_salesforce_response, sample_case_data):
    """Test data fetching functionality."""
    processor = SalesforceDataProcessor(mock_salesforce_auth)
    
    # Mock the execute_soql_query function
    def mock_execute_query(*args):
        return mock_salesforce_response['records']
    
    processor.sf_connection.query = mock_execute_query
    
    # Test data fetching
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    df, emails_df, comments_df, history_df, attachments_df = processor.fetch_data(
        customers=['Customer1'],
        start_date=start_date,
        end_date=end_date
    )
    
    assert df is not None
    assert not df.empty
    assert len(df) == len(mock_salesforce_response['records'])
    assert 'Id' in df.columns
    assert 'CaseNumber' in df.columns
    assert 'Status' in df.columns

def test_preprocess_data(mock_salesforce_auth, sample_case_data, sample_email_data, sample_comment_data):
    """Test data preprocessing functionality."""
    processor = SalesforceDataProcessor(mock_salesforce_auth)
    
    # Test preprocessing
    processed_df = processor._preprocess_data(
        sample_case_data,
        sample_email_data,
        sample_comment_data
    )
    
    assert processed_df is not None
    assert not processed_df.empty
    assert len(processed_df) == len(sample_case_data)
    
    # Check if email content was merged
    if not sample_email_data.empty:
        assert 'email_content' in processed_df.columns
    
    # Check if comment content was merged
    if not sample_comment_data.empty:
        assert 'comment_content' in processed_df.columns
    
    # Check if classification fields were cleaned
    for field in ['Product_Area__c', 'Product_Feature__c', 'RCA__c']:
        if field in processed_df.columns:
            assert not processed_df[field].isna().any()
            assert 'Unknown' in processed_df[field].unique()

def test_calculate_case_metrics(mock_salesforce_auth, sample_case_data):
    """Test case metrics calculation."""
    processor = SalesforceDataProcessor(mock_salesforce_auth)
    
    # Calculate metrics
    metrics = processor.calculate_case_metrics(sample_case_data)
    
    assert metrics is not None
    assert isinstance(metrics, dict)
    
    # Check basic metrics
    assert 'total_cases' in metrics
    assert metrics['total_cases'] == len(sample_case_data)
    
    assert 'open_cases' in metrics
    assert metrics['open_cases'] == len(sample_case_data[sample_case_data['Status'] == 'Open'])
    
    assert 'closed_cases' in metrics
    assert metrics['closed_cases'] == len(sample_case_data[sample_case_data['Status'] == 'Closed'])
    
    # Check CSAT metrics if available
    if 'CSAT__c' in sample_case_data.columns:
        assert 'avg_csat' in metrics
        assert 'csat_responses' in metrics
        
        csat_scores = sample_case_data['CSAT__c'].dropna()
        assert metrics['avg_csat'] == pytest.approx(csat_scores.mean(), rel=1e-10)
        assert metrics['csat_responses'] == len(csat_scores)
    
    # Check response time metrics if available
    if 'First_Response_Time__c' in sample_case_data.columns:
        assert 'avg_response_time' in metrics
        assert 'median_response_time' in metrics
        
        response_times = sample_case_data['First_Response_Time__c'].dropna()
        assert metrics['avg_response_time'] == pytest.approx(response_times.mean(), rel=1e-10)
        assert metrics['median_response_time'] == pytest.approx(response_times.median(), rel=1e-10)

def test_error_handling(mock_salesforce_auth):
    """Test error handling in the processor."""
    processor = SalesforceDataProcessor(mock_salesforce_auth)
    
    # Test with invalid DataFrame
    with pytest.raises(ValueError):
        processor.calculate_case_metrics(None)
    
    # Test with empty DataFrame
    empty_metrics = processor.calculate_case_metrics(pd.DataFrame())
    assert empty_metrics['total_cases'] == 0
    
    # Test with invalid date range
    end_date = datetime.now() - timedelta(days=30)
    start_date = datetime.now()  # Start date after end date
    
    result = processor.fetch_data(
        customers=['Customer1'],
        start_date=start_date,
        end_date=end_date
    )
    assert result == (None, None, None, None, None) 
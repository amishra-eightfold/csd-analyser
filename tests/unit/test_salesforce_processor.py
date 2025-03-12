"""Unit tests for the SalesforceDataProcessor class."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from processors.salesforce_processor import SalesforceDataProcessor

def test_init(mock_salesforce_auth, mock_salesforce):
    """Test processor initialization."""
    # Test with auth config
    with patch('processors.salesforce_processor.Salesforce') as mock_sf_class:
        mock_sf_instance = MagicMock()
        mock_sf_class.return_value = mock_sf_instance
        
        processor = SalesforceDataProcessor(mock_salesforce_auth)
        assert processor.sf_connection is mock_sf_instance
        
        # Verify Salesforce was initialized with correct parameters
        mock_sf_class.assert_called_once_with(
            username=mock_salesforce_auth['username'],
            password=mock_salesforce_auth['password'],
            security_token=mock_salesforce_auth['security_token'],
            domain=mock_salesforce_auth['domain']
        )
    
    # Test with direct connection
    processor = SalesforceDataProcessor(mock_salesforce)
    assert processor.sf_connection is mock_salesforce
    
    # Test with invalid input
    with pytest.raises(ValueError):
        SalesforceDataProcessor(None)
    
    # Test with empty dict
    with pytest.raises(ValueError):
        SalesforceDataProcessor({})

def test_fetch_data(mock_salesforce_auth, mock_salesforce, mock_salesforce_response, sample_case_data):
    """Test data fetching functionality."""
    processor = SalesforceDataProcessor(mock_salesforce)
    
    # Test successful data fetching
    mock_salesforce.query.return_value = mock_salesforce_response
    
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    df = processor.fetch_data(start_date, end_date)
    
    # Verify the query was called with correct date formatting
    expected_query = (
        "SELECT Id, CaseNumber, Subject, Status, Priority, CreatedDate, ClosedDate, "
        "Product_Area__c, Product_Feature__c, RCA__c, Internal_Priority__c, "
        "First_Response_Time__c, CSAT__c, IsEscalated, Account.Name "
        "FROM Case "
        "WHERE CreatedDate >= 2024-01-01T00:00:00Z "
        "AND CreatedDate <= 2024-12-31T23:59:59Z"
    )
    mock_salesforce.query.assert_called_with(expected_query)
    
    assert not df.empty
    assert len(df) == len(mock_salesforce_response['records'])
    assert 'Account.Name' in df.columns
    assert 'CreatedDate' in df.columns
    assert 'Status' in df.columns
    
    # Test error handling for query failure
    mock_salesforce.query.side_effect = Exception("SOQL query failed")
    with pytest.raises(ValueError, match="SOQL query failed"):
        processor.fetch_data(start_date, end_date)
    
    # Test error handling for invalid response format
    mock_salesforce.query.side_effect = None
    mock_salesforce.query.return_value = None
    with pytest.raises(ValueError, match="Invalid response format"):
        processor.fetch_data(start_date, end_date)
    
    # Test error handling for missing records field
    mock_salesforce.query.return_value = {}
    with pytest.raises(ValueError, match="No 'records' field"):
        processor.fetch_data(start_date, end_date)
    
    # Test empty records handling
    mock_salesforce.query.return_value = {'records': []}
    empty_df = processor.fetch_data(start_date, end_date)
    assert empty_df.empty

def test_preprocess_data(mock_salesforce_auth, mock_salesforce, sample_case_data, sample_email_data, sample_comment_data):
    """Test data preprocessing functionality."""
    processor = SalesforceDataProcessor(mock_salesforce)
    
    # Test preprocessing with all data
    processed_df = processor._preprocess_data(sample_case_data, sample_email_data, sample_comment_data)
    assert not processed_df.empty
    assert len(processed_df) == len(sample_case_data)
    
    # Test preprocessing with missing data
    processed_df = processor._preprocess_data(sample_case_data)
    assert not processed_df.empty
    assert len(processed_df) == len(sample_case_data)
    
    # Test error handling
    with pytest.raises(ValueError):
        processor._preprocess_data(None)
    with pytest.raises(ValueError):
        processor._preprocess_data(pd.DataFrame())

def test_calculate_case_metrics(mock_salesforce_auth, mock_salesforce, sample_case_data):
    """Test case metrics calculation."""
    processor = SalesforceDataProcessor(mock_salesforce)
    
    metrics = processor.calculate_case_metrics(sample_case_data)
    assert isinstance(metrics, dict)
    assert 'total_cases' in metrics
    assert 'open_cases' in metrics
    assert 'closed_cases' in metrics
    assert 'escalated_cases' in metrics
    assert metrics['total_cases'] == len(sample_case_data)
    
    # Test error handling
    with pytest.raises(ValueError):
        processor.calculate_case_metrics(None)
    with pytest.raises(ValueError):
        processor.calculate_case_metrics(pd.DataFrame())

def test_error_handling(mock_salesforce_auth, mock_salesforce):
    """Test error handling in the processor."""
    # Test with invalid auth config
    with pytest.raises(ValueError):
        SalesforceDataProcessor({})
    
    # Test with invalid connection
    with pytest.raises(ValueError):
        SalesforceDataProcessor(None)
    
    # Test with valid connection
    processor = SalesforceDataProcessor(mock_salesforce)
    assert processor.sf_connection is not None
    
    # Test fetch_data error handling
    mock_salesforce.query.side_effect = Exception("Test error")
    with pytest.raises(ValueError):
        processor.fetch_data("2023-01-01", "2023-12-31") 
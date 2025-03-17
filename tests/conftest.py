"""Test fixtures for the CSD Analyser tests."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock

@pytest.fixture
def sample_case_data():
    """Create sample case data for testing."""
    return pd.DataFrame({
        'Id': ['case1', 'case2'],
        'CaseNumber': ['00001', '00002'],
        'Subject': ['Test Case 1', 'Test Case 2'],
        'Description': ['Description 1', 'Description 2'],
        'Account_Name': ['Customer1', 'Customer1'],
        'Created Date': [
            datetime.now() - timedelta(days=5),
            datetime.now() - timedelta(days=3)
        ],
        'Closed Date': [
            datetime.now() - timedelta(days=2),
            None
        ],
        'Status': ['Closed', 'Open'],
        'Priority': ['P1', 'P2'],
        'Product Area': ['Area1', 'Area2'],
        'Product Feature': ['Feature1', 'Feature2'],
        'Root Cause': ['Bug', 'Not Specified'],
        'First Response Time': [
            datetime.now() - timedelta(days=4),
            datetime.now() - timedelta(days=2)
        ],
        'CSAT': [4.0, None],
        'IsEscalated': [True, False],
        'Resolution Time (Days)': [3.0, None]
    })

@pytest.fixture
def sample_comment_data():
    """Create sample comment data for testing."""
    return pd.DataFrame({
        'Id': ['comment1', 'comment2'],
        'ParentId': ['case1', 'case2'],
        'CommentBody': ['Test comment 1', 'Test comment 2'],
        'CreatedDate': [
            datetime.now() - timedelta(days=4),
            datetime.now() - timedelta(days=2)
        ]
    })

@pytest.fixture
def sample_email_data():
    """Create sample email data for testing."""
    return pd.DataFrame({
        'Id': ['email1', 'email2'],
        'ParentId': ['case1', 'case2'],
        'Subject': ['RE: Test Case 1', 'RE: Test Case 2'],
        'TextBody': ['Email body 1', 'Email body 2'],
        'CreatedDate': [
            datetime.now() - timedelta(days=4),
            datetime.now() - timedelta(days=2)
        ]
    })

@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    return {
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
    }

@pytest.fixture
def mock_salesforce_response():
    """Create a mock Salesforce API response."""
    return {
        'records': [
            {
                'Id': 'case1',
                'CaseNumber': '00001',
                'Subject': 'Test Case 1',
                'Description': 'Description 1',
                'Account': {'Account_Name__c': 'Customer1'},
                'CreatedDate': (datetime.now() - timedelta(days=5)).isoformat(),
                'ClosedDate': (datetime.now() - timedelta(days=2)).isoformat(),
                'Status': 'Closed',
                'Internal_Priority__c': 'P1',
                'Product_Area__c': 'Area1',
                'Product_Feature__c': 'Feature1',
                'RCA__c': 'Bug',
                'First_Response_Time__c': (datetime.now() - timedelta(days=4)).isoformat(),
                'CSAT__c': 4.0,
                'IsEscalated': True
            },
            {
                'Id': 'case2',
                'CaseNumber': '00002',
                'Subject': 'Test Case 2',
                'Description': 'Description 2',
                'Account': {'Account_Name__c': 'Customer1'},
                'CreatedDate': (datetime.now() - timedelta(days=3)).isoformat(),
                'ClosedDate': None,
                'Status': 'Open',
                'Internal_Priority__c': 'P2',
                'Product_Area__c': 'Area2',
                'Product_Feature__c': 'Feature2',
                'RCA__c': 'Not Specified',
                'First_Response_Time__c': (datetime.now() - timedelta(days=2)).isoformat(),
                'CSAT__c': None,
                'IsEscalated': False
            }
        ]
    }

@pytest.fixture
def mock_salesforce():
    """Create a mock Salesforce connection."""
    mock = MagicMock()
    mock.query.return_value = mock_salesforce_response
    return mock

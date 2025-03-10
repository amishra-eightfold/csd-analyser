"""Test configuration and fixtures for the CSD Analyzer application."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_case_data():
    """Create sample case data for testing."""
    return pd.DataFrame({
        'Id': ['case1', 'case2', 'case3'],
        'CaseNumber': ['C-001', 'C-002', 'C-003'],
        'Subject': ['Issue 1', 'Issue 2', 'Issue 3'],
        'Description': ['Problem with feature A', 'Error in module B', 'Question about C'],
        'Account.Name': ['Customer1', 'Customer2', 'Customer1'],
        'CreatedDate': [
            datetime.now() - timedelta(days=5),
            datetime.now() - timedelta(days=3),
            datetime.now() - timedelta(days=1)
        ],
        'ClosedDate': [
            datetime.now() - timedelta(days=4),
            datetime.now() - timedelta(days=2),
            None
        ],
        'Status': ['Closed', 'Closed', 'Open'],
        'Internal_Priority__c': ['P1', 'P2', 'P1'],
        'Product_Area__c': ['Area1', 'Area2', 'Area1'],
        'Product_Feature__c': ['Feature1', 'Feature2', 'Feature1'],
        'RCA__c': ['Bug', 'Configuration', 'Documentation'],
        'First_Response_Time__c': [1.5, 2.0, 1.0],
        'CSAT__c': [5, 4, None],
        'IsEscalated': [False, True, False]
    })

@pytest.fixture
def sample_comment_data():
    """Create sample comment data for testing."""
    return pd.DataFrame({
        'Id': ['comment1', 'comment2', 'comment3'],
        'ParentId': ['case1', 'case1', 'case2'],
        'CommentBody': [
            'Initial response',
            'Follow up',
            'Resolution details'
        ],
        'CreatedDate': [
            datetime.now() - timedelta(days=5),
            datetime.now() - timedelta(days=4),
            datetime.now() - timedelta(days=2)
        ],
        'CreatedById': ['user1', 'user1', 'user2']
    })

@pytest.fixture
def sample_email_data():
    """Create sample email data for testing."""
    return pd.DataFrame({
        'Id': ['email1', 'email2', 'email3'],
        'ParentId': ['case1', 'case2', 'case3'],
        'Subject': ['RE: Issue 1', 'RE: Issue 2', 'RE: Issue 3'],
        'TextBody': [
            'Here is the solution',
            'Need more information',
            'Let me check'
        ],
        'FromAddress': ['support@company.com'] * 3,
        'ToAddress': ['customer1@email.com', 'customer2@email.com', 'customer1@email.com'],
        'MessageDate': [
            datetime.now() - timedelta(days=5),
            datetime.now() - timedelta(days=3),
            datetime.now() - timedelta(days=1)
        ],
        'Status': ['Sent', 'Sent', 'Sent'],
        'HasAttachment': [False, True, False]
    })

@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    return {
        'choices': [{
            'message': {
                'content': """Here are the key insights from the support tickets:

1. Issue Categories:
   - 33% Bug-related issues
   - 33% Configuration issues
   - 33% Documentation-related queries

2. Response Times:
   - Average first response time: 1.5 hours
   - 67% of cases resolved within 24 hours

3. Customer Satisfaction:
   - Average CSAT score: 4.5/5
   - 100% resolution rate for closed cases

4. Priority Distribution:
   - 67% P1 (high priority) cases
   - 33% P2 (medium priority) cases

5. Areas of Focus:
   - Area1 shows highest case volume (67%)
   - Feature1 most frequently referenced

Recommendations:
1. Review documentation for Feature1
2. Investigate bug patterns in Area1
3. Maintain strong response times
4. Continue monitoring P1 case volume"""
            }
        }]
    }

@pytest.fixture
def mock_salesforce_auth():
    """Create mock Salesforce authentication credentials."""
    return {
        'username': 'test@example.com',
        'password': 'test_password',
        'security_token': 'test_token',
        'client_id': 'test_client_id',
        'client_secret': 'test_client_secret'
    }

@pytest.fixture
def mock_salesforce_response():
    """Create a mock Salesforce API response."""
    return {
        'totalSize': 3,
        'done': True,
        'records': [
            {
                'Id': 'case1',
                'CaseNumber': 'C-001',
                'Subject': 'Issue 1',
                'Status': 'Closed'
            },
            {
                'Id': 'case2',
                'CaseNumber': 'C-002',
                'Subject': 'Issue 2',
                'Status': 'Closed'
            },
            {
                'Id': 'case3',
                'CaseNumber': 'C-003',
                'Subject': 'Issue 3',
                'Status': 'Open'
            }
        ]
    }

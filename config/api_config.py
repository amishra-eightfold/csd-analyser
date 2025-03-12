"""Configuration settings for API interactions."""

import os
from typing import Dict, Any

# OpenAI API settings
OPENAI_SETTINGS = {
    'model': 'gpt-4',
    'temperature': 0.7,
    'max_tokens': 2000,
    'top_p': 1.0,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}

# Salesforce API settings
SALESFORCE_SETTINGS = {
    'version': '57.0',
    'domain': 'login',
    'timeout': 30,
    'retry_count': 3,
    'retry_delay': 1.0,
    'batch_size': 2000
}

# API endpoints
API_ENDPOINTS = {
    'salesforce': {
        'base_url': 'https://{domain}.salesforce.com',
        'auth_url': '/services/oauth2/token',
        'query_url': '/services/data/v{version}/query',
        'bulk_url': '/services/data/v{version}/jobs/query'
    }
}

# Query templates
QUERY_TEMPLATES = {
    'cases': """
        SELECT 
            Id, CaseNumber, Subject, Description,
            Account.Name, CreatedDate, ClosedDate, Status, Internal_Priority__c,
            Product_Area__c, Product_Feature__c, RCA__c,
            First_Response_Time__c, CSAT__c, IsEscalated
        FROM Case
        WHERE Account.Name IN ({customers})
        AND CreatedDate >= {start_date}T00:00:00Z
        AND CreatedDate <= {end_date}T23:59:59Z
    """,
    
    'comments': """
        SELECT 
            Id, ParentId, CommentBody, CreatedDate, CreatedById
        FROM CaseComment
        WHERE ParentId IN ({case_ids})
        ORDER BY CreatedDate ASC
    """,
    
    'emails': """
        SELECT 
            Id, ParentId, Subject, TextBody, HtmlBody, FromAddress, ToAddress,
            CcAddress, BccAddress, MessageDate, Status, HasAttachment
        FROM EmailMessage
        WHERE ParentId IN ({case_ids})
        ORDER BY MessageDate ASC
    """,
    
    'history': """
        SELECT 
            Id, CaseId, Field, OldValue, NewValue, CreatedDate, CreatedById
        FROM CaseHistory
        WHERE CaseId IN ({case_ids})
        ORDER BY CreatedDate ASC
    """
}

# Error messages
ERROR_MESSAGES = {
    'api_key_missing': "API key not found. Please add it to your environment variables or configuration.",
    'auth_failed': "Authentication failed. Please check your credentials.",
    'rate_limit': "Rate limit exceeded. Please try again later.",
    'timeout': "Request timed out. Please try again.",
    'invalid_response': "Invalid response received from the API.",
    'query_error': "Error executing query: {error}",
    'connection_error': "Connection error: {error}"
}

# Rate limiting settings
RATE_LIMITS = {
    'openai': {
        'requests_per_minute': 60,
        'tokens_per_minute': 90000
    },
    'salesforce': {
        'requests_per_day': 15000,
        'batch_size': 2000,
        'concurrent_requests': 5
    }
}

# Retry settings
RETRY_SETTINGS = {
    'max_retries': 3,
    'retry_delay': 1.0,
    'retry_errors': [
        'ConnectionError',
        'Timeout',
        'RateLimitError',
        'ServerError'
    ]
}

# Authentication settings
def get_auth_settings() -> Dict[str, Any]:
    """
    Get authentication settings from environment variables.
    
    Returns:
        Dict[str, Any]: Dictionary of authentication settings
    """
    return {
        'openai': {
            'api_key': os.getenv('OPENAI_API_KEY')
        },
        'salesforce': {
            'username': os.getenv('SALESFORCE_USERNAME'),
            'password': os.getenv('SALESFORCE_PASSWORD'),
            'security_token': os.getenv('SALESFORCE_SECURITY_TOKEN'),
            'client_id': os.getenv('SALESFORCE_CLIENT_ID'),
            'client_secret': os.getenv('SALESFORCE_CLIENT_SECRET')
        }
    }

# Cache settings
CACHE_SETTINGS = {
    'enabled': True,
    'ttl': 3600,  # 1 hour
    'max_size': 1000,  # Maximum number of items to cache
    'excluded_endpoints': [
        '/auth',
        '/logout'
    ]
}

# Logging settings
LOGGING_SETTINGS = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'api.log',
    'max_size': 10485760,  # 10MB
    'backup_count': 5
}

# Timeout settings
TIMEOUT_SETTINGS = {
    'connect': 5,  # Connection timeout in seconds
    'read': 30,    # Read timeout in seconds
    'total': 35    # Total timeout in seconds
} 
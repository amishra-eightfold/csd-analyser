"""Text processing utilities for cleaning and PII removal."""

import re
from typing import Optional, Union, List, Dict

def clean_text(text: str) -> str:
    """Clean text by removing URLs, special characters, and extra whitespace.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ''
        
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '', text)
    
    # Remove browser agent strings
    text = re.sub(r'Mozilla/[\d\.]+ \(.*?\)', '', text)
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    
    return text.lower()

def remove_pii(text: str) -> str:
    """Remove personally identifiable information (PII) from text.
    
    Args:
        text (str): Text to process
        
    Returns:
        str: Text with PII removed
    """
    if not isinstance(text, str):
        return ''
        
    # Initialize cleaned text
    cleaned_text = text
    
    # Remove email addresses
    cleaned_text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', cleaned_text)
    
    # Remove phone numbers (various formats)
    phone_patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Standard US format
        r'\+\d{1,3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{4}\b',  # International format
        r'\(\d{3}\)\s*\d{3}[-\s]?\d{4}\b',  # (123) 456-7890
        r'\b\d{10}\b'  # Plain 10 digits
    ]
    for pattern in phone_patterns:
        cleaned_text = re.sub(pattern, '[PHONE]', cleaned_text)
    
    # Remove URLs
    cleaned_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', cleaned_text)
    
    # Remove IP addresses
    cleaned_text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP_ADDRESS]', cleaned_text)
    
    # Remove credit card numbers
    cleaned_text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[CREDIT_CARD]', cleaned_text)
    
    # Remove social security numbers
    cleaned_text = re.sub(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', '[SSN]', cleaned_text)
    
    # Remove names (common name patterns)
    name_patterns = [
        r'(?i)(?:mr\.|mrs\.|ms\.|dr\.|prof\.)\s+[a-z]+',  # Titles with names
        r'(?i)(?:first|last|full)\s+name\s*(?::|is|=)\s*[a-z\s]+',  # Name declarations
        r'(?i)sincerely,\s+[a-z\s]+',  # Email signatures
        r'(?i)regards,\s+[a-z\s]+',  # Email signatures
        r'(?i)best,\s+[a-z\s]+'  # Email signatures
    ]
    for pattern in name_patterns:
        cleaned_text = re.sub(pattern, '[NAME]', cleaned_text)
    
    # Remove common password patterns
    cleaned_text = re.sub(r'(?i)password\s*(?::|is|=)\s*\S+', '[PASSWORD]', cleaned_text)
    
    # Remove dates of birth
    dob_patterns = [
        r'\b\d{2}[-/]\d{2}[-/]\d{4}\b',  # MM/DD/YYYY or DD/MM/YYYY
        r'\b\d{4}[-/]\d{2}[-/]\d{2}\b',  # YYYY/MM/DD
        r'(?i)(?:date\s+of\s+birth|dob|birth\s+date)\s*(?::|is|=)\s*[a-z0-9\s,]+' # DOB declarations
    ]
    for pattern in dob_patterns:
        cleaned_text = re.sub(pattern, '[DOB]', cleaned_text)
    
    return cleaned_text

def prepare_text_for_ai(data: Union[str, List, Dict, 'pd.DataFrame']) -> Union[str, List, Dict, 'pd.DataFrame']:
    """Prepare text data for AI analysis by removing PII.
    
    Args:
        data: Input data (can be string, list, dict, or DataFrame)
        
    Returns:
        Cleaned data with same structure as input
    """
    if isinstance(data, dict):
        # Handle dictionary input
        cleaned_data = {}
        for key, value in data.items():
            if isinstance(value, (str, list, dict)):
                cleaned_data[key] = prepare_text_for_ai(value)
            else:
                cleaned_data[key] = value
        return cleaned_data
    elif isinstance(data, list):
        # Handle list input
        return [prepare_text_for_ai(item) for item in data]
    elif isinstance(data, str):
        # Handle string input
        return remove_pii(data)
    elif str(type(data)).endswith("'pandas.core.frame.DataFrame'>"):
        # Handle DataFrame input
        cleaned_df = data.copy()
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].apply(remove_pii)
        return cleaned_df
    else:
        # Return as is for other types
        return data

def get_technical_stopwords() -> set:
    """Get set of technical stopwords for text analysis.
    
    Returns:
        set: Set of technical stopwords
    """
    return {
        'eightfold', 'mailto', 'chrome', 'firefox', 'safari', 'edge', 'opera',
        'webkit', 'mozilla', 'browser', 'agent', 'http', 'https', 'www',
        'com', 'net', 'org', 'html', 'htm', 'php', 'asp', 'aspx',
        'user-agent', 'useragent', 'version', 'windows', 'macintosh', 'linux',
        'unix', 'android', 'ios', 'mobile', 'desktop', 'platform',
        'application', 'software', 'browser-agent', 'browseragent'
    }

def get_highest_priority_from_history(sf_connection, case_id: str) -> str:
    """Get the highest priority a case reached during its lifecycle.
    
    Args:
        sf_connection: Salesforce connection object
        case_id (str): The ID of the case to check
        
    Returns:
        str: The highest priority the case reached (P0, P1, P2, P3, or original priority if no changes)
    """
    from salesforce_config import execute_soql_query
    
    # Priority order from lowest to highest
    priority_order = {'P3': 0, 'P2': 1, 'P1': 2, 'P0': 3}
    
    # Query case history for priority changes
    query = f"""
        SELECT Id, CaseId, Field, NewValue, OldValue, CreatedDate 
        FROM CaseHistory 
        WHERE Field = 'Internal_Priority__c' 
        AND CaseId = '{case_id}'
        ORDER BY CreatedDate ASC
    """
    
    try:
        history_records = execute_soql_query(sf_connection, query)
        if not history_records:
            return None
            
        # Collect all priority values (both old and new)
        priorities = []
        for record in history_records:
            if record.get('OldValue'):
                priorities.append(record['OldValue'])
            if record.get('NewValue'):
                priorities.append(record['NewValue'])
                
        # Filter valid priorities and get the highest
        valid_priorities = [p for p in priorities if p in priority_order]
        if valid_priorities:
            return max(valid_priorities, key=lambda x: priority_order.get(x, -1))
            
        return None
            
    except Exception as e:
        print(f"Error querying case history: {str(e)}")
        return None 
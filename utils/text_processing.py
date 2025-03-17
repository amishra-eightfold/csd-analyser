"""Text processing utilities for cleaning and PII removal."""

import re
from typing import Optional, Union, List, Dict, Set
import pandas as pd
from simple_salesforce import Salesforce
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import numpy as np

# Initialize NLTK resources
def initialize_nltk():
    """Initialize required NLTK resources."""
    resources = [
        'punkt',
        'stopwords',
        'averaged_perceptron_tagger'
    ]
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

# Call initialization on module import
initialize_nltk()

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

def remove_stopwords(text: str, custom_stopwords: Optional[Set[str]] = None) -> str:
    """Remove stopwords from text to reduce token usage.
    
    Args:
        text (str): Text to process
        custom_stopwords (Set[str], optional): Additional custom stopwords
        
    Returns:
        str: Text with stopwords removed
    """
    if not isinstance(text, str) or not text.strip():
        return ''
    
    # Get standard stopwords
    stop_words = set(stopwords.words('english'))
    
    # Add technical stopwords
    stop_words.update(get_technical_stopwords())
    
    # Add custom stopwords if provided
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    
    # Tokenize and filter
    word_tokens = word_tokenize(text.lower())
    filtered_text = ' '.join([word for word in word_tokens if word.lower() not in stop_words and len(word) > 1])
    
    return filtered_text

def prepare_text_for_ai(data: Union[str, List, Dict, 'pd.DataFrame'], remove_stops: bool = True) -> Union[str, List, Dict, 'pd.DataFrame']:
    """Prepare text data for AI analysis by removing PII and optionally stopwords.
    
    Args:
        data: Input data (can be string, list, dict, or DataFrame)
        remove_stops: Whether to remove stopwords (default: True)
        
    Returns:
        Cleaned data with same structure as input
    """
    if isinstance(data, dict):
        # Handle dictionary input
        cleaned_data = {}
        for key, value in data.items():
            if isinstance(value, (str, list, dict)):
                cleaned_data[key] = prepare_text_for_ai(value, remove_stops)
            else:
                cleaned_data[key] = value
        return cleaned_data
    elif isinstance(data, list):
        # Handle list input
        return [prepare_text_for_ai(item, remove_stops) for item in data]
    elif isinstance(data, str):
        # Handle string input
        text = remove_pii(data)
        if remove_stops:
            # Only remove stopwords from longer text fields to preserve meaning in short fields
            if len(text.split()) > 5:
                return remove_stopwords(text)
        return text
    elif str(type(data)).endswith("'pandas.core.frame.DataFrame'>"):
        # Handle DataFrame input
        cleaned_df = data.copy()
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        
        # Text fields likely to contain important descriptive content
        descriptive_columns = ['Subject', 'Description', 'Comments', 'Notes', 'Root Cause']
        
        for col in text_columns:
            # Apply PII removal to all text columns
            cleaned_df[col] = cleaned_df[col].apply(remove_pii)
            
            # Apply stopword removal only to descriptive columns with remove_stops enabled
            if remove_stops and col in descriptive_columns:
                cleaned_df[col] = cleaned_df[col].apply(lambda x: remove_stopwords(x) if isinstance(x, str) and len(str(x).split()) > 5 else x)
        
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

def get_highest_priority_from_history(sf_connection: Salesforce, case_id: str) -> Optional[str]:
    """Get the highest priority a case reached during its lifecycle.
    
    Args:
        sf_connection (Salesforce): Salesforce connection object
        case_id (str): The ID of the case to check
        
    Returns:
        Optional[str]: The highest priority the case reached (P0, P1, P2, P3), or None if no valid priority found
    """
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
        # Execute query directly using sf_connection
        history_records = sf_connection.query(query)['records']
        print(f"Found {len(history_records)} priority history records for case {case_id}")
        
        if not history_records:
            print(f"No priority history found for case {case_id}")
            return None
            
        # Collect all priority values (both old and new)
        priorities = []
        for record in history_records:
            if record.get('OldValue'):
                priorities.append(record['OldValue'])
                print(f"Case {case_id}: Found old priority value: {record['OldValue']}")
            if record.get('NewValue'):
                priorities.append(record['NewValue'])
                print(f"Case {case_id}: Found new priority value: {record['NewValue']}")
                
        # Filter valid priorities and get the highest
        valid_priorities = [p for p in priorities if p in priority_order]
        print(f"Case {case_id}: Valid priorities found: {valid_priorities}")
        
        if valid_priorities:
            highest_priority = max(valid_priorities, key=lambda x: priority_order.get(x, -1))
            print(f"Case {case_id}: Highest priority determined: {highest_priority}")
            return highest_priority
            
        print(f"Case {case_id}: No valid priorities found in history")
        return None
            
    except Exception as e:
        print(f"Error querying case history for case {case_id}: {str(e)}")
        return None
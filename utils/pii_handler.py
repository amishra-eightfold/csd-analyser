"""Privacy and PII Protection module for Support Ticket Analysis application."""

import re
import json
import logging
import datetime
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd
import hashlib
from pathlib import Path

# Configure logging for PII audit trail
logging.basicConfig(level=logging.INFO)
pii_logger = logging.getLogger('pii_audit')
pii_logger.setLevel(logging.INFO)

# Create a file handler
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
file_handler = logging.FileHandler(log_dir / 'pii_audit.log')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
pii_logger.addHandler(file_handler)

class PIIHandler:
    """Handles PII detection, removal, validation, and audit logging."""
    
    # PII Patterns
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone_us': r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        'phone_intl': r'\b\+?[1-9][0-9]{7,14}\b',
        'ssn': r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
        'credit_card': r'\b\d{4}[-. ]?\d{4}[-. ]?\d{4}[-. ]?\d{4}\b',
        'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        'url': r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*',
        'date_of_birth': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
        'name_with_title': r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
        'password': r'\b(?i)password[\s:]+\S+\b',
        'address': r'\b\d{1,5}\s+[A-Za-z\s]{1,30}\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Court|Ct|Circle|Cir|Trail|Trl|Parkway|Pkwy|Place|Pl)\b',
    }
    
    # Replacement templates
    REPLACEMENTS = {
        'email': '[EMAIL]',
        'phone_us': '[PHONE]',
        'phone_intl': '[PHONE]',
        'ssn': '[SSN]',
        'credit_card': '[CREDIT_CARD]',
        'ip_address': '[IP_ADDRESS]',
        'url': '[URL]',
        'date_of_birth': '[DOB]',
        'name_with_title': '[NAME]',
        'password': '[PASSWORD]',
        'address': '[ADDRESS]',
    }
    
    def __init__(self):
        """Initialize the PII handler."""
        self.pii_stats = {
            'total_detected': 0,
            'by_type': {},
            'validation_score': 0.0
        }
        self.audit_records = []
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text and return matches by type.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of PII type to list of matches
        """
        if not isinstance(text, str):
            return {}
            
        matches = {}
        for pii_type, pattern in self.PII_PATTERNS.items():
            found = re.findall(pattern, text)
            if found:
                matches[pii_type] = found
                self.pii_stats['total_detected'] += len(found)
                self.pii_stats['by_type'][pii_type] = self.pii_stats['by_type'].get(pii_type, 0) + len(found)
        
        return matches
    
    def remove_pii(self, text: str, audit: bool = True) -> Tuple[str, Dict]:
        """Remove PII from text and optionally log the operation.
        
        Args:
            text: Text to process
            audit: Whether to create audit record
            
        Returns:
            Tuple of (processed text, detection statistics)
        """
        if not isinstance(text, str):
            return str(text), {}
            
        processed_text = text
        stats = {'detected': 0, 'by_type': {}}
        
        # Create a hash of the original text for audit purposes
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, processed_text)
            if matches:
                stats['detected'] += len(matches)
                stats['by_type'][pii_type] = len(matches)
                
                # Replace each match with the corresponding template
                for match in matches:
                    processed_text = processed_text.replace(match, self.REPLACEMENTS[pii_type])
        
        if audit and stats['detected'] > 0:
            audit_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'text_hash': text_hash,
                'stats': stats
            }
            self.audit_records.append(audit_record)
            pii_logger.info(f"PII Removal - Hash: {text_hash}, "
                          f"Detected: {stats['detected']}, "
                          f"Types: {list(stats['by_type'].keys())}")
        
        return processed_text, stats
    
    def validate_pii_removal(self, processed_text: str) -> float:
        """Validate effectiveness of PII removal.
        
        Args:
            processed_text: Text to validate
            
        Returns:
            Validation score between 0 and 1
        """
        if not isinstance(processed_text, str):
            return 1.0
            
        total_potential_pii = 0
        remaining_potential_pii = 0
        
        for pattern in self.PII_PATTERNS.values():
            matches = re.findall(pattern, processed_text)
            remaining_potential_pii += len(matches)
        
        # Check for any replacement templates
        for template in self.REPLACEMENTS.values():
            total_potential_pii += processed_text.count(template)
        
        if total_potential_pii == 0:
            return 1.0
            
        score = 1 - (remaining_potential_pii / (total_potential_pii + remaining_potential_pii))
        self.pii_stats['validation_score'] = score
        return score
    
    def process_dataframe(self, df: pd.DataFrame, text_columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """Process PII in a DataFrame's text columns.
        
        Args:
            df: DataFrame to process
            text_columns: List of column names containing text
            
        Returns:
            Tuple of (processed DataFrame, processing statistics)
        """
        if df.empty or not text_columns:
            return df.copy(), {}
            
        processed_df = df.copy()
        stats = {
            'total_rows': len(df),
            'processed_columns': text_columns,
            'pii_detected': 0,
            'by_column': {}
        }
        
        for column in text_columns:
            if column in processed_df.columns:
                column_stats = {'detected': 0, 'by_type': {}}
                
                # Process each cell in the column
                for idx, value in processed_df[column].items():
                    # Remove PII and get stats
                    processed_text, cell_stats = self.remove_pii(str(value), audit=True)
                    
                    # Update column statistics
                    column_stats['detected'] += cell_stats['detected']
                    for pii_type, count in cell_stats['by_type'].items():
                        if pii_type not in column_stats['by_type']:
                            column_stats['by_type'][pii_type] = 0
                        column_stats['by_type'][pii_type] += count
                    
                    # Calculate validation score for the processed text
                    validation_score = self.validate_pii_removal(processed_text)
                    self.pii_stats['validation_score'] = min(
                        self.pii_stats['validation_score'] or 1.0,
                        validation_score
                    )
                    
                    # Update the cell with processed text
                    processed_df.at[idx, column] = processed_text
                
                # Update overall statistics
                stats['pii_detected'] += column_stats['detected']
                stats['by_column'][column] = column_stats
        
        return processed_df, stats
    
    def get_audit_summary(self) -> Dict:
        """Get summary of PII detection and removal operations.
        
        Returns:
            Dictionary containing audit summary
        """
        return {
            'total_operations': len(self.audit_records),
            'total_pii_detected': self.pii_stats['total_detected'],
            'by_type': self.pii_stats['by_type'],
            'latest_validation_score': self.pii_stats['validation_score'],
            'timestamp': datetime.datetime.now().isoformat()
        }

def get_privacy_status_indicator(validation_score: float) -> Dict[str, str]:
    """Get privacy status indicator based on validation score.
    
    Args:
        validation_score: PII removal validation score
        
    Returns:
        Dictionary containing status information
    """
    # Handle None or 0.0 validation score (initial state)
    if validation_score is None or validation_score == 0.0:
        return {
            'status': 'Not Started',
            'color': 'grey',
            'message': 'PII analysis has not been performed yet'
        }
    
    if validation_score >= 0.99:
        return {
            'status': 'Excellent',
            'color': 'green',
            'message': 'PII protection is excellent'
        }
    elif validation_score >= 0.95:
        return {
            'status': 'Good',
            'color': 'blue',
            'message': 'PII protection is good'
        }
    elif validation_score >= 0.90:
        return {
            'status': 'Fair',
            'color': 'orange',
            'message': 'PII protection needs improvement'
        }
    else:
        return {
            'status': 'Poor',
            'color': 'red',
            'message': 'PII protection requires immediate attention'
        } 
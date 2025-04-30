"""Priority handling utilities for support ticket analysis."""

import pandas as pd
import traceback
from typing import Dict, Any, Optional
import logging
import streamlit as st

# Import logging configuration
from config.logging_config import get_logger

# Initialize logger
logger = get_logger('priority')

def debug(message: str, data: Optional[Dict[str, Any]] = None, category: str = "priority") -> None:
    """Log debug information to the debug logger and file logger.
    
    Args:
        message: The message to log
        data: Optional dictionary of data to include with the message
        category: The log category to use
    """
    if hasattr(st.session_state, 'debug_logger'):
        st.session_state.debug_logger.log(message, data, category)
    
    # Log to file logger
    current_logger = get_logger(category)
    if data is not None:
        try:
            import json
            current_logger.info(f"{message} - {json.dumps(data)}")
        except:
            current_logger.info(f"{message} - {str(data)}")
    else:
        current_logger.info(message)

def get_highest_priority(case_id: str, history_df: pd.DataFrame, current_priority: str) -> str:
    """Get the highest priority from history data.
    
    Analyzes the case history to determine the highest priority that was
    ever assigned to a case, which can be different from the current priority.
    
    Args:
        case_id: The ID of the case to analyze
        history_df: DataFrame containing case history records
        current_priority: The current priority of the case
        
    Returns:
        The highest priority found in history or the current priority
        if no higher priority is found
    """
    try:
        # Initialize DataFrames and dictionaries
        highest_priorities = {}
        case_history = history_df[history_df['CaseId'] == case_id]
        
        # Calculate highest priority for each case
        for _, row in case_history.iterrows():
            if row['Field'] == 'Internal_Priority__c':
                # Check if this priority is higher (lower number is higher priority)
                # P1 > P2 > P3 > P4
                if row['NewValue'] != current_priority:
                    current_value = highest_priorities.get(case_id, current_priority)
                    
                    # Simple priority comparison, assuming format is 'P1', 'P2', etc.
                    # Lower number means higher priority
                    try:
                        new_priority_num = int(row['NewValue'].replace('P', ''))
                        current_priority_num = int(current_value.replace('P', ''))
                        
                        if new_priority_num < current_priority_num:
                            highest_priorities[case_id] = row['NewValue']
                    except (ValueError, AttributeError):
                        # If priority format is not as expected, just use the current value
                        continue
        
        return highest_priorities.get(case_id, current_priority)
    except Exception as e:
        error_msg = f"Error processing priority for case {case_id}: {str(e)}"
        debug(error_msg, {'traceback': traceback.format_exc()}, category="error")
        logger.error(error_msg, exc_info=True)
        return current_priority

def calculate_priority_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Calculate the distribution of priorities in the dataset.
    
    Args:
        df: DataFrame containing ticket data with a 'Highest_Priority' column
        
    Returns:
        Dictionary mapping priority values to counts
    """
    if 'Highest_Priority' not in df.columns:
        return {}
    
    try:
        priority_counts = df['Highest_Priority'].value_counts().to_dict()
        return priority_counts
    except Exception as e:
        error_msg = f"Error calculating priority distribution: {str(e)}"
        debug(error_msg, {'traceback': traceback.format_exc()}, category="error")
        logger.error(error_msg, exc_info=True)
        return {} 
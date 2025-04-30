"""Data fetching module for retrieving data from Salesforce."""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import traceback
from datetime import datetime

# Import custom modules
from salesforce_config import execute_soql_query
from config.logging_config import get_logger

# Initialize logger
logger = get_logger('data_fetcher')

def debug(message, data=None, category="app"):
    """Log debug information to the logger."""
    if hasattr(st.session_state, 'debug_logger'):
        st.session_state.debug_logger.log(message, data, category)
    
    # Log to file logger
    logger = get_logger(category)
    if data is not None:
        # Convert data to string if needed for logging
        if isinstance(data, dict):
            try:
                import json
                # Convert NumPy types to Python native types
                sanitized_data = {}
                for k, v in data.items():
                    if hasattr(v, 'dtype'):  # Check if it's a NumPy type
                        if np.issubdtype(v.dtype, np.integer):
                            sanitized_data[k] = int(v)
                        elif np.issubdtype(v.dtype, np.floating):
                            sanitized_data[k] = float(v)
                        elif np.issubdtype(v.dtype, np.bool_):
                            sanitized_data[k] = bool(v)
                        else:
                            sanitized_data[k] = str(v)
                    else:
                        sanitized_data[k] = v
                logger.info(f"{message} - {json.dumps(sanitized_data)}")
            except:
                logger.info(f"{message} - {str(data)}")
        else:
            logger.info(f"{message} - {str(data)}")
    else:
        logger.info(message)

def fetch_data() -> Optional[pd.DataFrame]:
    """Fetch data from Salesforce based on session state settings.
    
    Returns:
        DataFrame containing the fetched data or None if no data was found or 
        an error occurred.
    """
    try:
        if not st.session_state.selected_customers:
            st.warning("No customers selected")
            debug("No customers selected for data fetch", category="app")
            return None
            
        customer_list = "'" + "','".join(st.session_state.selected_customers) + "'"
        start_date, end_date = st.session_state.date_range
        
        debug("Fetching data with parameters", {
            'customers': st.session_state.selected_customers,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        })
        
        # Main case query
        query = f"""
            SELECT 
                Id, CaseNumber, Subject, Description,
                Account.Account_Name__c, CreatedDate, ClosedDate, Status, Internal_Priority__c,
                Product_Area__c, Product_Feature__c, RCA__c,
                First_Response_Time__c, CSAT__c, IsEscalated, Case_Type__c, Type
            FROM Case
            WHERE Account.Account_Name__c IN ({customer_list})
            AND CreatedDate >= {start_date.strftime('%Y-%m-%d')}T00:00:00Z
            AND CreatedDate <= {end_date.strftime('%Y-%m-%d')}T23:59:59Z
            AND Is_Case_L1_Triaged__c = false
            AND RecordTypeId = '0123m000000U8CCAA0'
        """
        
        debug("Executing SOQL query", {'query': query}, category="api")
        
        records = execute_soql_query(st.session_state.sf_connection, query)
        if not records:
            st.warning("No data found for the selected criteria")
            debug("No records returned from query", category="app")
            return pd.DataFrame()
            
        df = pd.DataFrame(records)
        
        # Extract Account Name from nested structure
        if 'Account' in df.columns and isinstance(df['Account'].iloc[0], dict):
            df['Account_Name'] = df['Account'].apply(lambda x: x.get('Account_Name__c') if isinstance(x, dict) else None)
            df = df.drop('Account', axis=1)
        
        # Fetch case history for priority tracking
        try:
            case_ids = "'" + "','".join(df['Id'].tolist()) + "'"
            history_query = f"""
                SELECT Id, CaseId, Field, OldValue, NewValue, CreatedDate
                FROM CaseHistory
                WHERE CaseId IN ({case_ids})
                AND Field = 'Internal_Priority__c'
                ORDER BY CreatedDate ASC
            """
            
            debug("Fetching case history data", {
                'query': history_query,
                'case_count': len(df['Id'].unique())
            })
            
            try:
                history_records = execute_soql_query(st.session_state.sf_connection, history_query)
                if history_records:
                    history_df = pd.DataFrame(history_records)
                    debug(f"Retrieved {len(history_df)} history records")
                else:
                    debug("No history records found")
                    history_df = pd.DataFrame(columns=['Id', 'CaseId', 'Field', 'OldValue', 'NewValue', 'CreatedDate'])
            except Exception as query_error:
                logger.error(f"Error fetching history data: {str(query_error)}", exc_info=True)
                history_df = pd.DataFrame(columns=['Id', 'CaseId', 'Field', 'OldValue', 'NewValue', 'CreatedDate'])
            
            # Import the priority handler function
            from utils.priority_handler import get_highest_priority
            
            # Calculate highest priority for each case
            priority_stats = {'success': 0, 'error': 0, 'unchanged': 0}
            
            for case_id in df['Id']:
                try:
                    current_priority = df.loc[df['Id'] == case_id, 'Internal_Priority__c'].iloc[0]
                    highest_priority = get_highest_priority(case_id, history_df, current_priority)
                    
                    if highest_priority != current_priority:
                        priority_stats['success'] += 1
                    else:
                        priority_stats['unchanged'] += 1
                        
                    df.loc[df['Id'] == case_id, 'Highest_Priority'] = highest_priority
                    
                except Exception as e:
                    priority_stats['error'] += 1
                    error_msg = f"Error processing priority for case {case_id}: {str(e)}"
                    debug(error_msg, {'traceback': traceback.format_exc()}, category="error")
                    logger.error(error_msg, exc_info=True)
                    df.loc[df['Id'] == case_id, 'Highest_Priority'] = current_priority
            
            debug("Priority analysis completed", {
                'total_cases': len(df),
                'priority_stats': priority_stats
            })
            
        except Exception as e:
            error_msg = f"Error fetching priority history: {str(e)}"
            debug(error_msg, {'traceback': traceback.format_exc()}, category="error")
            logger.error(error_msg, exc_info=True)
            df['Highest_Priority'] = df['Internal_Priority__c']
        
        # Handle missing values
        df['Subject'] = df['Subject'].fillna('')
        df['Description'] = df['Description'].fillna('')
        df['Product_Area__c'] = df['Product_Area__c'].fillna('Unspecified')
        df['Product_Feature__c'] = df['Product_Feature__c'].fillna('Unspecified')
        df['RCA__c'] = df['RCA__c'].fillna('Not Specified')
        df['Internal_Priority__c'] = df['Internal_Priority__c'].fillna('Not Set')
        df['Status'] = df['Status'].fillna('Unknown')
        df['CSAT__c'] = pd.to_numeric(df['CSAT__c'], errors='coerce')
        df['IsEscalated'] = df['IsEscalated'].fillna(False)
        
        # Convert date columns and ensure timezone consistency
        date_columns = ['CreatedDate', 'ClosedDate', 'First_Response_Time__c']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)
        
        # Calculate resolution time
        mask = df['ClosedDate'].notna() & df['CreatedDate'].notna()
        df.loc[mask, 'Resolution Time (Days)'] = (
            df.loc[mask, 'ClosedDate'] - df.loc[mask, 'CreatedDate']
        ).dt.total_seconds() / (24 * 60 * 60)
        
        # Rename columns for consistency
        df = df.rename(columns={
            'CreatedDate': 'Created Date',
            'ClosedDate': 'Closed Date',
            'Product_Area__c': 'Product Area',
            'Product_Feature__c': 'Product Feature',
            'RCA__c': 'Root Cause',
            'First_Response_Time__c': 'First Response Time',
            'CSAT__c': 'CSAT',
            'Internal_Priority__c': 'Priority'
        })
        
        debug("Data processing completed", {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'date_range': f"{df['Created Date'].min()} to {df['Created Date'].max()}"
        })
        
        if df.empty:
            st.warning("No data available after processing")
            debug("Empty DataFrame after processing", category="error")
            return pd.DataFrame()
            
        return df
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        debug(f"Error in fetch_data: {str(e)}", {'traceback': traceback.format_exc()}, category="error")
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.exception(e)
        return pd.DataFrame() 
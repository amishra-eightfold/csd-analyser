"""Data handling and processing functions for the application.

This module provides functions for:
- Fetching data from Salesforce
- Processing and preparing data for analysis
- Handling data transformations for display
"""

import streamlit as st
import pandas as pd
import traceback
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging

from config.logging_config import get_logger, log_error
from salesforce_config import execute_soql_query
from utils.time_analysis import calculate_first_response_time, calculate_sla_breaches
from utils.text_processing import get_highest_priority_from_history

# Initialize logger
logger = get_logger('data')

def fetch_data(
    start_date: datetime,
    end_date: datetime,
    selected_customers: List[str] = None,
    include_comments: bool = True,
    include_history: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch support ticket data from Salesforce within the specified parameters.
    
    Args:
        start_date: The start date for the query
        end_date: The end date for the query
        selected_customers: List of customer names to filter by
        include_comments: Whether to fetch case comments
        include_history: Whether to fetch case history
        
    Returns:
        Tuple containing three DataFrames:
        - Main case data
        - Case comments (if requested)
        - Case history (if requested)
    """
    # Create progress placeholder
    progress_placeholder = st.empty()
    progress_placeholder.markdown(
        """
        <div class='loading status-indicator processing'>
            <p>🔄 Fetching data from Salesforce...</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    try:
        # Format date range for SOQL query
        formatted_start_date = start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        formatted_end_date = end_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        
        # Build the customer filter condition
        customer_filter = ""
        if selected_customers and len(selected_customers) > 0:
            customer_names = "', '".join(selected_customers)
            customer_filter = f"AND Account.Account_Name__c IN ('{customer_names}')"
        
        # Construct main query for cases
        case_query = f"""
            SELECT Id, CaseNumber, Account.Account_Name__c, CreatedDate, ClosedDate, Status,
                   Subject, Description, Priority, Internal_Priority__c, Product_Area__c, Product_Feature__c, RCA__c,
                   CSAT__c, First_Response_Time__c, Case_Type__c
            FROM Case
            WHERE CreatedDate >= {formatted_start_date}
            AND CreatedDate <= {formatted_end_date}
            AND Is_Case_L1_Triaged__c = false 
            AND RecordTypeId = '0123m000000U8CCAA0'
            AND Case_Owner__c != 'Support'
            AND Status != 'Merged'
            {customer_filter}
            ORDER BY CreatedDate DESC
        """
        
        # Execute case query
        progress_placeholder.markdown(
            """
            <div class='loading status-indicator processing'>
                <p>🔄 Fetching case data...</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        cases = execute_soql_query(st.session_state.sf_connection, case_query)
        
        # Create DataFrame for case data
        if cases:
            cases_df = pd.DataFrame(cases)
            
            # Process date columns
            if 'CreatedDate' in cases_df.columns:
                cases_df['Created Date'] = pd.to_datetime(cases_df['CreatedDate'])
            if 'ClosedDate' in cases_df.columns:
                cases_df['Closed Date'] = pd.to_datetime(cases_df['ClosedDate'])
            
            # Process First Response Time
            if 'First_Response_Time__c' in cases_df.columns:
                cases_df['First Response Time'] = pd.to_datetime(cases_df['First_Response_Time__c'], errors='coerce')
                logger.info(f"Processed First Response Time from First_Response_Time__c column. Found {cases_df['First Response Time'].notna().sum()} valid values.")
            
            # Process Internal Priority
            if 'Internal_Priority__c' in cases_df.columns:
                cases_df['Highest Priority'] = cases_df['Internal_Priority__c']
                logger.info(f"Using Internal_Priority__c for priority analysis.")
            
            # Extract Account Name
            if 'Account' in cases_df.columns:
                cases_df['Account_Name'] = cases_df['Account'].apply(
                    lambda x: x.get('Account_Name__c') if x else None
                )
            
            # Add Case Number column
            if 'CaseNumber' in cases_df.columns:
                cases_df['Case Number'] = cases_df['CaseNumber']
            
            # Calculate resolution time in days
            cases_df['Resolution Time (Days)'] = None
            mask = (~cases_df['Closed Date'].isna()) & (~cases_df['Created Date'].isna())
            if any(mask):
                cases_df.loc[mask, 'Resolution Time (Days)'] = (
                    (cases_df.loc[mask, 'Closed Date'] - cases_df.loc[mask, 'Created Date'])
                    .dt.total_seconds() / (24 * 3600)
                ).round(1)
            
            # CSAT handling
            if 'CSAT__c' in cases_df.columns:
                cases_df['CSAT'] = pd.to_numeric(cases_df['CSAT__c'], errors='coerce')
                
            # RCA handling
            if 'RCA__c' in cases_df.columns:
                cases_df['Root Cause'] = cases_df['RCA__c'].fillna('Not Specified')
        else:
            cases_df = pd.DataFrame()
            progress_placeholder.markdown(
                """
                <div class='status-indicator error'>
                    <p>⚠️ No cases found for the selected criteria</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            time.sleep(2)
            progress_placeholder.empty()
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Fetch comments if requested
        comments_df = pd.DataFrame()
        if include_comments and not cases_df.empty:
            progress_placeholder.markdown(
                """
                <div class='loading status-indicator processing'>
                    <p>🔄 Fetching case comments...</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Get case IDs from the main query
            case_ids = "', '".join(cases_df['Id'].tolist())
            
            comments_query = f"""
                SELECT Id, ParentId, CreatedDate, CommentBody, CreatedBy.Name
                FROM CaseComment
                WHERE ParentId IN ('{case_ids}')
                ORDER BY CreatedDate ASC
            """
            
            comments = execute_soql_query(st.session_state.sf_connection, comments_query)
            
            if comments:
                comments_df = pd.DataFrame(comments)
                comments_df['Created Date'] = pd.to_datetime(comments_df['CreatedDate'])
                comments_df['Author'] = comments_df['CreatedBy'].apply(
                    lambda x: x.get('Name') if x else None
                )
        
        # Fetch history if requested
        history_df = pd.DataFrame()
        if include_history and not cases_df.empty:
            progress_placeholder.markdown(
                """
                <div class='loading status-indicator processing'>
                    <p>🔄 Fetching case history...</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            history_query = f"""
                SELECT Id, CaseId, Field, OldValue, NewValue, CreatedDate, CreatedById
                FROM CaseHistory
                WHERE CaseId IN ('{case_ids}')
                ORDER BY CreatedDate ASC
            """
            
            history = execute_soql_query(st.session_state.sf_connection, history_query)
            
            if history:
                history_df = pd.DataFrame(history)
                history_df['Created Date'] = pd.to_datetime(history_df['CreatedDate'])
        
        # Combine comments into a single field
        if not cases_df.empty and not comments_df.empty:
            # Group comments by case ID
            grouped_comments = comments_df.groupby('ParentId').apply(
                lambda x: '\n\n'.join(
                    f"[{row['Created Date'].strftime('%Y-%m-%d %H:%M')} - {row['Author']}]\n{row['CommentBody']}"
                    for _, row in x.iterrows()
                )
            ).reset_index()
            
            grouped_comments.columns = ['Id', 'Comments']
            
            # Merge with the main cases DataFrame
            cases_df = pd.merge(cases_df, grouped_comments, on='Id', how='left')
        else:
            cases_df['Comments'] = None
        
        # Process case history for highest priority
        if not cases_df.empty and not history_df.empty:
            # Filter history for priority changes
            priority_history = history_df[history_df['Field'] == 'Priority']
            
            if not priority_history.empty:
                # Get highest priority for each case
                highest_priorities = {}
                for case_id in cases_df['Id'].unique():
                    current_priority = cases_df.loc[cases_df['Id'] == case_id, 'Priority'].iloc[0]
                    highest_priority = get_highest_priority_from_history(
                        case_id, priority_history, current_priority
                    )
                    highest_priorities[case_id] = highest_priority
                
                # Add highest priority to cases DataFrame
                cases_df['Highest Priority'] = cases_df['Id'].map(highest_priorities)
            else:
                cases_df['Highest Priority'] = cases_df['Priority']
        else:
            cases_df['Highest Priority'] = cases_df['Priority']
        
        # Calculate response and resolution metrics
        if not cases_df.empty and not comments_df.empty:
            try:
                # Calculate first response time
                response_hours, stats = calculate_first_response_time(cases_df, allow_synthetic=False)
                cases_df['First Response Time (Hours)'] = response_hours
                
                # Calculate SLA breaches
                if 'Highest Priority' in cases_df.columns:
                    sla_data = calculate_sla_breaches(response_hours, cases_df['Highest Priority'])
                    cases_df = pd.merge(
                        cases_df, 
                        sla_data.rename(columns={'Priority': 'Highest Priority'}),
                        on='Highest Priority', 
                        how='left'
                    )
            except Exception as e:
                error_msg = f"Warning: Could not calculate response time metrics: {str(e)}"
                logger.warning(error_msg)
                # Continue with other data processing
        
        # Show success message
        progress_placeholder.markdown(
            f"""
            <div class='status-indicator success'>
                <p>✅ Data fetched successfully</p>
                <p>{len(cases_df)} cases loaded</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Keep the data in session state
        st.session_state.data_loaded = True
        
        return cases_df, comments_df, history_df
    
    except Exception as e:
        # Log and show error
        error_msg = f"Error fetching data: {str(e)}"
        progress_placeholder.markdown(
            f"""
            <div class='status-indicator error'>
                <p>❌ Error fetching data</p>
                <p>{str(e)}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        log_error(error_msg, traceback.format_exc())
        logger.error(error_msg, exc_info=True)
        
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    finally:
        # Clear placeholder after a delay
        time.sleep(2)
        progress_placeholder.empty()

def prepare_data_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the data for analysis by converting types and calculating additional metrics.
    
    Args:
        df: The raw DataFrame containing case data
        
    Returns:
        The processed DataFrame ready for analysis
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Convert date columns
    date_columns = ['Created Date', 'Closed Date']
    for col in date_columns:
        if col in processed_df.columns:
            processed_df[col] = pd.to_datetime(processed_df[col])
    
    # Convert numeric columns
    if 'CSAT' in processed_df.columns:
        processed_df['CSAT'] = pd.to_numeric(processed_df['CSAT'], errors='coerce')
    
    if 'Resolution Time (Days)' in processed_df.columns:
        processed_df['Resolution Time (Days)'] = pd.to_numeric(
            processed_df['Resolution Time (Days)'], 
            errors='coerce'
        )
    
    # Create date-based features for analysis
    if 'Created Date' in processed_df.columns:
        processed_df['Creation Year'] = processed_df['Created Date'].dt.year
        processed_df['Creation Month'] = processed_df['Created Date'].dt.month
        processed_df['Creation Week'] = processed_df['Created Date'].dt.isocalendar().week
        processed_df['Creation Day'] = processed_df['Created Date'].dt.day
        processed_df['Creation Weekday'] = processed_df['Created Date'].dt.day_name()
    
    # Create status category
    if 'Status' in processed_df.columns:
        processed_df['Status Category'] = processed_df['Status'].apply(
            lambda x: 'Open' if x in ['New', 'In Progress', 'On Hold'] else 
                     ('Closed' if x in ['Closed', 'Resolved'] else 'Other')
        )
    
    # Create priority category if not already present
    if 'Priority' in processed_df.columns and 'Priority Category' not in processed_df.columns:
        priority_map = {
            'P1': 'Critical',
            'P2': 'High',
            'P3': 'Medium',
            'P4': 'Low'
        }
        processed_df['Priority Category'] = processed_df['Priority'].map(priority_map)
    
    # Create root cause category if needed
    if 'RCA__c' in processed_df.columns and 'Root Cause Category' not in processed_df.columns:
        # Group common root causes into categories
        bug_related = ['Bug', 'Regression', 'Defect']
        config_related = ['Configuration', 'Setup', 'Installation']
        user_related = ['User Error', 'Training', 'Documentation']
        
        def categorize_root_cause(rca):
            if pd.isna(rca) or rca == 'Not Specified':
                return 'Not Specified'
            elif any(term in rca for term in bug_related):
                return 'Bug Related'
            elif any(term in rca for term in config_related):
                return 'Configuration Related'
            elif any(term in rca for term in user_related):
                return 'User Related'
            else:
                return 'Other'
        
        processed_df['Root Cause Category'] = processed_df['RCA__c'].apply(categorize_root_cause)
    
    return processed_df

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a summary of key metrics from the data.
    
    Args:
        df: DataFrame containing case data
        
    Returns:
        Dictionary containing summary metrics
    """
    if df.empty:
        return {
            'total_tickets': 0,
            'open_tickets': 0,
            'closed_tickets': 0,
            'avg_resolution_time': 0,
            'avg_csat': 0,
            'priority_distribution': {},
            'status_distribution': {},
            'product_area_distribution': {},
            'root_cause_distribution': {}
        }
    
    summary = {}
    
    # Basic counts
    summary['total_tickets'] = len(df)
    summary['open_tickets'] = len(df[df['Status'].isin(['Open', 'In Progress', 'On Hold', 'Pending'])]) 
    summary['closed_tickets'] = len(df[df['Status'] == 'Closed'])
    
    # Average resolution time
    closed_tickets = df[df['Status'] == 'Closed']
    if not closed_tickets.empty and 'Resolution Time (Days)' in closed_tickets.columns:
        avg_resolution = closed_tickets['Resolution Time (Days)'].mean()
        summary['avg_resolution_time'] = round(avg_resolution, 1)
    else:
        summary['avg_resolution_time'] = None
    
    # Average CSAT
    if 'CSAT' in df.columns:
        csat_data = df['CSAT'].dropna()
        if not csat_data.empty:
            summary['avg_csat'] = round(csat_data.mean(), 2)
        else:
            summary['avg_csat'] = None
    else:
        summary['avg_csat'] = None
    
    # Priority distribution
    if 'Priority' in df.columns:
        priority_counts = df['Priority'].value_counts().to_dict()
        summary['priority_distribution'] = priority_counts
    else:
        summary['priority_distribution'] = {}
    
    # Status distribution
    if 'Status' in df.columns:
        status_counts = df['Status'].value_counts().to_dict()
        summary['status_distribution'] = status_counts
    else:
        summary['status_distribution'] = {}
    
    # Product area distribution
    if 'Product_Area__c' in df.columns:
        product_area_counts = df['Product_Area__c'].value_counts().to_dict()
        summary['product_area_distribution'] = product_area_counts
    else:
        summary['product_area_distribution'] = {}
    
    # Root cause distribution
    if 'RCA__c' in df.columns:
        root_cause_counts = df['RCA__c'].value_counts().to_dict()
        summary['root_cause_distribution'] = root_cause_counts
    else:
        summary['root_cause_distribution'] = {}
    
    # Date range
    if 'Created Date' in df.columns:
        summary['first_date'] = df['Created Date'].min()
        summary['last_date'] = df['Created Date'].max()
    
    return summary 
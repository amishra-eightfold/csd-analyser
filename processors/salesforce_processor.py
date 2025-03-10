"""Salesforce data processor for the CSD Analyzer application."""

import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from base.data_processor import BaseDataProcessor
from utils.error_handlers import handle_errors, track_progress
from utils.data_validation import validate_dataframe_decorator
from config.api_config import QUERY_TEMPLATES
from salesforce_config import execute_soql_query

class SalesforceDataProcessor(BaseDataProcessor):
    """Class for processing Salesforce data with common operations."""
    
    def __init__(self, sf_connection):
        """
        Initialize the Salesforce data processor.
        
        Args:
            sf_connection: Salesforce connection object
        """
        super().__init__()
        self.sf_connection = sf_connection
    
    @handle_errors(custom_message="Error fetching Salesforce data")
    @track_progress(total_steps=4)
    def fetch_data(self,
                  customers: List[str],
                  start_date: datetime,
                  end_date: datetime,
                  progress_callback: Optional[callable] = None) -> Tuple[pd.DataFrame, ...]:
        """
        Fetch data from Salesforce based on selected customers and date range.
        
        Args:
            customers (List[str]): List of customer names
            start_date (datetime): Start date for data fetch
            end_date (datetime): End date for data fetch
            progress_callback (Optional[callable]): Callback for progress updates
            
        Returns:
            Tuple[pd.DataFrame, ...]: Tuple of DataFrames (cases, emails, comments, history, attachments)
        """
        # Format customer list for query
        customer_list = "'" + "','".join(customers) + "'"
        
        # Format dates for query
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch cases
        query = QUERY_TEMPLATES['cases'].format(
            customers=customer_list,
            start_date=start_str,
            end_date=end_str
        )
        records = execute_soql_query(self.sf_connection, query)
        if not records:
            return None, None, None, None, None
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        progress_callback()
        
        # Process Account Name
        if 'Account' in df.columns and isinstance(df['Account'].iloc[0], dict):
            df['Account_Name'] = df['Account'].apply(lambda x: x.get('Name') if isinstance(x, dict) else None)
            df = df.drop('Account', axis=1)
        
        # Process date columns
        date_columns = ['CreatedDate', 'ClosedDate', 'First_Response_Time__c']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Get case IDs for related data
        case_ids = df['Id'].tolist()
        progress_callback()
        
        # Fetch related data
        emails_df, comments_df, history_df, attachments_df = self._fetch_related_data(case_ids, progress_callback)
        
        # Clean and preprocess data
        df = self._preprocess_data(df, emails_df, comments_df)
        progress_callback()
        
        return df, emails_df, comments_df, history_df, attachments_df
    
    @handle_errors(custom_message="Error fetching related Salesforce data")
    def _fetch_related_data(self,
                          case_ids: List[str],
                          progress_callback: Optional[callable] = None) -> Tuple[pd.DataFrame, ...]:
        """
        Fetch related data for given case IDs.
        
        Args:
            case_ids (List[str]): List of case IDs
            progress_callback (Optional[callable]): Callback for progress updates
            
        Returns:
            Tuple[pd.DataFrame, ...]: Tuple of related DataFrames
        """
        if not case_ids:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Format IDs for query
        case_ids_str = "'" + "','".join(case_ids) + "'"
        
        # Fetch emails
        email_query = QUERY_TEMPLATES['emails'].format(case_ids=case_ids_str)
        email_records = execute_soql_query(self.sf_connection, email_query)
        emails_df = pd.DataFrame(email_records) if email_records else pd.DataFrame()
        
        # Fetch comments
        comment_query = QUERY_TEMPLATES['comments'].format(case_ids=case_ids_str)
        comment_records = execute_soql_query(self.sf_connection, comment_query)
        comments_df = pd.DataFrame(comment_records) if comment_records else pd.DataFrame()
        
        # Fetch history
        history_query = QUERY_TEMPLATES['history'].format(case_ids=case_ids_str)
        history_records = execute_soql_query(self.sf_connection, history_query)
        history_df = pd.DataFrame(history_records) if history_records else pd.DataFrame()
        
        # Fetch attachments
        attachment_query = f"""
            SELECT Id, ParentId, Name, ContentType, CreatedDate
            FROM Attachment
            WHERE ParentId IN ({case_ids_str})
        """
        attachment_records = execute_soql_query(self.sf_connection, attachment_query)
        attachments_df = pd.DataFrame(attachment_records) if attachment_records else pd.DataFrame()
        
        if progress_callback:
            progress_callback()
        
        return emails_df, comments_df, history_df, attachments_df
    
    @handle_errors(custom_message="Error preprocessing data")
    def _preprocess_data(self,
                        cases_df: pd.DataFrame,
                        emails_df: Optional[pd.DataFrame] = None,
                        comments_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Preprocess case data with related information.
        
        Args:
            cases_df (pd.DataFrame): Cases DataFrame
            emails_df (Optional[pd.DataFrame]): Emails DataFrame
            comments_df (Optional[pd.DataFrame]): Comments DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        df = cases_df.copy()
        
        # Clean classification fields
        classification_fields = ['Product_Area__c', 'Product_Feature__c', 'RCA__c']
        for field in classification_fields:
            if field in df.columns:
                df[field] = df[field].fillna('Unknown')
                df[field] = df[field].astype(str)
        
        # Combine email content
        if emails_df is not None and not emails_df.empty and 'Id' in df.columns:
            email_content = emails_df.groupby('ParentId')['TextBody'].apply(
                lambda x: ' '.join(x) if isinstance(x, (list, pd.Series)) else str(x)
            ).reset_index()
            email_content.columns = ['Id', 'email_content']
            df = pd.merge(df, email_content, on='Id', how='left')
        
        # Combine comment content
        if comments_df is not None and not comments_df.empty and 'Id' in df.columns:
            comment_content = comments_df.groupby('ParentId')['CommentBody'].apply(
                lambda x: ' '.join(x) if isinstance(x, (list, pd.Series)) else str(x)
            ).reset_index()
            comment_content.columns = ['Id', 'comment_content']
            df = pd.merge(df, comment_content, on='Id', how='left')
        
        # Fill NA values in text fields
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            df[col] = df[col].fillna('')
        
        return df
    
    @handle_errors(custom_message="Error calculating case metrics")
    def calculate_case_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate key metrics from case data.
        
        Args:
            df (pd.DataFrame): Cases DataFrame
            
        Returns:
            Dict[str, Any]: Dictionary of calculated metrics
        """
        metrics = {}
        
        # Basic counts
        metrics['total_cases'] = len(df)
        metrics['open_cases'] = len(df[df['Status'] == 'Open'])
        metrics['closed_cases'] = len(df[df['Status'] == 'Closed'])
        
        # Response times
        if 'First_Response_Time__c' in df.columns:
            response_times = df['First_Response_Time__c'].dropna()
            metrics['avg_response_time'] = response_times.mean() if not response_times.empty else 0
            metrics['median_response_time'] = response_times.median() if not response_times.empty else 0
        
        # CSAT scores
        if 'CSAT__c' in df.columns:
            csat_scores = df['CSAT__c'].dropna()
            metrics['avg_csat'] = csat_scores.mean() if not csat_scores.empty else 0
            metrics['csat_responses'] = len(csat_scores)
        
        # Priority distribution
        if 'Internal_Priority__c' in df.columns:
            metrics['priority_distribution'] = df['Internal_Priority__c'].value_counts().to_dict()
        
        # Product area distribution
        if 'Product_Area__c' in df.columns:
            metrics['product_area_distribution'] = df['Product_Area__c'].value_counts().to_dict()
        
        return metrics 
"""Salesforce data processor for CSD Analyzer."""
from typing import Dict, List, Optional
import pandas as pd
from simple_salesforce import Salesforce
from utils.error_handlers import handle_errors
from datetime import datetime

class SalesforceDataProcessor:
    """Process Salesforce data for analysis."""

    def __init__(self, auth_config_or_connection):
        """Initialize the processor with Salesforce authentication or connection.
        
        Args:
            auth_config_or_connection: Either a dictionary containing Salesforce authentication details
                                     or a direct Salesforce connection object
        """
        try:
            if isinstance(auth_config_or_connection, dict):
                # Handle authentication config
                self.sf_connection = Salesforce(
                    username=auth_config_or_connection.get('username'),
                    password=auth_config_or_connection.get('password'),
                    security_token=auth_config_or_connection.get('security_token'),
                    domain=auth_config_or_connection.get('domain', 'test')
                )
            else:
                # Handle direct connection object
                self.sf_connection = auth_config_or_connection
                
            if not self.sf_connection:
                raise ValueError("Invalid Salesforce connection")
                
        except Exception as e:
            self.sf_connection = None
            raise ValueError(f"Failed to initialize Salesforce connection: {str(e)}")

    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch case data from Salesforce.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame containing case data
            
        Raises:
            ValueError: If connection is not initialized or if there's an error fetching data
        """
        if not self.sf_connection:
            raise ValueError("Salesforce connection not initialized")
            
        try:
            # Format dates for SOQL query
            start_datetime = f"{start_date}T00:00:00Z"
            end_datetime = f"{end_date}T23:59:59Z"
            
            # Build query with consistent formatting
            query = (
                "SELECT Id, CaseNumber, Subject, Status, Priority, CreatedDate, ClosedDate, "
                "Product_Area__c, Product_Feature__c, RCA__c, Internal_Priority__c, "
                "First_Response_Time__c, CSAT__c, IsEscalated, Account.Name "
                "FROM Case "
                f"WHERE CreatedDate >= {start_datetime} "
                f"AND CreatedDate <= {end_datetime}"
            )
            
            try:
                result = self.sf_connection.query(query)
            except Exception as query_error:
                raise ValueError(f"SOQL query failed: {str(query_error)}\nQuery: {query}")
                
            if not isinstance(result, dict):
                raise ValueError("Invalid response format")
                
            if 'records' not in result:
                raise ValueError("No 'records' field")
                
            if not result:
                raise ValueError("Invalid response format")
                
            records = result.get('records', [])
            if not records:
                return pd.DataFrame()  # Return empty DataFrame for no records
            
            # Flatten nested fields
            flattened_records = []
            for record in records:
                flat_record = {}
                for key, value in record.items():
                    if isinstance(value, dict):
                        for nested_key, nested_value in value.items():
                            flat_record[f"{key}.{nested_key}"] = nested_value
                    else:
                        flat_record[key] = value
                flattened_records.append(flat_record)
            
            df = pd.DataFrame(flattened_records)
            
            # Convert date fields to datetime
            date_columns = ['CreatedDate', 'ClosedDate']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Convert boolean fields
            bool_columns = ['IsEscalated']
            for col in bool_columns:
                if col in df.columns:
                    df[col] = df[col].astype(bool)
            
            # Convert numeric fields
            numeric_columns = ['First_Response_Time__c', 'CSAT__c']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except ValueError as ve:
            # Re-raise ValueError with original message
            raise ve
        except Exception as e:
            error_msg = f"Failed to fetch data from Salesforce: {str(e)}"
            if hasattr(e, 'content'):
                error_msg += f"\nSalesforce error content: {e.content}"
            raise ValueError(error_msg)
            
        return pd.DataFrame()  # Fallback empty DataFrame

    def _preprocess_data(self, cases_df: pd.DataFrame, 
                        emails_df: Optional[pd.DataFrame] = None,
                        comments_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Preprocess case data with related emails and comments."""
        if cases_df is None or cases_df.empty:
            raise ValueError("Invalid or empty case data")

        # Clean classification fields
        classification_fields = ['Product_Area__c', 'Product_Feature__c', 'RCA__c']
        for field in classification_fields:
            if field in cases_df.columns:
                cases_df[field] = cases_df[field].fillna('Unknown')

        # Merge email content if available
        if emails_df is not None and not emails_df.empty:
            email_content = emails_df.groupby('ParentId')['TextBody'].apply(lambda x: ' '.join(x)).reset_index()
            cases_df = cases_df.merge(email_content, left_on='Id', right_on='ParentId', how='left')
            cases_df.rename(columns={'TextBody': 'email_content'}, inplace=True)
            cases_df['email_content'] = cases_df['email_content'].fillna('')

        # Merge comment content if available
        if comments_df is not None and not comments_df.empty:
            comment_content = comments_df.groupby('ParentId')['CommentBody'].apply(lambda x: ' '.join(x)).reset_index()
            cases_df = cases_df.merge(comment_content, left_on='Id', right_on='ParentId', how='left')
            cases_df.rename(columns={'CommentBody': 'comment_content'}, inplace=True)
            cases_df['comment_content'] = cases_df['comment_content'].fillna('')

        return cases_df

    def calculate_case_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate case metrics from the DataFrame."""
        if df is None:
            raise ValueError("DataFrame cannot be None")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")

        metrics = {
            'total_cases': len(df),
            'open_cases': len(df[df['Status'] == 'Open']),
            'closed_cases': len(df[df['Status'] == 'Closed']),
            'escalated_cases': len(df[df['IsEscalated'] == True])
        }

        # Calculate CSAT metrics if available
        if 'CSAT__c' in df.columns:
            csat_scores = df['CSAT__c'].dropna()
            if not csat_scores.empty:
                metrics['avg_csat'] = float(csat_scores.mean())
                metrics['csat_responses'] = int(len(csat_scores))

        # Calculate response time metrics if available
        if 'First_Response_Time__c' in df.columns:
            response_times = df['First_Response_Time__c'].dropna()
            if not response_times.empty:
                metrics['avg_response_time'] = float(response_times.mean())
                metrics['max_response_time'] = float(response_times.max())

        return metrics 
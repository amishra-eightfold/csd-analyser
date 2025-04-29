"""Salesforce data processing module."""

from typing import Dict, Union, List
from datetime import datetime
import pandas as pd
from simple_salesforce import Salesforce

class SalesforceDataProcessor:
    """Processes Salesforce case data for analysis."""
    
    def __init__(self, sf_connection):
        """Initialize the processor with a Salesforce connection."""
        if isinstance(sf_connection, dict):
            try:
                self.sf_connection = Salesforce(**sf_connection)
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Salesforce: {str(e)}")
        else:
            self.sf_connection = sf_connection

    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch case data from Salesforce.
        
        Args:
            start_date: Start date for data fetch (YYYY-MM-DD)
            end_date: End date for data fetch (YYYY-MM-DD)
            
        Returns:
            DataFrame containing case data
        """
        # Format dates for SOQL query
        start_date = f"{start_date}T00:00:00Z"
        end_date = f"{end_date}T23:59:59Z"
        
        # Construct SOQL query
        query = (
            "SELECT Id, CaseNumber, Subject, Status, Priority, CreatedDate, ClosedDate, "
            "Product_Area__c, Product_Feature__c, RCA__c, Internal_Priority__c, "
            "First_Response_Time__c, CSAT__c, IsEscalated, Account.Name "
            "FROM Case "
            f"WHERE CreatedDate >= {start_date} "
            f"AND CreatedDate <= {end_date}"
        )
        
        try:
            # Execute query and get results
            results = self.sf_connection.query(query)
            
            if not results['records']:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(results['records'])
            
            # Extract Account.Name from nested structure
            if 'Account' in df.columns:
                df['Account.Name'] = df['Account'].apply(lambda x: x.get('Name') if x else None)
                df.drop('Account', axis=1, inplace=True)
            
            # Remove attributes column if present
            if 'attributes' in df.columns:
                df.drop('attributes', axis=1, inplace=True)
            
            return df
        
        except Exception as e:
            raise ConnectionError(f"Failed to fetch data from Salesforce: {str(e)}")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess case data.
        
        Args:
            df: Raw case data DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if df.empty:
            return df
        
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Convert date columns to datetime
        date_columns = ['CreatedDate', 'ClosedDate', 'First_Response_Time__c']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Calculate resolution time in days
        if 'CreatedDate' in df.columns and 'ClosedDate' in df.columns:
            df['Resolution_Time_Days__c'] = (
                df['ClosedDate'] - df['CreatedDate']
            ).dt.total_seconds() / (24 * 3600)
        
        return df

    def calculate_case_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate key metrics from case data.
        
        Args:
            df: Preprocessed case data DataFrame
            
        Returns:
            Dictionary containing calculated metrics
        """
        if df.empty:
            return {
                'total_cases': 0,
                'open_cases': 0,
                'avg_resolution_time': 0,
                'priority_distribution': {},
                'product_area_distribution': {}
            }
        
        metrics = {
            'total_cases': len(df),
            'open_cases': len(df[df['Status'] != 'Closed']),
            'avg_resolution_time': df['Resolution_Time_Days__c'].mean(),
            'priority_distribution': df['Priority'].value_counts().to_dict(),
            'product_area_distribution': df['Product_Area__c'].value_counts().to_dict()
        }
        
        return metrics 
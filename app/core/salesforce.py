"""Salesforce integration module for CSD Analyzer."""

from typing import Optional, List, Dict, Any
import pandas as pd
from datetime import datetime
from simple_salesforce import Salesforce
import logging
from .config import config

logger = logging.getLogger(__name__)

class SalesforceClient:
    """Manages Salesforce connection and data retrieval."""
    
    def __init__(self):
        """Initialize Salesforce client."""
        self._client: Optional[Salesforce] = None
        self._connected = False
        
    @property
    def is_connected(self) -> bool:
        """Check if connected to Salesforce."""
        return self._connected and self._client is not None
    
    def connect(self) -> bool:
        """
        Establish connection to Salesforce.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self._client = Salesforce(
                username=config.salesforce_username,
                password=config.salesforce_password,
                security_token=config.salesforce_security_token
            )
            self._connected = True
            logger.info("Successfully connected to Salesforce")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Salesforce: {str(e)}")
            self._connected = False
            return False
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute SOQL query.
        
        Args:
            query (str): SOQL query to execute
            
        Returns:
            List[Dict[str, Any]]: Query results
            
        Raises:
            ConnectionError: If not connected to Salesforce
            Exception: For other query execution errors
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Salesforce")
            
        try:
            result = self._client.query_all(query)
            return result.get('records', [])
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    def get_customers(self) -> List[str]:
        """
        Get list of active customers.
        
        Returns:
            List[str]: List of customer names
        """
        try:
            query = """
                SELECT Account.Account_Name__c 
                FROM Account 
                WHERE Account.Account_Name__c != null
                AND Active_Contract__c = 'Yes'
                ORDER BY Account.Account_Name__c
            """
            records = self.execute_query(query)
            return [record['Account_Name__c'] for record in records if record.get('Account_Name__c')]
        except Exception as e:
            logger.error(f"Failed to fetch customers: {str(e)}")
            return []
    
    def get_support_tickets(self, 
                          customers: List[str],
                          start_date: datetime,
                          end_date: datetime) -> pd.DataFrame:
        """
        Fetch support tickets for specified customers and date range.
        
        Args:
            customers (List[str]): List of customer names
            start_date (datetime): Start date for ticket fetch
            end_date (datetime): End date for ticket fetch
            
        Returns:
            pd.DataFrame: DataFrame containing support tickets
        """
        try:
            customer_list = "'" + "','".join(customers) + "'"
            query = f"""
                SELECT 
                    Id, CaseNumber, Subject, Description,
                    Account.Account_Name__c, CreatedDate, ClosedDate, Status, Internal_Priority__c,
                    Product_Area__c, Product_Feature__c, RCA__c,
                    First_Response_Time__c, CSAT__c, IsEscalated
                FROM Case
                WHERE Account.Account_Name__c IN ({customer_list})
                AND CreatedDate >= {start_date.strftime('%Y-%m-%d')}T00:00:00Z
                AND CreatedDate <= {end_date.strftime('%Y-%m-%d')}T23:59:59Z
            """
            
            records = self.execute_query(query)
            if not records:
                logger.warning("No tickets found for the specified criteria")
                return pd.DataFrame()
                
            df = pd.DataFrame(records)
            
            # Extract Account Name from nested structure
            if 'Account' in df.columns and isinstance(df['Account'].iloc[0], dict):
                df['Account_Name'] = df['Account'].apply(
                    lambda x: x.get('Account_Name__c') if isinstance(x, dict) else None
                )
                df = df.drop('Account', axis=1)
            
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
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch support tickets: {str(e)}")
            return pd.DataFrame()

# Create global Salesforce client instance
salesforce = SalesforceClient()

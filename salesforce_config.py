import os
from simple_salesforce import Salesforce
from dotenv import load_dotenv
import streamlit as st

def init_salesforce():
    """Initialize Salesforce connection using environment variables or Streamlit secrets."""
    try:
        # First try to load from .env file
        load_dotenv()
        
        # Try to get credentials from environment variables or Streamlit secrets
        username = os.getenv('SF_USERNAME') or st.secrets.get('SF_USERNAME')
        password = os.getenv('SF_PASSWORD') or st.secrets.get('SF_PASSWORD')
        security_token = os.getenv('SF_SECURITY_TOKEN') or st.secrets.get('SF_SECURITY_TOKEN')
        domain = os.getenv('SF_DOMAIN', 'login') or st.secrets.get('SF_DOMAIN', 'login')
        
        if not all([username, password, security_token]):
            return None
            
        sf = Salesforce(
            username=username,
            password=password,
            security_token=security_token,
            domain=domain
        )
        return sf
    except Exception as e:
        st.error(f"Error connecting to Salesforce: {str(e)}")
        return None

def execute_soql_query(sf, query):
    """Execute a SOQL query and return results as a pandas DataFrame."""
    try:
        if not sf:
            st.error("Salesforce connection not initialized")
            return None
            
        # Execute the query
        results = sf.query_all(query)
        
        if not results.get('records'):
            return None
            
        # Convert to list of dictionaries, removing attributes
        records = []
        for record in results['records']:
            record_dict = {k: v for k, v in record.items() if k != 'attributes'}
            records.append(record_dict)
            
        return records
    except Exception as e:
        st.error(f"Error executing SOQL query: {str(e)}")
        return None 
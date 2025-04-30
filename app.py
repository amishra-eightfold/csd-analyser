import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback
import os
import time
from typing import Dict, List, Any, Tuple, Optional

# Import from new module structure
from src.visualization.app_visualizations import (
    create_ticket_volume_chart,
    create_resolution_time_chart,
    create_wordcloud,
    create_priority_distribution_chart,
    create_product_area_chart,
    create_csat_distribution_chart,
    create_root_cause_chart,
    create_root_cause_product_heatmap,
    create_first_response_time_chart,
    create_priority_time_chart,
    create_status_time_chart
)

from src.data.app_data import (
    fetch_data,
    prepare_data_for_analysis,
    get_data_summary
)

from src.ui.app_components import (
    debug,
    setup_application_sidebar,
    display_privacy_status,
    process_pii_in_dataframe,
    display_connection_status,
    apply_custom_css
)

from src.visualization.dashboard_manager import (
    display_visualization_dashboard,
    display_detailed_analysis
)

from salesforce_config import init_salesforce
from utils.pii_handler import PIIHandler
from utils.debug_logger import DebugLogger
from config.logging_config import get_logger

# Initialize loggers
logger = get_logger('app')
api_logger = get_logger('api')
error_logger = get_logger('error')

# Set Seaborn and Matplotlib style
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")

# Page Configuration
st.set_page_config(
    page_title="Support Ticket Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

def initialize_session_state() -> None:
    """Initialize all session state variables."""
    if 'customers' not in st.session_state:
        st.session_state.customers = None
    if 'selected_customers' not in st.session_state:
        st.session_state.selected_customers = []
    if 'date_range' not in st.session_state:
        # Initialize with default date range (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        st.session_state.date_range = (start_date, end_date)
    if 'sf_connection' not in st.session_state:
        st.session_state.sf_connection = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'pii_handler' not in st.session_state:
        st.session_state.pii_handler = PIIHandler()
    if 'enable_pii_processing' not in st.session_state:
        st.session_state.enable_pii_processing = False
    if 'enable_detailed_analysis' not in st.session_state:
        st.session_state.enable_detailed_analysis = True
    if 'enable_ai_analysis' not in st.session_state:
        st.session_state.enable_ai_analysis = False
    if 'debug_logger' not in st.session_state:
        st.session_state.debug_logger = DebugLogger()

def setup_salesforce_connection() -> bool:
    """
    Set up the connection to Salesforce.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    if st.session_state.sf_connection is None:
        connection_status = display_connection_status("🔌 Connecting to Salesforce...")
        try:
            st.session_state.sf_connection = init_salesforce()
            if st.session_state.sf_connection is None:
                connection_status.markdown(
                    """
                    <div class='status-indicator error'>
                        <p>❌ Failed to connect to Salesforce</p>
                        <p>Please check your credentials.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                return False
            else:
                connection_status.markdown(
                    """
                    <div class='status-indicator success'>
                        <p>✅ Connected to Salesforce</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                time.sleep(1)
                connection_status.empty()
                return True
        except Exception as e:
            connection_status.markdown(
                f"""
                <div class='status-indicator error'>
                    <p>❌ Connection Error</p>
                    <p>{str(e)}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            return False
    return True

def fetch_customer_list() -> bool:
    """
    Fetch the list of customers from Salesforce.
    
    Returns:
        bool: True if successful, False otherwise
    """
    if st.session_state.customers is None:
        try:
            debug("Fetching customer list")
            query = """
                SELECT Account.Account_Name__c 
                FROM Account 
                WHERE Account.Account_Name__c != null
                AND Active_Contract__c = 'Yes'
                ORDER BY Account.Account_Name__c
            """
            from salesforce_config import execute_soql_query
            records = execute_soql_query(st.session_state.sf_connection, query)
            if records:
                st.session_state.customers = [record['Account_Name__c'] for record in records]
                debug(f"Loaded {len(st.session_state.customers)} customers")
                return True
            else:
                st.session_state.customers = []
                debug("No customers found", category="error")
                return False
        except Exception as e:
            st.error(f"Error fetching customers: {str(e)}")
            debug(f"Error fetching customers: {str(e)}", {'traceback': traceback.format_exc()}, category="error")
            return False
    return True

def main() -> None:
    """Main application entry point."""
    st.title("Support Ticket Analytics")
    
    # Initialize session state
    initialize_session_state()
    
    # Setup Salesforce connection
    if not setup_salesforce_connection():
        return
    
    # Fetch customers
    if not fetch_customer_list():
        return
    
    # Create application sidebar and get settings
    settings = setup_application_sidebar()
    
    # Update session state with sidebar settings
    st.session_state.enable_detailed_analysis = settings.get('enable_detailed_analysis', True)
    st.session_state.enable_ai_analysis = settings.get('enable_ai_analysis', False)
    st.session_state.enable_pii_processing = settings.get('enable_pii_processing', False)
    
    # Display privacy status if PII processing is enabled
    if st.session_state.enable_pii_processing:
        display_privacy_status()
    
    # Check if we need to fetch data
    if settings.get('fetch_data', False):
        # Get parameters for data fetching
        start_date = settings.get('start_date')
        end_date = settings.get('end_date')
        selected_customers = settings.get('selected_customers', [])
        
        # Fetch data
        cases_df, comments_df, history_df = fetch_data(
            start_date=start_date,
            end_date=end_date,
            selected_customers=selected_customers
        )
        
        if cases_df is not None and not cases_df.empty:
            # Process PII if enabled
            if st.session_state.enable_pii_processing:
                cases_df = process_pii_in_dataframe(cases_df)
                if comments_df is not None:
                    comments_df = process_pii_in_dataframe(comments_df)
            
            # Prepare data for analysis
            df = prepare_data_for_analysis(cases_df)
            
            # Display data summary
            st.subheader("Data Summary")
            summary = get_data_summary(df)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tickets", summary.get('total_tickets', 0))
            with col2:
                st.metric("Avg. Resolution Time", f"{summary.get('avg_resolution_time', 0):.1f} days")
            with col3:
                st.metric("CSAT", f"{summary.get('avg_csat', 0):.1f}")
            with col4:
                st.metric("Open Tickets", summary.get('open_tickets', 0))
            
            # Display data table with filter options
            display_data_table(df, st.session_state.enable_pii_processing)
            
            # Display visualizations
            display_visualization_dashboard(df)
            
            # Display detailed analysis if enabled
            if st.session_state.enable_detailed_analysis:
                display_detailed_analysis(
                    df, 
                    st.session_state.enable_ai_analysis,
                    st.session_state.enable_pii_processing
                )
        else:
            st.warning("No data found for the selected criteria.")
    else:
        st.info("Select customers and date range, then click 'Load Data' to begin analysis.")

def display_data_table(df: pd.DataFrame, enable_pii_processing: bool = False) -> None:
    """
    Display the data table with filtering options.
    
    Args:
        df: DataFrame to display
        enable_pii_processing: Whether PII processing is enabled
    """
    with st.expander("Data Table", expanded=False):
        # Filter options
        st.subheader("Filter Options")
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_status = st.multiselect(
                "Status",
                options=df['Status'].unique(),
                default=df['Status'].unique()
            )
        with col2:
            filter_priority = st.multiselect(
                "Priority",
                options=df['Priority'].unique(),
                default=df['Priority'].unique()
            )
        with col3:
            filter_product = st.multiselect(
                "Product Area",
                options=df['Product Area'].unique(),
                default=df['Product Area'].unique()
            )
        
        # Apply filters
        filtered_df = df[
            (df['Status'].isin(filter_status)) &
            (df['Priority'].isin(filter_priority)) &
            (df['Product Area'].isin(filter_product))
        ]
        
        # Display filtered data
        st.write(f"Showing {len(filtered_df)} of {len(df)} tickets")
        st.dataframe(filtered_df)
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export to CSV"):
                # Create timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tickets_{timestamp}.csv"
                
                # Export
                filtered_df.to_csv(filename, index=False)
                
                # Download
                with open(filename, "rb") as file:
                    st.download_button(
                        label="Download CSV",
                        data=file,
                        file_name=filename,
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main() 
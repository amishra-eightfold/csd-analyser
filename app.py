import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback
import os
import time
from typing import Dict, List, Any, Tuple, Optional

# Import authentication module
import auth

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

def display_user_header() -> None:
    """Display user information and logout button in the header."""
    user_info = auth.get_current_user()
    if user_info:
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.markdown(f"**Welcome, {user_info.get('name', 'User')}** ({user_info.get('email', '')})")
        
        with col2:
            st.markdown(f"*Logged in via Google OAuth*")
        
        with col3:
            if st.button("🚪 Logout", key="logout_button"):
                auth.logout()

def display_data_filters() -> Dict[str, List[str]]:
    """Display data filters in the main area and return selected values."""
    # Check if data is loaded
    data_loaded = st.session_state.get('data_loaded', False)
    
    if data_loaded and 'analysis_data' in st.session_state and st.session_state.analysis_data is not None:
        st.markdown("### 🔍 Data Filters")
        
        df = st.session_state.analysis_data
        
        # Create three columns for the filters
        col1, col2, col3 = st.columns(3)
        
        # Get reset counter for dynamic widget keys
        reset_counter = st.session_state.get('filter_reset_counter', 0)
        
        with col1:
            # Product Area filter - include null/empty values
            product_areas = sorted(df['Product_Area__c'].fillna('(Not Specified)').unique().tolist())
            selected_product_areas = st.multiselect(
                "Product Area",
                options=product_areas,
                default=[],  # Always start empty
                key=f"product_area_filter_{reset_counter}",
                help="Select product areas to filter by (leave empty to show all)"
            )
        
        with col2:
            # Product Feature filter - include null/empty values
            product_features = sorted(df['Product_Feature__c'].fillna('(Not Specified)').unique().tolist())
            selected_product_features = st.multiselect(
                "Product Feature", 
                options=product_features,
                default=[],  # Always start empty
                key=f"product_feature_filter_{reset_counter}",
                help="Select product features to filter by (leave empty to show all)"
            )
        
        with col3:
            # Case Status filter
            case_statuses = sorted(df['Status'].unique().tolist())
            selected_statuses = st.multiselect(
                "Case Status",
                options=case_statuses,
                default=[],  # Always start empty
                key=f"case_status_filter_{reset_counter}",
                help="Select case statuses to filter by (leave empty to show all)"
            )
        
        # Add Apply Filters and Clear Filters buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            pass  # Empty column for spacing
        with col2:
            apply_filters = st.button("🔍 Apply Filters", type="primary", use_container_width=True)
        with col3:
            clear_filters = st.button("🧹 Clear Filters", use_container_width=True)
        
        # Handle button clicks
        if apply_filters:
            st.session_state.selected_product_areas = selected_product_areas
            st.session_state.selected_product_features = selected_product_features  
            st.session_state.selected_statuses = selected_statuses
            st.success("✅ Filters applied successfully!")
        
        if clear_filters:
            # Clear stored filters immediately
            if hasattr(st.session_state, 'selected_product_areas'):
                del st.session_state.selected_product_areas
            if hasattr(st.session_state, 'selected_product_features'):
                del st.session_state.selected_product_features
            if hasattr(st.session_state, 'selected_statuses'):
                del st.session_state.selected_statuses
            
            # Increment filter reset counter to force widget reset
            if 'filter_reset_counter' not in st.session_state:
                st.session_state.filter_reset_counter = 0
            st.session_state.filter_reset_counter += 1
            
            st.success("🧹 Filters cleared! Showing all data.")
            st.rerun()  # Refresh to show cleared filters
        
        # Show current filter status if filters have been applied
        if hasattr(st.session_state, 'selected_product_areas'):
            current_areas = st.session_state.get('selected_product_areas', [])
            current_features = st.session_state.get('selected_product_features', [])
            current_statuses = st.session_state.get('selected_statuses', [])
            
            total_cases = len(df)
            
            # Check if any filters are actually applied
            any_filters_applied = bool(current_areas or current_features or current_statuses)
            
            if any_filters_applied:
                # Apply filters step by step (same logic as in main display)
                filtered_df = df.copy()
                
                if current_areas:
                    df_temp = filtered_df.copy()
                    df_temp['Product_Area__c'] = df_temp['Product_Area__c'].fillna('(Not Specified)')
                    filtered_df = df_temp[df_temp['Product_Area__c'].isin(current_areas)]
                
                if current_features:
                    df_temp = filtered_df.copy()
                    df_temp['Product_Feature__c'] = df_temp['Product_Feature__c'].fillna('(Not Specified)')
                    filtered_df = df_temp[df_temp['Product_Feature__c'].isin(current_features)]
                    
                if current_statuses:
                    filtered_df = filtered_df[filtered_df['Status'].isin(current_statuses)]
                
                filtered_cases = len(filtered_df)
                
                # Build filter description
                active_filters = []
                if current_areas:
                    active_filters.append(f"Product Area: {', '.join(current_areas[:2])}{'...' if len(current_areas) > 2 else ''}")
                if current_features:
                    active_filters.append(f"Product Feature: {', '.join(current_features[:2])}{'...' if len(current_features) > 2 else ''}")
                if current_statuses:
                    active_filters.append(f"Status: {', '.join(current_statuses[:2])}{'...' if len(current_statuses) > 2 else ''}")
                
                filter_description = " | ".join(active_filters)
                st.info(f"📊 Filters applied: Showing {filtered_cases} of {total_cases} cases ({filter_description})")
            else:
                st.info("📊 No filters applied - showing all cases")
        
        return {
            'product_areas': selected_product_areas,
            'product_features': selected_product_features,
            'statuses': selected_statuses
        }
    else:
        # This should not be called when no data is loaded
        return {
            'product_areas': [],
            'product_features': [],
            'statuses': []
        }

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
    if 'history_df' not in st.session_state:
        st.session_state.history_df = pd.DataFrame()
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None

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
    
    # Handle authentication first
    if not auth.handle_auth():
        return  # Stop execution if not authenticated
    
    # Add user info and logout button to the top of the page
    display_user_header()
    
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
            selected_customers=selected_customers,
            include_history=settings.get('include_history_data', True)
        )
        
        if cases_df is not None and not cases_df.empty:
            # Store history data in session state
            st.session_state.history_df = history_df
            
            # Process PII if enabled
            if st.session_state.enable_pii_processing:
                cases_df, _ = process_pii_in_dataframe(cases_df)
                if comments_df is not None:
                    comments_df, _ = process_pii_in_dataframe(comments_df)
            
            # Prepare data for analysis
            df = prepare_data_for_analysis(cases_df)
            
            # Store the processed data in session state (no filtering at this stage)
            st.session_state.analysis_data = df
            st.session_state.data_loaded = True
            
            # Debug: Show actual status values in the data
            if st.session_state.debug_mode:
                st.write("🔍 **Debug: Data Analysis**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Status Values:**")
                    status_counts = df['Status'].value_counts()
                    st.write(status_counts)
                
                with col2:
                    st.write("**Product Area Values:**")
                    area_counts = df['Product_Area__c'].fillna('(NULL)').value_counts()
                    st.write(area_counts)
                
                with col3:
                    st.write("**Product Feature Values:**")  
                    feature_counts = df['Product_Feature__c'].fillna('(NULL)').value_counts()
                    st.write(feature_counts)
                
                # Show null value analysis
                st.write("**Null Value Analysis:**")
                null_analysis = {
                    'Product_Area__c': df['Product_Area__c'].isnull().sum(),
                    'Product_Feature__c': df['Product_Feature__c'].isnull().sum(),
                    'Status': df['Status'].isnull().sum()
                }
                st.write(f"Total cases: {len(df)}")
                st.write(f"Cases with null Product Area: {null_analysis['Product_Area__c']}")
                st.write(f"Cases with null Product Feature: {null_analysis['Product_Feature__c']}")
                st.write(f"Cases with null Status: {null_analysis['Status']}")
            
            st.success(f"✅ Data loaded successfully! {len(df)} cases available. Use the status filter to narrow down your view.")
        else:
            st.warning("No data found for the selected criteria.")
            st.session_state.data_loaded = False
    
    # Handle current session export if requested
    if settings.get('export_current_data', False) and st.session_state.get('data_loaded', False):
        export_format = settings.get('export_format', 'CSV')
        try:
            from src.data.data_handler import export_data
            df = st.session_state.analysis_data
            success, filename, file_data = export_data(df, export_format)
            
            if success:
                st.success(f"Data exported successfully!")
                # Set appropriate MIME type
                if export_format == "Excel":
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                elif export_format == "PowerPoint":
                    mime_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                elif export_format == "JSON":
                    mime_type = "application/json"
                else:  # CSV
                    mime_type = "text/csv"
                
                st.download_button(
                    label=f"Download {export_format} File",
                    data=file_data,
                    file_name=filename,
                    mime=mime_type
                )
            else:
                st.error("Failed to export data. Please try again.")
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")
            if st.session_state.debug_mode:
                st.exception(e)
    
    # Display data filters and data if available (either freshly loaded or from session state)
    if st.session_state.get('data_loaded', False) and 'analysis_data' in st.session_state:
        # Show data filters section
        display_data_filters()
        # Get the full dataset
        full_df = st.session_state.analysis_data
        
        # Apply filters from session state (only if they exist)
        if hasattr(st.session_state, 'selected_product_areas'):
            selected_product_areas = st.session_state.get('selected_product_areas', [])
            selected_product_features = st.session_state.get('selected_product_features', [])
            selected_statuses = st.session_state.get('selected_statuses', [])
            
            # Apply filters properly - only filter if values are selected
            df = full_df.copy()
            
            if selected_product_areas:
                # Handle null values by replacing them temporarily
                df_temp = df.copy()
                df_temp['Product_Area__c'] = df_temp['Product_Area__c'].fillna('(Not Specified)')
                df = df_temp[df_temp['Product_Area__c'].isin(selected_product_areas)]
            
            if selected_product_features:
                # Handle null values by replacing them temporarily
                df_temp = df.copy()
                df_temp['Product_Feature__c'] = df_temp['Product_Feature__c'].fillna('(Not Specified)')
                df = df_temp[df_temp['Product_Feature__c'].isin(selected_product_features)]
                
            if selected_statuses:
                df = df[df['Status'].isin(selected_statuses)]
            
            # Check if filtering actually reduced the dataset
            is_filtered = len(df) < len(full_df)
        else:
            # No filters applied yet
            df = full_df
            is_filtered = False
        
        # Display data summary
        st.subheader("Data Summary")
        
        summary = get_data_summary(df)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            label = "Filtered Tickets" if is_filtered else "Total Tickets"
            st.metric(label, summary.get('total_tickets', 0))
        with col2:
            resolution_time = summary.get('avg_resolution_time')
            st.metric("Avg. Resolution Time", f"{resolution_time:.1f} days" if resolution_time is not None else "No data")
        with col3:
            csat_value = summary.get('avg_csat')
            st.metric("CSAT", f"{csat_value:.1f}" if csat_value is not None else "No data")
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
        # Show disabled filters when no data is loaded
        st.markdown("### 🔍 Data Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.multiselect("Product Area", options=[], disabled=True, key="product_area_filter_disabled")
        with col2:
            st.multiselect("Product Feature", options=[], disabled=True, key="product_feature_filter_disabled") 
        with col3:
            st.multiselect("Case Status", options=[], disabled=True, key="case_status_filter_disabled")
        
        # Show disabled Apply button
        col_button = st.columns([1, 1, 1])[1]
        with col_button:
            st.button("🔍 Apply Filters", disabled=True, use_container_width=True)
        
        st.info("🔄 Load data to enable filtering options")
        st.info("Select customers and date range, then click 'Fetch Data' to begin analysis.")

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
                default=df['Status'].unique(),
                help="Additional status filtering (already filtered for active cases if enabled)"
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
                options=df['Product_Area__c'].unique(),
                default=df['Product_Area__c'].unique()
            )
        
        # Apply filters
        filtered_df = df[
            (df['Status'].isin(filter_status)) &
            (df['Priority'].isin(filter_priority)) &
            (df['Product_Area__c'].isin(filter_product))
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
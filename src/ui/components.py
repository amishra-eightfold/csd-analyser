"""UI components for the support ticket analysis dashboard."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import traceback
from typing import Dict, List, Any, Tuple, Optional
import logging
from config.logging_config import get_logger

# Initialize logger
logger = get_logger('ui')

def debug(message, data=None, category="ui"):
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
                logger.info(f"{message} - {json.dumps(data)}")
            except:
                logger.info(f"{message} - {str(data)}")
        else:
            logger.info(f"{message} - {str(data)}")
    else:
        logger.info(message)

def create_sidebar() -> Dict[str, Any]:
    """Create the sidebar with filters and options.
    
    Returns:
        A dictionary containing the selected filter values
    """
    settings = {}
    
    st.sidebar.title("Support Ticket Analysis")
    
    # Date Range Selection
    st.sidebar.subheader("Date Range")
    
    # Initialize default date range if not in session state
    if 'date_range' not in st.session_state:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=90)
        st.session_state.date_range = (start_date, end_date)
    
    # Date range selector
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=st.session_state.date_range,
        min_value=datetime(2010, 1, 1).date(),
        max_value=datetime.now().date(),
        key="date_range_input"
    )
    
    # Handle single date selection
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range
    
    settings['start_date'] = start_date
    settings['end_date'] = end_date
    
    debug("Date range selected", {
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat()
    })
    
    # Customer Selection
    st.sidebar.subheader("Customer")
    
    # Get available customers if fetched
    all_customers = ["All Customers"]
    if 'available_customers' in st.session_state:
        customer_list = st.session_state.available_customers
        all_customers.extend(customer_list)
    
    selected_customer = st.sidebar.selectbox(
        "Select Customer",
        options=all_customers,
        index=0,
        key="customer_selector"
    )
    
    settings['selected_customer'] = selected_customer
    debug("Customer selected", {'customer': selected_customer})
    
    # Advanced Options
    st.sidebar.subheader("Analysis Options")
    
    # Enable AI Analysis
    enable_ai_analysis = st.sidebar.checkbox("Enable AI Analysis", value=False, key="enable_ai")
    settings['enable_ai_analysis'] = enable_ai_analysis
    
    # Enable PII Protection
    enable_pii_processing = st.sidebar.checkbox("Enable PII Protection", value=True, key="enable_pii")
    settings['enable_pii_processing'] = enable_pii_processing
    
    # Debug Mode
    if st.sidebar.checkbox("Debug Mode", value=False, key="debug_mode"):
        st.session_state.debug_mode = True
        debug("Debug mode enabled")
    else:
        st.session_state.debug_mode = False
    
    # Fetch Data Button
    if st.sidebar.button("Fetch Data", key="fetch_data_button"):
        settings['fetch_data'] = True
        debug("Fetch data button clicked")
    else:
        settings['fetch_data'] = False
    
    # Export Options
    st.sidebar.subheader("Export Options")
    export_format = st.sidebar.radio(
        "Export Format",
        options=["CSV", "Excel", "JSON"],
        index=0,
        key="export_format"
    )
    settings['export_format'] = export_format
    
    # Export button
    if st.sidebar.button("Export Data", key="export_button"):
        settings['export_data'] = True
        debug("Export data button clicked", {'format': export_format})
    else:
        settings['export_data'] = False
    
    # Application Information
    with st.sidebar.expander("About", expanded=False):
        st.write("""
        **Support Ticket Analyzer**
        
        Analyze support ticket data to identify patterns,
        track resolution times, and improve customer satisfaction.
        
        Version: 1.0.0
        """)
    
    return settings


def display_header(df: Optional[pd.DataFrame] = None) -> None:
    """Display the header section with key metrics.
    
    Args:
        df: DataFrame containing the support ticket data (optional)
    """
    st.title("Support Ticket Analysis Dashboard")
    
    # Display key metrics if data is available
    if df is not None and not df.empty:
        try:
            # Create columns for metrics
            col1, col2, col3, col4 = st.columns(4)
            
            # Total tickets
            with col1:
                st.metric("Total Tickets", len(df))
            
            # Open tickets
            with col2:
                open_tickets = len(df[df['Status'].isin(['Open', 'In Progress', 'Pending'])])
                st.metric("Open Tickets", open_tickets)
            
            # Average resolution time
            with col3:
                closed_tickets = df[df['Status'] == 'Closed'].copy()
                if not closed_tickets.empty:
                    avg_resolution_days = (
                        (closed_tickets['Closed Date'] - closed_tickets['Created Date']).dt.total_seconds() / (24 * 3600)
                    ).mean()
                    st.metric("Avg Resolution (Days)", f"{avg_resolution_days:.1f}")
                else:
                    st.metric("Avg Resolution (Days)", "N/A")
            
            # Average CSAT
            with col4:
                if 'CSAT' in df.columns:
                    csat_data = df['CSAT'].dropna()
                    if not csat_data.empty:
                        avg_csat = csat_data.mean()
                        st.metric("Average CSAT", f"{avg_csat:.2f}/5")
                    else:
                        st.metric("Average CSAT", "N/A")
                else:
                    st.metric("Average CSAT", "N/A")
            
            # Add date range information
            st.caption(f"Data from {df['Created Date'].min().strftime('%Y-%m-%d')} to {df['Created Date'].max().strftime('%Y-%m-%d')}")
        
        except Exception as e:
            error_msg = f"Error displaying header metrics: {str(e)}"
            st.error(error_msg)
            debug(error_msg, {'traceback': traceback.format_exc()}, category="error")
            logger.error(error_msg, exc_info=True)
    else:
        # Show placeholder information when no data is available
        st.info("Welcome to the Support Ticket Analysis Dashboard. Please use the sidebar to fetch data.")


def display_data_table(df: pd.DataFrame, enable_pii_processing: bool = False) -> None:
    """Display the data table with ticket information.
    
    Args:
        df: DataFrame containing the support ticket data
        enable_pii_processing: Whether to enable PII protection
    """
    st.header("Ticket Data")
    
    try:
        # Process PII if enabled
        display_df = df.copy()
        if enable_pii_processing and hasattr(st.session_state, 'pii_handler'):
            display_df, _ = st.session_state.pii_handler.process_dataframe(
                display_df,
                ['Subject', 'Description', 'Comments', 'Account_Name']
            )
            st.info("🔒 PII Protection is enabled. Some data is masked for privacy.")
        
        # Select columns for display
        display_columns = [
            'Case Number', 'Subject', 'Status', 'Priority', 'Created Date', 
            'Account_Name', 'Product Area', 'Resolution Time (Days)', 'CSAT'
        ]
        
        available_columns = [col for col in display_columns if col in display_df.columns]
        
        if available_columns:
            # Add search box
            search_term = st.text_input("Search tickets", key="ticket_search")
            
            filtered_df = display_df
            if search_term:
                search_mask = filtered_df.astype(str).apply(
                    lambda row: row.str.contains(search_term, case=False).any(), 
                    axis=1
                )
                filtered_df = filtered_df[search_mask]
                st.caption(f"Found {len(filtered_df)} matching tickets")
            
            # Display data with pagination
            rows_per_page = st.slider("Rows per page", min_value=5, max_value=50, value=10, step=5)
            
            # Calculate pagination
            total_pages = max(1, (len(filtered_df) + rows_per_page - 1) // rows_per_page)
            
            # Add page selector
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 1
                
            # Ensure current page is valid
            if st.session_state.current_page > total_pages:
                st.session_state.current_page = 1
                
            # Page navigation
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                if st.button("Previous", disabled=(st.session_state.current_page <= 1)):
                    st.session_state.current_page -= 1
                    
            with col2:
                st.write(f"Page {st.session_state.current_page} of {total_pages}")
                
            with col3:
                if st.button("Next", disabled=(st.session_state.current_page >= total_pages)):
                    st.session_state.current_page += 1
            
            # Calculate slice indices
            start_idx = (st.session_state.current_page - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(filtered_df))
            
            # Display data
            page_df = filtered_df.iloc[start_idx:end_idx]
            display_df = page_df[available_columns].copy()
            
            # Format dates for display
            for col in display_df.columns:
                if 'Date' in col and pd.api.types.is_datetime64_any_dtype(display_df[col]):
                    display_df[col] = display_df[col].dt.strftime('%Y-%m-%d')
            
            # Display the dataframe
            st.dataframe(display_df, use_container_width=True)
            
            # Log activity
            debug("Data table displayed", {
                'total_rows': len(filtered_df),
                'displayed_rows': len(page_df),
                'current_page': st.session_state.current_page,
                'total_pages': total_pages,
                'search_applied': bool(search_term)
            })
        else:
            st.warning("No data columns available for display.")
    
    except Exception as e:
        error_msg = f"Error displaying data table: {str(e)}"
        st.error(error_msg)
        debug(error_msg, {'traceback': traceback.format_exc()}, category="error")
        logger.error(error_msg, exc_info=True)
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.exception(e)
            
def display_debug_info() -> None:
    """Display debug information if debug mode is enabled."""
    if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
        st.markdown("---")
        with st.expander("Debug Information", expanded=False):
            st.write("### Session State Variables")
            # Display session state variables excluding large objects
            session_vars = {
                key: value for key, value in st.session_state.items()
                if not isinstance(value, (pd.DataFrame, dict, list)) or key in ['date_range', 'debug_mode']
            }
            st.json(session_vars)
            
            # Display the current timestamp
            st.write(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Display basic system info
            st.write("### System Information")
            import platform
            system_info = {
                "Platform": platform.platform(),
                "Python Version": platform.python_version(),
                "Streamlit Version": st.__version__
            }
            st.json(system_info)
            
            # Display logger information
            st.write("### Logger Status")
            if hasattr(st.session_state, 'debug_logger'):
                if st.session_state.debug_logger.get_logs():
                    # Display the most recent logs (last 10)
                    st.write("#### Recent Logs")
                    for log in st.session_state.debug_logger.get_logs()[-10:]:
                        st.text(f"{log['timestamp']} - {log['category']} - {log['message']}")
                    
                    # Button to clear logs
                    if st.button("Clear Debug Logs"):
                        st.session_state.debug_logger.clear_logs()
                        st.success("Debug logs cleared.")
                else:
                    st.info("No debug logs available.")
            else:
                st.warning("Debug logger not initialized.") 
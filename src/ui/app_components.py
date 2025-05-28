"""UI components specifically for the main application interface.

This module provides UI components for the main application interface, including:
- Sidebar configuration and rendering
- Debug UI components
- Status indicators and visual feedback
- Main page layout helpers
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import traceback
import time
import json
import numpy as np
import os
from typing import Dict, List, Any, Tuple, Optional
import logging

from utils.pii_handler import get_privacy_status_indicator
from utils.debug_logger import DebugLogger
from config.logging_config import get_logger, log_error

# Initialize logger
logger = get_logger('ui')

def debug(message, data=None, category="app"):
    """Enhanced debug function that uses both DebugLogger and file logging."""
    # Log to DebugLogger if available
    if hasattr(st.session_state, 'debug_logger'):
        st.session_state.debug_logger.log(message, data, category)

    # Log to file logger
    logger = get_logger(category)
    if data is not None:
        # Convert NumPy types to Python native types
        if isinstance(data, dict):
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
        else:
            logger.info(f"{message} - {str(data)}")
    else:
        logger.info(message)

def setup_application_sidebar():
    """Create the application sidebar with all options and settings.
    
    Returns:
        Dict[str, Any]: A dictionary containing all the selected settings
    """
    settings = {}
    
    st.sidebar.title("Settings")
    
    # Analysis Options
    st.sidebar.header("Analysis Options")
    settings['enable_detailed_analysis'] = st.sidebar.checkbox(
        "Enable Detailed Analysis",
        value=True,
        help="Show comprehensive metrics and visualizations"
    )
    settings['enable_ai_analysis'] = st.sidebar.checkbox(
        "Enable AI Analysis",
        value=False,
        help="Use AI to generate insights and recommendations"
    )
    settings['enable_pii_processing'] = st.sidebar.checkbox(
        "Enable PII Protection",
        value=False,
        help="Remove sensitive information before analysis"
    )
    settings['include_history_data'] = st.sidebar.checkbox(
        "Include History Data",
        value=True,
        help="Load case history data for priority and status time analysis"
    )
    
    # Export Section in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("📥 Exports")
    
    # Check if data is loaded in current session
    current_session_has_data = st.session_state.get('data_loaded', False)
    
    if current_session_has_data:
        # Show current session export options
        st.sidebar.markdown("### Current Session")
        export_format = st.sidebar.selectbox(
            "Export Format",
            ["Excel", "CSV", "PowerPoint"],
            help="Export current analysis data"
        )
        
        if st.sidebar.button("Export Current Data", help="Export data from current analysis"):
            settings['export_current_data'] = True
            settings['export_format'] = export_format
        else:
            settings['export_current_data'] = False
    
    # Show previous exports if any exist
    from utils.data_export import get_available_exports
    exports = get_available_exports()
    
    if exports:
        st.sidebar.markdown("### Previous Exports")
        with st.sidebar.expander("📂 View Previous Exports", expanded=False):
            for export in exports:
                # Format timestamp safely
                timestamp_display = (
                    export['timestamp'].strftime('%Y-%m-%d %H:%M') 
                    if export['timestamp'] is not None 
                    else export['timestamp_str']
                )
                
                with st.sidebar.expander(f"{export['customer_name']} - {timestamp_display}"):
                    # Show export details
                    st.write(f"Files available:")
                    for file in export['files']:
                        file_size_mb = file['size'] / (1024 * 1024)
                        st.write(f"- {file['name']} ({file_size_mb:.1f} MB)")
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        # Individual file downloads
                        for file in export['files']:
                            with open(file['path'], 'rb') as f:
                                st.download_button(
                                    label=f"📄 {file['type']}",
                                    data=f,
                                    file_name=file['name'],
                                    mime=f"text/{file['type'].lower()}"
                                )
                    
                    with col2:
                        # Download all as ZIP
                        try:
                            from utils.data_export import create_customer_export_zip
                            zip_buffer, zip_filename = create_customer_export_zip(
                                export['customer_name'],
                                export['timestamp_str']
                            )
                            st.download_button(
                                label="📦 Download All",
                                data=zip_buffer,
                                file_name=zip_filename,
                                mime="application/zip",
                                help="Download all files as ZIP"
                            )
                        except Exception as e:
                            st.error("Error creating ZIP file")
                            if st.session_state.debug_mode:
                                st.exception(e)
    else:
        st.sidebar.info("No previous exports found.")
    
    # Date Range Selection
    st.sidebar.markdown("---")
    st.sidebar.header("Date Range")
    
    # Default date range if not already set
    if 'date_range' not in st.session_state:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        st.session_state.date_range = (start_date, end_date)
    
    date_range = st.sidebar.date_input(
        "Select date range",
        value=st.session_state.date_range,
        min_value=datetime(2020, 1, 1),
        max_value=datetime.now()
    )
    
    # Handle both single date and date range cases
    if isinstance(date_range, tuple) and len(date_range) == 2:
        settings['start_date'] = date_range[0]
        settings['end_date'] = date_range[1]
    else:
        settings['start_date'] = date_range
        settings['end_date'] = date_range
    

    # Customer Selection
    st.sidebar.markdown("---")
    st.sidebar.header("Customer Selection")
    
    if st.session_state.customers:
        multiselect = st.sidebar.multiselect(
            "Select Customers",
            options=st.session_state.customers,
            help="Select one or more customers to analyze"
        )
        settings['selected_customers'] = multiselect
    else:
        st.sidebar.warning("No customers available.")
        settings['selected_customers'] = []
    
    # Debug Mode Section
    if os.getenv('ENVIRONMENT', 'development').lower() != 'production':
        st.sidebar.markdown("---")
        st.sidebar.header("Developer Options")
        was_debug_enabled = st.session_state.get('debug_mode', False)
        st.session_state.debug_mode = st.sidebar.checkbox(
            "Enable Debug Mode",
            value=st.session_state.debug_mode,
            help="Show additional debugging information and detailed error messages",
            key="debug_mode_checkbox"
        )
        
        # Log initial debug message only when first enabled
        if st.session_state.debug_mode and not was_debug_enabled:
            debug("Debug mode enabled")
            debug("Application startup", {
                'environment': os.getenv('ENVIRONMENT', 'development'),
                'session_state_keys': list(st.session_state.keys())
            })
    
    # Add the Fetch Data button
    st.sidebar.markdown("---")
    settings['fetch_data'] = st.sidebar.button("Fetch Data", key="fetch_data_button")
    
    return settings

def display_privacy_status():
    """Display privacy status indicator in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.header("Privacy Status")
    
    # Get latest validation score and status
    audit_summary = st.session_state.pii_handler.get_audit_summary()
    validation_score = audit_summary['latest_validation_score']
    status = get_privacy_status_indicator(validation_score)
    
    # Display status with appropriate color and animation
    st.sidebar.markdown(
        f"""
        <div class='privacy-status {status["status"].lower()}'>
            <h3 style='color: {status["color"]}; margin: 0;'>{status["status"]}</h3>
            <p style='margin: 5px 0 0 0;'>{status["message"]}</p>
            <div style='margin-top: 10px;'>
                <div style='background: #eee; height: 4px; border-radius: 2px;'>
                    <div style='background: {status["color"]}; width: {validation_score * 100}%; height: 100%; border-radius: 2px; transition: width 0.5s ease;'></div>
                </div>
                <p style='text-align: right; font-size: 0.8em; margin: 2px 0;'>Score: {validation_score:.2f}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display audit summary with animation
    if st.sidebar.checkbox("Show Privacy Audit Summary"):
        st.sidebar.markdown(
            f"""
            <div class='section-transition'>
                <div class='status-indicator success'>
                    <h4 style='margin: 0;'>PII Detection Summary</h4>
                    <p style='margin: 5px 0;'>Total operations: {audit_summary['total_operations']}</p>
                    <p style='margin: 5px 0;'>Total PII detected: {audit_summary['total_pii_detected']}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if audit_summary['by_type']:
            st.sidebar.markdown("<div class='section-transition'>", unsafe_allow_html=True)
            st.sidebar.write("PII Types Detected:")
            for pii_type, count in audit_summary['by_type'].items():
                st.sidebar.markdown(
                    f"""
                    <div class='status-indicator processing' style='margin: 4px 0;'>
                        {pii_type}: {count}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.sidebar.markdown("</div>", unsafe_allow_html=True)
        
        # Show debug information if debug mode is enabled
        if st.session_state.debug_mode:
            st.sidebar.markdown(
                """
                <div class='section-transition'>
                    <div class='status-indicator processing'>
                        <h4 style='margin: 0;'>Debug Information</h4>
                """,
                unsafe_allow_html=True
            )
            st.sidebar.write("- PII Handler Stats:", st.session_state.pii_handler.pii_stats)
            st.sidebar.write("- Latest Audit Record:", st.session_state.pii_handler.audit_records[-1] if st.session_state.pii_handler.audit_records else "No audit records")
            st.sidebar.markdown("</div></div>", unsafe_allow_html=True)

def process_pii_in_dataframe(df):
    """Process PII in DataFrame with visual feedback.
    
    Args:
        df: The pandas DataFrame to process
        
    Returns:
        Tuple[pd.DataFrame, Dict]: The processed DataFrame and PII stats
    """
    if not st.session_state.enable_pii_processing:
        return df, {}
    
    # Create a placeholder for the progress
    progress_placeholder = st.empty()
    progress_placeholder.markdown(
        """
        <div class='loading status-indicator processing'>
            <p>🔍 Scanning for PII...</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    try:
        # Process the DataFrame
        processed_df, pii_stats = st.session_state.pii_handler.process_dataframe(
            df,
            ['Subject', 'Description', 'Comments', 'Account_Name', 'Product_Area__c', 'Product_Feature__c']
        )
        
        # Show success message with stats
        pii_count = pii_stats.get('pii_detected', 0)
        progress_placeholder.markdown(
            f"""
            <div class='status-indicator success'>
                <p>✅ PII Processing Complete</p>
                <p>Found and processed {pii_count} instances of PII</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        return processed_df, pii_stats
    except Exception as e:
        # Show error message
        progress_placeholder.markdown(
            f"""
            <div class='status-indicator error'>
                <p>❌ Error Processing PII</p>
                <p>{str(e)}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        log_error(f"Error processing PII: {str(e)}", traceback.format_exc())
        raise e
    finally:
        # Clear the progress placeholder after a delay
        time.sleep(2)
        progress_placeholder.empty()

def display_connection_status(message, status_type="processing"):
    """Display connection status with appropriate styling.
    
    Args:
        message: The message to display
        status_type: The type of status (processing, success, error)
    
    Returns:
        streamlit.empty: The status placeholder for further updates
    """
    status_placeholder = st.empty()
    status_placeholder.markdown(
        f"""
        <div class='loading status-indicator {status_type}'>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    return status_placeholder

def apply_custom_css():
    """Apply custom CSS styling to the application."""
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .stSelectbox>div>div>input {
        border-radius: 5px;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    h1 {
        color: #1E3D59;
        font-weight: bold;
        padding-bottom: 1rem;
        border-bottom: 2px solid #1E3D59;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stPlotlyChart {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    .stPlotlyChart:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Status Indicators */
    .status-indicator {
        padding: 8px 12px;
        border-radius: 4px;
        margin: 8px 0;
        transition: all 0.3s ease;
    }
    .status-indicator.processing {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        color: #856404;
    }
    .status-indicator.success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        color: #155724;
    }
    .status-indicator.error {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .loading {
        animation: pulse 1.5s infinite;
        padding: 1rem;
        border-radius: 5px;
        background-color: #f8f9fa;
        text-align: center;
    }
    
    /* Section Transitions */
    .section-transition {
        opacity: 0;
        transform: translateY(20px);
        animation: fadeInUp 0.5s forwards;
    }
    @keyframes fadeInUp {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Privacy Status Styles */
    .privacy-status {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .privacy-status.excellent {
        background-color: rgba(40, 167, 69, 0.1);
        border: 1px solid #28a745;
    }
    .privacy-status.good {
        background-color: rgba(0, 123, 255, 0.1);
        border: 1px solid #007bff;
    }
    .privacy-status.fair {
        background-color: rgba(255, 193, 7, 0.1);
        border: 1px solid #ffc107;
    }
    .privacy-status.poor {
        background-color: rgba(220, 53, 69, 0.1);
        border: 1px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True) 
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from io import BytesIO
from pptx import Presentation
from pptx.util import Inches, Pt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from salesforce_config import init_salesforce, execute_soql_query
import openai
from openai import OpenAI
import time
import json
import re
import os
import html
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os.path
import tiktoken
from textblob import TextBlob
from utils.text_processing import clean_text, remove_pii, prepare_text_for_ai, get_technical_stopwords, get_highest_priority_from_history
from processors.salesforce_processor import SalesforceDataProcessor
from visualizers.salesforce_visualizer import SalesforceVisualizer
from visualizers.advanced_visualizations import create_csat_analysis, create_word_clouds, create_root_cause_analysis, create_first_response_analysis
from utils.pii_handler import PIIHandler, get_privacy_status_indicator
from typing import Tuple, Dict
from utils.token_manager import TokenManager, TokenInfo, convert_value_for_json
import logging

# Set Seaborn and Matplotlib style
sns.set_theme(style="whitegrid")

# Custom color palettes for different visualizations
VIRIDIS_PALETTE = ["#440154", "#3B528B", "#21918C", "#5EC962", "#FDE725"]  # Viridis colors
AQUA_PALETTE = ["#E0F7FA", "#80DEEA", "#26C6DA", "#00ACC1", "#00838F", "#006064"]   # Material Cyan/Aqua
PRIORITY_COLORS = {
    'P1': VIRIDIS_PALETTE[0],
    'P2': VIRIDIS_PALETTE[1],
    'P3': VIRIDIS_PALETTE[2],
    'P4': VIRIDIS_PALETTE[3]
}

# Define custom color palettes for each visualization type
VOLUME_PALETTE = [AQUA_PALETTE[2], AQUA_PALETTE[4]]  # Two distinct colors for Created/Closed
PRIORITY_PALETTE = VIRIDIS_PALETTE  # Viridis for priority levels
CSAT_PALETTE = AQUA_PALETTE  # Aqua palette for CSAT
HEATMAP_PALETTE = "Viridis"  # Viridis colorscale for heatmaps

# Create an extended palette for root causes
ROOT_CAUSE_PALETTE = VIRIDIS_PALETTE

# Set default style
plt.style.use("seaborn-v0_8-whitegrid")

# Check for debug mode
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() in ('true', '1', 't')

# Page Configuration
st.set_page_config(
    page_title="Support Ticket Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'customers' not in st.session_state:
    st.session_state.customers = None
if 'selected_customers' not in st.session_state:
    st.session_state.selected_customers = []
if 'date_range' not in st.session_state:
    st.session_state.date_range = None
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

# Custom CSS
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

# Helper function to conditionally display debug information
def debug(message, data=None):
    """Enhanced debug function that logs to console, file and Streamlit."""
    if not hasattr(st.session_state, 'debug_mode') or not st.session_state.debug_mode:
        return
        
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Format the debug message
    if data is not None:
        if isinstance(data, dict):
            data_str = "\n  " + "\n  ".join([f"{k}: {v}" for k, v in data.items()])
        else:
            data_str = str(data)
        log_message = f"[DEBUG] {timestamp} - {message}:{data_str}"
    else:
        log_message = f"[DEBUG] {timestamp} - {message}"
    
    # Print to console
    print(log_message)
    
    # Write to file
    try:
        with open('debug.log', 'a') as f:
            f.write(log_message + '\n')
    except Exception as e:
        print(f"Error writing to debug log file: {str(e)}")
    
    # Show in Streamlit debug container
    if not hasattr(st.session_state, 'debug_container'):
        st.session_state.debug_container = []
    
    # Add to debug container and keep only last 1000 messages
    st.session_state.debug_container.append(log_message)
    if len(st.session_state.debug_container) > 1000:
        st.session_state.debug_container = st.session_state.debug_container[-1000:]

def process_pii_in_dataframe(df):
    """Process PII in DataFrame with visual feedback."""
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
        progress_placeholder.markdown(
            f"""
            <div class='status-indicator success'>
                <p>✅ PII Processing Complete</p>
                <p>Found and processed {pii_stats['pii_detected']} instances of PII</p>
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
        raise e
    finally:
        # Clear the progress placeholder after a delay
        time.sleep(2)
        progress_placeholder.empty()

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

# Initialize OpenAI client
client = OpenAI()

def main():
    st.title("Support Ticket Analytics")
    
    # Initialize debug container if not exists
    if 'debug_container' not in st.session_state:
        st.session_state.debug_container = []
    
    # Initialize debug mode if in development environment
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
        
        # Show debug container if debug mode is enabled
        if st.session_state.debug_mode:
            # Log initial debug message only when first enabled
            if not was_debug_enabled:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                st.session_state.debug_container.extend([
                    f"[DEBUG] {timestamp} - Debug mode enabled",
                    f"[DEBUG] {timestamp} - Application startup - Environment: {os.getenv('ENVIRONMENT', 'development')}, Session state keys: {list(st.session_state.keys())}"
                ])
            
            if st.sidebar.button("Clear Debug Log"):
                st.session_state.debug_container = []
            
            # Create a container for debug output
            with st.expander("Debug Log", expanded=True):
                if st.session_state.debug_container:
                    for msg in st.session_state.debug_container:
                        st.code(msg, language="text")
                else:
                    st.info("No debug messages yet.")
    
    # Add debug message for Salesforce connection attempt
    if st.session_state.debug_mode:
        debug("Attempting Salesforce connection")
    
    # Initialize Salesforce connection if not already done
    if st.session_state.sf_connection is None:
        connection_status = st.empty()
        connection_status.markdown(
            """
            <div class='loading status-indicator processing'>
                <p>🔌 Connecting to Salesforce...</p>
            </div>
            """,
            unsafe_allow_html=True
        )
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
                return
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
            return
    
    # Fetch customers if not already loaded
    if st.session_state.customers is None:
        try:
            query = """
                SELECT Account.Account_Name__c 
                FROM Account 
                WHERE Account.Account_Name__c != null
                AND Active_Contract__c = 'Yes'
                ORDER BY Account.Account_Name__c
            """
            records = execute_soql_query(st.session_state.sf_connection, query)
            if records:
                st.session_state.customers = [record['Account_Name__c'] for record in records]
            else:
                st.session_state.customers = []
        except Exception as e:
            st.error(f"Error fetching customers: {str(e)}")
            return
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Analysis Options
    st.sidebar.header("Analysis Options")
    st.session_state.enable_detailed_analysis = st.sidebar.checkbox("Enable Detailed Analysis", value=True)
    st.session_state.enable_ai_analysis = st.sidebar.checkbox("Enable AI Analysis", value=False)
    st.session_state.enable_pii_processing = st.sidebar.checkbox("Enable PII Protection", value=False)
    
    # Date Range Selection
    st.sidebar.markdown("---")
    st.sidebar.header("Date Range")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(datetime.now() - timedelta(days=30), datetime.now()),
        max_value=datetime.now()
    )
    st.session_state.date_range = date_range
    
    # Customer Selection
    st.sidebar.header("Customer Selection")
    if st.session_state.customers:
        selected = st.sidebar.multiselect(
            "Select Customers",
            options=st.session_state.customers,
            default=st.session_state.selected_customers,
            help="Choose one or more customers to analyze"
        )
        st.session_state.selected_customers = selected
    else:
        st.sidebar.warning("No customers found. Please check your Salesforce connection.")
    
    # Export Options
    st.sidebar.header("Export Options")
    if st.sidebar.button("Export Analysis"):
        if st.session_state.data_loaded and len(st.session_state.selected_customers) > 0:
            export_analysis()
        else:
            st.sidebar.error("Please load data and select at least one customer first")
    
    # Help Section
    with st.sidebar.expander("Help"):
        st.markdown("""
        ### How to use this tool:
        1. Select one or more customers from the dropdown
        2. Choose your desired date range
        3. Enable/disable analysis options as needed
        4. Click 'Generate Analysis' to view results
        5. Use 'Export Analysis' to download results
        
        ### Analysis Options:
        - **Detailed Analysis**: Shows comprehensive metrics and visualizations
        - **AI Analysis**: Provides AI-generated insights and recommendations
        - **PII Protection**: Removes sensitive information before analysis
        
        Need help? Contact support@company.com
        """)
    
    # Main Content Area
    if len(st.session_state.selected_customers) > 0:
        with st.spinner("Fetching data..."):
            try:
                df = fetch_data()
                st.session_state.data_loaded = True
                if st.session_state.enable_detailed_analysis:
                    display_detailed_analysis(
                        df, 
                        enable_ai_analysis=st.session_state.enable_ai_analysis,
                        enable_pii_processing=st.session_state.enable_pii_processing
                    )
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                if st.session_state.debug_mode:
                    st.exception(e)
    else:
        st.info("Please select at least one customer to begin analysis")

def fetch_data():
    """Fetch data from Salesforce based on session state."""
    try:
        if not st.session_state.selected_customers:
            st.warning("No customers selected")
            return None
            
        customer_list = "'" + "','".join(st.session_state.selected_customers) + "'"
        start_date, end_date = st.session_state.date_range
        
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
        
        records = execute_soql_query(st.session_state.sf_connection, query)
        if not records:
            st.warning("No data found for the selected criteria")
            return pd.DataFrame()  # Return empty DataFrame instead of None
        
        df = pd.DataFrame(records)
        
        # Extract Account Name from nested structure
        if 'Account' in df.columns and isinstance(df['Account'].iloc[0], dict):
            df['Account_Name'] = df['Account'].apply(lambda x: x.get('Account_Name__c') if isinstance(x, dict) else None)
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
                # Convert to datetime and localize to UTC if naive
                df[col] = pd.to_datetime(df[col], utc=True)
        
        # Calculate resolution time (both dates are now in UTC)
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
        
        if df.empty:
            st.warning("No data available after processing")
            return pd.DataFrame()
            
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        if st.session_state.debug_mode:
            st.exception(e)
        return pd.DataFrame()  # Return empty DataFrame instead of None

def generate_ai_insights(cases_df: pd.DataFrame,
                      comments_df: pd.DataFrame = None,
                      emails_df: pd.DataFrame = None) -> dict:
    """Generate AI insights from ticket data using OpenAI."""
    # Initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    try:
        # Create a summary of the data with error handling
        try:
            summary_stats = {
                'total_tickets': int(len(cases_df)),
                'status_distribution': {k: convert_value_for_json(v) for k, v in cases_df['Status'].value_counts().to_dict().items()},
                'priority_distribution': {k: convert_value_for_json(v) for k, v in cases_df['Priority'].value_counts().to_dict().items()},
                'product_areas': {k: convert_value_for_json(v) for k, v in cases_df['Product Area'].value_counts().to_dict().items()},
                'features': {k: convert_value_for_json(v) for k, v in cases_df['Product Feature'].value_counts().to_dict().items()},
                'root_causes': {k: convert_value_for_json(v) for k, v in cases_df['Root Cause'].value_counts().to_dict().items()}
            }
        except Exception as e:
            logger.error(f"Error creating summary stats: {str(e)}")
            summary_stats = {'total_tickets': int(len(cases_df))}

        # Calculate total tickets and maximum to analyze
        total_tickets = len(cases_df)
        max_tickets = min(50, total_tickets)  # Reduced from 100 to ensure we don't exceed token limits
        analyzed_tickets = min(total_tickets, max_tickets)
        
        logger.info(f"Processing {analyzed_tickets} tickets out of {total_tickets} total")
        
        # Sort by priority and date for better analysis
        priority_map = {'P0': 4, 'P1': 3, 'P2': 2, 'P3': 1, 'Not Set': 0}  # P0 (Critical) to P3 (Low)
        cases_df['priority_score'] = cases_df['Priority'].map(priority_map)
        cases_df = cases_df.sort_values(['priority_score', 'Created Date'], ascending=[False, True])
        
        # Select evenly distributed samples
        if total_tickets > max_tickets:
            step = total_tickets // max_tickets
            selected_indices = list(range(0, total_tickets, step))[:max_tickets]
            analysis_df = cases_df.iloc[selected_indices]
        else:
            analysis_df = cases_df

        # Prepare cases for analysis
        cases_for_analysis = []
        for _, case in analysis_df[['Subject', 'Description', 'Status', 'Priority', 
                                  'Product Area', 'Root Cause', 'Created Date']].iterrows():
            case_dict = {}
            for col, val in case.items():
                case_dict[col] = convert_value_for_json(val)
            cases_for_analysis.append(case_dict)

        # Split cases into smaller chunks to avoid token limits
        chunk_size = 5  # Process 5 tickets at a time
        chunks = [cases_for_analysis[i:i + chunk_size] for i in range(0, len(cases_for_analysis), chunk_size)]
        logger.info(f"Created {len(chunks)} chunks for analysis")
        
        # Initialize combined insights
        all_patterns = []
        all_recommendations = set()
        chunk_summaries = []
        
        # Process each chunk
        for chunk_index, chunk in enumerate(chunks, 1):
            try:
                # Prepare the prompt
                prompt = f"""Analyze these support tickets and provide insights.
                
                Priority Levels:
                - P0: Critical (Highest priority, severe business impact)
                - P1: Urgent (High priority, significant impact)
                - P2: Normal (Standard priority, moderate impact)
                - P3: Low (Lowest priority, minimal impact)

                Overall Statistics:
                {json.dumps(summary_stats, indent=2)}

                Tickets to Analyze:
                {json.dumps(chunk, indent=2)}

                Provide your analysis in the following JSON format:
                {{
                    "chunk_summary": "A summary of key insights from this chunk of tickets",
                    "patterns": [
                        {{"pattern": "Pattern Description", "frequency": "Frequency Description"}}
                    ],
                    "recommendations": [
                        {{"title": "Recommendation Title", "description": "Recommendation Details", "priority": "High/Medium/Low"}}
                    ]
                }}
                """

                # Make API call
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert support ticket analyst. Your responses must be in valid JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000,
                    response_format={ "type": "json_object" }
                )
                
                chunk_response = response.choices[0].message.content.strip()
                
                # Parse response
                try:
                    chunk_insights = json.loads(chunk_response)
                    
                    if isinstance(chunk_insights, dict):
                        if 'patterns' in chunk_insights and isinstance(chunk_insights['patterns'], list):
                            all_patterns.extend(chunk_insights['patterns'])
                        if 'recommendations' in chunk_insights and isinstance(chunk_insights['recommendations'], list):
                            all_recommendations.update([json.dumps(r) for r in chunk_insights['recommendations']])
                        if 'chunk_summary' in chunk_insights and isinstance(chunk_insights['chunk_summary'], str):
                            chunk_summaries.append(chunk_insights['chunk_summary'])
                        
                        logger.info(f"Successfully processed chunk {chunk_index}")
                    
                except json.JSONDecodeError as je:
                    logger.error(f"JSON decode error in chunk {chunk_index}: {str(je)}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_index}: {str(e)}")
                continue

        # Generate final insights
        try:
            final_prompt = f"""Based on the analysis of {analyzed_tickets} tickets out of {total_tickets} total tickets,
            provide a comprehensive analysis. Here are the key points found so far:

            Priority Levels:
            - P0: Critical (Highest priority, severe business impact)
            - P1: Urgent (High priority, significant impact)
            - P2: Normal (Standard priority, moderate impact)
            - P3: Low (Lowest priority, minimal impact)

            Chunk Summaries:
            {json.dumps(chunk_summaries, indent=2)}

            Patterns Found:
            {json.dumps(all_patterns, indent=2)}

            Overall Statistics:
            {json.dumps(summary_stats, indent=2)}

            Provide a final analysis in this JSON format:
            {{
                "key_findings": [
                    {{"title": "Finding Title", "description": "Finding Description"}}
                ],
                "common_patterns": [
                    {{"pattern": "Pattern Description", "frequency": "Frequency Description"}}
                ],
                "recommendations": [
                    {{"title": "Recommendation Title", "description": "Recommendation Details", "priority": "High/Medium/Low"}}
                ],
                "summary": "Overall summary of findings"
            }}
            """

            final_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert support ticket analyst. Your responses must be in valid JSON format."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
                response_format={ "type": "json_object" }
            )
            
            final_insights = json.loads(final_response.choices[0].message.content.strip())
            return final_insights

        except Exception as e:
            logger.error(f"Error generating final insights: {str(e)}")
            # Return a structured error response
            return {
                'key_findings': [{'title': 'Analysis Error', 'description': 'Failed to generate final insights'}],
                'common_patterns': [p for p in all_patterns if isinstance(p, dict)][:5],
                'recommendations': [json.loads(r) for r in list(all_recommendations)[:5]],
                'summary': 'Analysis completed with partial results. Some insights may be missing.'
            }

    except Exception as e:
        logger.error(f"Critical error in generate_ai_insights: {str(e)}")
        return {
            'key_findings': [{'title': 'Error', 'description': 'Failed to generate insights'}],
            'common_patterns': [],
            'recommendations': [],
            'summary': 'Analysis failed due to an error. Please check logs for details.'
        }

def get_highest_priority(case_id, history_df, current_priority):
    """
    Determine the highest priority a ticket reached during its lifecycle.
    Priority order from lowest to highest: P3, P2, P1, P0
    
    Args:
        case_id: The ID of the case
        history_df: DataFrame containing case history records
        current_priority: The current priority of the case
    """
    priority_order = {'P3': 0, 'P2': 1, 'P1': 2, 'P0': 3}
    
    # Start with current priority
    all_priorities = []
    if current_priority in priority_order:
        all_priorities.append(current_priority)
    
    # Get priority changes from history
    if case_id in history_df['CaseId'].values:
        priority_changes = history_df[
            (history_df['CaseId'] == case_id) & 
            (history_df['Field'] == 'Internal_Priority__c')
        ]
        
        if not priority_changes.empty:
            # Add old values
            all_priorities.extend([p for p in priority_changes['OldValue'].dropna() if p in priority_order])
            # Add new values
            all_priorities.extend([p for p in priority_changes['NewValue'].dropna() if p in priority_order])
    
    # If we found any valid priorities, get the highest
    if all_priorities:
        return max(all_priorities, key=lambda x: priority_order.get(x, -1))
    
    return current_priority

def analyze_unspecified_root_causes(df: pd.DataFrame) -> None:
    """Analyze tickets with unspecified root causes and output debug information."""
    try:
        # Filter out open and pending tickets
        closed_statuses = ['Closed', 'Resolved', 'Completed']
        df_closed = df[df['Status'].isin(closed_statuses)]
        
        # Filter tickets with 'Not Specified' root cause
        unspecified_df = df_closed[df_closed['Root Cause'].isin(['Not Specified', 'Not specified', 'not specified', None, np.nan])]
        
        if len(unspecified_df) == 0:
            st.warning("No closed tickets found with unspecified root causes.")
            return
            
        # Calculate percentage
        total_closed_tickets = len(df_closed)
        unspecified_count = len(unspecified_df)
        percentage = (unspecified_count / total_closed_tickets) * 100
        
        st.write(f"### Unspecified Root Cause Analysis (Closed Tickets Only)")
        st.write(f"Found {unspecified_count} closed tickets ({percentage:.1f}%) with unspecified root causes")
        
        # Group by priority
        st.write("\nDistribution by Priority:")
        priority_dist = unspecified_df['Priority'].value_counts()
        st.dataframe(priority_dist)
        
        # Export to CSV
        csv_data = unspecified_df.to_csv(index=False)
        st.download_button(
            label="Download Unspecified Root Cause Tickets (Closed Only)",
            data=csv_data,
            file_name=f"unspecified_root_cause_tickets_closed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Debug output if enabled
        if st.session_state.debug_mode:
            st.write("\n### Debug Information for Unspecified Root Cause Tickets")
            debug_cols = ['Id', 'CaseNumber', 'Subject', 'Status', 'Priority', 
                         'Product Area', 'Product Feature', 'Created Date', 
                         'Closed Date', 'First Response Time', 'CSAT']
            
            # Display available fields
            st.write("\nAvailable Fields in Dataset:")
            st.write(list(unspecified_df.columns))
            
            # Display sample tickets
            st.write("\nSample Tickets (First 10):")
            debug_df = unspecified_df[debug_cols] if all(col in unspecified_df.columns for col in debug_cols) else unspecified_df
            st.dataframe(debug_df.head(10))
            
            # Additional statistics
            st.write("\nAdditional Statistics:")
            st.write(f"- Average Resolution Time: {unspecified_df['Resolution Time (Days)'].mean():.1f} days")
            if 'CSAT' in unspecified_df.columns:
                st.write(f"- Average CSAT: {unspecified_df['CSAT'].mean():.2f}")
            
            # Log to debug
            debug("Unspecified Root Cause Analysis Summary (Closed Tickets):")
            debug(f"Total closed tickets: {total_closed_tickets}")
            debug(f"Unspecified root cause tickets: {unspecified_count}")
            debug(f"Percentage: {percentage:.1f}%")
            debug("Priority distribution:", priority_dist.to_dict())
            
    except Exception as e:
        st.error(f"Error analyzing unspecified root causes: {str(e)}")
        if st.session_state.debug_mode:
            st.exception(e)

def display_visualizations(df, customers):
    """Display visualizations using the dataset."""
    try:
        debug("Starting display_visualizations function")
        
        if df is None or df.empty:
            debug("DataFrame is None or empty")
            st.warning("No data available for visualization.")
            return
            
        # Make a defensive copy
        df = df.copy()
        
        # Add Root Cause Analysis section
        st.markdown("---")
        st.subheader("Root Cause Analysis")
        analyze_unspecified_root_causes(df)
        
        # Convert date columns
        date_cols = ['Created Date', 'Closed Date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Calculate highest priority for each case
        debug("Calculating highest priorities")
        df['Highest_Priority'] = df['Id'].apply(
            lambda case_id: get_highest_priority_from_history(st.session_state.sf_connection, case_id) or df.loc[df['Id'] == case_id, 'Priority'].iloc[0]
        )
        debug("Highest priorities calculated")
        
        # Log priority changes for debugging
        priority_changes = df[df['Highest_Priority'] != df['Priority']]
        if not priority_changes.empty:
            debug(f"Found {len(priority_changes)} cases with priority changes:")
            for _, case in priority_changes.iterrows():
                debug(f"Case {case['Id']}: Initial Priority={case['Priority']}, Highest Priority={case['Highest_Priority']}")
        
        # 1. Ticket Volume by Customer (Bar Chart)
        st.subheader("Ticket Distribution")
        # Create a mapping of truncated names to ticket counts
        ticket_counts = df.groupby('Account_Name', dropna=False).size().reset_index(name='Ticket_Count')
        ticket_counts['Truncated_Name'] = ticket_counts['Account_Name'].apply(lambda x: truncate_string(x, 20))
        
        fig_counts = go.Figure(data=[
            go.Bar(
                x=ticket_counts['Truncated_Name'],
                y=ticket_counts['Ticket_Count'],
                text=ticket_counts['Ticket_Count'],
                textposition='auto',
                marker_color=AQUA_PALETTE[2],
                hovertext=ticket_counts['Account_Name']  # Show full name on hover
            )
        ])
        
        fig_counts.update_layout(
            title='Ticket Count by Customer',
            xaxis_title='Customer',
            yaxis_title='Number of Tickets',
            showlegend=False,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_counts)
        
        # 2. Monthly Ticket Trends (Bar Chart)
        st.subheader("Monthly Ticket Trends")
        
        df['Month'] = df['Created Date'].dt.to_period('M')
        df['Month_Closed'] = df['Closed Date'].dt.to_period('M')
        
        monthly_created = df.groupby('Month').size().reset_index(name='Created')
        monthly_created['Month'] = monthly_created['Month'].astype(str)
        
        monthly_closed = df.groupby('Month_Closed').size().reset_index(name='Closed')
        monthly_closed['Month_Closed'] = monthly_closed['Month_Closed'].astype(str)
        
        monthly_trends = pd.merge(
            monthly_created, 
            monthly_closed.rename(columns={'Month_Closed': 'Month'}), 
            on='Month', 
            how='outer'
        ).fillna(0)
        
        monthly_trends['Month'] = pd.to_datetime(monthly_trends['Month'])
        monthly_trends = monthly_trends.sort_values('Month')
        monthly_trends['Month'] = monthly_trends['Month'].dt.strftime('%Y-%m')
        
        fig_trends = go.Figure()
        
        fig_trends.add_trace(go.Bar(
            name='Created',
            x=monthly_trends['Month'],
            y=monthly_trends['Created'],
            marker_color=VOLUME_PALETTE[0]
        ))
        
        fig_trends.add_trace(go.Bar(
            name='Closed',
            x=monthly_trends['Month'],
            y=monthly_trends['Closed'],
            marker_color=VOLUME_PALETTE[1]
        ))
        
        fig_trends.update_layout(
            title='Monthly Ticket Volume',
            xaxis_title='Month',
            yaxis_title='Number of Tickets',
            barmode='group',
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_trends)
        
        # 3. Resolution Time Analysis
        st.subheader("Resolution Time Analysis")
        
        df['resolution_time_days'] = (df['Closed Date'] - df['Created Date']).dt.total_seconds() / (24 * 3600)
        df['Month'] = df['Created Date'].dt.strftime('%Y-%m')
        
        # Debug logging for priorities
        debug("All priorities in dataset:", df['Highest_Priority'].value_counts().to_dict())
        debug("All Internal Priorities:", df['Priority'].value_counts().to_dict())
        
        # Filter out tickets with unspecified priority and invalid resolution times
        valid_priority_df = df[
            (df['resolution_time_days'].notna()) & 
            (df['resolution_time_days'] > 0) &  # Ensure positive resolution time
            (df['Highest_Priority'].notna()) & 
            (~df['Highest_Priority'].isin(['Unspecified', '', ' ', None]))
        ]
        
        # Debug logging for valid priorities
        debug("Valid priorities after filtering:", valid_priority_df['Highest_Priority'].value_counts().to_dict())
        debug("Sample of valid priority records:", valid_priority_df[['Id', 'Highest_Priority', 'Priority', 'resolution_time_days']].head().to_dict('records'))
        
        if len(valid_priority_df) > 0:
            # Create box plot
            fig_resolution = go.Figure()
            
            # Get all unique priorities and sort them
            all_priorities = sorted(valid_priority_df['Highest_Priority'].unique())
            debug("Unique priorities for plotting:", all_priorities)
            
            for priority in all_priorities:
                priority_data = valid_priority_df[valid_priority_df['Highest_Priority'] == priority]
                debug(f"Data points for priority {priority}:", len(priority_data))
                
                if len(priority_data) > 0:  # Only add trace if we have data
                    fig_resolution.add_trace(go.Box(
                        y=priority_data['resolution_time_days'],
                        name=f'Priority {priority}',
                        marker_color=PRIORITY_COLORS.get(priority, VIRIDIS_PALETTE[0]),
                        boxpoints='outliers'  # Show outliers
                    ))
            
            fig_resolution.update_layout(
                title='Resolution Time Distribution by Highest Priority',
                yaxis_title='Resolution Time (Days)',
                showlegend=True,
                boxmode='group'
            )
            
            st.plotly_chart(fig_resolution)
            
            # Display summary statistics
            st.write("### Resolution Time Summary")
            summary_stats = valid_priority_df.groupby('Highest_Priority').agg({
                'resolution_time_days': ['count', 'mean', 'median']
            }).round(2)
            summary_stats.columns = ['Count', 'Mean Days', 'Median Days']
            st.write(summary_stats)
            
            # Display data quality metrics
            st.write("### Data Quality Metrics")
            total_tickets = len(df)
            missing_resolution = df['resolution_time_days'].isna().sum()
            invalid_resolution = (df['resolution_time_days'] <= 0).sum() if 'resolution_time_days' in df.columns else 0
            missing_priority = df['Highest_Priority'].isna().sum()
            invalid_priority = df['Highest_Priority'].isin(['Unspecified', '', ' ']).sum()
            
            st.write(f"- Total tickets: {total_tickets}")
            st.write(f"- Missing resolution times: {missing_resolution}")
            st.write(f"- Invalid resolution times (<=0): {invalid_resolution}")
            st.write(f"- Missing priorities: {missing_priority}")
            st.write(f"- Invalid priorities: {invalid_priority}")
            st.write(f"- Valid tickets for analysis: {len(valid_priority_df)} ({(len(valid_priority_df)/total_tickets*100):.1f}%)")
        else:
            st.warning("No tickets found with both valid priority and resolution time data.")
        
        # 4. Resolution Time Heatmap
        st.subheader("Resolution Time Distribution")
        
        # Product Area Heatmap
        resolution_by_area = df[df['resolution_time_days'].notna()].pivot_table(
            values='resolution_time_days',
            index='Product Area',
            columns='Highest_Priority',
            aggfunc='mean'
        ).fillna(0)
        
        # Truncate product area names
        truncated_areas = [truncate_string(area, 20) for area in resolution_by_area.index]
        
        fig_heatmap_area = go.Figure(data=go.Heatmap(
            z=resolution_by_area.values,
            x=resolution_by_area.columns,
            y=truncated_areas,
            colorscale=HEATMAP_PALETTE,
            text=np.round(resolution_by_area.values, 1),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertext=[[f"{area}<br>{col}: {val:.1f}" 
                       for col, val in zip(resolution_by_area.columns, row)] 
                      for area, row in zip(resolution_by_area.index, resolution_by_area.values)]
        ))
        
        fig_heatmap_area.update_layout(
            title='Resolution Time by Product Area and Highest Priority (Days)',
            xaxis_title='Highest Priority',
            yaxis_title='Product Area'
        )
        
        st.plotly_chart(fig_heatmap_area)
        
        # 5. Ticket Volume Heatmaps
        st.subheader("Ticket Volume Distribution")
        
        # Product Area vs Feature Heatmap
        volume_heatmap = df.pivot_table(
            values='Id',
            index='Product Area',
            columns='Product Feature',
            aggfunc='count',
            fill_value=0
        )
        
        # Truncate both product area and feature names
        truncated_areas = [truncate_string(area, 20) for area in volume_heatmap.index]
        truncated_features = [truncate_string(feature, 20) for feature in volume_heatmap.columns]
        
        fig_volume_heatmap = go.Figure(data=go.Heatmap(
            z=volume_heatmap.values,
            x=truncated_features,
            y=truncated_areas,
            colorscale=HEATMAP_PALETTE,
            text=volume_heatmap.values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertext=[[f"{area}<br>{feature}: {val}" 
                       for feature, val in zip(volume_heatmap.columns, row)] 
                      for area, row in zip(volume_heatmap.index, volume_heatmap.values)]
        ))
        
        fig_volume_heatmap.update_layout(
            title='Ticket Volume by Product Area and Feature',
            xaxis_title='Product Feature',
            yaxis_title='Product Area'
        )
        
        st.plotly_chart(fig_volume_heatmap)
        
        # 6. Ticket Volume by RCA
        st.subheader("Root Cause Analysis")
        
        rca_counts = df['Root Cause'].value_counts().reset_index()
        rca_counts.columns = ['RCA', 'Count']
        rca_counts['Truncated_RCA'] = rca_counts['RCA'].apply(lambda x: truncate_string(x, 20))
        
        fig_rca = go.Figure(data=[
            go.Bar(
                x=rca_counts['Truncated_RCA'],
                y=rca_counts['Count'],
                marker_color=VIRIDIS_PALETTE[2],
                text=rca_counts['Count'],
                textposition='auto',
                hovertext=rca_counts['RCA']  # Show full RCA name on hover
            )
        ])
        
        fig_rca.update_layout(
            title='Ticket Volume by Root Cause',
            xaxis_title='Root Cause',
            yaxis_title='Number of Tickets',
            xaxis_tickangle=-45,
            height=500  # Make it taller to accommodate RCA labels
        )
        
        st.plotly_chart(fig_rca)
        
        # After the existing Resolution Time Analysis section
        st.write("---")
        st.subheader("Customer Satisfaction Analysis")
        try:
            csat_fig, csat_stats = create_csat_analysis(df)
            st.plotly_chart(csat_fig)
            
            # Display CSAT statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall CSAT", f"{csat_stats['overall_mean']:.2f}")
            with col2:
                st.metric("Response Rate", f"{csat_stats['response_rate']:.1f}%")
            with col3:
                st.metric("Trend", csat_stats['trend'])
        except Exception as e:
            st.error(f"Error generating CSAT analysis: {str(e)}")

        st.write("---")
        st.subheader("Text Analysis")
        try:
            word_clouds = create_word_clouds(df)
            if word_clouds:
                col1, col2 = st.columns(2)
                with col1:
                    if 'Subject' in word_clouds:
                        st.pyplot(word_clouds['Subject'])
                with col2:
                    if 'Description' in word_clouds:
                        st.pyplot(word_clouds['Description'])
            else:
                st.warning("No text data available for word cloud generation")
        except Exception as e:
            st.error(f"Error generating word clouds: {str(e)}")

        st.write("---")
        st.subheader("Root Cause Analysis")
        try:
            rca_figures, rca_stats = create_root_cause_analysis(df)
            
            # Display RCA trend over time
            st.plotly_chart(rca_figures['trends'])
            
            # Display resolution time by root cause
            st.plotly_chart(rca_figures['resolution_time'])
            
            # Display RCA statistics
            st.write("### Root Cause Distribution")
            rca_dist = pd.DataFrame.from_dict(rca_stats['rca_distribution'], 
                                            orient='index', 
                                            columns=['Count']).reset_index()
            rca_dist.columns = ['Root Cause', 'Count']
            st.dataframe(rca_dist)
            
            st.write("### Average Resolution Time by Root Cause")
            avg_res = pd.DataFrame.from_dict(rca_stats['avg_resolution_by_rca'], 
                                           orient='index', 
                                           columns=['Days']).reset_index()
            avg_res.columns = ['Root Cause', 'Average Days']
            avg_res['Average Days'] = avg_res['Average Days'].round(1)
            st.dataframe(avg_res)
        except Exception as e:
            st.error(f"Error generating root cause analysis: {str(e)}")

        # Add First Response Time Analysis
        st.markdown("---")
        st.subheader("First Response Time Analysis")
        try:
            frt_fig, frt_stats = create_first_response_analysis(df)
            
            # Display the box plot
            st.plotly_chart(frt_fig)
            
            # Display response time statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Mean Response Time", f"{frt_stats['overall_mean']:.1f} hours")
            with col2:
                st.metric("Overall Median Response Time", f"{frt_stats['overall_median']:.1f} hours")
            with col3:
                st.metric("Within 24h SLA", f"{frt_stats['sla_compliance']['within_24h']:.1f}%")
            
            # Display detailed statistics by priority
            st.write("### Response Time Statistics by Priority")
            # Convert the multi-level dictionary to a DataFrame
            priority_stats = pd.DataFrame(frt_stats['by_priority'])
            priority_stats = priority_stats['first_response_hours'].reset_index()
            priority_stats.columns = ['Priority', 'Count', 'Mean (hours)', 'Median (hours)', 'Std Dev']
            st.dataframe(priority_stats)
            
        except Exception as e:
            st.error(f"Error generating first response time analysis: {str(e)}")

    except Exception as e:
        error_msg = f"Error in display_visualizations: {str(e)}"
        debug(error_msg)
        debug("Full traceback:", traceback.format_exc())
        st.error(error_msg)

def generate_wordcloud(text_data, title, additional_stopwords=None):
    """Generate a word cloud from the given text data."""
    if not isinstance(text_data, str):
        text_data = ' '.join(str(x) for x in text_data if pd.notna(x))
    
    # Get default STOPWORDS from wordcloud
    stopwords = set(STOPWORDS)
    
    # Add custom stopwords specific to support tickets
    custom_stopwords = {
        # Common words
        'nan', 'none', 'null', 'unspecified', 'undefined', 'unknown',
        # Support-specific terms
        'ticket', 'case', 'issue', 'problem', 'request', 'support',
        'customer', 'user', 'client', 'team', 'please', 'thanks',
        'hello', 'hi', 'hey', 'dear', 'greetings', 'regards',
        'morning', 'afternoon', 'evening',
        # Browser and OS-related terms
        'Mozilla', 'KHTML', 'Gecko', 'NT', 'Windows', 'Apple', 'Safari',
        'AppleWebkit', 'Win32', 'Win64',
        # Common verbs and prepositions
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'should', 'could', 'may', 'might',
        'must', 'can', 'cannot', 'cant', 'wont', 'want',
        'need', 'needed', 'needs', 'required', 'requiring',
        'facing', 'seeing', 'looking', 'trying', 'tried',
        'getting', 'got', 'received', 'receiving',
        # Common articles and conjunctions
        'the', 'a', 'an', 'and', 'or', 'but', 'nor', 'for',
        'yet', 'so', 'although', 'because', 'before', 'after',
        'when', 'while', 'where', 'why', 'how',
        # Numbers and common symbols
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '@', '#', '$', '%', '&', '*', '(', ')', '-', '_',
        '+', '=', '[', ']', '{', '}', '|', '\\', '/', '<', '>',
        # Time-related terms
        'today', 'yesterday', 'tomorrow', 'week', 'month', 'year',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
        'saturday', 'sunday', 'jan', 'feb', 'mar', 'apr', 'may',
        'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    }
    
    # Update stopwords with custom ones
    stopwords.update(custom_stopwords)
    
    if additional_stopwords:
        stopwords.update(additional_stopwords)
    
    try:
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            collocations=False,
            stopwords=stopwords,
            min_font_size=10,
            max_font_size=50
        ).generate(text_data)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title)
        
        return fig
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")
        return None

def generate_powerpoint(filtered_df, active_accounts, escalation_rate, pii_summary: str = None):
    """Generate PowerPoint presentation with charts and statistics.
    
    Args:
        filtered_df: DataFrame containing filtered data
        active_accounts: Number of active accounts
        escalation_rate: Percentage of escalated tickets
        pii_summary: Optional privacy protection summary
    
    Returns:
        PowerPoint presentation as bytes
    """
    try:
        # Create presentation
        prs = Presentation()
        
        # Set slide width and height (16:9 aspect ratio)
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
        
        # Title slide
        title_slide_layout = prs.slide_layouts[0]
        slide = prs.slides.add_slide(title_slide_layout)
        title = slide.shapes.title
        subtitle = slide.placeholders[1]
        title.text = "Customer Support Ticket Analysis"
        subtitle.text = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Privacy Summary slide (if provided)
        if pii_summary:
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            title = slide.shapes.title
            content = slide.placeholders[1]
            title.text = "Privacy Protection Summary"
            content.text = pii_summary
        
        # Overview Statistics slide
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        title = slide.shapes.title
        title.text = "Overview Statistics"
        content = slide.placeholders[1]
        
        # Calculate average CSAT if available
        avg_csat_text = "N/A"
        if 'CSAT__c' in filtered_df.columns:
            try:
                # Convert to numeric and filter for valid range
                csat_numeric = pd.to_numeric(filtered_df['CSAT__c'], errors='coerce')
                # Only include non-null values in the valid range
                valid_csat = csat_numeric[(csat_numeric >= 0) & (csat_numeric <= 5) & csat_numeric.notna()]
                if not valid_csat.empty:
                    avg_csat = valid_csat.mean()
                    avg_csat_text = f"{avg_csat:.2f}"
                    debug(f"PowerPoint CSAT calculation: {len(valid_csat)} valid values, average: {avg_csat:.2f}")
                else:
                    debug("PowerPoint CSAT calculation: No valid CSAT values found")
            except Exception as e:
                debug(f"Error calculating CSAT for PowerPoint: {str(e)}")
        
        # Add statistics as bullet points
        stats_text = (
            f"• Total Cases: {len(filtered_df)}\n"
            f"• Active Accounts: {active_accounts}\n"
            f"• Product Areas: {filtered_df['Product Area'].nunique()}\n"
            f"• Average CSAT: {avg_csat_text}\n"
            f"• Escalation Rate: {escalation_rate:.1f}%"
        )
        content.text = stats_text
        
        # Save presentation
        pptx_output = BytesIO()
        prs.save(pptx_output)
        return pptx_output.getvalue()
    except Exception as e:
        raise Exception(f"Error generating PowerPoint: {str(e)}")

def export_data(df, format, customers):
    """Export data with PII protection."""
    try:
        # Only process PII if enabled
        if st.session_state.enable_pii_processing:
            export_df, pii_stats = process_pii_in_dataframe(df)
            pii_summary = f"""
            Privacy Protection Summary:
            - PII instances detected and removed: {pii_stats['pii_detected']}
            - Columns processed: {', '.join(pii_stats['processed_columns'])}
            - Processing timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
        else:
            export_df = df.copy()
            pii_summary = "PII protection disabled"
        
        if format == "Excel":
            output = BytesIO()
            try:
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Add PII Summary sheet
                    pd.DataFrame({
                        'Privacy Summary': [pii_summary]
                    }).to_excel(writer, sheet_name='Privacy Summary', index=False)
                    
                    # Summary sheet
                    summary_data = {
                        'Customer': [],
                        'Total Tickets': [],
                        'Avg Response Time (hrs)': [],
                        'Avg Resolution Time (days)': [],
                        'Avg CSAT (0-5)': []
                    }
                    
                    for customer in customers:
                        # Use string comparison for customer filtering
                        customer_df = export_df[export_df['Account_Name'].str.lower() == customer.lower()]
                        summary_data['Customer'].append(customer)
                        summary_data['Total Tickets'].append(len(customer_df))
                        
                        # Response Time calculation
                        if 'First_Response_Time__c' in customer_df.columns and 'Created Date' in customer_df.columns:
                            try:
                                response_times = pd.to_datetime(customer_df['First_Response_Time__c']) - pd.to_datetime(customer_df['Created Date'])
                                resp_time = response_times.dt.total_seconds().mean() / 3600
                                summary_data['Avg Response Time (hrs)'].append(round(resp_time, 2) if pd.notna(resp_time) else 'N/A')
                            except Exception as e:
                                debug(f"Error calculating response time: {str(e)}")
                                summary_data['Avg Response Time (hrs)'].append('N/A')
                        else:
                            summary_data['Avg Response Time (hrs)'].append('N/A')
                        
                        # Resolution Time calculation
                        if 'Closed Date' in customer_df.columns and 'Created Date' in customer_df.columns:
                            try:
                                resolution_times = pd.to_datetime(customer_df['Closed Date']) - pd.to_datetime(customer_df['Created Date'])
                                res_time = resolution_times.dt.total_seconds().mean() / (24 * 3600)
                                summary_data['Avg Resolution Time (days)'].append(round(res_time, 2) if pd.notna(res_time) else 'N/A')
                            except Exception as e:
                                debug(f"Error calculating resolution time: {str(e)}")
                                summary_data['Avg Resolution Time (days)'].append('N/A')
                        else:
                            summary_data['Avg Resolution Time (days)'].append('N/A')
                        
                        # CSAT calculation
                        if 'CSAT__c' in customer_df.columns:
                            try:
                                csat_numeric = pd.to_numeric(customer_df['CSAT__c'], errors='coerce')
                                valid_csat = csat_numeric[(csat_numeric >= 0) & (csat_numeric <= 5) & csat_numeric.notna()]
                                if not valid_csat.empty:
                                    avg_csat = valid_csat.mean()
                                    summary_data['Avg CSAT (0-5)'].append(round(avg_csat, 2) if pd.notna(avg_csat) else 'N/A')
                                else:
                                    summary_data['Avg CSAT (0-5)'].append('N/A')
                            except Exception as e:
                                debug(f"Error calculating CSAT: {str(e)}")
                                summary_data['Avg CSAT (0-5)'].append('N/A')
                        else:
                            summary_data['Avg CSAT (0-5)'].append('N/A')
                    
                    # Write summary sheet
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Write detailed data sheets
                    for customer in customers:
                        # Use string comparison for customer filtering
                        customer_df = export_df[export_df['Account_Name'].str.lower() == customer.lower()]
                        sheet_name = truncate_string(customer, 31)  # Excel sheet names limited to 31 chars
                        customer_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Offer download
                st.download_button(
                    label="Download Excel Report",
                    data=output.getvalue(),
                    file_name=f"support_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as excel_error:
                debug(f"Excel export failed: {str(excel_error)}")
                st.error(f"Excel export failed. Error: {str(excel_error)}")
                # Fallback to CSV
                st.info("Attempting CSV export as fallback...")
                export_data(df, "CSV", customers)
                
        elif format == "PowerPoint":
            pptx_data = generate_powerpoint(export_df, len(export_df['Account_Name'].unique()), 
                                          export_df['IsEscalated'].astype(bool).mean() * 100,
                                          pii_summary=pii_summary)
            st.download_button(
                label="Download PowerPoint",
                data=pptx_data,
                file_name=f"support_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
        elif format == "CSV":
            output = BytesIO()
            export_df.to_csv(output, index=False)
            output_str = f"# {pii_summary}\n" + output.getvalue().decode('utf-8')
            output = BytesIO(output_str.encode('utf-8'))
            st.download_button(
                label="Download CSV",
                data=output.getvalue(),
                file_name=f"support_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error exporting data: {str(e)}")
        st.error("Please check the data format and try again.")

def truncate_string(s, max_length=30, add_ellipsis=True):
    """Truncate a string to specified length and optionally add ellipsis if needed.
    
    Args:
        s: String to truncate
        max_length: Maximum length of the string
        add_ellipsis: Whether to add '...' to truncated strings
        
    Returns:
        Truncated string
    """
    if not isinstance(s, str):
        s = str(s)
    if len(s) > max_length:
        return s[:max_length] + ('...' if add_ellipsis else '')
    return s

def display_detailed_analysis(df: pd.DataFrame, enable_ai_analysis: bool = False, enable_pii_processing: bool = False):
    """Display detailed analysis of support tickets."""
    
    # Create a container for analysis
    analysis_container = st.container()
    
    with analysis_container:
        st.header("Detailed Analysis")
        
        with st.spinner("Processing data..."):
            # Monthly ticket trends
            st.subheader("Monthly Ticket Volume")
            monthly_data = df.copy()
            monthly_data['Created Date'] = pd.to_datetime(monthly_data['Created Date'])
            monthly_data['Closed Date'] = pd.to_datetime(monthly_data['Closed Date'])
            
            # Group by month and count tickets
            monthly_created = monthly_data.groupby(monthly_data['Created Date'].dt.to_period('M')).size()
            monthly_closed = monthly_data.groupby(monthly_data['Closed Date'].dt.to_period('M')).size()
            
            # Create figure
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[str(x) for x in monthly_created.index],
                y=monthly_created.values,
                name='Created Tickets',
                marker_color='blue'
            ))
            fig.add_trace(go.Bar(
                x=[str(x) for x in monthly_closed.index],
                y=monthly_closed.values,
                name='Closed Tickets',
                marker_color='green'
            ))
            
            fig.update_layout(
                title='Monthly Ticket Volume',
                xaxis_title='Month',
                yaxis_title='Number of Tickets',
                barmode='group'
            )
            
            st.plotly_chart(fig)
            
            # Resolution Time Analysis
            st.subheader("Resolution Time Analysis")
            
            # Filter for tickets with valid priority and resolution time
            resolution_data = df[df['Priority'].notna() & df['Resolution Time (Days)'].notna()]
            
            if not resolution_data.empty:
                # Create box plot
                fig = go.Figure()
                for priority in sorted(resolution_data['Priority'].unique()):
                    priority_data = resolution_data[resolution_data['Priority'] == priority]
                    fig.add_trace(go.Box(
                        y=priority_data['Resolution Time (Days)'],
                        name=f'Priority {priority}',
                        boxpoints='outliers'
                    ))
                
                fig.update_layout(
                    title='Resolution Time Distribution by Priority',
                    yaxis_title='Resolution Time (Days)',
                    showlegend=True
                )
                
                st.plotly_chart(fig)
                
                # Summary statistics
                st.subheader("Resolution Time Summary")
                summary_stats = resolution_data.groupby('Priority')['Resolution Time (Days)'].agg([
                    'count', 'mean', 'median', 'std', 'min', 'max'
                ]).round(2)
                
                st.dataframe(summary_stats)
            
            # AI Analysis Section
            if enable_ai_analysis:
                st.markdown("---")
                st.markdown("""
                <h2 style='text-align: center;'>🤖 AI Analysis</h2>
                """, unsafe_allow_html=True)
                
                with st.spinner("Generating AI insights..."):
                    try:
                        # Process PII if enabled
                        analysis_df = df.copy()
                        if enable_pii_processing:
                            analysis_df = st.session_state.pii_handler.process_dataframe(analysis_df)
                        
                        # Generate AI insights
                        insights = generate_ai_insights(analysis_df)
                        
                        if insights and isinstance(insights, dict):
                            # Key Findings
                            if 'key_findings' in insights:
                                st.markdown("""
                                <h3 style='color: #1E88E5;'>
                                    <i class='material-icons'>💡 Key Findings</i>
                                </h3>
                                """, unsafe_allow_html=True)
                                
                                for finding in insights['key_findings']:
                                    with st.container():
                                        st.markdown(f"""
                                        <div style='background-color: #E3F2FD; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                                            <h4 style='color: #1565C0; margin: 0;'>{finding['title']}</h4>
                                            <p style='margin: 5px 0 0 0;'>{finding['description']}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            # Common Patterns
                            if 'common_patterns' in insights:
                                st.markdown("""
                                <h3 style='color: #43A047;'>
                                    <i class='material-icons'>📊 Common Patterns</i>
                                </h3>
                                """, unsafe_allow_html=True)
                                
                                for pattern in insights['common_patterns']:
                                    with st.container():
                                        st.markdown(f"""
                                        <div style='background-color: #E8F5E9; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                                            <p style='margin: 0;'><b>Pattern:</b> {pattern['pattern']}</p>
                                            <p style='margin: 5px 0 0 0; color: #2E7D32;'><b>Frequency:</b> {pattern['frequency']}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            # Recommendations
                            if 'recommendations' in insights:
                                st.markdown("""
                                <h3 style='color: #FB8C00;'>
                                    <i class='material-icons'>⚡ Recommendations</i>
                                </h3>
                                """, unsafe_allow_html=True)
                                
                                for rec in insights['recommendations']:
                                    priority_color = {
                                        'High': '#F44336',
                                        'Medium': '#FB8C00',
                                        'Low': '#7CB342'
                                    }.get(rec['priority'], '#757575')
                                    
                                    with st.container():
                                        st.markdown(f"""
                                        <div style='background-color: #FFF3E0; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                                <h4 style='color: #E65100; margin: 0;'>{rec['title']}</h4>
                                                <span style='color: {priority_color}; font-weight: bold;'>Priority: {rec['priority']}</span>
                                            </div>
                                            <p style='margin: 5px 0 0 0;'>{rec['description']}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            # Overall Summary
                            if 'summary' in insights:
                                st.markdown("""
                                <h3 style='color: #5E35B1;'>
                                    <i class='material-icons'>📝 Overall Summary</i>
                                </h3>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                <div style='background-color: #EDE7F6; padding: 15px; border-radius: 5px; margin: 10px 0;'>
                                    <p style='margin: 0; font-size: 1.1em;'>{insights['summary']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.warning("No AI insights were generated. Please try again with a different data selection.")
                    
                    except Exception as e:
                        st.error("Error generating AI insights. Please try again or contact support if the issue persists.")
                        if st.session_state.debug_mode:
                            st.exception(e)
            
            # Add CSAT Analysis
            st.markdown("---")
            st.subheader("Customer Satisfaction Analysis")
            try:
                csat_fig, csat_stats = create_csat_analysis(df)
                st.plotly_chart(csat_fig)
                
                # Display CSAT statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall CSAT", f"{csat_stats['overall_mean']:.2f}")
                with col2:
                    st.metric("Response Rate", f"{csat_stats['response_rate']:.1f}%")
                with col3:
                    st.metric("Trend", csat_stats['trend'])
            except Exception as e:
                st.error(f"Error generating CSAT analysis: {str(e)}")

            # Add Text Analysis
            st.markdown("---")
            st.subheader("Text Analysis")
            try:
                word_clouds = create_word_clouds(df)
                if word_clouds:
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'Subject' in word_clouds:
                            st.pyplot(word_clouds['Subject'])
                    with col2:
                        if 'Description' in word_clouds:
                            st.pyplot(word_clouds['Description'])
                else:
                    st.warning("No text data available for word cloud generation")
            except Exception as e:
                st.error(f"Error generating word clouds: {str(e)}")

            # Add Root Cause Analysis
            st.markdown("---")
            st.subheader("Root Cause Analysis")
            try:
                rca_figures, rca_stats = create_root_cause_analysis(df)
                
                # Display RCA trend over time
                st.plotly_chart(rca_figures['trends'])
                
                # Display resolution time by root cause
                st.plotly_chart(rca_figures['resolution_time'])
                
                # Display RCA statistics
                st.write("### Root Cause Distribution")
                rca_dist = pd.DataFrame.from_dict(rca_stats['rca_distribution'], 
                                                orient='index', 
                                                columns=['Count']).reset_index()
                rca_dist.columns = ['Root Cause', 'Count']
                st.dataframe(rca_dist)
                
                st.write("### Average Resolution Time by Root Cause")
                avg_res = pd.DataFrame.from_dict(rca_stats['avg_resolution_by_rca'], 
                                               orient='index', 
                                               columns=['Days']).reset_index()
                avg_res.columns = ['Root Cause', 'Average Days']
                avg_res['Average Days'] = avg_res['Average Days'].round(1)
                st.dataframe(avg_res)
            except Exception as e:
                st.error(f"Error generating root cause analysis: {str(e)}")

            # Add First Response Time Analysis
            st.markdown("---")
            st.subheader("First Response Time Analysis")
            try:
                frt_fig, frt_stats = create_first_response_analysis(df)
                
                # Display the box plot
                st.plotly_chart(frt_fig)
                
                # Display response time statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Mean Response Time", f"{frt_stats['overall_mean']:.1f} hours")
                with col2:
                    st.metric("Overall Median Response Time", f"{frt_stats['overall_median']:.1f} hours")
                with col3:
                    st.metric("Within 24h SLA", f"{frt_stats['sla_compliance']['within_24h']:.1f}%")
                
                # Display detailed statistics by priority
                st.write("### Response Time Statistics by Priority")
                # Convert the multi-level dictionary to a DataFrame
                priority_stats = pd.DataFrame(frt_stats['by_priority'])
                priority_stats = priority_stats['first_response_hours'].reset_index()
                priority_stats.columns = ['Priority', 'Count', 'Mean (hours)', 'Median (hours)', 'Std Dev']
                st.dataframe(priority_stats)
                
            except Exception as e:
                st.error(f"Error generating first response time analysis: {str(e)}")

def export_analysis():
    """Export analysis data in various formats."""
    try:
        # Get current data
        df = fetch_data()
        if df is None or df.empty:
            st.error("No data available to export")
            return
            
        # Create export format selector
        format_options = ["Excel", "CSV", "PowerPoint"]
        export_format = st.sidebar.selectbox(
            "Select Export Format",
            options=format_options
        )
        
        # Export data in selected format
        export_data(df, export_format, st.session_state.selected_customers)
        
    except Exception as e:
        st.error(f"Error during export: {str(e)}")
        if st.session_state.debug_mode:
            st.exception(e)

if __name__ == "__main__":
    main() 
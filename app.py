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
from visualizers.pattern_evolution import analyze_pattern_evolution
from utils.pii_handler import PIIHandler, get_privacy_status_indicator
from typing import Tuple, Dict, Any, List
from utils.token_manager import TokenManager, TokenInfo, convert_value_for_json
import logging
from utils.ai_analysis import AIAnalyzer
from utils.exporter import BaseExporter
from utils.visualization_helpers import truncate_string
from utils.debug_logger import DebugLogger

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

# Initialize debug logger
if 'debug_logger' not in st.session_state:
    st.session_state.debug_logger = DebugLogger()

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

def debug(message, data=None, category="app"):
    """Enhanced debug function that uses the centralized DebugLogger."""
    if hasattr(st.session_state, 'debug_logger'):
        st.session_state.debug_logger.log(message, data, category)

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
        
        # Display debug UI if debug mode is enabled
        if st.session_state.debug_mode:
            # Log initial debug message only when first enabled
            if not was_debug_enabled:
                debug("Debug mode enabled")
                debug("Application startup", {
                    'environment': os.getenv('ENVIRONMENT', 'development'),
                    'session_state_keys': list(st.session_state.keys())
                })
            
            # Display debug UI
            st.session_state.debug_logger.display_debug_ui()
    
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
            debug("Fetching customer list")
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
                debug(f"Loaded {len(st.session_state.customers)} customers")
            else:
                st.session_state.customers = []
                debug("No customers found", category="error")
        except Exception as e:
            st.error(f"Error fetching customers: {str(e)}")
            debug(f"Error fetching customers: {str(e)}", {'traceback': traceback.format_exc()}, category="error")
            return
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Analysis Options
    st.sidebar.header("Analysis Options")
    st.session_state.enable_detailed_analysis = st.sidebar.checkbox(
        "Enable Detailed Analysis",
        value=True,
        help="Show comprehensive metrics and visualizations"
    )
    st.session_state.enable_ai_analysis = st.sidebar.checkbox(
        "Enable AI Analysis",
        value=False,
        help="Use AI to generate insights and recommendations"
    )
    st.session_state.enable_pii_processing = st.sidebar.checkbox(
        "Enable PII Protection",
        value=False,
        help="Remove sensitive information before analysis"
    )
    
    # Date Range Selection
    st.sidebar.markdown("---")
    st.sidebar.header("Date Range")
    
    # Get current dates from session state and ensure they are datetime objects
    if 'date_range' not in st.session_state:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        st.session_state.date_range = (start_date, end_date)
    
    current_start, current_end = st.session_state.date_range
    if not isinstance(current_start, datetime):
        current_start = pd.to_datetime(current_start)
    if not isinstance(current_end, datetime):
        current_end = pd.to_datetime(current_end)
    
    # Create separate date inputs for start and end dates
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=current_start.date(),
            max_value=datetime.now().date(),
            key='start_date_input'
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=current_end.date(),
            max_value=datetime.now().date(),
            min_value=start_date,  # Ensure end date is not before start date
            key='end_date_input'
        )
    
    # Update session state with new date range
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.max.time())
    st.session_state.date_range = (start_date, end_date)
    
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
                
                if df is not None and not df.empty:
                    # Display basic visualizations first
                    debug("Starting basic visualizations")
                    display_visualizations(df, st.session_state.selected_customers)
                    
                    # Display detailed analysis if enabled
                    if st.session_state.enable_detailed_analysis:
                        debug("Starting detailed analysis")
                        display_detailed_analysis(
                            df, 
                            enable_ai_analysis=st.session_state.enable_ai_analysis,
                            enable_pii_processing=st.session_state.enable_pii_processing
                        )
                else:
                    debug("No data available for analysis", category="error")
                    st.warning("No data available for the selected criteria. Please adjust your filters and try again.")
                    
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                debug(f"Error in main analysis flow: {str(e)}", 
                      {'traceback': traceback.format_exc()}, 
                      category="error")
                if st.session_state.debug_mode:
                    st.exception(e)
    else:
        st.info("Please select at least one customer to begin analysis")

def fetch_data():
    """Fetch data from Salesforce based on session state."""
    try:
        if not st.session_state.selected_customers:
            st.warning("No customers selected")
            debug("No customers selected for data fetch", category="app")
            return None
            
        customer_list = "'" + "','".join(st.session_state.selected_customers) + "'"
        start_date, end_date = st.session_state.date_range
        
        debug("Fetching data with parameters", {
            'customers': st.session_state.selected_customers,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        })
        
        query = f"""
            SELECT 
                Id, CaseNumber, Subject, Description,
                Account.Account_Name__c, CreatedDate, ClosedDate, Status, Internal_Priority__c,
                Product_Area__c, Product_Feature__c, RCA__c,
                First_Response_Time__c, CSAT__c, IsEscalated, Case_Type__c
            FROM Case
            WHERE Account.Account_Name__c IN ({customer_list})
            AND CreatedDate >= {start_date.strftime('%Y-%m-%d')}T00:00:00Z
            AND CreatedDate <= {end_date.strftime('%Y-%m-%d')}T23:59:59Z
            AND Case_Type__c = 'Support Request'
            AND Is_Case_L1_Triaged__c = false
            AND RecordTypeId = '0123m000000U8CCAA0'
        """
        
        debug("Executing SOQL query", {'query': query}, category="api")
        
        records = execute_soql_query(st.session_state.sf_connection, query)
        if not records:
            st.warning("No data found for the selected criteria")
            debug("No records returned from query", category="app")
            return pd.DataFrame()
        
        debug(f"Retrieved {len(records)} records from Salesforce", category="api")
        # Dump raw records to CSV for debugging/backup
        debug("Dumping raw records to CSV", {'record_count': len(records)}, category="data")
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'data/raw_records_{timestamp}.csv'
            os.makedirs('data', exist_ok=True)
            
            # Convert records to DataFrame for easy CSV export
            temp_df = pd.DataFrame(records)
            temp_df.to_csv(filename, index=False)
            debug(f"Successfully saved records to {filename}", category="data")
        except Exception as e:
            debug(f"Failed to save records to CSV: {str(e)}", 
                  {'traceback': traceback.format_exc()}, 
                  category="error")
        # Add data validation check
        st.write(f"Found {len(records)} tickets in total")
        if len(records) > 0:
            st.write("Sample ticket fields:", list(records[0].keys()))
            
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
        
        debug("Data processing completed", {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'date_range': f"{df['Created Date'].min()} to {df['Created Date'].max()}"
        })
        
        if df.empty:
            st.warning("No data available after processing")
            debug("Empty DataFrame after processing", category="error")
            return pd.DataFrame()
            
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        debug(f"Error in fetch_data: {str(e)}", {'traceback': traceback.format_exc()}, category="error")
        if st.session_state.debug_mode:
            st.exception(e)
        return pd.DataFrame()

def generate_ai_insights(cases_df: pd.DataFrame, client: OpenAI) -> Dict[str, Any]:
    """Generate AI insights from case data with enhanced pattern recognition."""
    logger = logging.getLogger(__name__)
    try:
        analyzer = AIAnalyzer(client)
        insights = analyzer.analyze_tickets(cases_df)
        
        if not insights or 'error' in insights:
            logger.error(f"Failed to generate insights: {insights.get('error', 'Unknown error')}")
            return {
                'status': 'error',
                'message': 'Failed to generate insights',
                'error': insights.get('error', 'Unknown error'),
                'timestamp': datetime.now().isoformat()
            }
            
        # Transform insights into the expected format
        formatted_insights = {
            'key_findings': insights['executive_summary']['key_findings'],
            'patterns': {
                'recurring_issues': insights['pattern_insights']['recurring_issues'],
                'high_confidence': insights['pattern_insights']['confidence_levels']['high_confidence'],
                'evolution': insights['trend_analysis']['pattern_evolution']
            },
            'recommendations': insights['recommendations'],
            'summary': {
                'trends': insights['trend_analysis'],
                'customer_impact': insights['customer_impact_analysis'],
                'next_steps': insights['next_steps']
            },
            'metadata': insights['metadata']
        }
        
        return {
            'status': 'success',
            'insights': formatted_insights,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating AI insights: {str(e)}")
        return {
            'status': 'error',
            'message': 'Error generating AI insights',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
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
        
        st.write("### Unspecified Root Cause Analysis (Closed Tickets Only)")
        st.write(f"Found {unspecified_count} closed tickets ({percentage:.1f}%) with unspecified root causes")
        
        # Always show the ticket details table
        st.write("#### Tickets Missing Root Cause")
        ticket_details = unspecified_df[['CaseNumber', 'Subject', 'Status', 'Priority', 'Created Date', 'Closed Date']].copy()
        # Convert timezone-aware dates to naive for display
        for col in ['Created Date', 'Closed Date']:
            if col in ticket_details.columns and ticket_details[col].dt.tz is not None:
                ticket_details[col] = ticket_details[col].dt.tz_localize(None)
        st.dataframe(ticket_details)
        
        # Log to debug
        debug("Tickets with unspecified root causes:", {
            'total_unspecified': unspecified_count,
            'ticket_numbers': unspecified_df['CaseNumber'].tolist(),
            'ticket_details': unspecified_df[['CaseNumber', 'Subject', 'Status', 'Priority', 'Created Date']].to_dict('records')
        })
        
        # Group by priority
        st.write("\nDistribution by Priority:")
        priority_dist = unspecified_df['Priority'].value_counts()
        st.dataframe(priority_dist)
        
        # Create trend analysis visualization
        st.write("\n### Root Cause Trend Analysis")
        
        try:
            # Convert dates to timezone-naive for Excel compatibility and create month periods
            df_closed = df_closed.copy()
            unspecified_df = unspecified_df.copy()
            
            # Ensure Created Date is datetime
            df_closed['Created Date'] = pd.to_datetime(df_closed['Created Date'])
            unspecified_df['Created Date'] = pd.to_datetime(unspecified_df['Created Date'])
            
            # Remove timezone if present
            if df_closed['Created Date'].dt.tz is not None:
                df_closed['Created Date'] = df_closed['Created Date'].dt.tz_localize(None)
            if unspecified_df['Created Date'].dt.tz is not None:
                unspecified_df['Created Date'] = unspecified_df['Created Date'].dt.tz_localize(None)
            
            # Create month periods
            df_closed['Month'] = df_closed['Created Date'].dt.to_period('M')
            unspecified_df['Month'] = unspecified_df['Created Date'].dt.to_period('M')
            
            # Calculate monthly percentages of unspecified root causes
            monthly_stats = pd.DataFrame()
            monthly_stats['Total'] = df_closed.groupby('Month').size()
            monthly_stats['Unspecified'] = unspecified_df.groupby('Month').size()
            monthly_stats['Percentage'] = (monthly_stats['Unspecified'] / monthly_stats['Total'] * 100).round(1)
            monthly_stats = monthly_stats.fillna(0)
            
            # Create trend visualization
            fig_trend = go.Figure()
            
            # Add bar chart for counts
            fig_trend.add_trace(go.Bar(
                name='Unspecified Root Causes',
                x=monthly_stats.index.astype(str),
                y=monthly_stats['Unspecified'],
                marker_color=VIRIDIS_PALETTE[0]
            ))
            
            # Add line for percentage
            fig_trend.add_trace(go.Scatter(
                name='Percentage of Total',
                x=monthly_stats.index.astype(str),
                y=monthly_stats['Percentage'],
                yaxis='y2',
                line=dict(color=VIRIDIS_PALETTE[2], width=2),
                mode='lines+markers'
            ))
            
            fig_trend.update_layout(
                title='Unspecified Root Causes Trend',
                xaxis_title='Month',
                yaxis_title='Number of Tickets',
                yaxis2=dict(
                    title='Percentage of Total Tickets',
                    overlaying='y',
                    side='right',
                    ticksuffix='%'
                ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_trend)
            
            debug("Root cause trend analysis completed", {
                'months_analyzed': len(monthly_stats),
                'avg_percentage': monthly_stats['Percentage'].mean(),
                'trend': 'increasing' if monthly_stats['Percentage'].iloc[-1] > monthly_stats['Percentage'].iloc[0] else 'decreasing'
            })
            
        except Exception as e:
            st.error(f"Error generating root cause trend analysis: {str(e)}")
            debug(f"Error in root cause trend analysis: {str(e)}", 
                  {'traceback': traceback.format_exc()}, 
                  category="error")
            if st.session_state.debug_mode:
                st.exception(e)

        # Export options in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export summary to CSV
            csv_buffer = BytesIO()
            # Convert timezone-aware datetimes to naive for CSV export
            export_df = unspecified_df.copy()
            for col in export_df.select_dtypes(include=['datetime64[ns, UTC]']).columns:
                export_df[col] = export_df[col].dt.tz_localize(None)
            export_df.to_csv(csv_buffer, index=False, encoding='utf-8')
            csv_buffer.seek(0)
            st.download_button(
                label="Download Summary CSV",
                data=csv_buffer,
                file_name=f"unspecified_root_cause_tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export raw data to Excel
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # Write summary sheet
                summary_df = pd.DataFrame({
                    'Metric': ['Total Closed Tickets', 'Unspecified Root Cause Tickets', 'Percentage'],
                    'Value': [total_closed_tickets, unspecified_count, f"{percentage:.1f}%"]
                })
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Write priority distribution
                priority_dist.to_frame('Count').to_excel(writer, sheet_name='Priority Distribution')
                
                # Write trend analysis
                monthly_stats.to_excel(writer, sheet_name='Monthly Trends')
                
                # Write raw data - ensure datetime columns are timezone-naive
                export_df = unspecified_df.copy()
                for col in export_df.select_dtypes(include=['datetime64[ns, UTC]']).columns:
                    export_df[col] = export_df[col].dt.tz_localize(None)
                export_df.to_excel(writer, sheet_name='Raw Data', index=False)
            
            excel_buffer.seek(0)
            st.download_button(
                label="Download Detailed Excel",
                data=excel_buffer.getvalue(),
                file_name=f"unspecified_root_cause_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            # Export all raw data
            all_data_buffer = BytesIO()
            # Convert timezone-aware datetimes to naive for CSV export
            export_df = df.copy()
            for col in export_df.select_dtypes(include=['datetime64[ns, UTC]']).columns:
                export_df[col] = export_df[col].dt.tz_localize(None)
            export_df.to_csv(all_data_buffer, index=False, encoding='utf-8')
            all_data_buffer.seek(0)
            st.download_button(
                label="Download All Raw Data",
                data=all_data_buffer,
                file_name=f"all_tickets_raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
            debug_df = unspecified_df[debug_cols] if all(col in unspecified_df.columns for col in debug_cols) else unspecified_df
            st.dataframe(debug_df.head(10))
            
            # Additional statistics
            st.write("\nAdditional Statistics:")
            st.write(f"- Average Resolution Time: {unspecified_df['Resolution Time (Days)'].mean():.1f} days")
            if 'CSAT' in unspecified_df.columns:
                st.write(f"- Average CSAT: {unspecified_df['CSAT'].mean():.2f}")
            
            # Log to debug
            debug("Unspecified Root Cause Analysis Summary (Closed Tickets):", {
                'total_closed': total_closed_tickets,
                'unspecified_count': unspecified_count,
                'percentage': percentage,
                'priority_distribution': priority_dist.to_dict(),
                'monthly_trend': monthly_stats.to_dict()
            })
            
    except Exception as e:
        st.error(f"Error analyzing unspecified root causes: {str(e)}")
        debug(f"Error in root cause analysis: {str(e)}", {'traceback': traceback.format_exc()}, category="error")
        if st.session_state.debug_mode:
            st.exception(e)

def display_visualizations(df, customers):
    """Display visualizations using the dataset."""
    try:
        debug("Starting display_visualizations function", {
            'data_shape': df.shape if df is not None else None,
            'customer_count': len(customers) if customers else 0
        })
        
        if df is None or df.empty:
            debug("DataFrame is None or empty", category="error")
            st.warning("No data available for visualization.")
            return
            
        # Make a defensive copy
        df = df.copy()
        
        # Add Root Cause Analysis section
        st.markdown("---")
        st.subheader("Root Cause Analysis")
        debug("Starting Root Cause Analysis")
        analyze_unspecified_root_causes(df)
        
        # Convert date columns
        debug("Converting date columns")
        date_cols = ['Created Date', 'Closed Date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Calculate highest priority for each case
        debug("Calculating highest priorities")
        df['Highest_Priority'] = df['Id'].apply(
            lambda case_id: get_highest_priority_from_history(st.session_state.sf_connection, case_id) or df.loc[df['Id'] == case_id, 'Priority'].iloc[0]
        )
        debug("Highest priorities calculated", {
            'priority_distribution': df['Highest_Priority'].value_counts().to_dict()
        })
        
        # Log priority changes for debugging
        priority_changes = df[df['Highest_Priority'] != df['Priority']]
        if not priority_changes.empty:
            debug(f"Found {len(priority_changes)} cases with priority changes:", {
                'changes': priority_changes[['Id', 'Priority', 'Highest_Priority']].to_dict('records')[:5]  # Show first 5 changes
            })
        
        # 1. Ticket Volume by Customer (Bar Chart)
        st.subheader("Ticket Distribution")
        debug("Generating ticket distribution visualization")
        
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
        debug("Ticket distribution visualization completed")
        
        # 2. Monthly Ticket Trends (Bar Chart)
        st.subheader("Monthly Ticket Trends")
        debug("Generating monthly trends visualization")
        
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
        
        debug("Monthly trends data prepared", {
            'months_available': len(monthly_trends),
            'date_range': f"{monthly_trends['Month'].iloc[0]} to {monthly_trends['Month'].iloc[-1]}"
        })
        
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
        debug("Monthly trends visualization completed")
        
        # 3. Resolution Time Analysis
        st.subheader("Resolution Time Analysis")
        debug("Starting resolution time analysis")
        
        df['resolution_time_days'] = (df['Closed Date'] - df['Created Date']).dt.total_seconds() / (24 * 3600)
        df['Month'] = df['Created Date'].dt.strftime('%Y-%m')
        
        # Debug logging for priorities
        debug("Priority data for resolution time analysis", {
            'all_priorities': df['Highest_Priority'].value_counts().to_dict(),
            'internal_priorities': df['Priority'].value_counts().to_dict()
        })
        
        # Filter out tickets with unspecified priority and invalid resolution times
        valid_priority_df = df[
            (df['resolution_time_days'].notna()) & 
            (df['resolution_time_days'] > 0) &  # Ensure positive resolution time
            (df['Highest_Priority'].notna()) & 
            (~df['Highest_Priority'].isin(['Unspecified', '', ' ', None]))
        ]
        
        # Debug logging for valid priorities
        debug("Valid priority data after filtering", {
            'valid_priorities': valid_priority_df['Highest_Priority'].value_counts().to_dict(),
            'valid_records': len(valid_priority_df),
            'total_records': len(df)
        })
        
        if len(valid_priority_df) > 0:
            # Create box plot
            fig_resolution = go.Figure()
            
            # Get all unique priorities and sort them
            all_priorities = sorted(valid_priority_df['Highest_Priority'].unique())
            debug("Unique priorities for plotting", {
                'priorities': all_priorities,
                'priority_counts': {p: len(valid_priority_df[valid_priority_df['Highest_Priority'] == p]) for p in all_priorities}
            })
            
            for priority in all_priorities:
                priority_data = valid_priority_df[valid_priority_df['Highest_Priority'] == priority]
                debug(f"Data points for priority {priority}", {
                    'count': len(priority_data),
                    'mean_resolution_time': priority_data['resolution_time_days'].mean(),
                    'median_resolution_time': priority_data['resolution_time_days'].median()
                })
                
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
            debug("Resolution time visualization completed")
            
            # Display summary statistics
            st.write("### Resolution Time Summary")
            summary_stats = valid_priority_df.groupby('Highest_Priority').agg({
                'resolution_time_days': ['count', 'mean', 'median']
            }).round(2)
            summary_stats.columns = ['Count', 'Mean Days', 'Median Days']
            st.write(summary_stats)
            debug("Resolution time summary statistics displayed")
            
            # Display data quality metrics
            st.write("### Data Quality Metrics")
            total_tickets = len(df)
            missing_resolution = df['resolution_time_days'].isna().sum()
            invalid_resolution = (df['resolution_time_days'] <= 0).sum() if 'resolution_time_days' in df.columns else 0
            missing_priority = df['Highest_Priority'].isna().sum()
            invalid_priority = df['Highest_Priority'].isin(['Unspecified', '', ' ']).sum()
            
            quality_metrics = {
                'total_tickets': total_tickets,
                'missing_resolution': missing_resolution,
                'invalid_resolution': invalid_resolution,
                'missing_priority': missing_priority,
                'invalid_priority': invalid_priority,
                'valid_tickets': len(valid_priority_df),
                'valid_percentage': (len(valid_priority_df)/total_tickets*100)
            }
            
            st.write(f"- Total tickets: {total_tickets}")
            st.write(f"- Missing resolution times: {missing_resolution}")
            st.write(f"- Invalid resolution times (<=0): {invalid_resolution}")
            st.write(f"- Missing priorities: {missing_priority}")
            st.write(f"- Invalid priorities: {invalid_priority}")
            st.write(f"- Valid tickets for analysis: {len(valid_priority_df)} ({(len(valid_priority_df)/total_tickets*100):.1f}%)")
            
            debug("Data quality metrics", quality_metrics)
        else:
            st.warning("No tickets found with both valid priority and resolution time data.")
            debug("No valid tickets for resolution time analysis", category="error")
        
        # Add First Response Time Analysis
        st.write("---")
        st.subheader("First Response Time Analysis")
        debug("Starting first response time analysis")
        
        try:
            # Calculate first response time in hours
            df['First Response Hours'] = (df['First Response Time'] - df['Created Date']).dt.total_seconds() / 3600
            
            # Filter valid response times
            valid_response_df = df[
                (df['First Response Hours'].notna()) & 
                (df['First Response Hours'] > 0) &
                (df['Highest_Priority'].notna()) & 
                (~df['Highest_Priority'].isin(['Unspecified', '', ' ', None]))
            ]
            
            if len(valid_response_df) > 0:
                # Create box plot for response times
                fig_response = go.Figure()
                
                # Get all unique priorities and sort them
                all_priorities = sorted(valid_response_df['Highest_Priority'].unique())
                debug("Unique priorities for response time plotting", {
                    'priorities': all_priorities,
                    'priority_counts': {p: len(valid_response_df[valid_response_df['Highest_Priority'] == p]) for p in all_priorities}
                })
                
                for priority in all_priorities:
                    priority_data = valid_response_df[valid_response_df['Highest_Priority'] == priority]
                    if len(priority_data) > 0:
                        fig_response.add_trace(go.Box(
                            y=priority_data['First Response Hours'],
                            name=f'Priority {priority}',
                            marker_color=PRIORITY_COLORS.get(priority, VIRIDIS_PALETTE[0]),
                            boxpoints='outliers'
                        ))
                
                fig_response.update_layout(
                    title='First Response Time Distribution by Highest Priority',
                    yaxis_title='Response Time (Hours)',
                    showlegend=True,
                    boxmode='group'
                )
                
                st.plotly_chart(fig_response)
                
                # Display summary statistics
                st.write("### First Response Time Summary")
                response_stats = valid_response_df.groupby('Highest_Priority').agg({
                    'First Response Hours': ['count', 'mean', 'median']
                }).round(2)
                response_stats.columns = ['Count', 'Mean Hours', 'Median Hours']
                st.write(response_stats)
                
                # Display data quality metrics
                st.write("### Data Quality Metrics")
                total_tickets = len(df)
                missing_response = df['First Response Hours'].isna().sum()
                invalid_response = (df['First Response Hours'] <= 0).sum() if 'First Response Hours' in df.columns else 0
                
                quality_metrics = {
                    'total_tickets': total_tickets,
                    'missing_response': missing_response,
                    'invalid_response': invalid_response,
                    'valid_tickets': len(valid_response_df),
                    'valid_percentage': (len(valid_response_df)/total_tickets*100)
                }
                
                st.write(f"- Total tickets: {total_tickets}")
                st.write(f"- Missing response times: {missing_response}")
                st.write(f"- Invalid response times (<=0): {invalid_response}")
                st.write(f"- Valid tickets for analysis: {len(valid_response_df)} ({(len(valid_response_df)/total_tickets*100):.1f}%)")
                
                debug("First response time analysis completed", quality_metrics)
            else:
                st.warning("No tickets found with both valid priority and first response time data.")
                debug("No valid tickets for first response time analysis", category="error")
                
        except Exception as e:
            st.error("Error generating first response time analysis")
            debug(f"Error in first response time analysis: {str(e)}", 
                  {'traceback': traceback.format_exc()}, 
                  category="error")
            if st.session_state.debug_mode:
                st.exception(e)
        
        # Add CSAT Analysis
        st.write("---")
        st.subheader("Customer Satisfaction Analysis")
        debug("Starting CSAT analysis")
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
                
            debug("CSAT analysis completed", csat_stats)
        except Exception as e:
            st.error(f"Error generating CSAT analysis: {str(e)}")
            debug(f"Error in CSAT analysis: {str(e)}", {'traceback': traceback.format_exc()}, category="error")
        
        # Add Word Cloud Analysis
        st.write("---")
        st.subheader("Text Analysis")
        debug("Starting text analysis")
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
                debug("Word cloud analysis completed")
            else:
                st.warning("No text data available for word cloud generation")
                debug("No text data for word clouds", category="error")
        except Exception as e:
            st.error(f"Error generating word clouds: {str(e)}")
            debug(f"Error in word cloud generation: {str(e)}", {'traceback': traceback.format_exc()}, category="error")
        
        debug("All visualizations completed successfully")
        
        # Add Product Analysis Heatmaps
        st.write("---")
        st.subheader("Product Analysis Heatmaps")
        debug("Starting product analysis heatmaps")
        
        try:
            # Prepare data for heatmaps
            # Handle missing values in product fields
            df['Product Area'] = df['Product Area'].fillna('Unspecified')
            df['Product Feature'] = df['Product Feature'].fillna('Unspecified')
            
            # 1. Resolution Time Heatmap
            resolution_pivot = df.pivot_table(
                values='Resolution Time (Days)',
                index='Product Area',
                columns='Product Feature',
                aggfunc='mean',
                fill_value=0
            )
            
            fig_resolution_heatmap = go.Figure(data=go.Heatmap(
                z=resolution_pivot.values,
                x=resolution_pivot.columns,
                y=resolution_pivot.index,
                colorscale=HEATMAP_PALETTE,
                text=np.round(resolution_pivot.values, 1),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title='Days')
            ))
            
            fig_resolution_heatmap.update_layout(
                title='Average Resolution Time by Product Area and Feature',
                xaxis_title='Product Feature',
                yaxis_title='Product Area',
                width=800,
                height=600
            )
            
            st.plotly_chart(fig_resolution_heatmap)
            
            # 2. Ticket Volume Heatmap
            volume_pivot = df.pivot_table(
                values='Id',
                index='Product Area',
                columns='Product Feature',
                aggfunc='count',
                fill_value=0
            )
            
            fig_volume_heatmap = go.Figure(data=go.Heatmap(
                z=volume_pivot.values,
                x=volume_pivot.columns,
                y=volume_pivot.index,
                colorscale=HEATMAP_PALETTE,
                text=volume_pivot.values.astype(int),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title='Count')
            ))
            
            fig_volume_heatmap.update_layout(
                title='Ticket Volume by Product Area and Feature',
                xaxis_title='Product Feature',
                yaxis_title='Product Area',
                width=800,
                height=600
            )
            
            st.plotly_chart(fig_volume_heatmap)
            
            # Add summary statistics
            st.write("### Product Analysis Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Top Product Areas by Volume:")
                top_areas = df['Product Area'].value_counts().head(5)
                for area, count in top_areas.items():
                    st.write(f"- {area}: {count} tickets")
            
            with col2:
                st.write("Top Product Features by Volume:")
                top_features = df['Product Feature'].value_counts().head(5)
                for feature, count in top_features.items():
                    st.write(f"- {feature}: {count} tickets")
            
            debug("Product analysis heatmaps completed", {
                'product_areas': len(df['Product Area'].unique()),
                'product_features': len(df['Product Feature'].unique()),
                'total_combinations': len(df.groupby(['Product Area', 'Product Feature']).size())
            })
            
        except Exception as e:
            st.error("Error generating product analysis heatmaps")
            debug(f"Error in product analysis heatmaps: {str(e)}", 
                  {'traceback': traceback.format_exc()}, 
                  category="error")
            if st.session_state.debug_mode:
                st.exception(e)
        
    except Exception as e:
        st.error(f"Error in visualizations: {str(e)}")
        debug(f"Error in display_visualizations: {str(e)}", {'traceback': traceback.format_exc()}, category="error")
        if st.session_state.debug_mode:
            st.exception(e)
            st.write("DataFrame columns:", df.columns.tolist() if df is not None else "None")
            st.write("DataFrame info:", df.info() if df is not None else "None")

def display_detailed_analysis(df: pd.DataFrame, enable_ai_analysis: bool = False, enable_pii_processing: bool = False):
    """Display detailed analysis of support tickets."""
    try:
        st.header("Detailed Analysis")
        
        debug("Starting detailed analysis", {
            'enable_ai_analysis': enable_ai_analysis,
            'enable_pii_processing': enable_pii_processing,
            'data_shape': df.shape
        })
        
        # Pattern Evolution Analysis
        st.markdown("---")
        st.subheader("Pattern Evolution Analysis")
        
        # Ensure we have the required columns
        required_columns = ['Created Date', 'Resolution Time (Days)', 'CSAT', 'Root Cause']
        
        if all(col in df.columns for col in required_columns):
            # Call the dedicated function for pattern evolution analysis
            debug("Calling pattern evolution analysis function")
            display_pattern_evolution(df)
        else:
            missing_cols = [col for col in required_columns if col not in df.columns]
            st.warning(f"Cannot display pattern evolution analysis. Missing columns: {', '.join(missing_cols)}")
            debug("Missing columns for pattern evolution analysis", {
                'missing_columns': missing_cols
            }, category="warning")

        # AI Analysis Section
        if enable_ai_analysis:
            st.markdown("---")
            st.markdown("""
            <h2 style='text-align: center;'>🤖 AI Analysis</h2>
            """, unsafe_allow_html=True)
            
            with st.spinner("Generating AI insights..."):
                try:
                    # Initialize OpenAI client
                    openai_api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get("OPENAI_API_KEY", None)
                    if not openai_api_key:
                        st.warning("OpenAI API key not found. Please add it to your environment variables or Streamlit secrets.")
                        return
                        
                    client = OpenAI(api_key=openai_api_key)
                    
                    # Process PII if enabled
                    analysis_df = df.copy()
                    if enable_pii_processing:
                        analysis_df, _ = st.session_state.pii_handler.process_dataframe(
                            analysis_df,
                            ['Subject', 'Description', 'Comments', 'Account_Name', 'Product_Area', 'Product_Feature']
                        )
                    
                    # Initialize AI Analyzer with proper configuration
                    analyzer = AIAnalyzer(client)
                    
                    # Generate insights with debug logging
                    if st.session_state.debug_mode:
                        st.write("Generating AI insights...")
                        st.write(f"Data shape: {analysis_df.shape}")
                        st.write("Columns:", list(analysis_df.columns))
                    
                    insights = analyzer.analyze_tickets(analysis_df)
                    
                    if insights and 'error' not in insights:
                        # Display Executive Summary
                        st.markdown("""
                        <h3 style='color: #1E88E5;'>
                            <i class='material-icons'>💡 Executive Summary</i>
                        </h3>
                        """, unsafe_allow_html=True)
                        
                        if 'executive_summary' in insights:
                            exec_summary = insights['executive_summary']
                            
                            # Key Findings
                            st.markdown("#### Key Findings")
                            for finding in exec_summary.get('key_findings', []):
                                st.markdown(f"- {finding}")
                            
                            # Critical Patterns
                            if 'critical_patterns' in exec_summary:
                                st.markdown("#### Critical Patterns")
                                for pattern in exec_summary['critical_patterns']:
                                    st.markdown(f"- {pattern}")
                            
                            # Risk Areas
                            if 'risk_areas' in exec_summary:
                                st.markdown("#### Risk Areas")
                                for risk in exec_summary['risk_areas']:
                                    st.markdown(f"- {risk}")
                        
                        # Pattern Insights
                        if 'pattern_insights' in insights:
                            st.markdown("""
                            <h3 style='color: #43A047;'>
                                <i class='material-icons'>📊 Pattern Analysis</i>
                            </h3>
                            """, unsafe_allow_html=True)
                            
                            pattern_insights = insights['pattern_insights']
                            
                            # Recurring Issues
                            if 'recurring_issues' in pattern_insights:
                                st.markdown("#### Recurring Issues")
                                for issue in pattern_insights['recurring_issues']:
                                    st.markdown(f"- {issue}")
                            
                            # Confidence Levels
                            if 'confidence_levels' in pattern_insights:
                                conf_levels = pattern_insights['confidence_levels']
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.markdown("#### High Confidence Patterns")
                                    for pattern in conf_levels.get('high_confidence', []):
                                        st.markdown(f"- {pattern}")
                                
                                with col2:
                                    st.markdown("#### Medium Confidence Patterns")
                                    for pattern in conf_levels.get('medium_confidence', []):
                                        st.markdown(f"- {pattern}")
                                
                                with col3:
                                    st.markdown("#### Low Confidence Patterns")
                                    for pattern in conf_levels.get('low_confidence', []):
                                        st.markdown(f"- {pattern}")
                        
                        # Customer Impact Analysis
                        if 'customer_impact_analysis' in insights:
                            st.markdown("""
                            <h3 style='color: #FB8C00;'>
                                <i class='material-icons'>👥 Customer Impact</i>
                            </h3>
                            """, unsafe_allow_html=True)
                            
                            impact = insights['customer_impact_analysis']
                            
                            # Display satisfaction trends
                            st.markdown("#### Customer Satisfaction Trends")
                            st.markdown(impact.get('satisfaction_trends', 'No satisfaction trend data available'))
                            
                            # Pain Points
                            if 'pain_points' in impact:
                                st.markdown("#### Customer Pain Points")
                                for point in impact['pain_points']:
                                    st.markdown(f"- {point}")
                            
                            # Improvement Opportunities
                            if 'improvement_opportunities' in impact:
                                st.markdown("#### Improvement Opportunities")
                                for opp in impact['improvement_opportunities']:
                                    st.markdown(f"- {opp}")
                        
                        # Recommendations
                        if 'recommendations' in insights:
                            st.markdown("""
                            <h3 style='color: #E91E63;'>
                                <i class='material-icons'>⚡ Recommendations</i>
                            </h3>
                            """, unsafe_allow_html=True)
                            
                            for rec in insights['recommendations']:
                                priority_color = {
                                    'High': '#F44336',
                                    'Medium': '#FB8C00',
                                    'Low': '#7CB342'
                                }.get(rec['priority'], '#757575')
                                
                                st.markdown(f"""
                                <div style='background-color: #FFF3E0; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                                        <h4 style='color: #E65100; margin: 0;'>{rec['title']}</h4>
                                        <span style='color: {priority_color}; font-weight: bold;'>Priority: {rec['priority']}</span>
                                    </div>
                                    <p style='margin: 5px 0;'>{rec['description']}</p>
                                    <p style='margin: 5px 0;'><strong>Impact:</strong> {rec['impact']}</p>
                                    <p style='margin: 5px 0;'><strong>Effort:</strong> {rec['effort']}</p>
                                    <p style='margin: 5px 0;'><strong>Timeline:</strong> {rec.get('timeline', 'Not specified')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Next Steps
                        if 'next_steps' in insights:
                            st.markdown("""
                            <h3 style='color: #5E35B1;'>
                                <i class='material-icons'>➡️ Next Steps</i>
                            </h3>
                            """, unsafe_allow_html=True)
                            
                            for step in insights['next_steps']:
                                st.markdown(f"- {step}")
                        
                        # Analysis Metadata
                        if 'metadata' in insights:
                            with st.expander("Analysis Metadata"):
                                meta = insights['metadata']
                                st.markdown(f"""
                                - Analysis Timestamp: {meta['analysis_timestamp']}
                                - Tickets Analyzed: {meta['tickets_analyzed']} of {meta['total_tickets']}
                                - Chunks Processed: {meta['chunks_processed']}
                                - Patterns Detected: {meta['patterns_detected']}
                                - Pattern Insights Generated: {meta['pattern_insights_generated']}
                                """)
                    else:
                        st.error("Failed to generate AI insights. Please check the logs for more information.")
                        if 'error' in insights:
                            st.error(f"Error: {insights['error']}")
                            if st.session_state.debug_mode:
                                st.write("Debug information:")
                                st.write(insights)
                
                except Exception as e:
                    st.error("Error generating AI insights. Please try again or contact support if the issue persists.")
                    if st.session_state.debug_mode:
                        st.exception(e)
                        st.write("Debug information:")
                        st.write({
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "traceback": traceback.format_exc()
                        })

    except Exception as e:
        st.error(f"Error in detailed analysis: {str(e)}")
        if st.session_state.debug_mode:
            st.exception(e)
            st.write("DataFrame columns:", df.columns.tolist())
            st.write("DataFrame info:", df.info())

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

def export_data(df: pd.DataFrame, format: str, customers: List[str]) -> None:
    """
    Export data to the selected format using BaseExporter.
    
    Args:
        df (pd.DataFrame): DataFrame to export
        format (str): Export format (Excel, CSV, or PowerPoint)
        customers (List[str]): List of selected customers
    """
    try:
        # Initialize exporter
        exporter = BaseExporter(df)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == "Excel":
            # Prepare sheets for Excel export
            sheets = {}
            
            # Summary sheet data
            summary_data = {
                'Customer': [],
                'Total Tickets': [],
                'Avg Response Time (hrs)': [],
                'Avg Resolution Time (days)': [],
                'Avg CSAT': []
            }
            
            for customer in customers:
                customer_df = df[df['Account_Name'] == customer]
                summary_data['Customer'].append(customer)
                summary_data['Total Tickets'].append(len(customer_df))
                
                # Response Time
                if 'First_Response_Time__c' in customer_df.columns:
                    resp_time = (customer_df['First_Response_Time__c'] - customer_df['CreatedDate']).dt.total_seconds().mean() / 3600
                    summary_data['Avg Response Time (hrs)'].append(round(resp_time, 2) if pd.notna(resp_time) else 'N/A')
                else:
                    summary_data['Avg Response Time (hrs)'].append('N/A')
                
                # Resolution Time
                if 'ClosedDate' in customer_df.columns:
                    res_time = (customer_df['ClosedDate'] - customer_df['CreatedDate']).dt.total_seconds().mean() / (24 * 3600)
                    summary_data['Avg Resolution Time (days)'].append(round(res_time, 2) if pd.notna(res_time) else 'N/A')
                else:
                    summary_data['Avg Resolution Time (days)'].append('N/A')
                
                # CSAT
                if 'CSAT__c' in customer_df.columns:
                    avg_csat = customer_df['CSAT__c'].mean()
                    summary_data['Avg CSAT'].append(round(avg_csat, 2) if pd.notna(avg_csat) else 'N/A')
                else:
                    summary_data['Avg CSAT'].append('N/A')
                
                # Add customer-specific sheet
                sheets[customer[:31]] = customer_df  # Excel sheet names limited to 31 chars
            
            # Add summary sheet
            sheets['Summary'] = summary_data
            
            # Export to Excel
            output = exporter.to_excel(
                filename=f"support_analysis_{timestamp}.xlsx",
                sheets=sheets,
                include_summary=True
            )
            
            # Offer download
            st.download_button(
                label="Download Excel Report",
                data=output.getvalue(),
                file_name=f"support_analysis_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        elif format == "CSV":
            # Export to CSV
            output = exporter.to_csv(filename=f"support_analysis_{timestamp}.csv")
            
            # Offer download
            st.download_button(
                label="Download CSV Report",
                data=output.getvalue(),
                file_name=f"support_analysis_{timestamp}.csv",
                mime="text/csv"
            )
            
        elif format == "PowerPoint":
            # Export to PowerPoint with custom title
            output = exporter.to_powerpoint(
                filename=f"support_analysis_{timestamp}.pptx",
                title="Support Ticket Analysis",
                include_charts=True
            )
            
            # Offer download
            st.download_button(
                label="Download PowerPoint Report",
                data=output.getvalue(),
                file_name=f"support_analysis_{timestamp}.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
            
    except Exception as e:
        st.error(f"Error exporting data: {str(e)}")
        if st.session_state.debug_mode:
            st.exception(e)

def display_pattern_evolution(df: pd.DataFrame) -> None:
    """Display pattern evolution analysis including root cause trends."""
    try:
        st.write("## Pattern Evolution Analysis")
        
        # Ensure required columns are present
        required_columns = ['Created Date', 'Resolution Time (Days)', 'CSAT', 'Priority', 'Root Cause', 'Status']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            st.error(f"Missing required columns for pattern analysis: {', '.join(missing_cols)}")
            return
            
        # Filter for closed cases
        closed_statuses = ['Closed', 'Resolved', 'Completed']
        df_closed = df[df['Status'].isin(closed_statuses)].copy()
        
        if len(df_closed) == 0:
            st.warning("No closed cases found for pattern analysis.")
            return
            
        # Create a copy of df_closed for unspecified analysis
        df_with_unspecified = df_closed.copy()
            
        # Further filter out unspecified root causes for main analysis
        df_closed = df_closed[~df_closed['Root Cause'].isin(['Not Specified', 'Not specified', 'not specified', None, np.nan])]
        
        if len(df_closed) == 0:
            st.warning("No closed cases found with specified root causes.")
            return
            
        # Prepare data for pattern analysis
        df_closed['Created Date'] = pd.to_datetime(df_closed['Created Date'])
        df_closed['Month'] = df_closed['Created Date'].dt.to_period('M')
        
        # Calculate monthly metrics
        monthly_metrics = pd.DataFrame()
        monthly_metrics['Ticket Count'] = df_closed.groupby('Month').size()
        monthly_metrics['Avg Resolution Time'] = df_closed.groupby('Month')['Resolution Time (Days)'].mean()
        monthly_metrics['Avg CSAT'] = df_closed.groupby('Month')['CSAT'].mean()
        
        # Handle NaN values
        monthly_metrics = monthly_metrics.fillna(0)
        
        # Root Cause Trend Analysis
        st.write("### Root Cause Distribution Over Time (Closed Cases Only)")
        try:
            # Calculate monthly root cause distribution
            root_cause_monthly = pd.crosstab(
                df_closed['Month'], 
                df_closed['Root Cause'], 
                normalize='index'
            ) * 100
            
            # Create line chart
            fig_root_cause = go.Figure()
            
            # Add a line for each root cause
            for column in root_cause_monthly.columns:
                fig_root_cause.add_trace(go.Scatter(
                    name=column,
                    x=root_cause_monthly.index.astype(str),
                    y=root_cause_monthly[column],
                    mode='lines+markers',  # Add markers for better visibility
                    line=dict(width=2),  # Slightly thicker lines
                    marker=dict(size=6),  # Reasonable marker size
                    hovertemplate="%{y:.1f}%<extra>%{fullData.name}</extra>"
                ))
            
            fig_root_cause.update_layout(
                title='Root Cause Distribution Trend (Closed Cases with Specified Root Causes)',
                xaxis_title='Month',
                yaxis_title='Percentage of Tickets',
                yaxis=dict(
                    ticksuffix='%',
                    range=[0, 100]  # Fix y-axis range from 0 to 100%
                ),
                hovermode='x unified'  # Show all values for a given x-coordinate
            )
            
            st.plotly_chart(fig_root_cause)
            
            # Summary statistics
            st.write("#### Root Cause Summary (Closed Cases with Specified Root Causes)")
            total_tickets = len(df_closed)
            root_cause_summary = df_closed['Root Cause'].value_counts()
            root_cause_percentages = (root_cause_summary / total_tickets * 100).round(1)
            
            summary_df = pd.DataFrame({
                'Count': root_cause_summary,
                'Percentage': root_cause_percentages
            })
            st.dataframe(summary_df)
            
            # Add context about filtered data
            total_closed = len(df[df['Status'].isin(closed_statuses)])
            st.info(f"""
                Data Summary:
                - Total closed cases: {total_closed}
                - Cases with specified root causes: {total_tickets} ({(total_tickets/total_closed*100):.1f}%)
                - Cases excluded (unspecified root causes): {total_closed - total_tickets} ({((total_closed-total_tickets)/total_closed*100):.1f}%)
            """)
            
            debug("Root cause pattern analysis completed", {
                'total_closed_tickets': total_closed,
                'tickets_with_root_causes': total_tickets,
                'excluded_tickets': total_closed - total_tickets,
                'total_root_causes': len(root_cause_summary),
                'top_root_cause': root_cause_summary.index[0],
                'top_root_cause_percentage': root_cause_percentages.iloc[0]
            })
            
            # Add analysis for unspecified root causes by month
            unspecified_count = total_closed - total_tickets
            if unspecified_count > 0:
                st.write("#### Unspecified Root Causes Analysis")
                
                # Prepare data for unspecified analysis
                df_with_unspecified['Created Date'] = pd.to_datetime(df_with_unspecified['Created Date'])
                df_with_unspecified['Month'] = df_with_unspecified['Created Date'].dt.to_period('M')
                
                # Filter for unspecified root causes
                unspecified_df = df_with_unspecified[df_with_unspecified['Root Cause'].isin(['Not Specified', 'Not specified', 'not specified', None, np.nan])]
                
                # Group by month
                monthly_unspecified = unspecified_df.groupby('Month').size()
                monthly_total = df_with_unspecified.groupby('Month').size()
                monthly_pct = (monthly_unspecified / monthly_total * 100).round(1)
                
                # Create dataframe for display
                unspecified_trend = pd.DataFrame({
                    'Total Cases': monthly_total,
                    'Unspecified Root Causes': monthly_unspecified,
                    'Percentage': monthly_pct
                })
                
                # Create chart showing unspecified root causes trend
                fig_unspecified = go.Figure()
                
                fig_unspecified.add_trace(go.Scatter(
                    x=monthly_pct.index.astype(str),
                    y=monthly_pct.values,
                    name='Unspecified Root Causes',
                    mode='lines+markers',
                    line=dict(width=2, color='red'),
                    marker=dict(size=8),
                    hovertemplate="%{y:.1f}%<extra>Unspecified Root Causes</extra>"
                ))
                
                fig_unspecified.update_layout(
                    title='Unspecified Root Causes Trend (Closed Cases)',
                    xaxis_title='Month',
                    yaxis_title='Percentage of Tickets',
                    yaxis=dict(
                        ticksuffix='%',
                        range=[0, 100]
                    ),
                    showlegend=True,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_unspecified)
                st.dataframe(unspecified_trend)
                
                # Add recommendations
                if monthly_pct.iloc[-1] > 20:  # If latest month has > 20% unspecified
                    st.warning("""
                        **High percentage of unspecified root causes detected in recent months.**
                        
                        Recommendations:
                        1. Review ticket closure process to ensure root causes are being properly documented
                        2. Consider implementing a validation rule in Salesforce to require root cause before closure
                        3. Provide additional training to support team on root cause analysis importance
                    """)
            
        except Exception as e:
            st.error(f"Error generating root cause analysis: {str(e)}")
            debug(f"Error in root cause analysis: {str(e)}", 
                {'traceback': traceback.format_exc()}, 
                category="error")
            if st.session_state.debug_mode:
                st.exception(e)
        
    except Exception as e:
        st.error(f"Error in pattern evolution analysis: {str(e)}")
        debug(f"Error in pattern evolution analysis: {str(e)}", 
              {'traceback': traceback.format_exc()}, 
              category="error")
        if st.session_state.debug_mode:
            st.exception(e)

if __name__ == "__main__":
    main() 
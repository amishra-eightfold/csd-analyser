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
    }
    .stPlotlyChart {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function to conditionally display debug information
def debug(message, data=None):
    """Enhanced debug function that logs to console, file and Streamlit."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Format the debug message
    if data is not None:
        log_message = f"[DEBUG] {timestamp} - {message}: {str(data)}"
    else:
        log_message = f"[DEBUG] {timestamp} - {message}"
    
    # Print to console
    print(log_message)
    
    # Write to file
    with open('debug.log', 'a') as f:
        f.write(log_message + '\n')
    
    # Show in Streamlit if debug mode is enabled
    if st.session_state.debug_mode:
        st.write(log_message)

def main():
    st.title("Support Ticket Analytics")
    
    # Initialize Salesforce connection if not already done
    if st.session_state.sf_connection is None:
        with st.spinner("Connecting to Salesforce..."):
            st.session_state.sf_connection = init_salesforce()
            if st.session_state.sf_connection is None:
                st.error("Failed to connect to Salesforce. Please check your credentials.")
                return
    
    # Sidebar for filters and controls
    st.sidebar.title("Controls")
    
    # Debug mode toggle
    st.session_state.debug_mode = st.sidebar.checkbox("Debug Mode", value=st.session_state.debug_mode, 
                                                     help="Show detailed debug information")
    
    # Date range selection
    st.sidebar.header("Date Range")
    
    # Default to last 90 days
    default_end_date = datetime.now()
    default_start_date = default_end_date - timedelta(days=90)
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start_date,
        max_value=default_end_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=default_end_date,
        min_value=start_date,
        max_value=datetime.now()
    )
    
    # Fetch customers if not already loaded
    if not st.session_state.customers:
        with st.spinner("Loading customers..."):
            customer_query = """
                SELECT Id, Account_Name__c 
                FROM Account 
                WHERE Type='Customer' 
                AND Active_Contract__c='Yes'
            """
            try:
                # Debug logging for customer query
                debug("Customer query", customer_query)
                
                records = execute_soql_query(st.session_state.sf_connection, customer_query)
                if records:
                    customers_df = pd.DataFrame(records)
                    debug("Customer data", customers_df.head())
                    st.session_state.customers = customers_df['Account_Name__c'].unique().tolist()
                    debug("Unique customers", len(st.session_state.customers))
                else:
                    st.error("No customers found in Salesforce.")
                    return
            except Exception as e:
                st.error(f"Error fetching customers: {str(e)}")
                return
    
    # Sidebar - Data Selection
    with st.sidebar:
        st.header("Data Selection")
        
        # Customer Selection
        selected_customers = st.multiselect(
            "Select Customers (max 10)",
            options=st.session_state.customers,
            max_selections=10,
            help="Choose up to 10 customers to analyze"
        )
        
        # Generate Analysis Button
        if selected_customers and start_date <= end_date:
            if st.button("Generate Analysis", use_container_width=True):
                with st.spinner("Fetching data..."):
                    df, emails_df, comments_df, history_df, attachments_df = fetch_data(selected_customers, start_date, end_date)
                    if df is not None and not df.empty:
                        st.session_state.data = df
                        st.session_state.data_loaded = True
                        st.rerun()
                    else:
                        st.error("No data found for the selected criteria.")
                        st.session_state.data_loaded = False
        
        # Export Options
        if st.session_state.data_loaded:
            st.header("Export Options")
            export_format = st.selectbox(
                "Select Format",
                ["Excel", "PowerPoint", "CSV"]
            )
            if st.button("Export Data"):
                try:
                    export_data(st.session_state.data, export_format, selected_customers)
                except Exception as e:
                    st.error(f"Error exporting data: {str(e)}")
        
        # Add a section for detailed analysis (only enabled when a single customer is selected)
        if len(selected_customers) == 1:
            st.sidebar.markdown("---")
            st.sidebar.header("Detailed Analysis")
            
            # Store detailed analysis settings in session state
            if 'detailed_analysis_enabled' not in st.session_state:
                st.session_state.detailed_analysis_enabled = False
            if 'ai_analysis_enabled' not in st.session_state:
                st.session_state.ai_analysis_enabled = False
                
            # Update session state based on checkbox values
            st.session_state.detailed_analysis_enabled = st.sidebar.checkbox(
                "Enable Detailed Ticket Analysis", 
                value=st.session_state.detailed_analysis_enabled,
                help="Fetch additional data and perform detailed analysis for the selected customer"
            )
            
            if st.session_state.detailed_analysis_enabled:
                st.session_state.ai_analysis_enabled = st.sidebar.checkbox(
                    "Enable AI-powered Analysis", 
                    value=st.session_state.ai_analysis_enabled,
                    help="Use OpenAI to analyze ticket patterns and provide insights"
                )
                
                if st.session_state.ai_analysis_enabled:
                    st.sidebar.info("AI-powered analysis is enabled. Scroll down to see AI insights after the detailed analysis.")
                    debug("AI analysis is enabled in main function")
    
    # Main Content
    if st.session_state.data_loaded and hasattr(st.session_state, 'data'):
        try:
            # Product Filters
            col1, col2 = st.columns(2)
            with col1:
                if 'Product_Area__c' in st.session_state.data.columns:
                    product_areas = st.multiselect(
                        "Filter by Product Area",
                        options=sorted(st.session_state.data['Product_Area__c'].unique())
                    )
            
            with col2:
                if 'Product_Feature__c' in st.session_state.data.columns:
                    # Create a mapping of truncated names to original names
                    feature_mapping = {truncate_string(feature): feature 
                                    for feature in sorted(st.session_state.data['Product_Feature__c'].unique())}
                    product_features = st.multiselect(
                        "Filter by Product Feature",
                        options=sorted(feature_mapping.keys())
                    )
                    # Convert truncated selections back to original values for filtering
                    selected_features = [feature_mapping[feature] for feature in product_features]
            
            # Apply filters
            df = st.session_state.data.copy()
            if product_areas:
                df = df[df['Product_Area__c'].isin(product_areas)]
            if selected_features:  # Use the mapped back full names
                df = df[df['Product_Feature__c'].isin(selected_features)]
            
            # Display visualizations
            if not df.empty:
                display_visualizations(df, selected_customers)
                
                # Detailed ticket analysis (only for single customer selection)
                if len(selected_customers) == 1 and st.session_state.detailed_analysis_enabled:
                    with st.spinner("Fetching detailed ticket information..."):
                        detailed_data = fetch_detailed_data(selected_customers[0], start_date, end_date)
                        if detailed_data is not None:
                            # Add debug output to help troubleshoot
                            debug("AI analysis enabled", st.session_state.ai_analysis_enabled)
                            debug("Detailed data structure", {k: type(v) for k, v in detailed_data.items()})
                            
                            # Display a clear message about AI status
                            if st.session_state.ai_analysis_enabled:
                                st.info("AI-powered analysis is enabled. Scroll down to see AI insights after the detailed analysis.")
                            
                            # Pass the AI analysis flag to the display function
                            display_detailed_analysis(detailed_data, st.session_state.ai_analysis_enabled, skip_impl_phase_analysis=True)
            else:
                st.warning("No data available after applying filters.")
        except Exception as e:
            st.error(f"Error displaying visualizations: {str(e)}")
            st.error("Please try refreshing the page or selecting different criteria.")

def fetch_data(customers: list, start_date: datetime, end_date: datetime) -> tuple:
    """Fetch case data from Salesforce."""
    try:
        if not st.session_state.sf_connection:
            raise ValueError("Salesforce connection not initialized")
            
        processor = SalesforceDataProcessor(st.session_state.sf_connection)
        cases_df = processor.fetch_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        debug("Initial fetch - shape:", cases_df.shape if cases_df is not None else "None")
        debug("Initial fetch - columns:", cases_df.columns.tolist() if cases_df is not None else "None")
        
        # Dump raw data to CSV for debugging
        if cases_df is not None and not cases_df.empty:
            debug("Dumping raw Salesforce data to CSV")
            raw_df = cases_df.copy()
            try:
                raw_df.to_csv('raw_salesforce_data.csv', index=False)
                debug("Raw data dumped to raw_salesforce_data.csv")
                # Log sample of First_Response_Time__c values
                if 'First_Response_Time__c' in raw_df.columns:
                    debug("First_Response_Time__c sample values:", raw_df['First_Response_Time__c'].head().tolist())
                    debug("First_Response_Time__c data type:", raw_df['First_Response_Time__c'].dtype)
            except Exception as e:
                debug(f"Error dumping raw data: {str(e)}")
        
        if cases_df is not None and not cases_df.empty:
            # Create a clean copy of the dataframe
            cases_df = cases_df.copy()
            
            # Handle nested Account structure
            debug("Processing Account structure")
            if 'Account' in cases_df.columns:
                try:
                    # Extract Account Name from nested dictionary
                    cases_df['Account_Name'] = cases_df['Account'].apply(
                        lambda x: x.get('Name', 'Unspecified') if isinstance(x, dict) else 'Unspecified'
                    )
                    # Drop the original nested columns
                    cases_df = cases_df.drop(['Account', 'Account.attributes'], axis=1, errors='ignore')
                except Exception as e:
                    debug(f"Error processing Account column: {str(e)}")
                    cases_df['Account_Name'] = 'Unspecified'
            
            # Handle Account.Name if present
            if 'Account.Name' in cases_df.columns:
                cases_df['Account_Name'] = cases_df['Account.Name']
                cases_df = cases_df.drop('Account.Name', axis=1, errors='ignore')
            
            # Ensure Account_Name exists and is properly formatted
            if 'Account_Name' not in cases_df.columns:
                cases_df['Account_Name'] = 'Unspecified'
            
            # Clean up attribute columns
            attribute_cols = [col for col in cases_df.columns if col.startswith('attributes.')]
            cases_df = cases_df.drop(attribute_cols, axis=1, errors='ignore')
            
            # Convert date columns - now including First_Response_Time__c
            date_cols = ['CreatedDate', 'ClosedDate', 'First_Response_Time__c']
            for col in date_cols:
                if col in cases_df.columns:
                    cases_df[col] = pd.to_datetime(cases_df[col], errors='coerce')
                    debug(f"Converted {col} to datetime")
            
            # Convert numeric columns - remove First_Response_Time__c from here
            numeric_cols = ['CSAT__c']
            for col in numeric_cols:
                if col in cases_df.columns:
                    cases_df[col] = pd.to_numeric(cases_df[col], errors='coerce')
            
            # Convert boolean columns
            bool_cols = ['IsEscalated']
            for col in bool_cols:
                if col in cases_df.columns:
                    cases_df[col] = cases_df[col].fillna(False).astype(bool)
            
            # Convert remaining string columns and handle NaN values
            string_cols = cases_df.select_dtypes(include=['object']).columns
            for col in string_cols:
                cases_df[col] = cases_df[col].fillna('Unspecified').astype(str)
            
            debug("Processed columns:", cases_df.columns.tolist())
            debug("Column types:", cases_df.dtypes.to_dict())
            
            # Filter by customers if provided
            if customers:
                # Ensure customer names are strings
                customers = [str(c) for c in customers]
                cases_df = cases_df[cases_df['Account_Name'].isin(customers)]
                debug("After customer filter - shape:", cases_df.shape)
        
        if cases_df is None or cases_df.empty:
            debug("No data found after filtering")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        # Fetch related data
        case_ids = cases_df['Id'].tolist()
        emails_df, comments_df, history_df, attachments_df = fetch_related_data(case_ids)
        
        return cases_df, emails_df, comments_df, history_df, attachments_df
        
    except Exception as e:
        debug(f"Error in fetch_data: {str(e)}")
        debug("Traceback:", traceback.format_exc())
        raise

def fetch_related_data(case_ids: list) -> tuple:
    """Fetch related data for a list of case IDs.
    
    Args:
        case_ids: List of case IDs to fetch related data for
        
    Returns:
        Tuple of (emails_df, comments_df, history_df, attachments_df)
    """
    if not case_ids:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    try:
        # Format case IDs for query
        case_ids_str = "'" + "','".join(case_ids) + "'"
        
        # Fetch email messages
        email_query = f"""
            SELECT 
                Id, ParentId, Subject, TextBody, HtmlBody, FromAddress, ToAddress,
                CcAddress, BccAddress, MessageDate, Status, HasAttachment
            FROM EmailMessage
            WHERE ParentId IN ({case_ids_str})
            ORDER BY MessageDate ASC
        """
        emails = execute_soql_query(st.session_state.sf_connection, email_query)
        emails_df = pd.DataFrame(emails) if emails else pd.DataFrame()
        
        # Convert date fields in emails
        if not emails_df.empty and 'MessageDate' in emails_df.columns:
            emails_df['MessageDate'] = pd.to_datetime(emails_df['MessageDate'], errors='coerce')
        
        # Fetch case comments
        comments_query = f"""
            SELECT 
                Id, ParentId, CommentBody, CreatedDate, CreatedById
            FROM CaseComment
            WHERE ParentId IN ({case_ids_str})
            ORDER BY CreatedDate ASC
        """
        comments = execute_soql_query(st.session_state.sf_connection, comments_query)
        comments_df = pd.DataFrame(comments) if comments else pd.DataFrame()
        
        # Convert date fields in comments
        if not comments_df.empty and 'CreatedDate' in comments_df.columns:
            comments_df['CreatedDate'] = pd.to_datetime(comments_df['CreatedDate'], errors='coerce')
        
        # Fetch case history
        history_query = f"""
            SELECT 
                Id, CaseId, Field, OldValue, NewValue, CreatedDate, CreatedById
            FROM CaseHistory
            WHERE CaseId IN ({case_ids_str})
            ORDER BY CreatedDate ASC
        """
        history = execute_soql_query(st.session_state.sf_connection, history_query)
        history_df = pd.DataFrame(history) if history else pd.DataFrame()
        
        # Convert date fields in history
        if not history_df.empty and 'CreatedDate' in history_df.columns:
            history_df['CreatedDate'] = pd.to_datetime(history_df['CreatedDate'], errors='coerce')
        
        # Fetch attachments
        attachments_query = f"""
            SELECT 
                Id, ParentId, Name, Description, ContentType, BodyLength, CreatedDate
            FROM Attachment
            WHERE ParentId IN ({case_ids_str})
            ORDER BY CreatedDate ASC
        """
        attachments = execute_soql_query(st.session_state.sf_connection, attachments_query)
        attachments_df = pd.DataFrame(attachments) if attachments else pd.DataFrame()
        
        # Convert date fields in attachments
        if not attachments_df.empty and 'CreatedDate' in attachments_df.columns:
            attachments_df['CreatedDate'] = pd.to_datetime(attachments_df['CreatedDate'], errors='coerce')
        
        return emails_df, comments_df, history_df, attachments_df
        
    except Exception as e:
        st.error(f"Error fetching related data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def display_basic_analysis(cases_df: pd.DataFrame) -> dict:
    """Display basic analysis of case data.
    
    Args:
        cases_df: DataFrame containing case data
        
    Returns:
        Dictionary containing basic metrics
    """
    processor = SalesforceDataProcessor(st.session_state.sf_connection)
    metrics = processor.calculate_case_metrics(cases_df)
    
    st.subheader("Basic Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cases", metrics['total_cases'])
    with col2:
        st.metric("Open Cases", metrics['open_cases'])
    with col3:
        st.metric("Closed Cases", metrics['closed_cases'])
    with col4:
        st.metric("Escalated Cases", metrics['escalated_cases'])
        
    if 'avg_csat' in metrics:
        st.metric("Average CSAT", f"{metrics['avg_csat']:.2f}")
        
    return metrics

def display_detailed_analysis(data, enable_ai_analysis=False, skip_impl_phase_analysis=True):
    """Display detailed analysis of the selected customer's tickets."""
    try:
        debug("Starting detailed analysis")
        insights = None  # Initialize insights variable
        cases_df = data['cases'] if isinstance(data, dict) else data
        
        # Process DataFrame
        debug("Processing cases DataFrame")
        cases_df['CreatedDate'] = pd.to_datetime(cases_df['CreatedDate'])
        cases_df['ClosedDate'] = pd.to_datetime(cases_df['ClosedDate'])
        cases_df['First_Response_Time__c'] = pd.to_datetime(cases_df['First_Response_Time__c'])
        cases_df['CSAT__c'] = pd.to_numeric(cases_df['CSAT__c'], errors='coerce')
        
        # Calculate highest priority for each case
        debug("Calculating highest priorities")
        cases_df['Highest_Priority'] = cases_df['Id'].apply(
            lambda case_id: get_highest_priority_from_history(st.session_state.sf_connection, case_id) or cases_df.loc[cases_df['Id'] == case_id, 'Internal_Priority__c'].iloc[0]
        )
        debug("Highest priorities calculated")

        # Response Time Analysis
        debug("Starting Response Time Analysis")
        response_time_df = cases_df.copy()
        
        # Log data state
        debug(f"Response Time Analysis - Initial data shape: {response_time_df.shape}")
        debug(f"First_Response_Time__c data type: {response_time_df['First_Response_Time__c'].dtype}")
        debug(f"First_Response_Time__c sample values:\n{response_time_df['First_Response_Time__c'].head()}")
        debug(f"Highest_Priority value counts:\n{response_time_df['Highest_Priority'].value_counts(dropna=False)}")
        
        # Calculate response time in hours
        response_time_df['response_time_hours'] = (response_time_df['First_Response_Time__c'] - response_time_df['CreatedDate']).dt.total_seconds() / 3600
        
        # Check for missing or invalid response times
        missing_response_times = response_time_df['First_Response_Time__c'].isna().sum()
        invalid_response_times = (response_time_df['response_time_hours'] <= 0).sum() if 'response_time_hours' in response_time_df.columns else 0
        debug(f"Missing response times: {missing_response_times}")
        debug(f"Invalid response times (<=0): {invalid_response_times}")
        
        # Filter valid data
        valid_response_time_df = response_time_df[
            (response_time_df['First_Response_Time__c'].notna()) & 
            (response_time_df['CreatedDate'].notna()) &
            (response_time_df['response_time_hours'] > 0) &
            (response_time_df['Highest_Priority'].notna())
        ]
        
        debug(f"Valid response time records: {len(valid_response_time_df)}")
        
        if len(valid_response_time_df) > 0:
            # Calculate statistics
            response_stats = valid_response_time_df.groupby('Highest_Priority').agg({
                'response_time_hours': ['count', 'mean', 'median'],
                'Id': 'count'
            }).round(2)
            
            debug(f"Response time statistics:\n{response_stats}")
            
            # Create visualization
            fig = go.Figure()
            
            priorities = valid_response_time_df['Highest_Priority'].unique()
            for priority in priorities:
                priority_data = valid_response_time_df[valid_response_time_df['Highest_Priority'] == priority]
                fig.add_trace(go.Box(
                    y=priority_data['response_time_hours'],
                    name=f"Priority {priority}",
                    marker_color=PRIORITY_COLORS.get(priority, VIRIDIS_PALETTE[0]),
                    boxpoints=False  # Remove scatter points
                ))
            
            fig.update_layout(
                title="Response Time Distribution by Highest Priority",
                yaxis_title="Response Time (hours)",
                showlegend=True
            )
            
            st.plotly_chart(fig)
            
            # Display summary statistics
            st.write("### Response Time Summary")
            st.write(f"- Total tickets with valid response time: {len(valid_response_time_df)}")
            st.write(f"- Overall mean response time: {valid_response_time_df['response_time_hours'].mean():.2f} hours")
            st.write(f"- Overall median response time: {valid_response_time_df['response_time_hours'].median():.2f} hours")
            
            # Display data quality metrics
            st.write("### Data Quality Metrics")
            total_tickets = len(response_time_df)
            st.write(f"- Total tickets: {total_tickets}")
            st.write(f"- Missing response times: {missing_response_times} ({(missing_response_times/total_tickets*100):.1f}%)")
            st.write(f"- Invalid response times (<=0): {invalid_response_times} ({(invalid_response_times/total_tickets*100):.1f}%)")
            st.write(f"- Valid response times: {len(valid_response_time_df)} ({(len(valid_response_time_df)/total_tickets*100):.1f}%)")
            
        else:
            st.warning("""
            No valid response time data available for analysis. This could be due to:
            - Missing First Response Time values ({} records)
            - Missing Created Date values
            - Invalid response times (negative or zero)
            - Missing priority information
            
            Please check the Salesforce configuration to ensure both First_Response_Time__c 
            and CreatedDate fields are being populated correctly.
            """.format(missing_response_times))

        # CSAT Analysis Section
        st.write("### CSAT Analysis")
        if 'CSAT__c' in cases_df.columns and 'CreatedDate' in cases_df.columns:
            # Prepare monthly CSAT data
            cases_df['Month'] = cases_df['CreatedDate'].dt.to_period('M')
            monthly_csat = cases_df.groupby('Month').agg({
                'CSAT__c': ['count', 'mean']
            }).reset_index()
            monthly_csat.columns = ['Month', 'CSAT_Returns', 'Average_CSAT']
            
            # Create two-line plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_csat['Month'].astype(str),
                y=monthly_csat['CSAT_Returns'],
                name='Number of Returns',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=monthly_csat['Month'].astype(str),
                y=monthly_csat['Average_CSAT'],
                name='Average Score',
                line=dict(color='green'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Monthly CSAT Returns and Average Scores',
                xaxis_title='Month',
                yaxis_title='Number of Returns',
                yaxis2=dict(
                    title='Average CSAT Score',
                    overlaying='y',
                    side='right'
                ),
                showlegend=True
            )
            st.plotly_chart(fig)
        else:
            st.info("No CSAT data available for analysis")

        # Add a clear separator before AI analysis
        if enable_ai_analysis:
            st.markdown("---")
            st.write("## 🤖 AI-Powered Analysis")
            
            with st.spinner("Generating AI insights..."):
                try:
                    debug("Starting AI analysis")
                    insights = generate_ai_insights(cases_df, data.get('comments', pd.DataFrame()), data.get('emails', pd.DataFrame()))
                    
                    if insights and isinstance(insights, dict):
                        # Display key insights
                        st.subheader("🔍 Key Insights")
                        st.write(insights.get('summary', 'No summary available'))
                        
                        # Display patterns
                        if 'patterns' in insights and insights['patterns']:
                            st.subheader("📊 Identified Patterns")
                            for pattern in insights['patterns']:
                                if isinstance(pattern, dict):
                                    st.markdown(f"**{pattern.get('title', '')}**")
                                    st.markdown(f"{pattern.get('description', '')}")
                                else:
                                    st.markdown(f"• {pattern}")
                        
                        # Display recommendations
                        if 'recommendations' in insights and insights['recommendations']:
                            st.subheader("💡 Recommendations")
                            for rec in insights['recommendations']:
                                st.markdown(f"• {rec}")
                    else:
                        st.warning("Unable to generate AI insights. Please check the debug logs for more information.")
                        debug("AI insights generation failed or returned invalid format")
                except Exception as e:
                    st.error(f"Error generating AI insights: {str(e)}")
                    debug(f"AI analysis error: {str(e)}")
                    debug(f"Traceback: {traceback.format_exc()}")
        
        return insights
            
    except Exception as e:
        debug(f"Error in display_detailed_analysis: {str(e)}")
        debug(f"Traceback: {traceback.format_exc()}")
        st.error(f"Error displaying detailed analysis: {str(e)}")
        return None

def generate_ai_insights(cases_df: pd.DataFrame,
                      comments_df: pd.DataFrame = None,
                      emails_df: pd.DataFrame = None) -> dict:
    """Generate AI insights from ticket data using OpenAI."""
    try:
        # Check if OpenAI package is installed
        try:
            from openai import OpenAI
            debug("OpenAI package imported successfully")
        except ImportError:
            debug("OpenAI import error - package not installed")
            return {
                'summary': "OpenAI package is not installed. Please install it to enable AI analysis.",
                'patterns': [],
                'recommendations': ["Install the OpenAI package using 'pip install openai'"]
            }
            
        # Check if OpenAI API key is available
        openai_api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get("OPENAI_API_KEY", None)
        if not openai_api_key:
            debug("OpenAI API key not found")
            return {
                'summary': "OpenAI API key not found. Please configure it to enable AI analysis.",
                'patterns': [],
                'recommendations': [
                    "Add OPENAI_API_KEY to your environment variables",
                    "Or add it to your Streamlit secrets.toml file"
                ]
            }

        debug("OpenAI setup validated successfully")

        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)
        
        # Ensure data is properly structured
        if cases_df is None or cases_df.empty:
            debug("No cases data found in input")
            return {
                'summary': "No ticket data available for analysis.",
                'patterns': [],
                'recommendations': ["Please ensure there is ticket data to analyze"]
            }

        # Create summary statistics with datetime handling
        def convert_value_for_json(val):
            if pd.isna(val):
                return None
            if isinstance(val, pd.Timestamp):
                return val.strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(val, (np.int64, np.float64)):
                return int(val) if val.is_integer() else float(val)
            return str(val)  # Convert all other values to strings for safe JSON serialization

        summary_stats = {
            'total_tickets': int(len(cases_df)),
            'status_distribution': {k: int(v) for k, v in cases_df['Status'].value_counts().to_dict().items()},
            'priority_distribution': {k: int(v) for k, v in cases_df['Internal_Priority__c'].value_counts().to_dict().items()},
            'product_areas': {k: int(v) for k, v in cases_df['Product_Area__c'].value_counts().to_dict().items()},
            'features': {k: int(v) for k, v in cases_df['Product_Feature__c'].value_counts().to_dict().items()},
            'root_causes': {k: int(v) for k, v in cases_df['RCA__c'].value_counts().to_dict().items()}
        }

        debug("Created summary statistics")

        # Prepare cases for analysis
        total_tickets = len(cases_df)
        max_tickets = 100
        analyzed_tickets = min(total_tickets, max_tickets)
        
        # Sort by created date to get a representative sample across time
        cases_df = cases_df.sort_values('CreatedDate')
        
        # If we have more than max_tickets, take evenly spaced samples
        if total_tickets > max_tickets:
            step = total_tickets // max_tickets
            selected_indices = list(range(0, total_tickets, step))[:max_tickets]
            analysis_df = cases_df.iloc[selected_indices]
        else:
            analysis_df = cases_df

        # Prepare cases for chunked analysis with datetime handling
        cases_for_analysis = []
        for _, case in analysis_df[['Subject', 'Description', 'Status', 'Internal_Priority__c', 
                                  'Product_Area__c', 'RCA__c', 'CreatedDate']].iterrows():
            case_dict = {}
            for col, val in case.items():
                case_dict[col] = convert_value_for_json(val)
            cases_for_analysis.append(case_dict)
        
        # Initialize combined insights
        all_patterns = []
        all_recommendations = set()  # Use set to avoid duplicates
        chunk_summaries = []
        
        # Process in chunks of 20 cases
        chunk_size = 20
        for i in range(0, len(cases_for_analysis), chunk_size):
            chunk = cases_for_analysis[i:i + chunk_size]
            
            # Prepare the prompt for this chunk
            prompt = f"""You are an expert support ticket analyst. Analyze the following support ticket data and provide insights.
            
            Overall Statistics:
            {json.dumps(summary_stats, indent=2)}

            Detailed Analysis of Ticket Chunk {i//chunk_size + 1} of {(len(cases_for_analysis) + chunk_size - 1)//chunk_size}:
            {json.dumps(chunk, indent=2)}

            Provide your analysis in the following JSON format EXACTLY (no other text before or after):
            {{
                "chunk_summary": "A summary of key insights from this chunk of tickets",
                "patterns": [
                    {{"title": "Pattern Title", "description": "Pattern Description"}}
                ],
                "recommendations": [
                    "Specific recommendation"
                ]
            }}

            Focus on practical insights that can help improve customer support operations.
            Your response must be valid JSON and follow the exact format above.
            """

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert support ticket analyst. Your responses must be in valid JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                
                chunk_response = response.choices[0].message.content.strip()
                debug(f"Raw chunk response: {chunk_response[:100]}...")  # Log first 100 chars of response
                
                # Extract JSON from response (handle potential extra text)
                try:
                    json_start = chunk_response.find('{')
                    json_end = chunk_response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        chunk_response = chunk_response[json_start:json_end]
                        chunk_insights = json.loads(chunk_response)
                    else:
                        raise ValueError("No JSON object found in response")
                    
                    # Validate chunk insights structure
                    if not isinstance(chunk_insights, dict):
                        raise ValueError("Response is not a dictionary")
                    
                    # Collect insights from this chunk
                    if 'patterns' in chunk_insights and isinstance(chunk_insights['patterns'], list):
                        all_patterns.extend(chunk_insights['patterns'])
                    if 'recommendations' in chunk_insights and isinstance(chunk_insights['recommendations'], list):
                        all_recommendations.update(chunk_insights['recommendations'])
                    if 'chunk_summary' in chunk_insights and isinstance(chunk_insights['chunk_summary'], str):
                        chunk_summaries.append(chunk_insights['chunk_summary'])
                    
                except json.JSONDecodeError as je:
                    debug(f"JSON decode error in chunk {i//chunk_size + 1}: {str(je)}")
                    debug(f"Problematic response: {chunk_response}")
                    continue
                except Exception as e:
                    debug(f"Error processing chunk {i//chunk_size + 1}: {str(e)}")
                    continue

            except Exception as e:
                debug(f"Error processing chunk {i//chunk_size + 1}: {str(e)}")
                continue

        # Generate final comprehensive analysis
        final_prompt = f"""Based on the analysis of {analyzed_tickets} tickets out of {total_tickets} total tickets,
        and the following chunk summaries, provide a comprehensive analysis:

        Chunk Summaries:
        {json.dumps(chunk_summaries, indent=2)}

        Overall Statistics:
        {json.dumps(summary_stats, indent=2)}

        Please provide your analysis in JSON format if possible, following this structure:
        {{
            "summary": "A comprehensive summary of all insights",
            "patterns": [
                {{"title": "Pattern Title", "description": "Pattern Description"}}
            ],
            "recommendations": [
                "Final recommendation"
            ]
        }}
        """

        try:
            # First attempt with JSON response format
            try:
                final_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert support ticket analyst providing a final comprehensive analysis. Try to provide your response in JSON format."},
                        {"role": "user", "content": final_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000,
                    response_format={ "type": "json_object" }  # Try JSON format first
                )
                
                final_content = final_response.choices[0].message.content.strip()
                debug(f"Raw final response (JSON attempt): {final_content[:100]}...")
                
                try:
                    final_analysis = json.loads(final_content)
                    if isinstance(final_analysis, dict):
                        # JSON parsing succeeded, add coverage info and return
                        coverage_info = f"\n\nAnalysis Coverage: Analyzed {analyzed_tickets} out of {total_tickets} total tickets "
                        coverage_info += f"({(analyzed_tickets/total_tickets*100):.1f}% coverage)"
                        if total_tickets > max_tickets:
                            coverage_info += f". Analysis was limited to {max_tickets} tickets for processing efficiency."
                        
                        final_analysis['summary'] = final_analysis.get('summary', '') + coverage_info
                        final_analysis['patterns'] = final_analysis.get('patterns', all_patterns)
                        final_analysis['recommendations'] = final_analysis.get('recommendations', list(all_recommendations))
                        
                        return final_analysis
                except json.JSONDecodeError:
                    debug("JSON response format failed, falling back to regular response")
                    raise  # This will trigger the fallback approach
                    
            except Exception as json_error:
                debug(f"JSON format attempt failed: {str(json_error)}")
                # Fallback to regular response without JSON format requirement
                final_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": """You are an expert support ticket analyst. 
                        Structure your response in three sections:
                        1. SUMMARY: A comprehensive summary of insights
                        2. PATTERNS: List each pattern with a title and description
                        3. RECOMMENDATIONS: List of specific recommendations"""},
                        {"role": "user", "content": final_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                
                final_content = final_response.choices[0].message.content.strip()
                debug(f"Raw final response (fallback): {final_content[:100]}...")
                
                # Parse the regular response into our expected structure
                sections = final_content.split('\n')
                summary = []
                patterns = []
                recommendations = []
                current_section = None
                
                for line in sections:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if 'SUMMARY:' in line.upper():
                        current_section = 'summary'
                    elif 'PATTERNS:' in line.upper():
                        current_section = 'patterns'
                    elif 'RECOMMENDATIONS:' in line.upper():
                        current_section = 'recommendations'
                    else:
                        if current_section == 'summary':
                            summary.append(line)
                        elif current_section == 'patterns':
                            if ':' in line:
                                title, desc = line.split(':', 1)
                                patterns.append({"title": title.strip(), "description": desc.strip()})
                            else:
                                patterns.append({"title": "Pattern", "description": line.strip()})
                        elif current_section == 'recommendations':
                            if line.startswith('•') or line.startswith('-'):
                                line = line[1:].strip()
                            recommendations.append(line)
                
                # Add coverage information
                coverage_info = f"\n\nAnalysis Coverage: Analyzed {analyzed_tickets} out of {total_tickets} total tickets "
                coverage_info += f"({(analyzed_tickets/total_tickets*100):.1f}% coverage)"
                if total_tickets > max_tickets:
                    coverage_info += f". Analysis was limited to {max_tickets} tickets for processing efficiency."
                
                # Construct final analysis dictionary
                final_analysis = {
                    'summary': ' '.join(summary) + coverage_info,
                    'patterns': patterns if patterns else all_patterns,
                    'recommendations': recommendations if recommendations else list(all_recommendations)
                }
                
                return final_analysis

        except Exception as e:
            debug(f"Error in final analysis: {str(e)}")
            # Return consolidated results with error note
            return {
                'summary': "Analysis completed with partial results due to API error.\n\n" + "\n".join(chunk_summaries),
                'patterns': all_patterns,
                'recommendations': list(all_recommendations)
            }

    except Exception as e:
        debug(f"General error in generate_ai_insights: {str(e)}")
        return {
            'summary': f"Error generating insights: {str(e)}",
            'patterns': [],
            'recommendations': ["Contact support for assistance"]
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
        
        # Convert date columns
        date_cols = ['CreatedDate', 'ClosedDate']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Calculate highest priority for each case
        debug("Calculating highest priorities")
        df['Highest_Priority'] = df['Id'].apply(
            lambda case_id: get_highest_priority_from_history(st.session_state.sf_connection, case_id) or df.loc[df['Id'] == case_id, 'Internal_Priority__c'].iloc[0]
        )
        debug("Highest priorities calculated")
        
        # Log priority changes for debugging
        priority_changes = df[df['Highest_Priority'] != df['Internal_Priority__c']]
        if not priority_changes.empty:
            debug(f"Found {len(priority_changes)} cases with priority changes:")
            for _, case in priority_changes.iterrows():
                debug(f"Case {case['Id']}: Initial Priority={case['Internal_Priority__c']}, Highest Priority={case['Highest_Priority']}")
        
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
        
        df['Month'] = df['CreatedDate'].dt.to_period('M')
        df['Month_Closed'] = df['ClosedDate'].dt.to_period('M')
        
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
        
        df['resolution_time_days'] = (df['ClosedDate'] - df['CreatedDate']).dt.total_seconds() / (24 * 3600)
        df['Month'] = df['CreatedDate'].dt.strftime('%Y-%m')
        
        # Debug logging for priorities
        debug("All priorities in dataset:", df['Highest_Priority'].value_counts().to_dict())
        debug("All Internal Priorities:", df['Internal_Priority__c'].value_counts().to_dict())
        
        # Filter out tickets with unspecified priority and invalid resolution times
        valid_priority_df = df[
            (df['resolution_time_days'].notna()) & 
            (df['resolution_time_days'] > 0) &  # Ensure positive resolution time
            (df['Highest_Priority'].notna()) & 
            (~df['Highest_Priority'].isin(['Unspecified', '', ' ', None]))
        ]
        
        # Debug logging for valid priorities
        debug("Valid priorities after filtering:", valid_priority_df['Highest_Priority'].value_counts().to_dict())
        debug("Sample of valid priority records:", valid_priority_df[['Id', 'Highest_Priority', 'Internal_Priority__c', 'resolution_time_days']].head().to_dict('records'))
        
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
            index='Product_Area__c',
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
            index='Product_Area__c',
            columns='Product_Feature__c',
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
        
        rca_counts = df['RCA__c'].value_counts().reset_index()
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

def generate_powerpoint(filtered_df, active_accounts, escalation_rate):
    """Generate PowerPoint presentation with charts and statistics."""
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
        
        # Add statistics as bullet points - CSAT reference added back
        stats_text = (
            f"• Total Cases: {len(filtered_df)}\n"
            f"• Active Accounts: {active_accounts}\n"
            f"• Product Areas: {filtered_df['Product_Area__c'].nunique()}\n"
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
    """Export data to the selected format."""
    try:
        # Create a copy of the dataframe to avoid modifying the original
        export_df = df.copy()
        
        # Convert all string columns to string type to avoid comparison issues
        for col in export_df.columns:
            if export_df[col].dtype == 'object':
                export_df[col] = export_df[col].astype(str)
        
        # Handle datetime columns - convert to strings for Excel compatibility
        datetime_cols = export_df.select_dtypes(include=['datetime']).columns.tolist()
        if datetime_cols:
            debug(f"Converting datetime columns for export: {datetime_cols}")
            for col in datetime_cols:
                export_df[col] = export_df[col].astype(str)
        
        # Handle CSAT column if it exists
        if 'CSAT__c' in export_df.columns:
            export_df['CSAT__c'] = pd.to_numeric(export_df['CSAT__c'], errors='coerce')
        
        if format == "Excel":
            output = BytesIO()
            try:
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
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
                        if 'First_Response_Time__c' in customer_df.columns and 'CreatedDate' in customer_df.columns:
                            try:
                                response_times = pd.to_datetime(customer_df['First_Response_Time__c']) - pd.to_datetime(customer_df['CreatedDate'])
                                resp_time = response_times.dt.total_seconds().mean() / 3600
                                summary_data['Avg Response Time (hrs)'].append(round(resp_time, 2) if pd.notna(resp_time) else 'N/A')
                            except Exception as e:
                                debug(f"Error calculating response time: {str(e)}")
                                summary_data['Avg Response Time (hrs)'].append('N/A')
                        else:
                            summary_data['Avg Response Time (hrs)'].append('N/A')
                        
                        # Resolution Time calculation
                        if 'ClosedDate' in customer_df.columns and 'CreatedDate' in customer_df.columns:
                            try:
                                resolution_times = pd.to_datetime(customer_df['ClosedDate']) - pd.to_datetime(customer_df['CreatedDate'])
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
                                          export_df['IsEscalated'].astype(bool).mean() * 100)
            st.download_button(
                label="Download PowerPoint",
                data=pptx_data,
                file_name=f"support_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
        elif format == "CSV":
            output = BytesIO()
            export_df.to_csv(output, index=False)
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

def fetch_detailed_data(customer, start_date, end_date):
    """Fetch detailed ticket information for a single customer."""
    try:
        # Create a progress bar
        progress_bar = st.progress(0)
        st.info("Fetching detailed ticket data. This may take a moment...")
        
        # Step 1: Get basic case information (25%)
        query = f"""
            SELECT Id, CaseNumber, Subject, Description,
                Account.Name, CreatedDate, ClosedDate, Status, Internal_Priority__c,
                Product_Area__c, Product_Feature__c, RCA__c,
                First_Response_Time__c, CSAT__c, IsEscalated,
                OwnerId, Origin, Type, Reason,
                Group_Id__c, Environment__c,
                IMPL_Phase__c, Case_Type__c
            FROM Case
            WHERE Account.Name = '{customer}'
            AND CreatedDate >= {start_date.strftime('%Y-%m-%d')}T00:00:00Z
            AND CreatedDate <= {end_date.strftime('%Y-%m-%d')}T23:59:59Z
            AND Case_Type__c = 'Support Request'
        """
        
        cases = execute_soql_query(st.session_state.sf_connection, query)
        if not cases:
            st.warning(f"No cases found for {customer} in the selected date range.")
            return None
        
        cases_df = pd.DataFrame(cases)
        
        # Debug logging for case types
        debug("Case types in dataset:", cases_df['Case_Type__c'].value_counts().to_dict())
        debug("Total cases fetched:", len(cases_df))
        
        # Extract Account Name from nested structure
        if 'Account' in cases_df.columns and isinstance(cases_df['Account'].iloc[0], dict):
            cases_df['Account_Name'] = cases_df['Account'].apply(lambda x: x.get('Name') if isinstance(x, dict) else None)
            cases_df = cases_df.drop('Account', axis=1)
        
        # Convert date fields to datetime objects
        date_columns = ['CreatedDate', 'ClosedDate']
        for col in date_columns:
            if col in cases_df.columns:
                try:
                    cases_df[col] = pd.to_datetime(cases_df[col], errors='coerce')
                except Exception as e:
                    debug(f"Error converting {col} to datetime: {str(e)}")
        
        progress_bar.progress(25)
        
        # Step 2: Get case comments (50%)
        case_ids = cases_df['Id'].tolist()
        comments_query = f"""
            SELECT 
                Id, ParentId, CommentBody, CreatedDate, CreatedById
            FROM CaseComment
            WHERE ParentId IN ('{"','".join(case_ids)}')
            ORDER BY CreatedDate ASC
        """
        
        comments = execute_soql_query(st.session_state.sf_connection, comments_query)
        comments_df = pd.DataFrame(comments) if comments else pd.DataFrame()
        
        # Convert date fields in comments
        if not comments_df.empty and 'CreatedDate' in comments_df.columns:
            try:
                comments_df['CreatedDate'] = pd.to_datetime(comments_df['CreatedDate'], errors='coerce')
            except Exception as e:
                debug(f"Error converting comments CreatedDate to datetime: {str(e)}")
        
        progress_bar.progress(50)
        
        # Step 3: Get case history (75%)
        history_query = f"""
            SELECT 
                Id, CaseId, Field, OldValue, NewValue, CreatedDate, CreatedById
            FROM CaseHistory
            WHERE CaseId IN ('{"','".join(case_ids)}')
            ORDER BY CreatedDate ASC
        """
        
        history = execute_soql_query(st.session_state.sf_connection, history_query)
        history_df = pd.DataFrame(history) if history else pd.DataFrame()
        
        # Convert date fields in history
        if not history_df.empty and 'CreatedDate' in history_df.columns:
            try:
                history_df['CreatedDate'] = pd.to_datetime(history_df['CreatedDate'], errors='coerce')
            except Exception as e:
                debug(f"Error converting history CreatedDate to datetime: {str(e)}")
        
        progress_bar.progress(75)
        
        # Step 4: Get email messages (100%)
        email_query = f"""
            SELECT 
                Id, ParentId, Subject, TextBody, HtmlBody, FromAddress, ToAddress,
                CcAddress, BccAddress, MessageDate, Status, HasAttachment
            FROM EmailMessage
            WHERE ParentId IN ('{"','".join(case_ids)}')
            ORDER BY MessageDate ASC
        """
        
        emails = execute_soql_query(st.session_state.sf_connection, email_query)
        emails_df = pd.DataFrame(emails) if emails else pd.DataFrame()
        
        # Convert date fields in emails
        if not emails_df.empty and 'MessageDate' in emails_df.columns:
            try:
                emails_df['MessageDate'] = pd.to_datetime(emails_df['MessageDate'], errors='coerce')
            except Exception as e:
                debug(f"Error converting emails MessageDate to datetime: {str(e)}")
        
        progress_bar.progress(100)
        
        # Combine all data into a dictionary
        detailed_data = {
            'cases': cases_df,
            'comments': comments_df,
            'history': history_df,
            'emails': emails_df
        }
        
        return detailed_data
    
    except Exception as e:
        st.error(f"Error fetching detailed data: {str(e)}")
        return None

if __name__ == "__main__":
    main() 
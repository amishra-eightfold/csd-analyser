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
from openai import OpenAI

# Set Seaborn and Matplotlib style
sns.set_theme(style="whitegrid")

# Custom color palettes for different visualizations
BLUES_PALETTE = ["#E3F2FD", "#90CAF9", "#42A5F5", "#1E88E5", "#1565C0", "#0D47A1"]  # Material Blues
AQUA_PALETTE = ["#E0F7FA", "#80DEEA", "#26C6DA", "#00ACC1", "#00838F", "#006064"]   # Material Cyan/Aqua
PURPLE_PALETTE = ["#F3E5F5", "#CE93D8", "#AB47BC", "#8E24AA", "#6A1B9A", "#4A148C"]  # Material Purple

# Define custom color palettes for each visualization type
VOLUME_PALETTE = [BLUES_PALETTE[2], AQUA_PALETTE[2]]  # Two distinct colors for Created/Closed
PRIORITY_PALETTE = BLUES_PALETTE[1:]  # Blues for priority levels
CSAT_PALETTE = sns.color_palette(AQUA_PALETTE)  # Aqua palette for CSAT
HEATMAP_PALETTE = sns.color_palette("YlGnBu", as_cmap=True)  # Yellow-Green-Blue for heatmaps

# Create an extended palette for root causes by combining multiple color palettes
ROOT_CAUSE_PALETTE = (
    BLUES_PALETTE[1:] +     # 5 blues
    AQUA_PALETTE[1:] +      # 5 aquas
    PURPLE_PALETTE[1:]      # 5 purples
)  # Total of 15 distinct colors

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
    """Display debug information only if debug mode is enabled."""
    if st.session_state.debug_mode:
        if data is not None:
            st.write(f"Debug - {message}:", data)
        else:
            st.write(f"Debug - {message}")

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

def fetch_data(customers, start_date, end_date):
    """Fetch data from Salesforce based on selected customers and date range."""
    try:
        # Create customer list for SQL query
        customer_list = "'" + "','".join(customers) + "'"
        
        debug(f"Fetching data for customers: {customers} from {start_date} to {end_date}")
        
        # Fetch tickets based on customer selection
        # Adding CSAT__c back to the query
        query = f"""
        SELECT 
            Id, CaseNumber, Subject, Description,
            Account.Name, CreatedDate, ClosedDate, Status, Internal_Priority__c,
            Product_Area__c, Product_Feature__c, RCA__c,
            First_Response_Time__c, CSAT__c, IsEscalated
        FROM Case
        WHERE Account.Name IN ({customer_list})
        AND CreatedDate >= {start_date.strftime('%Y-%m-%d')}T00:00:00Z
        AND CreatedDate <= {end_date.strftime('%Y-%m-%d')}T23:59:59Z
        """
        
        # Execute the SOQL query
        records = execute_soql_query(st.session_state.sf_connection, query)
        if not records:
            debug("No tickets found for the selected criteria")
            return None, None, None, None, None
        
        # Convert to dataframe
        df = pd.DataFrame(records)
        
        # Extract Account Name from nested structure
        if 'Account' in df.columns and isinstance(df['Account'].iloc[0], dict):
            df['Account_Name'] = df['Account'].apply(lambda x: x.get('Name') if isinstance(x, dict) else None)
            df = df.drop('Account', axis=1)
            
        # Safe handling of date columns
        date_columns = ['CreatedDate', 'ClosedDate', 'First_Response_Time__c']
        for col in date_columns:
            if col in df.columns:
                # Safely convert date columns to datetime, coercing errors to NaT
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Dump raw data to Excel before any processing
        try:
            # Generate a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_excel_file = f"raw_data_{timestamp}.xlsx"
            
            # Create a copy for export to avoid modifying original data
            export_df = df.copy()
            
            # Simpler, more robust approach to handle datetime columns for Excel
            datetime_cols = export_df.select_dtypes(include=['datetime']).columns.tolist()
            debug(f"Datetime columns detected: {datetime_cols}")
            
            # Convert all datetime columns to strings in ISO format for Excel compatibility
            for col in datetime_cols:
                export_df[col] = export_df[col].astype(str)
                debug(f"Converted {col} to string format for Excel compatibility")
            
            # Save the raw data to Excel
            try:
                export_df.to_excel(raw_excel_file, index=False)
                debug(f"Raw data saved to {raw_excel_file}")
                st.info(f"Raw data saved to: {raw_excel_file}")
            except Exception as excel_error:
                # Fallback to CSV if Excel export fails
                debug(f"Excel export failed: {str(excel_error)}")
                csv_file = f"raw_data_{timestamp}.csv"
                export_df.to_csv(csv_file, index=False)
                debug(f"Saved data as CSV instead: {csv_file}")
                st.info(f"Saved data as CSV (Excel export failed): {csv_file}")
            
        except Exception as e:
            debug(f"Error saving raw data: {str(e)}")
            import traceback
            debug(traceback.format_exc())
        
        # Get case IDs for fetching related data
        case_ids = df['Id'].tolist()
        
        # Fetch related data
        emails_df, comments_df, history_df, attachments_df = fetch_related_data(case_ids)
        
        # Post-process data - ensure NA values are properly handled
        
        # Clean classification fields to handle problematic values
        df = preprocess_classification_fields(df)
        
        # For ticket classification, also need to include emails and comments content
        if 'Id' in df.columns and emails_df is not None and not emails_df.empty:
            # Combine email content by case
            email_content = emails_df.groupby('ParentId')['TextBody'].apply(lambda x: ' '.join(x) if isinstance(x, (list, pd.Series)) else str(x)).reset_index()
            email_content.columns = ['Id', 'email_content']
            
            # Merge with main dataframe
            df = pd.merge(df, email_content, on='Id', how='left')
            
        if 'Id' in df.columns and comments_df is not None and not comments_df.empty:
            # Combine comment content by case
            comment_content = comments_df.groupby('ParentId')['CommentBody'].apply(lambda x: ' '.join(x) if isinstance(x, (list, pd.Series)) else str(x)).reset_index()
            comment_content.columns = ['Id', 'comment_content']
            
            # Merge with main dataframe
            df = pd.merge(df, comment_content, on='Id', how='left')
        
        # Apply ticket classification
        df = classify_tickets(df)
            
        # Convert any remaining NA values in text fields to empty strings to avoid boolean ambiguity
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('')
            
        debug(f"Retrieved {len(df)} tickets")
        
        return df, emails_df, comments_df, history_df, attachments_df
    
    except Exception as e:
        debug(f"Error fetching data: {str(e)}")
        import traceback
        debug(traceback.format_exc())
        return None, None, None, None, None

def fetch_related_data(case_ids):
    """Fetch related data for the given case IDs."""
    try:
        # If no case IDs, return empty dataframes
        if not case_ids:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Format IDs for SOQL query
        case_ids_str = "'" + "','".join(case_ids) + "'"
        
        # Fetch emails
        email_query = f"""
            SELECT Id, ParentId, Subject, TextBody, HtmlBody, FromAddress, ToAddress, CreatedDate
            FROM EmailMessage
            WHERE ParentId IN ({case_ids_str})
        """
        email_records = execute_soql_query(st.session_state.sf_connection, email_query)
        emails_df = pd.DataFrame(email_records) if email_records else pd.DataFrame()
        
        # Fetch comments
        comment_query = f"""
            SELECT Id, ParentId, CommentBody, CreatedDate, CreatedById
            FROM CaseComment
            WHERE ParentId IN ({case_ids_str})
        """
        comment_records = execute_soql_query(st.session_state.sf_connection, comment_query)
        comments_df = pd.DataFrame(comment_records) if comment_records else pd.DataFrame()
        
        # Fetch history
        history_query = f"""
            SELECT Id, CaseId, Field, OldValue, NewValue, CreatedDate
            FROM CaseHistory
            WHERE CaseId IN ({case_ids_str})
        """
        history_records = execute_soql_query(st.session_state.sf_connection, history_query)
        history_df = pd.DataFrame(history_records) if history_records else pd.DataFrame()
        
        # Fetch attachments
        attachment_query = f"""
            SELECT Id, ParentId, Name, ContentType, CreatedDate
            FROM Attachment
            WHERE ParentId IN ({case_ids_str})
        """
        attachment_records = execute_soql_query(st.session_state.sf_connection, attachment_query)
        attachments_df = pd.DataFrame(attachment_records) if attachment_records else pd.DataFrame()
        
        debug(f"Fetched related data: {len(emails_df)} emails, {len(comments_df)} comments, {len(history_df)} history records, {len(attachments_df)} attachments")
        
        return emails_df, comments_df, history_df, attachments_df
    
    except Exception as e:
        debug(f"Error fetching related data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def preprocess_classification_fields(df):
    """Clean and standardize classification fields.
    
    This function identifies and standardizes problematic values in the 
    classification fields (Product_Area__c, Product_Feature__c, RCA__c).
    
    Args:
        df (pandas.DataFrame): The dataframe containing ticket data
        
    Returns:
        pandas.DataFrame: The dataframe with cleaned classification fields
    """
    # Clone the dataframe to avoid modifying the original
    df = df.copy()
    
    # Define problematic values
    empty_values = ['', 'None', 'Unspecified', 'N/A', 'Unknown', 'Other', None]
    
    # Replace empty values with NaN for easier handling
    for field in ['Product_Area__c', 'Product_Feature__c', 'RCA__c']:
        if field in df.columns:
            df[field] = df[field].apply(lambda x: pd.NA if x in empty_values or pd.isna(x) else x)
            # Log the percentage of missing values
            missing_pct = df[field].isna().mean() * 100
            debug(f"Field {field} has {missing_pct:.2f}% missing values after preprocessing")
    
    return df

def create_text_classifier(X_text, y_labels):
    """Create a text classifier using scikit-learn.
    
    This function builds a machine learning pipeline that:
    1. Converts text to numerical features using TF-IDF vectorization
    2. Trains a Random Forest classifier on these features
    
    Args:
        X_text (Series): Text data for training
        y_labels (Series): Classification labels for training
        
    Returns:
        Pipeline: Trained scikit-learn pipeline for text classification
    """
    # Text vectorization with TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    # Create classification pipeline with Random Forest
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    debug(f"Training classifier on {len(X_text)} samples with {len(set(y_labels))} unique classes")
    pipeline.fit(X_text, y_labels)
    
    return pipeline

def get_cached_classifier(field, X_text=None, y_labels=None, force_retrain=False):
    """Get or create a classifier for a field, with optional caching"""
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{field}_classifier.pkl")
    
    # Check if we need to create a new model
    if force_retrain or not os.path.exists(model_path) or X_text is not None:
        if X_text is not None and y_labels is not None and len(X_text) >= 10:
            debug(f"Training new classifier for {field}")
            classifier = create_text_classifier(X_text, y_labels)
            
            # Save the model for future use
            try:
                joblib.dump(classifier, model_path)
                debug(f"Saved classifier to {model_path}")
            except Exception as e:
                debug(f"Error saving classifier: {str(e)}")
        else:
            debug(f"Insufficient data to train classifier for {field}")
            return None
    else:
        # Load existing model
        try:
            classifier = joblib.load(model_path)
            debug(f"Loaded existing classifier from {model_path}")
        except Exception as e:
            debug(f"Error loading classifier: {str(e)}")
            return None
            
    return classifier

def classify_tickets(data):
    """
    Classify tickets that have missing classification fields using text-based ML.
    Returns dataframe with additional enhanced classification fields.
    """
    if data is None or data.empty:
        return data

    debug("Starting ticket classification")
    
    # Create a copy of the data to avoid modifying the original
    data = data.copy()
    
    # Fields to potentially classify
    classification_fields = ['Product_Area__c', 'Product_Feature__c', 'RCA__c']
    
    # Prepare text features from the tickets
    text_features = []
    for _, row in data.iterrows():
        # Combine relevant text fields, safely handling NAs
        subject = str(row['Subject']) if pd.notna(row['Subject']) else ""
        description = str(row['Description']) if pd.notna(row['Description']) else ""
        
        # Create a combined feature
        combined_text = f"{subject} {description}"
        text_features.append(combined_text)
    
    # Add text features as a column for debugging
    data['combined_text_features'] = text_features
    
    # For each classification field, check if we can improve it
    for field in classification_fields:
        # Skip if field doesn't exist in the data
        if field not in data.columns:
            debug(f"Field {field} not in dataset, skipping classification")
            continue
        
        # Fill NA values with 'Unknown' to avoid boolean ambiguity issues
        data[field] = data[field].fillna('Unknown')
        
        # Create mask for missing or problematic values (now 'Unknown')
        missing_mask = data[field] == 'Unknown'
        
        # Get indices where the field is missing or problematic
        missing_indices = data[missing_mask].index
        
        # Only proceed if we have missing values
        if len(missing_indices) == 0:
            debug(f"No missing {field} values to classify")
            continue
        
        # Only proceed if we have enough known values to learn from
        known_indices = data[~missing_mask].index
        if len(known_indices) < 5:  # Arbitrary threshold to ensure we have enough data
            debug(f"Not enough known {field} values to train classifier ({len(known_indices)} found)")
            continue
        
        # Extract known data for training
        X_train = [text_features[i] for i in known_indices]
        y_train = data.loc[known_indices, field].values
        
        # Get text for prediction
        X_test = [text_features[i] for i in missing_indices]
        
        # Get or create classifier
        classifier = get_cached_classifier(field, X_train, y_train)
        if classifier is None:
            debug(f"Could not create classifier for {field}")
            continue
        
        # Make predictions with probabilities
        predictions = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test)
        confidences = [max(probs) for probs in probabilities]
        
        # Create a enhanced field name
        enhanced_field = f"{field}_enhanced"
        confidence_field = f"{field}_confidence"
        low_confidence_field = f"{field}_low_confidence"
        
        # Fill in the enhanced field with original values to start
        data[enhanced_field] = data[field]
        
        # Flag for low confidence predictions to help users identify uncertain predictions
        data[low_confidence_field] = False
        
        # Add confidence scores and handle NA values safely
        data[confidence_field] = pd.NA
        
        # For indices with predictions, update the values
        for i, (idx, pred, conf) in enumerate(zip(missing_indices, predictions, confidences)):
            # Update enhanced field with prediction
            data.loc[idx, enhanced_field] = pred
            
            # Store confidence score
            data.loc[idx, confidence_field] = conf
            
            # Flag low confidence predictions (threshold arbitrary, can be tuned)
            data.loc[idx, low_confidence_field] = conf < 0.7
        
        debug(f"Enhanced {field} with ML predictions for {len(missing_indices)} entries")
    
    # Final cleanup - make sure all enhanced fields don't have NA values
    for field in classification_fields:
        enhanced_field = f"{field}_enhanced"
        confidence_field = f"{field}_confidence"
        low_confidence_field = f"{field}_low_confidence"
        
        if enhanced_field in data.columns:
            # Replace any remaining NA values with 'Unspecified'
            data[enhanced_field] = data[enhanced_field].fillna('Unspecified')
            
        if confidence_field in data.columns:
            # Fill NA confidence with 0
            data[confidence_field] = data[confidence_field].fillna(0.0)
            
        if low_confidence_field in data.columns:
            # Ensure boolean field has no NAs
            data[low_confidence_field] = data[low_confidence_field].fillna(False)
    
    debug("Completed ticket classification")
    return data

def display_visualizations(df, customers):
    """Display visualizations using the dataset."""
    try:
        if df is None or df.empty:
            st.warning("No data available for visualization.")
            return
            
        # Make a defensive copy of the dataframe
        df = df.copy()
        
        # Pre-emptively handle NA values in ALL columns to avoid boolean ambiguity
        for col in df.columns:
            # Skip CSAT__c column as it needs special handling
            if col == 'CSAT__c':
                continue
            # For object (string) columns, fill NAs with 'Unspecified'
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].fillna('Unspecified')
            # For numeric columns, fill NAs with 0
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            # For boolean columns, fill NAs with False
            elif pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].fillna(False)
            # For datetime columns, don't modify (we'll handle them specially)
        
        # Debug CSAT data before any processing
        debug(f"CSAT data types before processing: {df['CSAT__c'].apply(type).value_counts()}")
        debug(f"CSAT null count before processing: {df['CSAT__c'].isna().sum()}")
        debug(f"CSAT value counts before processing: {df['CSAT__c'].value_counts().head()}")
        
        debug("Pre-processed visualization dataframe to handle NA values")
            
        # Check if we have enhanced classification fields and inform user
        has_enhanced_fields = any(col.endswith('_enhanced') for col in df.columns)
        if has_enhanced_fields:
            st.info("📊 Using AI-enhanced classification fields for visualizations where available.")
            
            # Show sample of reclassified tickets
            with st.expander("View AI-Enhanced Classifications"):
                # Get fields that have been enhanced
                enhanced_fields = [col[:-9] for col in df.columns if col.endswith('_enhanced')]
                
                if enhanced_fields:
                    # For each enhanced field, show before/after
                    for orig_field in enhanced_fields:
                        enhanced_field = f"{orig_field}_enhanced"
                        confidence_field = f"{orig_field}_confidence"
                        
                        # Skip fields that aren't actually present
                        if orig_field not in df.columns or enhanced_field not in df.columns:
                            continue
                            
                        # Only show samples where the original was Unspecified but we predicted it
                        # Handle NA values safely by using string comparison to avoid boolean issues
                        reclassified = df[(df[orig_field] == 'Unspecified') & (df[enhanced_field] != 'Unspecified')].copy()
                        
                        if not reclassified.empty:
                            st.write(f"### {orig_field} Reclassifications")
                            st.write(f"Found {len(reclassified)} tickets with missing {orig_field} that were automatically classified.")
                            
                            # Prepare a nice display table
                            display_df = reclassified[['CaseNumber', 'Subject', enhanced_field]].copy()
                            
                            # Add confidence if available
                            if confidence_field in reclassified.columns:
                                display_df['Confidence'] = reclassified[confidence_field].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                                
                            # Limit to 10 examples
                            if len(display_df) > 10:
                                display_df = display_df.head(10)
                                
                            st.dataframe(display_df)
        
        # Use enhanced fields when available for visualizations
        # Make sure to handle NAs appropriately in all visualizations
        area_field = 'Product_Area__c_enhanced' if 'Product_Area__c_enhanced' in df.columns else 'Product_Area__c'
        feature_field = 'Product_Feature__c_enhanced' if 'Product_Feature__c_enhanced' in df.columns else 'Product_Feature__c'
        rca_field = 'RCA__c_enhanced' if 'RCA__c_enhanced' in df.columns else 'RCA__c'
        
        # Fill NAs in classification fields to avoid boolean ambiguity issues
        for field in [area_field, feature_field, rca_field]:
            if field in df.columns:
                df[field] = df[field].fillna('Unspecified')
        
        # Count of tickets by customer
        st.subheader("Tickets by Customer")
        
        # Determine active accounts based on whether they have tickets in the dataset
        active_accounts = df['Account_Name'].unique()
        
        # Count tickets by account
        customer_df = df.copy()
        ticket_counts = customer_df.groupby('Account_Name').size().reset_index(name='Ticket_Count')
        
        plt.figure(figsize=(10, 6))
        bar_plot = sns.barplot(x='Account_Name', y='Ticket_Count', data=ticket_counts)
        plt.title('Ticket Count by Customer')
        plt.xlabel('Customer')
        plt.ylabel('Number of Tickets')
        plt.xticks(rotation=45)
        
        # Add count labels on top of each bar
        for i, count in enumerate(ticket_counts['Ticket_Count']):
            bar_plot.text(i, count + 0.1, str(count), ha='center')
            
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
        
        # Tickets over time
        st.subheader("Ticket Volume Over Time (Created vs Closed)")
        
        # Convert CreatedDate to month periods for time series analysis
        customer_df['Month'] = pd.to_datetime(customer_df['CreatedDate']).dt.to_period('M')
        ticket_trend = customer_df.groupby(['Month', 'Account_Name']).size().reset_index(name='Count')
        
        # Sort by month to ensure chronological order
        ticket_trend = ticket_trend.sort_values('Month')
        
        # Convert Month back to string for plotting
        ticket_trend['Month'] = ticket_trend['Month'].astype(str)
        ticket_trend['Type'] = 'Created'
        
        # Filter for closed tickets
        closed_df = df[df['Status'] == 'Closed'].copy()
        
        # Process closed tickets if available
        if not closed_df.empty:
            # Convert ClosedDate to month periods
            closed_df['Month'] = pd.to_datetime(closed_df['ClosedDate']).dt.to_period('M')
            closed_trend = closed_df.groupby(['Month', 'Account_Name']).size().reset_index(name='Count')
            
            # Sort by month to ensure chronological order
            closed_trend = closed_trend.sort_values('Month')
            
            # Convert Month back to string for plotting
            closed_trend['Month'] = closed_trend['Month'].astype(str)
            closed_trend['Type'] = 'Closed'
            
            # Combine created and closed ticket data
            combined_trend = pd.concat([ticket_trend, closed_trend], ignore_index=True)
            
            # Create the combined visualization
            plt.figure(figsize=(14, 7))
            
            # Use barplot with hue for ticket type (Created vs Closed)
            sns.barplot(data=combined_trend, x='Month', y='Count', hue='Type')
            
            plt.title('Ticket Volume Over Time (Created vs Closed)')
            plt.xlabel('Month')
            plt.ylabel('Number of Tickets')
            plt.xticks(rotation=45)
            plt.legend(title='Ticket Type')
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
        else:
            # If no closed tickets, just show created tickets
            plt.figure(figsize=(12, 6))
            sns.barplot(data=ticket_trend, x='Month', y='Count', hue='Account_Name')
            plt.title('Ticket Volume Over Time (Created)')
            plt.xlabel('Month')
            plt.ylabel('Number of Tickets')
            plt.xticks(rotation=45)
            plt.legend(title='Customer')
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
            st.info("No closed tickets found in the selected date range.")
        
        # First Response Time Analysis
        st.subheader("First Response Time Analysis")

        try:
            # Get case IDs for fetching related data
            case_ids = df['Id'].tolist()
            
            if not case_ids:
                st.info("No cases available for response time analysis.")
                return
                
            # Fetch related data (comments and emails)
            debug(f"Fetching related data for {len(case_ids)} cases")
            emails_df, comments_df, history_df, attachments_df = fetch_related_data(case_ids)
            
            # Calculate first response time
            response_times = []
            for idx, case in df.iterrows():
                if pd.isna(case['CreatedDate']):
                    continue
                    
                case_id = case['Id']
                created_date = pd.to_datetime(case['CreatedDate'])
                priority = case.get('Priority', 'Not Set')
                account_name = case['Account_Name']
                
                # Find first public comment or email
                first_response = None
                
                # Check comments
                if comments_df is not None and not comments_df.empty:
                    case_comments = comments_df[comments_df['ParentId'] == case_id]
                    if not case_comments.empty:
                        first_comment = pd.to_datetime(case_comments['CreatedDate'].min())
                        first_response = first_comment if first_response is None else min(first_response, first_comment)
                
                # Check emails
                if emails_df is not None and not emails_df.empty:
                    case_emails = emails_df[emails_df['ParentId'] == case_id]
                    if not case_emails.empty:
                        first_email = pd.to_datetime(case_emails['CreatedDate'].min())
                        first_response = first_email if first_response is None else min(first_response, first_email)
                
                if first_response is not None:
                    response_time = (first_response - created_date).total_seconds() / 3600  # Convert to hours
                    if response_time >= 0:  # Filter out negative response times (data errors)
                        response_times.append({
                            'Case_Id': case_id,
                            'Account_Name': account_name,
                            'Priority': priority,
                            'Response_Time_Hours': response_time
                        })
            
            if response_times:
                response_times_df = pd.DataFrame(response_times)
                
                # Original account-based analysis
                account_response_stats = response_times_df.groupby('Account_Name').agg({
                    'Response_Time_Hours': ['count', 'mean', 'median', 'std']
                }).round(2)
                
                account_response_stats.columns = ['Count', 'Mean Hours', 'Median Hours', 'Std Dev']
                st.write("### Response Time by Customer")
                st.dataframe(account_response_stats, use_container_width=True)
                
                # Priority-based Response Time Analysis
                if 'Priority' in df.columns:
                    st.write("### Response Time by Priority")
                    priority_stats = response_times_df.groupby('Priority').agg({
                        'Response_Time_Hours': ['count', 'mean', 'median', 'std']
                    }).round(2)
                    
                    priority_stats.columns = ['Count', 'Mean Hours', 'Median Hours', 'Std Dev']
                    st.dataframe(priority_stats, use_container_width=True)
                    
                    # Box plot of response times by priority
                    plt.figure(figsize=(12, 6))
                    sns.boxplot(data=response_times_df, x='Priority', y='Response_Time_Hours', hue='Priority', legend=False)
                    plt.title('Distribution of First Response Time by Priority')
                    plt.xlabel('Priority')
                    plt.ylabel('Response Time (Hours)')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(plt)
                    plt.close()
                    
                    # Trend analysis by priority
                    response_times_df['Month'] = pd.to_datetime(df.loc[response_times_df['Case_Id'], 'CreatedDate']).dt.to_period('M')
                    monthly_priority_stats = response_times_df.groupby(['Month', 'Priority'])['Response_Time_Hours'].mean().reset_index()
                    monthly_priority_stats['Month'] = monthly_priority_stats['Month'].astype(str)
                    
                    plt.figure(figsize=(12, 6))
                    for priority in monthly_priority_stats['Priority'].unique():
                        priority_data = monthly_priority_stats[monthly_priority_stats['Priority'] == priority]
                        plt.plot(priority_data['Month'], priority_data['Response_Time_Hours'], 
                                marker='o', label=priority)
                    
                    plt.title('Average First Response Time Trend by Priority')
                    plt.xlabel('Month')
                    plt.ylabel('Average Response Time (Hours)')
                    plt.legend(title='Priority')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(plt)
                    plt.close()
                    
                    # Add SLA analysis if available
                    if 'SLA_Response_Time__c' in df.columns:
                        st.write("### SLA Compliance Analysis")
                        response_times_df['SLA_Target'] = df.loc[response_times_df['Case_Id'], 'SLA_Response_Time__c'].values
                        response_times_df['Within_SLA'] = response_times_df['Response_Time_Hours'] <= response_times_df['SLA_Target']
                        
                        sla_stats = response_times_df.groupby('Priority').agg({
                            'Within_SLA': ['count', 'mean']
                        }).round(3)
                        
                        sla_stats.columns = ['Total Cases', 'SLA Compliance Rate']
                        sla_stats['SLA Compliance Rate'] = (sla_stats['SLA Compliance Rate'] * 100).round(1).astype(str) + '%'
                        st.dataframe(sla_stats, use_container_width=True)
            else:
                st.warning("No valid response time data available.")

        except Exception as e:
            st.error(f"Error analyzing First Response Time: {str(e)}")
            debug(f"First Response Time analysis error: {str(e)}")
            debug(traceback.format_exc())

        # Continue with Resolution Time Analysis if available
        st.subheader("Resolution Time Analysis")
        
        # Filter for resolved cases
        resolved_df = df[df['Status'] == 'Closed'].copy()
        
        if not resolved_df.empty:
            # Calculate resolution time in days
            resolved_df['Resolution_Time_Days'] = (pd.to_datetime(resolved_df['ClosedDate']) - 
                                                pd.to_datetime(resolved_df['CreatedDate'])).dt.total_seconds() / (60*60*24)
            
            # Filter out any negative resolution times (data errors)
            resolved_df = resolved_df[resolved_df['Resolution_Time_Days'] >= 0]
            
            # Add month for time series
            resolved_df['Month'] = resolved_df['CreatedDate'].dt.to_period('M').astype(str)
            
            # By Product Area - Use enhanced field if available
            if 'Product_Area__c_enhanced' in resolved_df.columns:
                product_field = 'Product_Area__c_enhanced'
                debug("Using enhanced Product Area field for visualizations")
            else:
                product_field = 'Product_Area__c'
            
            if product_field in resolved_df.columns:
                # Drop rows with null product area - explicitly fill NA values first
                resolved_df[product_field] = resolved_df[product_field].fillna('Unspecified')
                product_resolution = resolved_df[resolved_df[product_field] != 'Unspecified'].copy()
                
                if not product_resolution.empty:
                    plt.figure(figsize=(10, 6))
                    # Get unique values to set appropriate palette size
                    unique_areas = product_resolution[product_field].dropna().unique()
                    palette = sns.color_palette("Set2", n_colors=len(unique_areas))
                    
                    sns.boxplot(data=product_resolution, x=product_field, y='Resolution_Time_Days',
                              palette=palette)
                    plt.title('Resolution Time by Product Area')
                    plt.xlabel('Product Area')
                    plt.ylabel('Resolution Time (Days)')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(plt)
                    plt.close()
            
            # By Product Feature - Use enhanced field if available
            if feature_field in resolved_df.columns:
                # Drop rows with null product feature - explicitly fill NA values first
                resolved_df[feature_field] = resolved_df[feature_field].fillna('Unspecified')
                feature_resolution = resolved_df[resolved_df[feature_field] != 'Unspecified'].copy()
                
                if not feature_resolution.empty:
                    plt.figure(figsize=(10, 6))
                    # Get unique values to set appropriate palette size
                    unique_features = feature_resolution[feature_field].dropna().unique()
                    palette = sns.color_palette("Set2", n_colors=len(unique_features))
                    
                    sns.boxplot(data=feature_resolution, x=feature_field, y='Resolution_Time_Days',
                              palette=palette)
                    plt.title('Resolution Time by Product Feature')
                    plt.xlabel('Product Feature')
                    plt.ylabel('Resolution Time (Days)')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(plt)
                    plt.close()
            
            # Root Cause Analysis - Use enhanced field if available
            if rca_field in resolved_df.columns:
                st.subheader("Root Cause Analysis")
                
                # Drop rows with null RCA - explicitly fill NA values first
                resolved_df[rca_field] = resolved_df[rca_field].fillna('Unspecified')
                resolution_by_root = resolved_df[resolved_df[rca_field] != 'Unspecified'].copy()
                
                if not resolution_by_root.empty:
                    # Distribution of RCA
                    plt.figure(figsize=(10, 6))
                    root_counts = resolution_by_root[rca_field].value_counts()
                    # Only plot the top N causes to avoid cluttering
                    top_n = min(10, len(root_counts))
                    # Get unique values to set appropriate palette size
                    unique_rca = resolution_by_root[rca_field].unique()
                    palette = sns.color_palette("Set3", n_colors=len(unique_rca))
                    
                    sns.countplot(data=resolution_by_root, x=rca_field, 
                                order=root_counts.index[:top_n],
                                palette=palette)
                    plt.title('Distribution of Root Causes')
                    plt.xlabel('Root Cause')
                    plt.ylabel('Number of Tickets')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(plt)
                    plt.close()
                    
                    # Resolution time by RCA
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=resolution_by_root, x=rca_field, y='Resolution_Time_Days',
                              order=root_counts.index[:top_n],
                              palette=palette)
                    plt.title('Resolution Time by Root Cause')
                    plt.xlabel('Root Cause')
                    plt.ylabel('Resolution Time (Days)')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(plt)
                    plt.close()
                    
                    # In the Resolution Time Analysis section
                    if not resolved_df.empty:
                        # Priority-based Resolution Time Analysis
                        if 'Priority' in resolved_df.columns:
                            st.write("### Resolution Time by Priority")
                            priority_resolution_stats = resolved_df.groupby('Priority').agg({
                                'Resolution_Time_Days': ['count', 'mean', 'median', 'std']
                            }).round(2)
                            
                            priority_resolution_stats.columns = ['Count', 'Mean Days', 'Median Days', 'Std Dev']
                            st.dataframe(priority_resolution_stats, use_container_width=True)
                            
                            # Box plot of resolution times by priority
                            plt.figure(figsize=(12, 6))
                            sns.boxplot(data=resolved_df, x='Priority', y='Resolution_Time_Days')
                            plt.title('Distribution of Resolution Time by Priority')
                            plt.xlabel('Priority')
                            plt.ylabel('Resolution Time (Days)')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(plt)
                            plt.close()
                            
                            # Trend analysis by priority
                            resolved_df['Month'] = pd.to_datetime(resolved_df['CreatedDate']).dt.to_period('M')
                            monthly_priority_resolution = resolved_df.groupby(['Month', 'Priority'])['Resolution_Time_Days'].mean().reset_index()
                            monthly_priority_resolution['Month'] = monthly_priority_resolution['Month'].astype(str)
                            
                            plt.figure(figsize=(12, 6))
                            for priority in monthly_priority_resolution['Priority'].unique():
                                priority_data = monthly_priority_resolution[monthly_priority_resolution['Priority'] == priority]
                                plt.plot(priority_data['Month'], priority_data['Resolution_Time_Days'], 
                                        marker='o', label=priority)
                            
                            plt.title('Average Resolution Time Trend by Priority')
                            plt.xlabel('Month')
                            plt.ylabel('Average Resolution Time (Days)')
                            plt.legend(title='Priority')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(plt)
                            plt.close()
            
            # Add heatmap analysis for Product Area and Product Feature
            st.subheader("Product Area and Feature Analysis")
            
            if product_field in resolved_df.columns and feature_field in resolved_df.columns:
                # Create pivot tables for volume and resolution time
                volume_heatmap = pd.crosstab(resolved_df[product_field], resolved_df[feature_field])
                resolution_time_heatmap = pd.pivot_table(
                    resolved_df,
                    values='Resolution_Time_Days',
                    index=product_field,
                    columns=feature_field,
                    aggfunc='mean'
                )
                
                # Create two columns for the heatmaps
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Ticket Volume by Product Area and Feature")
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(volume_heatmap, annot=True, fmt='d', cmap='YlOrRd',
                               cbar_kws={'label': 'Number of Tickets'})
                    plt.title('Ticket Volume Heatmap')
                    plt.xlabel('Product Feature')
                    plt.ylabel('Product Area')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(plt)
                    plt.close()
                
                with col2:
                    st.write("#### Average Resolution Time by Product Area and Feature")
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(resolution_time_heatmap, annot=True, fmt='.1f', cmap='YlOrRd',
                               cbar_kws={'label': 'Average Resolution Time (Days)'})
                    plt.title('Resolution Time Heatmap')
                    plt.xlabel('Product Feature')
                    plt.ylabel('Product Area')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(plt)
                    plt.close()
                
                # Add summary statistics
                st.write("#### Summary Statistics")
                summary_stats = pd.DataFrame({
                    'Metric': [
                        'Total Unique Product Areas',
                        'Total Unique Features',
                        'Most Common Product Area',
                        'Most Common Feature',
                        'Highest Volume Combination',
                        'Longest Avg Resolution Time Combination'
                    ],
                    'Value': [
                        len(volume_heatmap.index),
                        len(volume_heatmap.columns),
                        volume_heatmap.sum(axis=1).idxmax(),
                        volume_heatmap.sum(axis=0).idxmax(),
                        f"{volume_heatmap.stack().idxmax()[0]} - {volume_heatmap.stack().idxmax()[1]} ({volume_heatmap.stack().max()} tickets)",
                        f"{resolution_time_heatmap.stack().idxmax()[0]} - {resolution_time_heatmap.stack().idxmax()[1]} ({resolution_time_heatmap.stack().max():.1f} days)"
                    ]
                })
                st.dataframe(summary_stats, use_container_width=True)
        else:
            st.info("No resolved tickets found in the selected date range.")
            
    except Exception as e:
        st.error(f"Error displaying visualizations: {str(e)}")
        debug(f"Error in display_visualizations: {str(e)}")
        if st.session_state.debug_mode:
            st.error(f"Debug - Visualization error details: {str(e)}")

def remove_pii(text):
    """Remove personally identifiable information (PII) from text."""
    if not isinstance(text, str):
        return ''
        
    # Initialize cleaned text
    cleaned_text = text
    
    # Remove email addresses
    cleaned_text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', cleaned_text)
    
    # Remove phone numbers (various formats)
    phone_patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Standard US format
        r'\+\d{1,3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{4}\b',  # International format
        r'\(\d{3}\)\s*\d{3}[-\s]?\d{4}\b',  # (123) 456-7890
        r'\b\d{10}\b'  # Plain 10 digits
    ]
    for pattern in phone_patterns:
        cleaned_text = re.sub(pattern, '[PHONE]', cleaned_text)
    
    # Remove URLs
    cleaned_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', cleaned_text)
    
    # Remove IP addresses
    cleaned_text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP_ADDRESS]', cleaned_text)
    
    # Remove credit card numbers
    cleaned_text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[CREDIT_CARD]', cleaned_text)
    
    # Remove social security numbers
    cleaned_text = re.sub(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', '[SSN]', cleaned_text)
    
    # Remove names (common name patterns)
    name_patterns = [
        r'(?i)(?:mr\.|mrs\.|ms\.|dr\.|prof\.)\s+[a-z]+',  # Titles with names
        r'(?i)(?:first|last|full)\s+name\s*(?::|is|=)\s*[a-z\s]+',  # Name declarations
        r'(?i)sincerely,\s+[a-z\s]+',  # Email signatures
        r'(?i)regards,\s+[a-z\s]+',  # Email signatures
        r'(?i)best,\s+[a-z\s]+'  # Email signatures
    ]
    for pattern in name_patterns:
        cleaned_text = re.sub(pattern, '[NAME]', cleaned_text)
    
    # Remove common password patterns
    cleaned_text = re.sub(r'(?i)password\s*(?::|is|=)\s*\S+', '[PASSWORD]', cleaned_text)
    
    # Remove dates of birth
    dob_patterns = [
        r'\b\d{2}[-/]\d{2}[-/]\d{4}\b',  # MM/DD/YYYY or DD/MM/YYYY
        r'\b\d{4}[-/]\d{2}[-/]\d{2}\b',  # YYYY/MM/DD
        r'(?i)(?:date\s+of\s+birth|dob|birth\s+date)\s*(?::|is|=)\s*[a-z0-9\s,]+' # DOB declarations
    ]
    for pattern in dob_patterns:
        cleaned_text = re.sub(pattern, '[DOB]', cleaned_text)
    
    return cleaned_text

def prepare_text_for_ai(data):
    """Prepare text data for AI analysis by removing PII."""
    if isinstance(data, dict):
        # Handle dictionary input
        cleaned_data = {}
        for key, value in data.items():
            if isinstance(value, (str, list, dict)):
                cleaned_data[key] = prepare_text_for_ai(value)
            else:
                cleaned_data[key] = value
        return cleaned_data
    elif isinstance(data, list):
        # Handle list input
        return [prepare_text_for_ai(item) for item in data]
    elif isinstance(data, str):
        # Handle string input
        return remove_pii(data)
    elif isinstance(data, pd.DataFrame):
        # Handle DataFrame input
        cleaned_df = data.copy()
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].apply(remove_pii)
        return cleaned_df
    else:
        # Return as is for other types
        return data

def calculate_tokens(text):
    """Calculate the number of tokens in a text string using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    except Exception as e:
        debug(f"Error calculating tokens: {str(e)}")
        # Fallback to approximate calculation (avg 4 chars per token)
        return len(text) // 4

def chunk_data(data, max_tokens=4000):
    """
    Intelligently chunk data to fit within token limits while preserving context.
    Returns a list of chunks and their token counts.
    """
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    # Helper function to get essential context
    def get_context(item):
        return {
            'id': item.get('Id', 'Unknown'),
            'case_number': item.get('CaseNumber', 'Unknown'),
            'account': item.get('Account_Name', 'Unknown'),
            'created_date': str(item.get('CreatedDate', 'Unknown'))
        }
    
    # Process each item
    for item in data:
        # Calculate tokens for this item
        item_text = json.dumps(item)
        item_tokens = calculate_tokens(item_text)
        
        # If single item exceeds max tokens, truncate it
        if item_tokens > max_tokens:
            debug(f"Truncating large item: {item.get('Id', 'Unknown')}")
            # Keep essential fields and truncate description/comments
            truncated_item = {
                **get_context(item),
                'Subject': item.get('Subject', '')[:200],
                'Description': item.get('Description', '')[:500],
                'truncated': True
            }
            item_text = json.dumps(truncated_item)
            item_tokens = calculate_tokens(item_text)
            item = truncated_item  # Replace the original item with truncated version
        
        # If adding this item would exceed limit, start new chunk
        if current_tokens + item_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        
        # Add item to current chunk
        current_chunk.append(item)
        current_tokens += item_tokens
    
    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    debug(f"Created {len(chunks)} chunks from {len(data)} items")
    return chunks

def process_chunks_with_retry(chunks, process_func, max_retries=3, backoff_factor=2):
    """
    Process chunks with retry mechanism and detailed progress tracking.
    Returns tuple of (results, processing_stats).
    """
    results = []
    processing_stats = {
        'total_chunks': len(chunks),
        'processed_chunks': 0,
        'successful_chunks': 0,
        'failed_chunks': 0,
        'retry_count': 0,
        'total_tokens': 0,
        'errors': [],
        'processing_times': [],
        'chunk_sizes': []
    }
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    error_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    try:
        for i, chunk_data in enumerate(chunks):
            chunk_size = len(chunk_data)
            processing_stats['chunk_sizes'].append(chunk_size)
            
            retry_count = 0
            success = False
            last_error = None
            chunk_start_time = time.time()
            
            # Update progress message
            status_text.text(f"Processing chunk {i+1}/{len(chunks)} (size: {chunk_size} records)")
            
            while retry_count < max_retries and not success:
                try:
                    if retry_count > 0:
                        # Calculate backoff time
                        wait_time = backoff_factor ** retry_count
                        status_text.text(f"Retrying chunk {i+1} (attempt {retry_count + 1}/{max_retries}) in {wait_time} seconds...")
                        time.sleep(wait_time)
                    
                    # Process the chunk
                    result = process_func(chunk_data)
                    results.append(result)
                    success = True
                    
                    # Update successful stats
                    processing_stats['successful_chunks'] += 1
                    
                except Exception as e:
                    last_error = str(e)
                    retry_count += 1
                    processing_stats['retry_count'] += 1
                    debug(f"Error processing chunk {i+1}: {str(e)}")
                    
                    if retry_count == max_retries:
                        error_msg = f"Failed to process chunk {i+1} after {max_retries} attempts: {last_error}"
                        processing_stats['errors'].append(error_msg)
                        processing_stats['failed_chunks'] += 1
                        error_placeholder.error(error_msg)
            
            # Record processing time
            chunk_time = time.time() - chunk_start_time
            processing_stats['processing_times'].append(chunk_time)
            
            # Update progress
            processing_stats['processed_chunks'] += 1
            progress = (i + 1) / len(chunks)
            progress_bar.progress(progress)
            
            # Update running statistics
            avg_time = sum(processing_stats['processing_times']) / len(processing_stats['processing_times'])
            success_rate = (processing_stats['successful_chunks'] / processing_stats['processed_chunks']) * 100
            
            stats_placeholder.info(f"""
            Processing Statistics:
            - Progress: {progress * 100:.1f}% ({processing_stats['processed_chunks']}/{processing_stats['total_chunks']} chunks)
            - Success Rate: {success_rate:.1f}%
            - Average Processing Time: {avg_time:.1f}s per chunk
            - Total Retries: {processing_stats['retry_count']}
            - Failed Chunks: {processing_stats['failed_chunks']}
            """)
    
    except Exception as e:
        error_msg = f"Critical error during chunk processing: {str(e)}"
        processing_stats['errors'].append(error_msg)
        error_placeholder.error(error_msg)
        debug(f"Error in process_chunks_with_retry: {str(e)}")
        debug(traceback.format_exc())
    
    finally:
        # Calculate final statistics
        if processing_stats['processing_times']:
            processing_stats['avg_processing_time'] = sum(processing_stats['processing_times']) / len(processing_stats['processing_times'])
            processing_stats['max_processing_time'] = max(processing_stats['processing_times'])
            processing_stats['min_processing_time'] = min(processing_stats['processing_times'])
        
        if processing_stats['chunk_sizes']:
            processing_stats['avg_chunk_size'] = sum(processing_stats['chunk_sizes']) / len(processing_stats['chunk_sizes'])
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show final summary
        if processing_stats['failed_chunks'] > 0:
            error_placeholder.error(f"Processing completed with {processing_stats['failed_chunks']} failed chunks")
        else:
            error_placeholder.success("Processing completed successfully")
        
        stats_placeholder.info(f"""
        Final Processing Statistics:
        - Total Chunks Processed: {processing_stats['processed_chunks']}/{processing_stats['total_chunks']}
        - Successful Chunks: {processing_stats['successful_chunks']} ({(processing_stats['successful_chunks']/processing_stats['total_chunks'])*100:.1f}%)
        - Failed Chunks: {processing_stats['failed_chunks']}
        - Total Retries: {processing_stats['retry_count']}
        - Average Processing Time: {processing_stats.get('avg_processing_time', 0):.1f}s per chunk
        - Average Chunk Size: {processing_stats.get('avg_chunk_size', 0):.1f} records
        """)
    
    return results, processing_stats

def validate_analysis_data(data):
    """Validate the analysis data structure and content."""
    try:
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }

        # Check if data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Expected DataFrame, got {type(data)}")
            return validation_results

        # Required columns
        required_columns = ['Subject', 'Description', 'CreatedDate', 'Status']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
            return validation_results

        # Check data types
        try:
            # Ensure CreatedDate is datetime
            if not pd.api.types.is_datetime64_any_dtype(data['CreatedDate']):
                data['CreatedDate'] = pd.to_datetime(data['CreatedDate'], errors='coerce')
            
            # Remove timezone information if present
            if data['CreatedDate'].dt.tz is not None:
                data['CreatedDate'] = data['CreatedDate'].dt.tz_localize(None)
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Error processing dates: {str(e)}")
            return validation_results

        # Check for empty DataFrame
        if len(data) == 0:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Empty DataFrame")
            return validation_results

        # Calculate statistics
        validation_results['stats'] = {
            'total_records': len(data),
            'null_counts': data[required_columns].isnull().sum().to_dict(),
            'status_distribution': data['Status'].value_counts().to_dict(),
            'date_range': {
                'start': data['CreatedDate'].min().strftime('%Y-%m-%d') if not pd.isna(data['CreatedDate'].min()) else None,
                'end': data['CreatedDate'].max().strftime('%Y-%m-%d') if not pd.isna(data['CreatedDate'].max()) else None
            }
        }

        # Add warnings for potential data quality issues
        if data['Subject'].isnull().any():
            validation_results['warnings'].append(f"Found {data['Subject'].isnull().sum()} records with missing Subject")
        if data['Description'].isnull().any():
            validation_results['warnings'].append(f"Found {data['Description'].isnull().sum()} records with missing Description")
        if data['CreatedDate'].isnull().any():
            validation_results['warnings'].append(f"Found {data['CreatedDate'].isnull().sum()} records with missing CreatedDate")

        return validation_results

    except Exception as e:
        return {
            'is_valid': False,
            'errors': [f"Validation error: {str(e)}"],
            'warnings': [],
            'stats': {}
        }

def merge_analysis_results(results):
    """
    Merge results from multiple chunks while handling duplicates and conflicts.
    Returns consolidated analysis and quality metrics.
    """
    merged = {
        'summary': '',
        'trends': [],
        'recommendations': [],
        'csat_analysis': [],
        'quality_metrics': {
            'chunk_count': len(results),
            'consistency_score': 0,
            'coverage_metrics': {}
        }
    }
    
    # Helper function to deduplicate insights while preserving unique information
    def deduplicate_insights(items):
        seen = set()
        unique_items = []
        for item in items:
            item_hash = hash(str(item).lower())
            if item_hash not in seen:
                seen.add(item_hash)
                unique_items.append(item)
        return unique_items
    
    # Combine summaries intelligently
    summaries = [r.get('summary', '') for r in results if r.get('summary')]
    if summaries:
        # Use the most comprehensive summary (usually the longest that's not excessive)
        merged['summary'] = max(summaries, key=len)
    
    # Combine and deduplicate trends and recommendations
    all_trends = [trend for r in results for trend in r.get('trends', [])]
    all_recommendations = [rec for r in results for rec in r.get('recommendations', [])]
    
    merged['trends'] = deduplicate_insights(all_trends)
    merged['recommendations'] = deduplicate_insights(all_recommendations)
    
    # Combine CSAT analysis while preserving temporal information
    all_csat = [csat for r in results for csat in r.get('csat_analysis', [])]
    merged['csat_analysis'] = deduplicate_insights(all_csat)
    
    # Calculate quality metrics
    merged['quality_metrics']['consistency_score'] = len(merged['trends']) / len(all_trends) if all_trends else 1
    merged['quality_metrics']['coverage_metrics'] = {
        'trends_coverage': len(merged['trends']) / len(results) if results else 0,
        'recommendations_coverage': len(merged['recommendations']) / len(results) if results else 0
    }
    
    return merged

def generate_ai_insights(data):
    """Generate AI insights from ticket data using OpenAI."""
    try:
        # Check if OpenAI package is installed
        try:
            from openai import OpenAI
            debug("OpenAI package imported successfully")
        except ImportError:
            st.error("OpenAI package is not installed. Please run 'pip install openai' to enable AI analysis.")
            debug("OpenAI import error - package not installed")
            return None
            
        # Check if OpenAI API key is available
        openai_api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get("OPENAI_API_KEY", None)
        if not openai_api_key:
            st.warning("OpenAI API key not found. Please add it to your environment variables or Streamlit secrets.")
            debug("OpenAI API key not found")
            return None

        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)

        # Ensure data is properly structured
        if not isinstance(data, dict):
            debug(f"Input data is not a dictionary, type: {type(data)}")
            return None

        debug(f"Input data keys: {list(data.keys())}")

        # Extract and normalize cases data
        if 'cases' not in data:
            debug("No cases data found in input")
            return None

        try:
            # Handle both DataFrame and list/dict cases
            if isinstance(data['cases'], pd.DataFrame):
                cases_df = data['cases']
            else:
                cases_df = pd.json_normalize(data['cases'])
            
            debug(f"Cases DataFrame columns: {cases_df.columns.tolist()}")
            
            # Extract required fields, handling nested structures
            required_fields = {
                'Subject': cases_df['Subject'] if 'Subject' in cases_df.columns else cases_df.get('attributes.Subject', ''),
                'Description': cases_df['Description'] if 'Description' in cases_df.columns else cases_df.get('attributes.Description', ''),
                'Status': cases_df['Status'] if 'Status' in cases_df.columns else cases_df.get('attributes.Status', ''),
                'Priority': cases_df.get('Internal_Priority__c', None),
                'Account': cases_df.get('Account.Name', None),
                'Product_Area__c': cases_df.get('Product_Area__c', None),
                'Product_Feature__c': cases_df.get('Product_Feature__c', None)
            }

            # Handle dates separately with timezone normalization
            created_date = cases_df['CreatedDate'] if 'CreatedDate' in cases_df.columns else cases_df.get('attributes.CreatedDate')
            if created_date is not None:
                try:
                    # Convert to datetime and normalize timezone to UTC
                    created_date = pd.to_datetime(created_date).dt.tz_localize(None)
                    required_fields['CreatedDate'] = created_date
                except Exception as e:
                    debug(f"Error processing CreatedDate: {str(e)}")
                    required_fields['CreatedDate'] = pd.NaT

            # Create analysis DataFrame
            analysis_df = pd.DataFrame(required_fields)
            
            # Add comments if available
            if 'comments' in data and data['comments'] is not None:
                comments_df = pd.json_normalize(data['comments'])
                if not comments_df.empty:
                    analysis_df['Comments'] = comments_df.groupby('ParentId')['CommentBody'].apply(list)

            # Fill missing values
            analysis_df['Subject'] = analysis_df['Subject'].fillna('')
            analysis_df['Description'] = analysis_df['Description'].fillna('')
            analysis_df['Status'] = analysis_df['Status'].fillna('Unknown')
            analysis_df['Priority'] = analysis_df['Priority'].fillna('Not Set')
            analysis_df['Product_Area__c'] = analysis_df['Product_Area__c'].fillna('Unspecified')
            analysis_df['Product_Feature__c'] = analysis_df['Product_Feature__c'].fillna('Unspecified')

            # Prepare summary statistics
            summary_stats = {
                'total_tickets': len(analysis_df),
                'status_distribution': analysis_df['Status'].value_counts().to_dict(),
                'priority_distribution': analysis_df['Priority'].value_counts().to_dict(),
                'product_areas': analysis_df['Product_Area__c'].value_counts().to_dict(),
                'avg_description_length': int(analysis_df['Description'].str.len().mean()),
                'date_range': {
                    'start': analysis_df['CreatedDate'].min().strftime('%Y-%m-%d'),
                    'end': analysis_df['CreatedDate'].max().strftime('%Y-%m-%d')
                }
            }

            # Prepare the prompt
            prompt = f"""
            Analyze the following support ticket data and provide insights:

            Summary Statistics:
            {json.dumps(summary_stats, indent=2)}

            Please provide a comprehensive analysis with:
            1. A summary of key insights and trends
            2. Notable patterns in ticket distribution and customer issues
            3. Actionable recommendations for improving customer support

            Format your response as a JSON object with these keys:
            - summary: Overall analysis and key findings
            - patterns: List of identified patterns and trends
            - recommendations: List of specific, actionable recommendations

            Focus on practical insights that can help improve customer support operations.
            """

            # Call OpenAI API
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert support ticket analyst. Analyze the provided data and extract meaningful insights."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                
                # Extract and parse the response
                ai_response = response.choices[0].message.content
                insights = json.loads(ai_response)
                
                return insights

            except Exception as e:
                st.error(f"Error calling OpenAI API: {str(e)}")
                debug(f"OpenAI API error: {str(e)}")
                return None

        except Exception as e:
            st.error(f"Error preparing data for analysis: {str(e)}")
            debug(f"Data preparation error: {str(e)}")
            return None

    except Exception as e:
        st.error(f"Error in generate_ai_insights: {str(e)}")
        debug(f"General error in generate_ai_insights: {str(e)}")
        return None

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
        
        # Handle datetime columns - convert to strings for Excel compatibility
        datetime_cols = export_df.select_dtypes(include=['datetime']).columns.tolist()
        if datetime_cols:
            debug(f"Converting datetime columns for export: {datetime_cols}")
            for col in datetime_cols:
                export_df[col] = export_df[col].astype(str)
        
        if format == "Excel":
            output = BytesIO()
            try:
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Summary sheet - CSAT removed from summary
                    summary_data = {
                        'Customer': [],
                        'Total Tickets': [],
                        'Avg Response Time (hrs)': [],
                        'Avg Resolution Time (days)': [],
                        'Avg CSAT (0-5)': []  # Add CSAT back to summary
                    }
                    
                    for customer in customers:
                        customer_df = export_df[export_df['Account_Name'] == customer]
                        summary_data['Customer'].append(customer)
                        summary_data['Total Tickets'].append(len(customer_df))
                        
                        # Response Time calculation using string dates
                        if 'First_Response_Time__c' in customer_df.columns and 'CreatedDate' in customer_df.columns:
                            try:
                                # Convert back to datetime for calculation
                                response_times = pd.to_datetime(customer_df['First_Response_Time__c']) - pd.to_datetime(customer_df['CreatedDate'])
                                # Calculate mean in hours
                                resp_time = response_times.dt.total_seconds().mean() / 3600
                                summary_data['Avg Response Time (hrs)'].append(round(resp_time, 2) if pd.notna(resp_time) else 'N/A')
                            except Exception as e:
                                debug(f"Error calculating response time: {str(e)}")
                                summary_data['Avg Response Time (hrs)'].append('N/A')
                        else:
                            summary_data['Avg Response Time (hrs)'].append('N/A')
                        
                        # Resolution Time calculation using string dates
                        if 'ClosedDate' in customer_df.columns and 'CreatedDate' in customer_df.columns:
                            try:
                                # Convert back to datetime for calculation
                                resolution_times = pd.to_datetime(customer_df['ClosedDate']) - pd.to_datetime(customer_df['CreatedDate'])
                                # Calculate mean in days
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
                                # Convert CSAT to numeric
                                csat_numeric = pd.to_numeric(customer_df['CSAT__c'], errors='coerce')
                                # Only include non-null values in the valid range
                                valid_csat = csat_numeric[(csat_numeric >= 0) & (csat_numeric <= 5) & csat_numeric.notna()]
                                if not valid_csat.empty:
                                    # Calculate average CSAT using only valid values
                                    avg_csat = valid_csat.mean()
                                    summary_data['Avg CSAT (0-5)'].append(round(avg_csat, 2) if pd.notna(avg_csat) else 'N/A')
                                    debug(f"Export CSAT for {customer}: {len(valid_csat)} valid values, average: {avg_csat:.2f}")
                                else:
                                    summary_data['Avg CSAT (0-5)'].append('N/A')
                                    debug(f"Export CSAT for {customer}: No valid CSAT values found")
                            except Exception as e:
                                debug(f"Error calculating CSAT for {customer}: {str(e)}")
                                summary_data['Avg CSAT (0-5)'].append('N/A')
                        else:
                            summary_data['Avg CSAT (0-5)'].append('N/A')
                    
                    # Write summary sheet
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Write detailed data sheets
                    for customer in customers:
                        customer_df = export_df[export_df['Account_Name'] == customer]
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
                                          export_df['IsEscalated'].mean() * 100)
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

def truncate_string(s, max_length=30):
    """Truncate a string to specified length and add ellipsis if needed."""
    return s[:max_length] + '...' if len(s) > max_length else s

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
                IMPL_Phase__c
            FROM Case
            WHERE Account.Name = '{customer}'
            AND CreatedDate >= {start_date.strftime('%Y-%m-%d')}T00:00:00Z
            AND CreatedDate <= {end_date.strftime('%Y-%m-%d')}T23:59:59Z
        """
        
        cases = execute_soql_query(st.session_state.sf_connection, query)
        if not cases:
            st.warning(f"No cases found for {customer} in the selected date range.")
            return None
        
        cases_df = pd.DataFrame(cases)
        
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

def display_detailed_analysis(data, enable_ai_analysis, skip_impl_phase_analysis=True):
    """Display detailed analysis of the selected customer's tickets."""
    try:
        # Check if data is None
        if data is None:
            st.warning("No data available for analysis.")
            return
            
        # Handle whether data is a dictionary or DataFrame
        if isinstance(data, dict):
            # Extract the cases DataFrame from the dictionary
            cases_df = data.get('cases')
            if cases_df is None or cases_df.empty:
                st.warning("No ticket data available for analysis.")
                return
                
            # Get references to other data elements
            comments_df = data.get('comments', pd.DataFrame())
            history_df = data.get('history', pd.DataFrame())
            emails_df = data.get('emails', pd.DataFrame())
        else:
            # Data is already a DataFrame (direct tickets)
            cases_df = data
            if cases_df.empty:
                st.warning("No data available for analysis.")
                return
            # Set others to empty DataFrames since they're not provided
            comments_df = pd.DataFrame()
            history_df = pd.DataFrame()
            emails_df = pd.DataFrame()

        # Safety - Create a defensive copy and ensure all NA values are handled
        cases_df = cases_df.copy()
        
        # Pre-emptively handle NA values in each column to avoid boolean ambiguity issues
        for col in cases_df.columns:
            # For object (string) columns, fill NAs with empty string
            if cases_df[col].dtype == 'object' or pd.api.types.is_string_dtype(cases_df[col]):
                cases_df[col] = cases_df[col].fillna('')
            # For numeric columns, fill NAs with 0
            elif pd.api.types.is_numeric_dtype(cases_df[col]):
                cases_df[col] = cases_df[col].fillna(0)
            # For boolean columns, fill NAs with False
            elif pd.api.types.is_bool_dtype(cases_df[col]):
                cases_df[col] = cases_df[col].fillna(False)
        
        # Set up progress tracking
        progress_bar = st.progress(0)
        st.write("## Support Ticket Analysis")
        
        # Display key metrics and statistics
        col1, col2, col3, col4 = st.columns(4)
        
        # 1. Total tickets
        with col1:
            st.metric("Total Tickets", len(cases_df))
            
        # 2. Open tickets
        open_tickets = cases_df['Status'].isin(['New', 'Open', 'In Progress', 'Reopened']).sum()
        with col2:
            st.metric("Open Tickets", open_tickets)
            
        # 3. Closed tickets
        closed_tickets = cases_df['Status'].isin(['Closed', 'Solved', 'Resolved']).sum()
        with col3:
            st.metric("Closed Tickets", closed_tickets)
            
        # 4. Escalated tickets
        escalated_tickets = cases_df['IsEscalated'].sum()
        with col4:
            st.metric("Escalated Tickets", escalated_tickets)
            
        progress_bar.progress(20)
        
        # Ticket breakdown by Product Area
        st.write("### Ticket Distribution by Product Area")
        
        # Use enhanced field if available
        area_field = 'Product_Area__c_enhanced' if 'Product_Area__c_enhanced' in cases_df.columns else 'Product_Area__c'
        
        # Distribution of tickets by product area
        if area_field in cases_df.columns:
            # Make sure we handle empty values
            cases_df[area_field] = cases_df[area_field].fillna('Unspecified')
            # Create a countplot
            fig, ax = plt.subplots(figsize=(10, 6))
            area_counts = cases_df[area_field].value_counts()
            sns.barplot(x=area_counts.index, y=area_counts.values)
            plt.title('Tickets by Product Area')
            plt.xlabel('Product Area')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        progress_bar.progress(40)
        
        # Ticket status distribution
        st.write("### Ticket Status Distribution")
        if 'Status' in cases_df.columns:
            # Make sure we handle empty values
            cases_df['Status'] = cases_df['Status'].fillna('Unknown')
            # Create a countplot
            fig, ax = plt.subplots(figsize=(10, 6))
            status_counts = cases_df['Status'].value_counts()
            sns.barplot(x=status_counts.index, y=status_counts.values)
            plt.title('Tickets by Status')
            plt.xlabel('Status')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        progress_bar.progress(60)
        
        # Skip Implementation Phase Analysis if requested
        if not skip_impl_phase_analysis and 'Implementation_Phase__c' in cases_df.columns:
            st.write("### Implementation Phase Analysis")
            
            # Make sure we handle empty values
            cases_df['Implementation_Phase__c'] = cases_df['Implementation_Phase__c'].fillna('Not Specified')
            
            # Create a countplot
            fig, ax = plt.subplots(figsize=(10, 6))
            impl_counts = cases_df['Implementation_Phase__c'].value_counts()
            sns.barplot(x=impl_counts.index, y=impl_counts.values)
            plt.title('Tickets by Implementation Phase')
            plt.xlabel('Implementation Phase')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Ticket status by implementation phase
            ct = pd.crosstab(cases_df['Implementation_Phase__c'], cases_df['Status'])
            fig, ax = plt.subplots(figsize=(12, 8))
            ct.plot(kind='bar', stacked=True, ax=ax)
            plt.title('Status by Implementation Phase')
            plt.xlabel('Implementation Phase')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Status')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        progress_bar.progress(80)
        
        # Text and Sentiment Analysis Section
        if not cases_df.empty:
            st.write("### Text and Sentiment Analysis")
            
            # Define additional stopwords for technical terms
            additional_stopwords = {
                'eightfold', 'mailto', 'chrome', 'firefox', 'safari', 'edge', 'opera',
                'webkit', 'mozilla', 'browser', 'agent', 'http', 'https', 'www',
                'com', 'net', 'org', 'html', 'htm', 'php', 'asp', 'aspx',
                'user-agent', 'useragent', 'version', 'windows', 'macintosh', 'linux',
                'unix', 'android', 'ios', 'mobile', 'desktop', 'platform',
                'application', 'software', 'browser-agent', 'browseragent'
            }
            
            # Combine text data from different sources
            text_data = []
            
            # Clean and add text data
            def clean_text(text):
                if not isinstance(text, str):
                    return ''
                # Remove URLs
                text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
                # Remove email addresses
                text = re.sub(r'[\w\.-]+@[\w\.-]+', '', text)
                # Remove browser agent strings
                text = re.sub(r'Mozilla/[\d\.]+ \(.*?\)', '', text)
                # Remove special characters and extra whitespace
                text = re.sub(r'[^\w\s]', ' ', text)
                text = ' '.join(text.split())
                return text.lower()
            
            text_data.extend(cases_df['Subject'].apply(clean_text))
            text_data.extend(cases_df['Description'].apply(clean_text))
            
            if not comments_df.empty:
                text_data.extend(comments_df['CommentBody'].apply(clean_text))
                
            if not emails_df.empty:
                text_data.extend(emails_df['TextBody'].apply(clean_text))
                
            combined_text = ' '.join(text_data)
            
            # Generate WordCloud with enhanced stopwords
            if combined_text.strip():
                st.write("#### Word Cloud Analysis")
                wordcloud_fig = generate_wordcloud(combined_text, 'Common Terms in All Communications', additional_stopwords)
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig)
                    plt.close()
            
            # Sentiment Analysis with Enhanced Visualization
            st.write("#### Sentiment Analysis")
            st.write("""
            The sentiment analysis uses TextBlob's algorithm, which:
            1. Performs part-of-speech tagging
            2. Extracts sentiment-bearing phrases
            3. Assigns polarity scores (-1 to +1) based on:
               - Word sentiment from built-in lexicon
               - Contextual modifiers (e.g., negations)
               - Intensifiers and diminishers
            4. Aggregates scores at phrase and text level
            
            Interpretation:
            - 🟢 Positive (0.2 to 1.0): Indicates satisfaction, praise, or positive experiences
            - 🟡 Neutral (-0.2 to 0.2): Factual or balanced content
            - 🔴 Negative (-1.0 to -0.2): Indicates dissatisfaction, complaints, or issues
            """)
            
            # Initialize TextBlob for sentiment analysis
            from textblob import TextBlob
            
            # Enhanced sentiment function with subjectivity
            def get_sentiment(text):
                if not isinstance(text, str) or not text.strip():
                    return {'polarity': 0, 'subjectivity': 0}
                analysis = TextBlob(text).sentiment
                return {
                    'polarity': analysis.polarity,
                    'subjectivity': analysis.subjectivity
                }
            
            # Calculate sentiment for each text source
            cases_df['subject_sentiment'] = cases_df['Subject'].apply(lambda x: get_sentiment(x)['polarity'])
            cases_df['description_sentiment'] = cases_df['Description'].apply(lambda x: get_sentiment(x)['polarity'])
            
            if not comments_df.empty:
                comments_df['sentiment'] = comments_df['CommentBody'].apply(lambda x: get_sentiment(x)['polarity'])
                
            if not emails_df.empty:
                emails_df['sentiment'] = emails_df['TextBody'].apply(lambda x: get_sentiment(x)['polarity'])
            
            # Calculate average sentiment per case
            case_sentiments = pd.DataFrame()
            case_sentiments['subject_sentiment'] = cases_df['subject_sentiment']
            case_sentiments['description_sentiment'] = cases_df['description_sentiment']
            
            if not comments_df.empty:
                comment_sentiments = comments_df.groupby('ParentId')['sentiment'].mean()
                case_sentiments = case_sentiments.join(comment_sentiments.rename('comments_sentiment'), how='left')
            
            if not emails_df.empty:
                email_sentiments = emails_df.groupby('ParentId')['sentiment'].mean()
                case_sentiments = case_sentiments.join(email_sentiments.rename('email_sentiment'), how='left')
            
            # Calculate overall sentiment
            case_sentiments['overall_sentiment'] = case_sentiments.mean(axis=1)
            
            # Create enhanced sentiment distribution plot with emojis
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create histogram with custom colors
            n, bins, patches = plt.hist(case_sentiments['overall_sentiment'], bins=20, 
                                      density=True, alpha=0.7)
            
            # Color the bars based on sentiment
            for i in range(len(patches)):
                if bins[i] < -0.2:
                    patches[i].set_facecolor('#ff6b6b')  # Red for negative
                elif bins[i] > 0.2:
                    patches[i].set_facecolor('#4ecdc4')  # Green for positive
                else:
                    patches[i].set_facecolor('#ffd93d')  # Yellow for neutral
            
            # Add emoji annotations
            plt.annotate('😊', xy=(0.5, plt.ylim()[1]), fontsize=20)
            plt.annotate('😐', xy=(0, plt.ylim()[1]), fontsize=20)
            plt.annotate('😠', xy=(-0.5, plt.ylim()[1]), fontsize=20)
            
            plt.title('Distribution of Overall Sentiment')
            plt.xlabel('Sentiment Score (-1: Very Negative, 1: Very Positive)')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            # If we have enough tickets, show sentiment trend over time
            if len(cases_df) >= 20:
                st.write("#### Sentiment Trend Over Time")
                
                # Add timestamps and sort
                sentiment_trend = pd.DataFrame({
                    'timestamp': cases_df['CreatedDate'],
                    'sentiment': case_sentiments['overall_sentiment']
                })
                sentiment_trend = sentiment_trend.sort_values('timestamp')
                
                # Create monthly averages
                monthly_sentiment = sentiment_trend.set_index('timestamp').resample('ME')['sentiment'].mean()
                
                # Create enhanced trend plot with emojis
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot line with gradient color based on sentiment
                for i in range(len(monthly_sentiment)-1):
                    sentiment_value = monthly_sentiment.iloc[i]
                    color = '#4ecdc4' if sentiment_value > 0.2 else '#ff6b6b' if sentiment_value < -0.2 else '#ffd93d'
                    plt.plot([monthly_sentiment.index[i], monthly_sentiment.index[i+1]], 
                            [monthly_sentiment.iloc[i], monthly_sentiment.iloc[i+1]], 
                            color=color, linewidth=2)
                
                # Add points with emojis
                for i, (date, sentiment) in enumerate(monthly_sentiment.items()):
                    if sentiment > 0.2:
                        emoji = '😊'
                    elif sentiment < -0.2:
                        emoji = '😠'
                    else:
                        emoji = '😐'
                    plt.annotate(emoji, (date, sentiment), fontsize=12)
                
                plt.title('Average Sentiment Trend by Month')
                plt.xlabel('Month')
                plt.ylabel('Average Sentiment')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # If CSAT scores are available, show correlation
                if 'CSAT__c' in cases_df.columns:
                    st.write("#### Sentiment vs CSAT Correlation")
                    
                    # Prepare data for correlation
                    correlation_data = pd.DataFrame({
                        'Sentiment': case_sentiments['overall_sentiment'],
                        'CSAT': pd.to_numeric(cases_df['CSAT__c'], errors='coerce')
                    })
                    
                    # Filter for valid CSAT values (0-5 range) and non-null values
                    correlation_data = correlation_data[
                        (correlation_data['CSAT'].notna()) & 
                        (correlation_data['CSAT'] >= 0) & 
                        (correlation_data['CSAT'] <= 5) &
                        (correlation_data['Sentiment'].notna())
                    ]
                    
                    if not correlation_data.empty:
                        # Calculate correlation
                        correlation = correlation_data['Sentiment'].corr(correlation_data['CSAT'])
                        
                        # Create enhanced scatter plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Plot points with custom colors
                        for sentiment, csat in zip(correlation_data['Sentiment'], correlation_data['CSAT']):
                            color = '#4ecdc4' if sentiment > 0.2 else '#ff6b6b' if sentiment < -0.2 else '#ffd93d'
                            plt.scatter(sentiment, csat, c=color, alpha=0.6)
                        
                        # Add trend line
                        z = np.polyfit(correlation_data['Sentiment'], correlation_data['CSAT'], 1)
                        p = np.poly1d(z)
                        plt.plot(correlation_data['Sentiment'], p(correlation_data['Sentiment']), 
                                "r--", alpha=0.8)
                        
                        plt.title(f'Sentiment vs CSAT (Correlation: {correlation:.2f})')
                        plt.xlabel('Sentiment Score')
                        plt.ylabel('CSAT Score')
                        plt.grid(True, alpha=0.3)
                        
                        # Add emoji annotations
                        plt.annotate('😊', xy=(max(correlation_data['Sentiment']), plt.ylim()[1]), fontsize=20)
                        plt.annotate('😠', xy=(min(correlation_data['Sentiment']), plt.ylim()[1]), fontsize=20)
                        
                        st.pyplot(fig)
                        plt.close()
                        
                        # Add correlation interpretation with emojis
                        if abs(correlation) > 0.5:
                            st.write("🎯 Strong correlation between sentiment and CSAT scores")
                        elif abs(correlation) > 0.3:
                            st.write("🔄 Moderate correlation between sentiment and CSAT scores")
                        else:
                            st.write("↔️ Weak correlation between sentiment and CSAT scores")
                            
                        # Add detailed interpretation
                        st.write(f"""
                        Correlation Analysis:
                        - Correlation Coefficient: {correlation:.2f}
                        - Direction: {'Positive' if correlation > 0 else 'Negative'}
                        - Strength: {'Strong' if abs(correlation) > 0.5 else 'Moderate' if abs(correlation) > 0.3 else 'Weak'}
                        
                        This means that {'higher sentiment scores tend to correspond with higher CSAT scores' if correlation > 0 else 'there is no strong relationship between sentiment and CSAT scores'}.
                        """)
                    else:
                        st.info("No valid CSAT scores found for correlation analysis. Please ensure CSAT scores are within the valid range (0-5).")
        
        # Complete progress bar and remove it
        progress_bar.progress(90)

        # AI Analysis Section
        if enable_ai_analysis:
            st.markdown("---")
            st.header("🤖 AI-Powered Insights")
            
            with st.spinner("Generating AI insights from ticket data..."):
                # Process data for AI analysis
                ai_data = {
                    'cases': cases_df,
                    'comments': comments_df,
                    'history': history_df,
                    'emails': emails_df
                }
                
                ai_insights = generate_ai_insights(ai_data)
                
                if ai_insights is not None:
                    st.success("AI analysis completed successfully!")
                    
                    # Display insights in an organized manner
                    st.subheader("📊 Key Insights")
                    if isinstance(ai_insights, pd.DataFrame):
                        # If the result is a DataFrame, display relevant statistics
                        st.write("Ticket Analysis Summary:")
                        st.dataframe(ai_insights.describe())
                    elif isinstance(ai_insights, dict):
                        # If the result is a dictionary, display structured insights
                        for key, value in ai_insights.items():
                            if key == 'summary':
                                st.write("### Summary")
                                st.write(value)
                            elif key == 'patterns':
                                st.write("### Identified Patterns")
                                for pattern in value:
                                    st.write(f"- {pattern}")
                            elif key == 'recommendations':
                                st.write("### Recommendations")
                                for rec in value:
                                    st.write(f"- {rec}")
                    else:
                        st.write(ai_insights)
                else:
                    st.warning("Unable to generate AI insights. This could be due to missing OpenAI API key or insufficient data.")
                    st.info("To enable AI insights, please ensure you have:")
                    st.markdown("""
                    1. Set up your OpenAI API key in your environment variables or Streamlit secrets
                    2. Have sufficient ticket data for analysis
                    3. Properly configured the AI analysis settings
                    """)

        # After sentiment analysis section but before final progress bar
        if not cases_df.empty and enable_ai_analysis:
            st.write("### AI-Powered Insights")
            with st.spinner("Generating AI insights from ticket data..."):
                debug("Generating AI insights")
                
                # Create a combined data dictionary with cases_df as the main dataframe
                analysis_data = {
                    'cases': cases_df,
                    'comments': comments_df,
                    'history': history_df,
                    'emails': emails_df
                }
                
                insights = generate_ai_insights(analysis_data)
                
                if insights is not None:
                    # Create tabs for different insight categories
                    insights_tab, patterns_tab, recommendations_tab = st.tabs(["Summary", "Patterns", "Recommendations"])
                    
                    with insights_tab:
                        st.markdown(insights.get('summary', 'No summary insights available.'))
                        
                    with patterns_tab:
                        patterns = insights.get('patterns', [])
                        if patterns:
                            for i, pattern in enumerate(patterns):
                                st.markdown(f"**Pattern {i+1}:** {pattern}")
                        else:
                            st.write("No patterns identified.")
                            
                    with recommendations_tab:
                        recommendations = insights.get('recommendations', [])
                        if recommendations:
                            for i, rec in enumerate(recommendations):
                                st.markdown(f"**Recommendation {i+1}:** {rec}")
                        else:
                            st.write("No recommendations available.")
                    
                    # Add download button for the full HTML report
                    if 'html' in insights:
                        st.download_button(
                            label="Download AI Analysis Report",
                            data=insights['html'],
                            file_name="ai_support_ticket_analysis.html",
                            mime="text/html"
                        )
                else:
                    # Fallback with sample insights when generation fails
                    st.info("AI analysis is enabled but no insights were generated. Here's a sample of what insights might look like:")
                    
                    # Sample data
                    st.markdown("""
                    #### Sample Insights Summary
                    
                    The support ticket analysis reveals several key trends and areas for attention. Most tickets are related to 
                    authentication issues and API integration problems. Response times are generally within SLA, but escalation 
                    rates are higher for enterprise customers. A significant portion of tickets are reopened after closure, 
                    suggesting potential issues with solution quality or completeness.
                    
                    #### Sample Identified Patterns
                    
                    * Authentication-related issues spike after system updates
                    * Enterprise customers experience more integration challenges
                    * Documentation questions frequently lead to feature requests
                    
                    #### Sample Recommendations
                    
                    * Expand authentication troubleshooting documentation
                    * Create dedicated integration solutions for enterprise customers
                    * Develop proactive notifications for system updates
                    """)
        elif enable_ai_analysis:
            st.info("No data available for AI analysis.")
            debug("AI analysis enabled but no data available")
        
        # Complete progress bar and remove it
        progress_bar.progress(100)
        time.sleep(0.5)  # Slight delay to show completion
        progress_bar.empty()
            
    except Exception as e:
        st.error(f"Error displaying detailed analysis: {str(e)}")
        debug(f"Error in display_detailed_analysis: {str(e)}")
        import traceback
        debug(traceback.format_exc())

if __name__ == "__main__":
    main() 
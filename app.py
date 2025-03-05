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
                    df = fetch_data(selected_customers, start_date, end_date)
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
            
            detailed_analysis = st.sidebar.checkbox("Enable Detailed Ticket Analysis", 
                                                   help="Fetch additional data and perform detailed analysis for the selected customer")
            
            if detailed_analysis:
                ai_analysis = st.sidebar.checkbox("Enable AI-powered Analysis", 
                                                 help="Use OpenAI to analyze ticket patterns and provide insights")
                
                if ai_analysis:
                    st.sidebar.info("AI analysis will process ticket data to identify patterns and provide recommendations.")
    
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
                if len(selected_customers) == 1 and detailed_analysis:
                    with st.spinner("Fetching detailed ticket information..."):
                        detailed_data = fetch_detailed_data(selected_customers[0], start_date, end_date)
                        if detailed_data is not None:
                            # Add debug output to help troubleshoot
                            debug("AI analysis enabled", ai_analysis)
                            debug("Detailed data structure", {k: type(v) for k, v in detailed_data.items()})
                            
                            # Make sure ai_analysis is properly passed
                            display_detailed_analysis(detailed_data, ai_analysis)
            else:
                st.warning("No data available after applying filters.")
        except Exception as e:
            st.error(f"Error displaying visualizations: {str(e)}")
            st.error("Please try refreshing the page or selecting different criteria.")

def fetch_data(customers, start_date, end_date):
    """Fetch data from Salesforce based on selected criteria."""
    try:
        customer_list = "'" + "','".join(customers) + "'"
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
        
        # Debug logging
        debug("SOQL Query", query)
        
        records = execute_soql_query(st.session_state.sf_connection, query)
        if not records:
            st.warning("No data found for the selected criteria.")
            return None
        
        df = pd.DataFrame(records)
        
        # Extract Account Name from nested structure
        if 'Account' in df.columns and isinstance(df['Account'].iloc[0], dict):
            df['Account_Name'] = df['Account'].apply(lambda x: x.get('Name') if isinstance(x, dict) else None)
            df = df.drop('Account', axis=1)
        
        # Debug logging
        debug("DataFrame columns", df.columns.tolist())
        debug("First few rows", df.head())
        debug("Number of records", len(df))
        
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
        
        # Convert date columns
        date_columns = ['CreatedDate', 'ClosedDate', 'First_Response_Time__c']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.error("Debug - Failed query:", query)
        return None

def display_visualizations(df, customers):
    """Display all visualizations according to PRD requirements."""
    try:
        if df is None or df.empty:
            st.warning("No data available for visualization.")
            return
        
        # Debug logging for data validation
        debug("Visualization data shape", df.shape)
        debug("Available columns", df.columns.tolist())
        
        # Create filtered dataframe excluding unspecified data
        total_records = len(df)
        df_filtered = df[
            (df['Product_Area__c'] != 'Unspecified') &
            (df['Product_Feature__c'] != 'Unspecified') &
            (df['RCA__c'] != 'Not Specified')
        ].copy()
        
        excluded_records = total_records - len(df_filtered)
        if excluded_records > 0:
            st.warning(
                f"ℹ️ {excluded_records} records ({(excluded_records/total_records)*100:.1f}%) "
                "are being excluded from visualizations due to unspecified values in Product Area, "
                "Product Feature, or Root Cause fields."
            )
        
        # Overview Statistics
        st.header("Overview Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Cases", len(df_filtered))
        with col2:
            st.metric("Active Accounts", len(customers))
        with col3:
            st.metric("Product Areas", df_filtered['Product_Area__c'].nunique())
        with col4:
            avg_csat = df_filtered[df_filtered['CSAT__c'].notna()]['CSAT__c'].mean()
            st.metric("Avg CSAT", f"{avg_csat:.2f}" if pd.notna(avg_csat) else "N/A")
        with col5:
            escalation_rate = (df_filtered['IsEscalated'].mean() * 100)
            st.metric("Escalation Rate", f"{escalation_rate:.1f}%" if pd.notna(escalation_rate) else "N/A")
        
        # 1. Ticket Volume Analysis
        st.header("Ticket Volume Analysis")
        for customer in customers:
            try:
                customer_df = df_filtered[df_filtered['Account_Name'] == customer].copy()
                if customer_df.empty:
                    st.warning(f"No data available for customer: {customer}")
                    continue
                
                # Created tickets
                customer_df['Month'] = pd.to_datetime(customer_df['CreatedDate']).dt.to_period('M')
                created_monthly = customer_df.groupby('Month').size().reset_index(name='Created')
                
                # Closed tickets
                closed_df = customer_df[customer_df['ClosedDate'].notna()].copy()
                if not closed_df.empty:
                    closed_df['Month'] = pd.to_datetime(closed_df['ClosedDate']).dt.to_period('M')
                    closed_monthly = closed_df.groupby('Month').size().reset_index(name='Closed')
                    monthly_data = pd.merge(created_monthly, closed_monthly, on='Month', how='outer').fillna(0)
                else:
                    monthly_data = created_monthly
                    monthly_data['Closed'] = 0
                
                monthly_data['Month'] = monthly_data['Month'].astype(str)
                
                # Create Seaborn plot
                plt.figure(figsize=(12, 6))
                monthly_data_melted = pd.melt(monthly_data, id_vars=['Month'], value_vars=['Created', 'Closed'])
                sns.barplot(data=monthly_data_melted, x='Month', y='value', hue='variable', palette=VOLUME_PALETTE)
                plt.title(f"{customer} - Ticket Volume Trends")
                plt.xlabel("Month")
                plt.ylabel("Number of Tickets")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(plt)
                plt.close()
                
            except Exception as e:
                st.error(f"Error displaying ticket volume analysis for {customer}: {str(e)}")
                continue
        
        # 2. Response Time Analysis
        st.header("Response Time Analysis")
        response_df = df_filtered[df_filtered['First_Response_Time__c'].notna()].copy()
        response_df['Response_Time_Hours'] = (
            response_df['First_Response_Time__c'] - response_df['CreatedDate']
        ).dt.total_seconds() / 3600
        
        # Count records before filtering
        total_records = len(response_df)
        
        # Filter out negative response times but include zero
        response_df = response_df[response_df['Response_Time_Hours'] >= 0]
        
        # Calculate and display filtered records
        filtered_records = total_records - len(response_df)
        if filtered_records > 0:
            st.warning(f"⚠️ {filtered_records} records ({(filtered_records/total_records)*100:.1f}%) were excluded due to negative response times, which likely indicates data anomalies.")
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=response_df, x='Account_Name', y='Response_Time_Hours', 
                   hue='Internal_Priority__c', palette=PRIORITY_PALETTE)
        plt.title('Time to First Response by Priority')
        plt.xlabel('Customer')
        plt.ylabel('Response Time (Hours)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
        
        # 3. CSAT Analysis
        st.header("Customer Satisfaction Analysis")
        csat_df = df_filtered[df_filtered['CSAT__c'].notna()].copy()
        
        if len(csat_df) == 0:
            st.info("ℹ️ No customer satisfaction scores are available for the selected time period. This could be because customers haven't provided CSAT responses yet.")
        else:
            # Sort by date to ensure chronological order
            csat_df['Month'] = csat_df['CreatedDate'].dt.to_period('M')
            csat_df = csat_df.sort_values('Month')  # Sort by date
            csat_df['Month'] = csat_df['Month'].astype(str)  # Convert to string after sorting
            
            plt.figure(figsize=(12, 6))
            g = sns.lineplot(data=csat_df, x='Month', y='CSAT__c', 
                            hue='Account_Name', marker='o', palette="Set2")
            plt.title('CSAT Scores Trend')
            plt.xlabel('Month')
            plt.ylabel('CSAT Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
            
            # Display summary statistics
            st.subheader("CSAT Summary Statistics")
            csat_summary = csat_df.groupby('Account_Name')['CSAT__c'].agg(['count', 'mean', 'min', 'max']).round(2)
            csat_summary.columns = ['Number of Responses', 'Average CSAT', 'Minimum CSAT', 'Maximum CSAT']
            st.dataframe(csat_summary)
        
        # 4. Resolution Time Analysis
        st.header("Resolution Time Analysis")
        resolved_df = df_filtered[df_filtered['ClosedDate'].notna()].copy()
        # Filter out records where priority is not set
        resolved_df = resolved_df[resolved_df['Internal_Priority__c'] != 'Not Set']
        resolved_df['Resolution_Time_Days'] = (
            resolved_df['ClosedDate'] - resolved_df['CreatedDate']
        ).dt.total_seconds() / (24 * 3600)
        resolved_df['Month'] = resolved_df['CreatedDate'].dt.to_period('M').astype(str)
        
        # Create FacetGrid for resolution time by priority analysis
        g = sns.FacetGrid(resolved_df, col='Account_Name', col_wrap=2, height=4, aspect=1.5)
        g.map_dataframe(sns.boxplot, x='Internal_Priority__c', y='Resolution_Time_Days', 
                       hue='Internal_Priority__c', palette=PRIORITY_PALETTE, legend=False)
        g.fig.suptitle('Resolution Time by Priority', y=1.02)
        plt.tight_layout()
        st.pyplot(g.fig)
        plt.close()
        
        # Add Resolution Time by Product Area visualization
        st.subheader("Resolution Time by Product Area")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=resolved_df, x='Product_Area__c', y='Resolution_Time_Days',
                   hue='Product_Area__c', palette=BLUES_PALETTE, legend=False)
        plt.title('Resolution Time Distribution by Product Area')
        plt.xlabel('Product Area')
        plt.ylabel('Resolution Time (Days)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
        
        # 5. Product Distribution Analysis
        st.header("Product Distribution Analysis")
        
        # Get top 15 product areas by ticket volume
        top_15_areas = df_filtered['Product_Area__c'].value_counts().head(15).index
        df_top_areas = df_filtered[df_filtered['Product_Area__c'].isin(top_15_areas)].copy()
        
        # Show percentage of data represented by top 15 areas
        total_area_cases = len(df_filtered)
        top_15_area_cases = len(df_top_areas)
        area_coverage_percent = (top_15_area_cases / total_area_cases) * 100
        st.info(f"📊 Showing top 15 product areas, representing {area_coverage_percent:.1f}% of all cases.")
        
        # Create cross-tabulation for heatmap with top 15 areas
        df_top_areas['Product_Feature_Truncated'] = df_top_areas['Product_Feature__c'].apply(lambda x: truncate_string(x, 30))
        product_heatmap = pd.crosstab(df_top_areas['Product_Area__c'], df_top_areas['Product_Feature_Truncated'])
        
        plt.figure(figsize=(15, 10))  # Increased height for better readability
        sns.heatmap(product_heatmap, annot=True, fmt='d', cmap=HEATMAP_PALETTE, 
                   cbar_kws={'label': 'Ticket Count'})
        plt.title('Top 15 Product Areas vs Feature Distribution')
        plt.xlabel('Product Feature')
        plt.ylabel('Product Area')
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
        
        # 6. Text Analysis
        st.header("Text Analysis")
        for customer in customers:
            customer_df = df_filtered[df_filtered['Account_Name'] == customer]
            st.subheader(f"{customer} - Text Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                subject_cloud = generate_wordcloud(
                    customer_df['Subject'].str.cat(sep=' '),
                    'Common Terms in Subjects'
                )
                if subject_cloud:
                    st.pyplot(subject_cloud)
            
            with col2:
                desc_cloud = generate_wordcloud(
                    customer_df['Description'].str.cat(sep=' '),
                    'Common Terms in Descriptions'
                )
                if desc_cloud:
                    st.pyplot(desc_cloud)
        
        # 7. Root Cause Analysis
        if 'RCA__c' in df_filtered.columns:
            st.header("Root Cause Analysis")
            
            # Get root cause counts and select top 15
            root_cause_counts = df_filtered['RCA__c'].value_counts()
            top_15_causes = root_cause_counts.head(15).index.tolist()
            
            # Filter dataframe to include only top 15 root causes
            df_top_causes = df_filtered[df_filtered['RCA__c'].isin(top_15_causes)].copy()
            
            # Show percentage of data represented by top 15 causes
            total_cases = len(df_filtered)
            top_15_cases = len(df_top_causes)
            coverage_percent = (top_15_cases / total_cases) * 100
            st.info(f"📊 Showing top 15 root causes, representing {coverage_percent:.1f}% of all cases.")
            
            # Distribution of root causes
            plt.figure(figsize=(12, 8))  # Increased height for better readability
            root_cause_counts = df_top_causes['RCA__c'].value_counts().reset_index()
            root_cause_counts.columns = ['RCA', 'Count']
            
            # Use a color palette that matches the number of categories
            n_colors = len(root_cause_counts)
            current_palette = sns.color_palette(ROOT_CAUSE_PALETTE[:n_colors])
            
            # Fix palette warning by using hue parameter
            sns.barplot(data=root_cause_counts, x='Count', y='RCA', 
                       hue='RCA', palette=current_palette, legend=False)
            plt.title('Distribution of Top 15 Root Causes')
            plt.xlabel('Number of Tickets')
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
            
            # Root Cause by Product Area heatmap
            root_cause_product = pd.crosstab(df_top_causes['RCA__c'], df_top_causes['Product_Area__c'])
            plt.figure(figsize=(14, 10))  # Increased size for better readability
            sns.heatmap(root_cause_product, annot=True, fmt='d', cmap=HEATMAP_PALETTE,
                       cbar_kws={'label': 'Ticket Count'})
            plt.title('Top 15 Root Causes by Product Area')
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
            
            # Resolution time by root cause
            resolution_by_root = df_top_causes[df_top_causes['ClosedDate'].notna()].copy()
            resolution_by_root['Resolution_Time_Days'] = (
                resolution_by_root['ClosedDate'] - resolution_by_root['CreatedDate']
            ).dt.total_seconds() / (24 * 3600)
            
            plt.figure(figsize=(14, 8))  # Increased width for better readability
            # Fix palette warning by using hue parameter
            sns.boxplot(data=resolution_by_root, x='RCA__c', y='Resolution_Time_Days',
                       hue='RCA__c', palette=current_palette, legend=False)
            plt.title('Resolution Time by Root Cause (Top 15)')
            plt.xlabel('Root Cause')
            plt.ylabel('Resolution Time (Days)')
            plt.xticks(rotation=45, ha='right')  # Improved label readability
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
        
        # 8. Configuration Analysis
        st.header("Configuration Analysis")
        
        # Define configuration categories
        config_categories = ['Configuration Implementation Miss', 'Net New Configuration']
        config_miss_category = ['Configuration Implementation Miss']
        
        # Create a copy of the dataframe for configuration analysis
        config_df = df_filtered.copy()
        
        # Filter for configuration-related tickets
        config_df['Is_Config_Related'] = config_df['RCA__c'].isin(config_categories)
        config_df['Is_Config_Miss'] = config_df['RCA__c'].isin(config_miss_category)
        
        # Get the first ticket date for each account
        account_first_tickets = config_df.groupby('Account_Name')['CreatedDate'].min()
        
        # Calculate the date range for each account (45 days from first ticket)
        account_date_ranges = pd.DataFrame({
            'First_Ticket': account_first_tickets,
            'Range_End': account_first_tickets + pd.Timedelta(days=45)
        })
        
        # Initialize lists to store data for visualization
        account_data = []
        
        for account in customers:
            if account in account_date_ranges.index:
                # Get date range for this account
                start_date = account_date_ranges.loc[account, 'First_Ticket']
                end_date = account_date_ranges.loc[account, 'Range_End']
                
                # Filter tickets for this account within the date range
                account_tickets = config_df[
                    (config_df['Account_Name'] == account) &
                    (config_df['CreatedDate'] >= start_date) &
                    (config_df['CreatedDate'] <= end_date)
                ]
                
                # Count configuration-related tickets
                config_count = account_tickets['Is_Config_Related'].sum()
                config_miss_count = account_tickets['Is_Config_Miss'].sum()
                total_count = len(account_tickets)
                
                # Calculate percentages
                config_percentage = (config_count / total_count * 100) if total_count > 0 else 0
                config_miss_percentage = (config_miss_count / total_count * 100) if total_count > 0 else 0
                
                # Calculate average resolution time for config miss cases
                config_miss_resolution = account_tickets[account_tickets['Is_Config_Miss']]
                avg_resolution_time = None
                if not config_miss_resolution.empty and 'ClosedDate' in config_miss_resolution.columns:
                    resolved_cases = config_miss_resolution[config_miss_resolution['ClosedDate'].notna()]
                    if not resolved_cases.empty:
                        avg_resolution_time = (resolved_cases['ClosedDate'] - resolved_cases['CreatedDate']).dt.total_seconds().mean() / (24 * 3600)
                
                # Store the data
                account_data.append({
                    'Account': account,
                    'Config_Tickets': config_count,
                    'Config_Miss_Tickets': config_miss_count,
                    'Total_Tickets': total_count,
                    'Config_Percentage': config_percentage,
                    'Config_Miss_Percentage': config_miss_percentage,
                    'Avg_Resolution_Days': avg_resolution_time,
                    'Start_Date': start_date.strftime('%Y-%m-%d'),
                    'End_Date': end_date.strftime('%Y-%m-%d')
                })
        
        if account_data:
            # Create DataFrame for visualization
            config_analysis_df = pd.DataFrame(account_data)
            
            # Display summary information
            st.info("📊 This analysis shows configuration-related tickets created within the first 45 days "
                   "of each account's first ticket, with special focus on Configuration Implementation Miss cases.")
            
            # Create three columns for the visualizations
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Bar chart showing absolute numbers
                plt.figure(figsize=(10, 6))
                
                # Create a categorical color palette for the bars
                config_palette = {
                    'Config Miss': BLUES_PALETTE[2],
                    'Other Config': AQUA_PALETTE[2],
                    'Other Tickets': PURPLE_PALETTE[2]
                }
                
                # Prepare data for seaborn
                plot_data = pd.DataFrame({
                    'Account': config_analysis_df['Account'].repeat(3),
                    'Type': ['Config Miss'] * len(config_analysis_df) + 
                           ['Other Config'] * len(config_analysis_df) + 
                           ['Other Tickets'] * len(config_analysis_df),
                    'Count': list(config_analysis_df['Config_Miss_Tickets']) + 
                            list(config_analysis_df['Config_Tickets'] - config_analysis_df['Config_Miss_Tickets']) +
                            list(config_analysis_df['Total_Tickets'] - config_analysis_df['Config_Tickets'])
                })
                
                # Use seaborn's barplot with proper hue
                sns.barplot(data=plot_data, x='Account', y='Count', 
                          hue='Type', palette=config_palette)
                
                plt.xlabel('Account')
                plt.ylabel('Number of Tickets')
                plt.title('Ticket Distribution\n(First 45 Days)')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='')
                plt.tight_layout()
                st.pyplot(plt)
                plt.close()
            
            with col2:
                # Percentage visualization
                plt.figure(figsize=(10, 6))
                # Fix palette warning by using hue parameter
                sns.barplot(data=config_analysis_df, x='Account', y='Config_Miss_Percentage',
                          hue='Account', palette=[BLUES_PALETTE[2]], legend=False)
                plt.xlabel('Account')
                plt.ylabel('Percentage of Config Miss Tickets')
                plt.title('Config Miss Percentage\n(First 45 Days)')
                plt.xticks(rotation=45, ha='right')
                plt.axhline(y=50, color='r', linestyle='--', alpha=0.5)
                plt.tight_layout()
                st.pyplot(plt)
                plt.close()
            
            with col3:
                # Resolution time visualization
                plt.figure(figsize=(10, 6))
                # Fix palette warning by using hue parameter
                sns.barplot(data=config_analysis_df, x='Account', y='Avg_Resolution_Days',
                          hue='Account', palette=[AQUA_PALETTE[2]], legend=False)
                plt.xlabel('Account')
                plt.ylabel('Average Resolution Time (Days)')
                plt.title('Config Miss Resolution Time\n(First 45 Days)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(plt)
                plt.close()
            
            # Display detailed statistics
            st.subheader("Configuration Analysis Details")
            
            # Format the DataFrame for display
            display_df = config_analysis_df.copy()
            display_df['Config_Percentage'] = display_df['Config_Percentage'].round(1).astype(str) + '%'
            display_df['Config_Miss_Percentage'] = display_df['Config_Miss_Percentage'].round(1).astype(str) + '%'
            display_df['Avg_Resolution_Days'] = display_df['Avg_Resolution_Days'].round(1)
            display_df.columns = ['Account', 'All Config Tickets', 'Config Miss Tickets', 'Total Tickets', 
                                'All Config %', 'Config Miss %', 'Avg Resolution (Days)', 'Analysis Start', 'Analysis End']
            
            st.dataframe(display_df.set_index('Account'))
            
            # Add summary statistics
            st.subheader("Summary Statistics")
            total_config_miss = config_analysis_df['Config_Miss_Tickets'].sum()
            total_tickets = config_analysis_df['Total_Tickets'].sum()
            overall_config_miss_percentage = (total_config_miss / total_tickets * 100) if total_tickets > 0 else 0
            avg_resolution_time = config_analysis_df['Avg_Resolution_Days'].mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Config Miss Tickets", total_config_miss)
            with col2:
                st.metric("Overall Config Miss %", f"{overall_config_miss_percentage:.1f}%")
            with col3:
                st.metric("Average Resolution Time", f"{avg_resolution_time:.1f} days")
        else:
            st.warning("No configuration-related tickets found in the first 45 days for any account.")
    except Exception as e:
        st.error(f"Error in visualization: {str(e)}")
        st.error("Please check your data or try different selection criteria.")

def generate_wordcloud(text_data, title):
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

def generate_powerpoint(filtered_df, active_accounts, avg_csat, escalation_rate):
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
        
        # Add statistics as bullet points
        stats_text = (
            f"• Total Cases: {len(filtered_df)}\n"
            f"• Active Accounts: {active_accounts}\n"
            f"• Product Areas: {filtered_df['Product_Area__c'].nunique()}\n"
            f"• Average CSAT: {avg_csat:.2f}\n"
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
        
        # Convert timezone-aware datetime columns to timezone-naive
        datetime_columns = export_df.select_dtypes(include=['datetime64[ns, UTC]']).columns
        for col in datetime_columns:
            export_df[col] = export_df[col].dt.tz_localize(None)
        
        if format == "Excel":
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {
                    'Customer': [],
                    'Total Tickets': [],
                    'Avg Response Time (hrs)': [],
                    'Avg Resolution Time (days)': [],
                    'Avg CSAT': []
                }
                
                for customer in customers:
                    customer_df = export_df[export_df['Account_Name'] == customer]
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
        elif format == "PowerPoint":
            pptx_data = generate_powerpoint(export_df, len(export_df['Account_Name'].unique()), 
                                          export_df['CSAT__c'].mean(), export_df['IsEscalated'].mean() * 100)
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
                ContactId, OwnerId, Origin, Type, Reason,
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

def display_detailed_analysis(data, enable_ai_analysis):
    """Display detailed analysis of ticket data."""
    st.header("Detailed Ticket Analysis")
    
    # Create a progress bar for the entire analysis process
    progress_bar = st.progress(0)
    
    cases_df = data['cases']
    comments_df = data['comments']
    history_df = data['history']
    emails_df = data['emails']
    
    # Update progress
    progress_bar.progress(10, text="Analyzing basic statistics...")
    
    # Basic statistics
    st.subheader("Ticket Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tickets", len(cases_df))
    with col2:
        closed_cases = cases_df[cases_df['Status'].isin(['Closed', 'Resolved'])]
        st.metric("Closed Tickets", len(closed_cases))
    with col3:
        avg_resolution_time = None
        if not closed_cases.empty:
            resolution_times = (closed_cases['ClosedDate'] - closed_cases['CreatedDate']).dt.total_seconds() / (24 * 3600)
            avg_resolution_time = resolution_times.mean()
        st.metric("Avg Resolution Time", f"{avg_resolution_time:.1f} days" if avg_resolution_time else "N/A")
    with col4:
        escalated_count = cases_df['IsEscalated'].sum()
        escalation_rate = (escalated_count / len(cases_df)) * 100 if len(cases_df) > 0 else 0
        st.metric("Escalation Rate", f"{escalation_rate:.1f}%")
    
    # Update progress
    progress_bar.progress(20, text="Analyzing implementation phases...")
    
    # Implementation Phase Analysis (if available)
    if 'IMPL_Phase__c' in cases_df.columns:
        st.subheader("Implementation Phase Analysis")
        
        # Count cases by implementation phase
        phase_counts = cases_df['IMPL_Phase__c'].fillna('Not Specified').value_counts()
        
        # Create a pie chart
        plt.figure(figsize=(10, 6))
        plt.pie(phase_counts, labels=phase_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=sns.color_palette('Blues', len(phase_counts)))
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Cases by Implementation Phase')
        st.pyplot(plt)
        plt.close()
        
        # Create a bar chart showing resolution time by implementation phase
        if not closed_cases.empty and 'IMPL_Phase__c' in closed_cases.columns:
            # Calculate resolution time
            closed_cases = closed_cases.copy()
            closed_cases['Resolution_Time_Days'] = (closed_cases['ClosedDate'] - closed_cases['CreatedDate']).dt.total_seconds() / (24 * 3600)
            
            # Group by implementation phase
            phase_resolution = closed_cases.groupby('IMPL_Phase__c')['Resolution_Time_Days'].mean().reset_index()
            phase_resolution = phase_resolution.sort_values('Resolution_Time_Days', ascending=False)
            
            # Create bar chart - fix the palette warning by setting hue and legend=False
            plt.figure(figsize=(10, 6))
            sns.barplot(data=phase_resolution, x='IMPL_Phase__c', y='Resolution_Time_Days', 
                       hue='IMPL_Phase__c', palette='Blues', legend=False)
            plt.title('Average Resolution Time by Implementation Phase')
            plt.xlabel('Implementation Phase')
            plt.ylabel('Resolution Time (Days)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
    
    # Update progress
    progress_bar.progress(30, text="Preparing ticket details...")
    
    # Ticket details
    with st.expander("Ticket Details", expanded=False):
        display_columns = ['CaseNumber', 'Subject', 'Status', 'Internal_Priority__c', 
                          'Product_Area__c', 'Product_Feature__c', 'CreatedDate', 'ClosedDate']
        
        # Add IMPL_Phase__c if available
        if 'IMPL_Phase__c' in cases_df.columns:
            display_columns.append('IMPL_Phase__c')
            
        st.dataframe(cases_df[display_columns])
    
    # Update progress
    progress_bar.progress(40, text="Analyzing comments...")
    
    # Comments analysis
    if not comments_df.empty:
        st.subheader("Comments Analysis")
        
        # Group comments by case
        comments_by_case = comments_df.groupby('ParentId').size().reset_index(name='comment_count')
        cases_with_comments = pd.merge(
            cases_df[['Id', 'CaseNumber', 'Subject']], 
            comments_by_case, 
            left_on='Id', 
            right_on='ParentId', 
            how='left'
        )
        cases_with_comments['comment_count'] = cases_with_comments['comment_count'].fillna(0).astype(int)
        
        # Visualize comment distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=cases_with_comments, x='comment_count', bins=10, kde=True)
        plt.title('Distribution of Comments per Ticket')
        plt.xlabel('Number of Comments')
        plt.ylabel('Number of Tickets')
        st.pyplot(plt)
        plt.close()
        
        # Show cases with most comments
        st.subheader("Tickets with Most Comments")
        top_commented = cases_with_comments.sort_values('comment_count', ascending=False).head(5)
        st.dataframe(top_commented[['CaseNumber', 'Subject', 'comment_count']])
    
    # Update progress
    progress_bar.progress(60, text="Analyzing email communications...")
    
    # Email analysis
    if not emails_df.empty:
        st.subheader("Email Communication Analysis")
        
        # Group emails by case
        emails_by_case = emails_df.groupby('ParentId').size().reset_index(name='email_count')
        cases_with_emails = pd.merge(
            cases_df[['Id', 'CaseNumber', 'Subject']], 
            emails_by_case, 
            left_on='Id', 
            right_on='ParentId', 
            how='left'
        )
        cases_with_emails['email_count'] = cases_with_emails['email_count'].fillna(0).astype(int)
        
        # Visualize email distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=cases_with_emails, x='email_count', bins=10, kde=True)
        plt.title('Distribution of Emails per Ticket')
        plt.xlabel('Number of Emails')
        plt.ylabel('Number of Tickets')
        st.pyplot(plt)
        plt.close()
    
    # Update progress
    progress_bar.progress(80, text="Analyzing status changes...")
    
    # Status change analysis
    if not history_df.empty:
        st.subheader("Status Change Analysis")
        
        # Filter for status changes
        status_changes = history_df[history_df['Field'] == 'Status']
        
        if not status_changes.empty:
            # Count status transitions
            status_transitions = status_changes.groupby(['OldValue', 'NewValue']).size().reset_index(name='count')
            
            # Create a heatmap of status transitions
            pivot_table = status_transitions.pivot_table(
                values='count', 
                index='OldValue', 
                columns='NewValue', 
                fill_value=0
            )
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='g')
            plt.title('Status Transition Heatmap')
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
    
    # AI Analysis
    if enable_ai_analysis:
        # Update progress
        progress_bar.progress(90, text="Generating AI-powered insights...")
        
        # Create a separate section for AI insights with a clear header
        st.markdown("---")
        st.header("AI-Powered Insights", help="Insights generated using OpenAI's GPT-4 model")
        debug("AI analysis enabled in display_detailed_analysis")
        
        # Use a spinner to indicate that AI analysis is in progress
        with st.spinner("🧠 Analyzing ticket data with AI... This may take a moment."):
            debug("Calling generate_ai_insights")
            ai_insights = generate_ai_insights(data)
            debug("generate_ai_insights returned", ai_insights is not None)
            
            if ai_insights and isinstance(ai_insights, dict):
                # Display AI insights
                st.subheader("Key Insights")
                st.markdown(ai_insights.get('summary', 'No summary available'))
                
                # Display patterns
                if 'patterns' in ai_insights and ai_insights['patterns']:
                    st.subheader("Identified Patterns")
                    for pattern in ai_insights['patterns']:
                        if isinstance(pattern, dict) and 'title' in pattern and 'description' in pattern:
                            st.markdown(f"- **{pattern['title']}**: {pattern['description']}")
                
                # Display recommendations
                if 'recommendations' in ai_insights and ai_insights['recommendations']:
                    st.subheader("Recommendations")
                    for rec in ai_insights['recommendations']:
                        st.markdown(f"- {rec}")
                
                # Add a download button for the HTML file
                if hasattr(st.session_state, 'latest_ai_html_file') and st.session_state.latest_ai_html_file:
                    with open(st.session_state.latest_ai_html_file, 'r') as f:
                        html_content = f.read()
                    st.download_button(
                        label="Download AI Analysis as HTML",
                        data=html_content,
                        file_name=os.path.basename(st.session_state.latest_ai_html_file),
                        mime="text/html"
                    )
            else:
                st.warning("No AI insights were generated. Please check the debug output for more information.")
                debug("No AI insights were generated or invalid format returned")
                
                # Provide a fallback sample response for demonstration purposes
                st.info("Showing sample AI insights for demonstration purposes.")
                
                sample_insights = {
                    "summary": "This is a sample AI analysis. To get real insights, please ensure the OpenAI API is properly configured.",
                    "patterns": [
                        {"title": "Sample Pattern", "description": "This is an example pattern that would be identified by the AI."},
                        {"title": "Demo Insight", "description": "In a real analysis, the AI would identify trends and patterns in your ticket data."}
                    ],
                    "recommendations": [
                        "This is a sample recommendation. Configure your OpenAI API key to get actual insights.",
                        "Another example recommendation that would be tailored to your specific data."
                    ]
                }
                
                # Display sample insights
                st.subheader("Sample Key Insights")
                st.markdown(sample_insights['summary'])
                
                st.subheader("Sample Identified Patterns")
                for pattern in sample_insights['patterns']:
                    st.markdown(f"- **{pattern['title']}**: {pattern['description']}")
                
                st.subheader("Sample Recommendations")
                for rec in sample_insights['recommendations']:
                    st.markdown(f"- {rec}")
    
    # Complete the progress bar
    progress_bar.progress(100, text="Analysis complete!")
    time.sleep(0.5)  # Short delay to show the completed progress
    progress_bar.empty()  # Remove the progress bar after completion

def generate_ai_insights(data):
    """Generate AI insights from ticket data using OpenAI."""
    try:
        # Check if OpenAI package is installed
        try:
            import openai
            debug("OpenAI package imported successfully")
        except ImportError:
            st.error("OpenAI package is not installed. Please run 'pip install openai' to enable AI analysis.")
            return None
            
        # Check if OpenAI API key is available
        openai_api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get("OPENAI_API_KEY", None)
        if not openai_api_key:
            st.warning("OpenAI API key not found. Please add it to your environment variables or Streamlit secrets.")
            debug("OpenAI API key not found")
            
            # Provide a sample response for demonstration purposes
            return {
                "summary": "This is a sample AI analysis. To get real insights, please configure your OpenAI API key.",
                "patterns": [
                    {"title": "Sample Pattern", "description": "This is an example pattern that would be identified by the AI."},
                    {"title": "Demo Insight", "description": "In a real analysis, the AI would identify trends and patterns in your ticket data."}
                ],
                "recommendations": [
                    "This is a sample recommendation. Configure your OpenAI API key to get actual insights.",
                    "Another example recommendation that would be tailored to your specific data."
                ]
            }
        
        debug("OpenAI API key found")
        
        # Set up OpenAI client
        try:
            client = openai.OpenAI(api_key=openai_api_key)
            debug("OpenAI client initialized successfully")
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {str(e)}")
            debug("OpenAI client initialization error", str(e))
            return None
        
        # Prepare data for analysis
        cases_df = data['cases']
        
        # Create a summary of the data
        case_summary = {
            "total_cases": len(cases_df),
            "product_areas": cases_df['Product_Area__c'].value_counts().to_dict(),
            "priorities": cases_df['Internal_Priority__c'].value_counts().to_dict(),
            "statuses": cases_df['Status'].value_counts().to_dict(),
            "root_causes": cases_df['RCA__c'].value_counts().to_dict(),
        }
        
        # Add implementation phase data if available
        if 'IMPL_Phase__c' in cases_df.columns:
            case_summary["implementation_phases"] = cases_df['IMPL_Phase__c'].fillna('Not Specified').value_counts().to_dict()
        
        # Sample of case subjects and descriptions (limit to 20 for API constraints)
        sample_columns = ['Subject', 'Description', 'RCA__c', 'Status']
        if 'IMPL_Phase__c' in cases_df.columns:
            sample_columns.append('IMPL_Phase__c')
            
        case_samples = cases_df.sample(min(20, len(cases_df)))[sample_columns].to_dict('records')
        
        debug("Data prepared for OpenAI", f"Summary contains {len(case_summary)} keys, {len(case_samples)} samples")
        
        # Prepare the prompt
        prompt = f"""
        Analyze the following support ticket data and provide insights:
        
        Summary Statistics:
        {json.dumps(case_summary, indent=2)}
        
        Sample Cases:
        {json.dumps(case_samples, indent=2)}
        
        Please provide:
        1. A summary of key insights from the data
        2. Patterns or trends you identify in the tickets
        3. Recommendations for improving customer support
        
        Format your response as JSON with the following structure:
        {{
            "summary": "Overall summary of insights",
            "patterns": [
                {{"title": "Pattern 1", "description": "Description of pattern 1"}},
                {{"title": "Pattern 2", "description": "Description of pattern 2"}}
            ],
            "recommendations": [
                "Recommendation 1",
                "Recommendation 2"
            ]
        }}
        """
        
        # Call OpenAI API
        try:
            debug("Calling OpenAI API")
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert support ticket analyst. Analyze the provided data and extract meaningful insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=2000
            )
            debug("OpenAI API call successful")
        except Exception as e:
            st.error(f"Error calling OpenAI API: {str(e)}")
            debug("OpenAI API call error", str(e))
            return None
        
        # Extract and parse the response
        ai_response = response.choices[0].message.content
        debug("OpenAI response received", ai_response[:100] + "..." if len(ai_response) > 100 else ai_response)
        
        # Save the raw response to an HTML file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>OpenAI Response - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                h1, h2 {{ color: #333; }}
                .metadata {{ color: #666; font-size: 0.9em; }}
                .json {{ color: #0066cc; }}
            </style>
        </head>
        <body>
            <h1>OpenAI Response</h1>
            <div class="metadata">
                <p>Generated on: {timestamp}</p>
                <p>Model: gpt-4</p>
            </div>
            <h2>Raw Response:</h2>
            <pre>{html.escape(ai_response)}</pre>
            
            <h2>Parsed JSON:</h2>
            <pre class="json" id="parsed-json"></pre>
            
            <script>
                try {{
                    // Try to extract and parse JSON
                    const jsonPattern = /({{\s*"summary".*}})/s;
                    const jsonMatch = `{html.escape(ai_response)}`.match(jsonPattern);
                    if (jsonMatch) {{
                        const parsedJson = JSON.parse(jsonMatch[1]);
                        document.getElementById('parsed-json').textContent = JSON.stringify(parsedJson, null, 2);
                    }} else {{
                        document.getElementById('parsed-json').textContent = "Could not extract JSON from response";
                    }}
                }} catch (e) {{
                    document.getElementById('parsed-json').textContent = "Error parsing JSON: " + e.message;
                }}
            </script>
        </body>
        </html>
        """
        
        # Save the HTML file
        html_filename = f"openai_response_{timestamp}.html"
        with open(html_filename, "w") as f:
            f.write(html_content)
        debug(f"Saved OpenAI response to {html_filename}")
        
        # Store the HTML file path in session state for download
        st.session_state.latest_ai_html_file = html_filename
        
        # Extract JSON from the response (in case there's additional text)
        json_match = re.search(r'({.*})', ai_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                insights = json.loads(json_str)
                debug("JSON parsed successfully")
                
                # Also save as JSON file for easier processing
                json_filename = f"openai_response_{timestamp}.json"
                with open(json_filename, "w") as f:
                    json.dump(insights, f, indent=2)
                debug(f"Saved parsed JSON to {json_filename}")
                
                return insights
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON response: {str(e)}")
                debug("JSON parsing error", str(e))
                return None
        else:
            st.warning("Could not parse AI response into the expected format.")
            debug("No JSON found in response")
            return None
            
    except Exception as e:
        st.error(f"Error generating AI insights: {str(e)}")
        debug("General error in generate_ai_insights", str(e))
        return None

if __name__ == "__main__":
    main() 
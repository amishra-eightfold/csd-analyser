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

def main():
    st.title("Support Ticket Analytics")
    
    # Initialize Salesforce connection
    if not st.session_state.sf_connection:
        with st.spinner("Connecting to Salesforce..."):
            st.session_state.sf_connection = init_salesforce()
            if not st.session_state.sf_connection:
                st.error("Failed to connect to Salesforce. Please check your credentials.")
                return
    
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
                st.write("Debug - Customer query:", customer_query)
                
                records = execute_soql_query(st.session_state.sf_connection, customer_query)
                if records:
                    customers_df = pd.DataFrame(records)
                    st.write("Debug - Customer data:", customers_df.head())
                    st.session_state.customers = customers_df['Account_Name__c'].unique().tolist()
                    st.write("Debug - Unique customers:", len(st.session_state.customers))
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
        
        # Date Range Selection
        st.subheader("Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now().date() - timedelta(days=90)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date()
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
        st.write("Debug - SOQL Query:", query)
        
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
        st.write("Debug - DataFrame columns:", df.columns.tolist())
        st.write("Debug - First few rows:", df.head())
        st.write("Debug - Number of records:", len(df))
        
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
        st.write("Debug - Visualization data shape:", df.shape)
        st.write("Debug - Available columns:", df.columns.tolist())
        
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
                       palette=PRIORITY_PALETTE)
        g.fig.suptitle('Resolution Time by Priority', y=1.02)
        plt.tight_layout()
        st.pyplot(g.fig)
        plt.close()
        
        # Add Resolution Time by Product Area visualization
        st.subheader("Resolution Time by Product Area")
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=resolved_df, x='Product_Area__c', y='Resolution_Time_Days',
                   palette=BLUES_PALETTE)
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
            current_palette = ROOT_CAUSE_PALETTE[:n_colors]
            
            sns.barplot(data=root_cause_counts, x='Count', y='RCA', 
                       palette=current_palette, legend=False)
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
            sns.boxplot(data=resolution_by_root, x='RCA__c', y='Resolution_Time_Days',
                       palette=current_palette)  # Use same palette as bar chart
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
        
        # Create a copy of the dataframe for configuration analysis
        config_df = df_filtered.copy()
        
        # Filter for configuration-related tickets
        config_df['Is_Config_Related'] = config_df['RCA__c'].isin(config_categories)
        
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
                total_count = len(account_tickets)
                
                # Calculate percentage
                config_percentage = (config_count / total_count * 100) if total_count > 0 else 0
                
                # Store the data
                account_data.append({
                    'Account': account,
                    'Config_Tickets': config_count,
                    'Total_Tickets': total_count,
                    'Config_Percentage': config_percentage,
                    'Start_Date': start_date.strftime('%Y-%m-%d'),
                    'End_Date': end_date.strftime('%Y-%m-%d')
                })
        
        if account_data:
            # Create DataFrame for visualization
            config_analysis_df = pd.DataFrame(account_data)
            
            # Display summary information
            st.info("📊 This analysis shows configuration-related tickets (Configuration Implementation Miss & Net New Configuration) "
                   "created within the first 45 days of each account's first ticket.")
            
            # Create two columns for the visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart showing absolute numbers
                plt.figure(figsize=(10, 6))
                x = range(len(config_analysis_df))
                width = 0.35
                
                plt.bar(x, config_analysis_df['Config_Tickets'], width, 
                       label='Configuration Tickets', color=BLUES_PALETTE[2])
                plt.bar([i + width for i in x], 
                       config_analysis_df['Total_Tickets'] - config_analysis_df['Config_Tickets'],
                       width, label='Other Tickets', color=AQUA_PALETTE[2])
                
                plt.xlabel('Account')
                plt.ylabel('Number of Tickets')
                plt.title('Configuration vs Other Tickets\n(First 45 Days)')
                plt.xticks([i + width/2 for i in x], config_analysis_df['Account'], 
                          rotation=45, ha='right')
                plt.legend()
                plt.tight_layout()
                st.pyplot(plt)
                plt.close()
            
            with col2:
                # Percentage visualization
                plt.figure(figsize=(10, 6))
                plt.bar(config_analysis_df['Account'], config_analysis_df['Config_Percentage'],
                       color=PURPLE_PALETTE[2])
                plt.xlabel('Account')
                plt.ylabel('Percentage of Configuration Tickets')
                plt.title('Configuration Tickets Percentage\n(First 45 Days)')
                plt.xticks(rotation=45, ha='right')
                plt.axhline(y=50, color='r', linestyle='--', alpha=0.5)
                plt.tight_layout()
                st.pyplot(plt)
                plt.close()
            
            # Display detailed statistics
            st.subheader("Configuration Analysis Details")
            
            # Format the DataFrame for display
            display_df = config_analysis_df.copy()
            display_df['Config_Percentage'] = display_df['Config_Percentage'].round(1).astype(str) + '%'
            display_df.columns = ['Account', 'Configuration Tickets', 'Total Tickets', 
                                'Configuration %', 'Analysis Start', 'Analysis End']
            
            st.dataframe(display_df.set_index('Account'))
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

if __name__ == "__main__":
    main() 
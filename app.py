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
from salesforce_config import init_salesforce, execute_soql_query

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
                    product_features = st.multiselect(
                        "Filter by Product Feature",
                        options=sorted(st.session_state.data['Product_Feature__c'].unique())
                    )
            
            # Apply filters
            df = st.session_state.data.copy()
            if product_areas:
                df = df[df['Product_Area__c'].isin(product_areas)]
            if product_features:
                df = df[df['Product_Feature__c'].isin(product_features)]
            
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
        
        # Overview Statistics
        st.header("Overview Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Cases", len(df))
        with col2:
            st.metric("Active Accounts", len(customers))
        with col3:
            st.metric("Product Areas", df['Product_Area__c'].nunique())
        with col4:
            avg_csat = df[df['CSAT__c'].notna()]['CSAT__c'].mean()
            st.metric("Avg CSAT", f"{avg_csat:.2f}" if pd.notna(avg_csat) else "N/A")
        with col5:
            escalation_rate = (df['IsEscalated'].mean() * 100)
            st.metric("Escalation Rate", f"{escalation_rate:.1f}%" if pd.notna(escalation_rate) else "N/A")
        
        # 1. Ticket Volume Analysis
        st.header("Ticket Volume Analysis")
        for customer in customers:
            try:
                customer_df = df[df['Account_Name'] == customer].copy()
                if customer_df.empty:
                    st.warning(f"No data available for customer: {customer}")
                    continue
                
                # Debug logging for customer data
                st.write(f"Debug - Processing customer: {customer}")
                st.write(f"Debug - Customer data shape: {customer_df.shape}")
                
                # Created tickets
                customer_df['Month'] = pd.to_datetime(customer_df['CreatedDate']).dt.to_period('M')
                created_monthly = customer_df.groupby('Month').size().reset_index(name='Created')
                
                # Closed tickets
                closed_df = customer_df[customer_df['ClosedDate'].notna()].copy()
                if not closed_df.empty:
                    closed_df['Month'] = pd.to_datetime(closed_df['ClosedDate']).dt.to_period('M')
                    closed_monthly = closed_df.groupby('Month').size().reset_index(name='Closed')
                    
                    # Merge created and closed
                    monthly_data = pd.merge(created_monthly, closed_monthly, on='Month', how='outer').fillna(0)
                else:
                    monthly_data = created_monthly
                    monthly_data['Closed'] = 0
                
                monthly_data['Month'] = monthly_data['Month'].astype(str)
                
                # Debug logging for monthly data
                st.write(f"Debug - Monthly data for {customer}:", monthly_data)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=monthly_data['Month'],
                    y=monthly_data['Created'],
                    name='Created',
                    marker_color='#2E86C1'  # Blue color for created tickets
                ))
                if 'Closed' in monthly_data.columns:
                    fig.add_trace(go.Bar(
                        x=monthly_data['Month'],
                        y=monthly_data['Closed'],
                        name='Closed',
                        marker_color='#27AE60'  # Green color for closed tickets
                    ))
                fig.update_layout(
                    title=f"{customer} - Ticket Volume Trends",
                    xaxis_title="Month",
                    yaxis_title="Number of Tickets",
                    barmode='group',  # Group bars side by side
                    bargap=0.2,       # Gap between bars in a group
                    bargroupgap=0.1,  # Gap between bar groups
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying ticket volume analysis for {customer}: {str(e)}")
                continue
        
        # 2. Response Time Analysis
        st.header("Response Time Analysis")
        response_df = df[df['First_Response_Time__c'].notna()].copy()
        response_df['Response_Time_Hours'] = (
            response_df['First_Response_Time__c'] - response_df['CreatedDate']
        ).dt.total_seconds() / 3600
        
        fig = px.box(
            response_df,
            x='Account_Name',
            y='Response_Time_Hours',
            color='Internal_Priority__c',
            title='Time to First Response by Priority',
            labels={'Response_Time_Hours': 'Response Time (Hours)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. CSAT Analysis
        st.header("Customer Satisfaction Analysis")
        csat_df = df[df['CSAT__c'].notna()].copy()
        csat_df['Month'] = csat_df['CreatedDate'].dt.to_period('M').astype(str)
        
        fig = px.line(
            csat_df,
            x='Month',
            y='CSAT__c',
            color='Account_Name',
            title='CSAT Scores Trend',
            markers=True,
            labels={'CSAT__c': 'CSAT Score'}
        )
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. Resolution Time Analysis
        st.header("Resolution Time Analysis")
        resolved_df = df[df['ClosedDate'].notna()].copy()
        resolved_df['Resolution_Time_Days'] = (
            resolved_df['ClosedDate'] - resolved_df['CreatedDate']
        ).dt.total_seconds() / (24 * 3600)
        resolved_df['Month'] = resolved_df['CreatedDate'].dt.to_period('M').astype(str)
        
        fig = px.box(
            resolved_df,
            x='Month',
            y='Resolution_Time_Days',
            color='Internal_Priority__c',
            facet_col='Account_Name',
            title='Resolution Time by Priority',
            labels={'Resolution_Time_Days': 'Resolution Time (Days)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 5. Product Area Distribution
        st.header("Product Distribution Analysis")
        
        # Feature Distribution by Product Area
        st.subheader("Feature Distribution by Product Area")
        
        # Create hierarchical data for the treemap
        feature_area_data = df.groupby(['Product_Area__c', 'Product_Feature__c']).size().reset_index(name='Count')
        
        # Create treemap with hierarchical structure
        fig = px.treemap(
            feature_area_data,
            path=['Product_Area__c', 'Product_Feature__c'],
            values='Count',
            title='Feature Distribution by Product Area',
            color='Count',
            color_continuous_scale='Teal',
            maxdepth=2
        )
        
        # Update layout and styling
        fig.update_layout(
            height=800,  # Increased height for better visibility
            margin=dict(t=50, l=25, r=25, b=25),
            uniformtext=dict(minsize=12)
        )
        
        # Update trace styling
        fig.update_traces(
            marker=dict(cornerradius=5),  # Rounded corners
            hovertemplate='<b>%{label}</b><br>Tickets: %{value}<extra></extra>',
            textposition="middle center"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Continue with other visualizations
        
        # 6. Text Analysis
        st.header("Text Analysis")
        for customer in customers:
            customer_df = df[df['Account_Name'] == customer]
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
        if 'RCA__c' in df.columns:
            st.header("Root Cause Analysis")
            
            # Distribution using treemap
            root_cause_data = df.groupby(['RCA__c']).size().reset_index(name='Count')
            
            fig = px.treemap(
                root_cause_data,
                path=['RCA__c'],
                values='Count',
                title='Distribution of Root Causes',
                color='Count',
                color_continuous_scale='viridis'
            )
            
            # Update layout and styling
            fig.update_layout(
                height=600,
                margin=dict(t=50, l=25, r=25, b=25),
                uniformtext=dict(minsize=14)
            )
            
            # Update trace styling
            fig.update_traces(
                marker=dict(cornerradius=5),
                hovertemplate='<b>%{label}</b><br>Tickets: %{value}<br>Percentage: %{percentParent:.1%}<extra></extra>',
                textposition="middle center"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Root Cause by Product Area using treemap
            root_cause_product_data = df.groupby(['RCA__c', 'Product_Area__c']).size().reset_index(name='Count')
            
            fig = px.treemap(
                root_cause_product_data,
                path=['RCA__c', 'Product_Area__c'],
                values='Count',
                title='Root Cause by Product Area',
                color='Count',
                color_continuous_scale='viridis'
            )
            
            # Update layout and styling
            fig.update_layout(
                height=700,
                margin=dict(t=50, l=25, r=25, b=25),
                uniformtext=dict(minsize=12)
            )
            
            # Update trace styling
            fig.update_traces(
                marker=dict(cornerradius=5),
                hovertemplate='<b>%{label}</b><br>Tickets: %{value}<br>Percentage: %{percentParent:.1%}<extra></extra>',
                textposition="middle center"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trends over time
            df['Month'] = df['CreatedDate'].dt.to_period('M').astype(str)
            root_cause_trend = pd.crosstab(df['Month'], df['RCA__c'])
            
            # Create stacked area chart
            fig = go.Figure()
            
            # Add area traces for each root cause
            for column in root_cause_trend.columns:
                fig.add_trace(
                    go.Scatter(
                        x=root_cause_trend.index,
                        y=root_cause_trend[column],
                        name=column,
                        mode='lines',
                        stackgroup='one',  # Enable stacking
                        hovertemplate="<b>%{x}</b><br>" +
                                    "Root Cause: " + column + "<br>" +
                                    "Tickets: %{y}<br>" +
                                    "<extra></extra>"
                    )
                )
            
            # Update layout with Viridis colors
            fig.update_layout(
                title='Root Cause Distribution Over Time',
                xaxis_title="Month",
                yaxis_title="Number of Tickets",
                showlegend=True,
                hovermode='x unified',
                xaxis={'tickangle': 45},
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Apply Viridis color scheme
            num_colors = len(root_cause_trend.columns)
            viridis_colors = px.colors.sequential.Viridis
            color_scale = [viridis_colors[int(i * (len(viridis_colors)-1) / (num_colors-1))] for i in range(num_colors)]
            
            for i, trace in enumerate(fig.data):
                trace.update(fillcolor=color_scale[i], line=dict(color=color_scale[i]))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Resolution time by root cause
            resolution_by_root = df[df['ClosedDate'].notna()].copy()
            resolution_by_root['Resolution_Time_Days'] = (
                resolution_by_root['ClosedDate'] - resolution_by_root['CreatedDate']
            ).dt.total_seconds() / (24 * 3600)
            
            fig = px.box(
                resolution_by_root,
                x='RCA__c',
                y='Resolution_Time_Days',
                title='Resolution Time by Root Cause',
                labels={'Resolution_Time_Days': 'Resolution Time (Days)'},
                color='RCA__c',
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig.update_layout(
                showlegend=False,
                xaxis={'tickangle': 45}
            )
            st.plotly_chart(fig, use_container_width=True)
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
                customer_df = df[df['Account_Name'] == customer]
                summary_data['Customer'].append(customer)
                summary_data['Total Tickets'].append(len(customer_df))
                
                # Response Time
                if 'First_Response_Time__c' in customer_df.columns:
                    resp_time = (customer_df['First_Response_Time__c'] - customer_df['CreatedDate']).dt.total_seconds().mean() / 3600
                    summary_data['Avg Response Time (hrs)'].append(round(resp_time, 2))
                else:
                    summary_data['Avg Response Time (hrs)'].append('N/A')
                
                # Resolution Time
                if 'ClosedDate' in customer_df.columns:
                    res_time = (customer_df['ClosedDate'] - customer_df['CreatedDate']).dt.total_seconds().mean() / (24 * 3600)
                    summary_data['Avg Resolution Time (days)'].append(round(res_time, 2))
                else:
                    summary_data['Avg Resolution Time (days)'].append('N/A')
                
                # CSAT
                if 'CSAT__c' in customer_df.columns:
                    avg_csat = customer_df['CSAT__c'].mean()
                    summary_data['Avg CSAT'].append(round(avg_csat, 2))
                else:
                    summary_data['Avg CSAT'].append('N/A')
            
            # Write summary sheet
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Write detailed data sheets
            for customer in customers:
                customer_df = df[df['Account_Name'] == customer]
                customer_df.to_excel(writer, sheet_name=customer[:31], index=False)  # Excel sheet names limited to 31 chars
        
        # Offer download
        st.download_button(
            label="Download Excel Report",
            data=output.getvalue(),
            file_name=f"support_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    elif format == "PowerPoint":
        pptx_data = generate_powerpoint(df, len(df['Account_Name'].unique()), df['CSAT__c'].mean(), df['IsEscalated'].mean() * 100)
        st.download_button(
            label="Download PowerPoint",
            data=pptx_data,
            file_name=f"support_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )
    elif format == "CSV":
        output = BytesIO()
        df.to_csv(output, index=False)
        st.download_button(
            label="Download CSV",
            data=output.getvalue(),
            file_name=f"support_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
    
    # Sample data format
    st.header("Expected File Format")
    st.markdown("""
    Your file (CSV or Excel) should contain the following columns:
    - `Id`: Case ID
    - `CaseNumber`: Case number
    - `AccountId`: Account ID
    - `Account.Account_Name__c`: Account name
    - `Group_Id__c`: Customer Group ID
    - `Subject`: Case subject
    - `Description`: Case description
    - `Product_Area__c`: Product area
    - `Product_Feature__c`: Product feature
    - `POD_Name__c`: POD name
    - `CreatedDate`: Date when the ticket was created
    - `ClosedDate`: Date when the ticket was closed
    - `Case_Type__c`: Type of case
    - `Age_days__c`: Age of the case in days
    - `IsEscalated`: Whether the case is escalated
    - `CSAT__c`: Customer satisfaction score
    - `Internal_Priority__c`: Internal priority level
    
    Supported file formats:
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    """) 
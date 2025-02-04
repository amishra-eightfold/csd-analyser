import pandas as pd
import numpy as np
from collections import Counter
import re
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from difflib import SequenceMatcher
import io
from fpdf import FPDF
import base64
from datetime import datetime
import plotly.io as pio

def load_data(file_path):
    """Load data from CSV or Excel file."""
    if file_path.name.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    return df

def analyze_field_completeness(df, fields):
    """Analyze completeness of specified fields."""
    completeness = {}
    for field in fields:
        total = len(df)
        missing = df[field].isna().sum()
        empty = len(df[df[field].isin(['', 'Unspecified', 'None', 'null', 'NA'])])
        filled = total - missing - empty
        
        completeness[field] = {
            'total': total,
            'filled': filled,
            'missing': missing,
            'empty': empty,
            'completion_rate': (filled / total) * 100
        }
    return completeness

def analyze_subject_quality(df):
    """Analyze quality of subject field."""
    subjects = df['Subject'].dropna()
    
    # Length analysis
    length_stats = {
        'avg_length': subjects.str.len().mean(),
        'min_length': subjects.str.len().min(),
        'max_length': subjects.str.len().max(),
    }
    
    # Common patterns
    common_starts = Counter(subjects.str[:20]).most_common(10)
    
    # Check for generic subjects
    generic_patterns = [
        r'^help\s+needed',
        r'^issue\s+with',
        r'^problem\s+with',
        r'^error',
        r'^question',
        r'^assistance\s+needed',
        r'^support\s+needed'
    ]
    
    generic_subjects = subjects[
        subjects.str.lower().str.contains('|'.join(generic_patterns), regex=True)
    ]
    
    return {
        'length_stats': length_stats,
        'common_starts': common_starts,
        'generic_count': len(generic_subjects),
        'generic_percentage': (len(generic_subjects) / len(subjects)) * 100
    }

def analyze_field_consistency(df, field):
    """Analyze consistency in field values."""
    values = df[field].dropna()
    
    # Case consistency
    case_variations = {}
    for value in values.unique():
        if pd.isna(value) or not isinstance(value, str):
            continue
        variations = values[values.str.lower() == value.lower()].unique()
        if len(variations) > 1:
            case_variations[value.lower()] = list(variations)
    
    # Value frequency
    value_counts = values.value_counts()
    
    # Spelling variations using difflib
    similar_values = {}
    unique_values = values.unique()
    for i, val1 in enumerate(unique_values):
        if pd.isna(val1) or not isinstance(val1, str):
            continue
        for val2 in unique_values[i+1:]:
            if pd.isna(val2) or not isinstance(val2, str):
                continue
            similarity = SequenceMatcher(None, val1.lower(), val2.lower()).ratio()
            if similarity > 0.8 and similarity < 1.0:  # High similarity but not identical
                key = min(val1, val2)
                if key not in similar_values:
                    similar_values[key] = []
                similar_values[key].append(max(val1, val2))
    
    return {
        'case_variations': case_variations,
        'value_counts': value_counts,
        'similar_values': similar_values
    }

def analyze_data_quality_issues(df):
    """Analyze data quality related issues in DP and Analytics tickets."""
    
    # Define comprehensive data quality related terms
    data_quality_terms = (
        r'data\s*quality|data\s*governance|data\s*validation|data\s*accuracy|'
        r'data\s*integrity|data\s*consistency|data\s*completeness|data\s*reliability|'
        r'data\s*cleansing|data\s*cleanup|data\s*standardization|data\s*enrichment|'
        r'data\s*profiling|data\s*monitoring|data\s*lineage|data\s*catalog|'
        r'data\s*classification|data\s*security|data\s*privacy|data\s*compliance|'
        r'data\s*protection|data\s*governance|data\s*stewardship|data\s*management|'
        r'data\s*issue|data\s*error|data\s*corruption|data\s*anomaly|'
        r'data\s*discrepancy|data\s*reconciliation|data\s*quality\s*check|'
        r'data\s*inconsistency|data\s*mismatch|data\s*incorrect|'
        r'incorrect\s*data|mismatched\s*data|inconsistent\s*data|'
        r'sync\s*issue|sync\s*error|synchronization|data\s*sync|'
        r'gdpr|data\s*protection\s*regulation|data\s*privacy|'
        r'wrong\s*data|invalid\s*data|corrupt\s*data'
    )
    
    # Pattern matching mask for DP/Analytics content
    dp_analytics_mask = (
        df['Product_Area__c'].str.contains('DP|Analytics', case=False, na=False) |
        df['Product_Feature__c'].str.contains('DP|Analytics', case=False, na=False) |
        df['Subject'].str.contains('DP|Analytics', case=False, na=False) |
        df['Description'].str.contains('DP|Analytics', case=False, na=False) |
        (df['POD_Name__c'] == 'CI-Analytics') |
        df['POD_Name__c'].str.startswith('DP', na=False)
    )
    
    if 'RCA__c' in df.columns:
        dp_analytics_mask = dp_analytics_mask | df['RCA__c'].str.contains('DP|Analytics', case=False, na=False)
    
    dp_analytics_tickets = df[dp_analytics_mask].copy()
    
    # Initialize a set to track unique ticket IDs
    unique_ticket_ids = set()
    deduplicated_issues = []
    
    # Search for data quality patterns across all fields
    fields_to_check = ['Subject', 'Product_Area__c', 'Product_Feature__c', 'Description', 'POD_Name__c']
    if 'RCA__c' in dp_analytics_tickets.columns:
        fields_to_check.append('RCA__c')
    
    for _, ticket in dp_analytics_tickets.iterrows():
        if ticket['Id'] not in unique_ticket_ids:
            # Check if any field contains data quality terms
            has_quality_issue = any(
                isinstance(ticket[field], str) and 
                re.search(data_quality_terms, ticket[field], re.IGNORECASE)
                for field in fields_to_check
            )
            
            if has_quality_issue:
                unique_ticket_ids.add(ticket['Id'])
                deduplicated_issues.append(ticket)
    
    # Convert deduplicated issues to DataFrame
    deduplicated_df = pd.DataFrame(deduplicated_issues)
    
    # Analyze issues over time
    time_series_data = None
    if not deduplicated_df.empty and 'CreatedDate' in deduplicated_df.columns:
        deduplicated_df['CreatedDate'] = pd.to_datetime(deduplicated_df['CreatedDate'])
        # Group by month and count issues
        monthly_issues = (
            deduplicated_df
            .groupby(deduplicated_df['CreatedDate'].dt.to_period('M'))
            .size()
            .reset_index()
        )
        monthly_issues.columns = ['Month', 'Issue Count']
        monthly_issues['Month'] = monthly_issues['Month'].astype(str)
        time_series_data = monthly_issues
    
    # Aggregate results
    results = {
        'total_dp_analytics_tickets': len(dp_analytics_tickets),
        'unique_data_quality_issues': len(deduplicated_df),
        'issues_by_pod': deduplicated_df['POD_Name__c'].value_counts().to_dict() if not deduplicated_df.empty else {},
        'issues_by_product_area': deduplicated_df['Product_Area__c'].value_counts().to_dict() if not deduplicated_df.empty else {},
        'time_series_data': time_series_data,
        'sample_tickets': deduplicated_df[['Id', 'Subject', 'Product_Area__c', 'Product_Feature__c', 'POD_Name__c', 'CreatedDate']].head(10).to_dict('records') if not deduplicated_df.empty else [],
        'raw_data': deduplicated_df
    }
    
    # Analyze common terms across all fields
    if not deduplicated_df.empty:
        all_text = ' '.join(
            deduplicated_df[fields_to_check]
            .fillna('')
            .astype(str)
            .values.flatten()
        )
        
        term_counts = {}
        patterns = data_quality_terms.split('|')
        for pattern in patterns:
            pattern = pattern.strip()
            count = len(re.findall(pattern, all_text, re.IGNORECASE))
            if count > 0:
                term_counts[pattern.replace(r'\s*', ' ').strip()] = count
        
        results['common_terms'] = term_counts
    else:
        results['common_terms'] = {}
    
    return results

def export_to_pdf(quality_results):
    """Create a PDF report of the dashboard data."""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Data Quality Analysis Report', 0, 1, 'C')
    pdf.ln(10)
    
    # Date
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'R')
    
    # Overview
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Overview', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Total DP/Analytics Tickets: {quality_results["total_dp_analytics_tickets"]}', 0, 1, 'L')
    pdf.cell(0, 10, f'Unique Data Quality Issues: {quality_results["unique_data_quality_issues"]}', 0, 1, 'L')
    
    # Issues by POD
    if quality_results['issues_by_pod']:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Issues by POD', 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        for pod, count in quality_results['issues_by_pod'].items():
            pdf.cell(0, 10, f'{pod}: {count}', 0, 1, 'L')
    
    # Common Terms
    if quality_results['common_terms']:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Common Data Quality Terms', 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        for term, count in quality_results['common_terms'].items():
            pdf.cell(0, 10, f'{term}: {count}', 0, 1, 'L')
    
    # Sample Tickets
    if quality_results['sample_tickets']:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Sample Tickets', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        for ticket in quality_results['sample_tickets'][:5]:  # Show first 5 tickets
            pdf.multi_cell(0, 10, f"ID: {ticket['Id']}\nSubject: {ticket['Subject']}\nPOD: {ticket['POD_Name__c']}\n", 0, 'L')
            pdf.ln(5)
    
    return pdf.output(dest='S').encode('latin-1')

def main():
    st.title("Data Quality Analysis Dashboard")
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            
            # Data Quality Issues Analysis
            st.header("Data Quality Issues Analysis")
            st.write("Analyzing data quality issues in Data Platform and Analytics tickets...")
            
            quality_results = analyze_data_quality_issues(df)
            
            # Add export options in the sidebar
            st.sidebar.header("Export Options")
            
            # Export filtered raw data as Excel
            if not quality_results['raw_data'].empty:
                st.sidebar.subheader("Export Raw Data")
                
                # Create Excel file in memory
                output_excel = io.BytesIO()
                with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
                    # Convert timezone-aware datetime to naive
                    export_df = quality_results['raw_data'].copy()
                    datetime_columns = export_df.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64[ns]']).columns
                    for col in datetime_columns:
                        export_df[col] = export_df[col].dt.tz_localize(None)
                    
                    export_df.to_excel(writer, index=False, sheet_name='Filtered Raw Data')
                
                st.sidebar.download_button(
                    "Download Filtered Raw Data (Excel)",
                    output_excel.getvalue(),
                    "filtered_raw_data.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key='download-raw-excel'
                )
            
            # Export dashboard as PDF
            st.sidebar.subheader("Export Dashboard")
            try:
                pdf_data = export_to_pdf(quality_results)
                st.sidebar.download_button(
                    "Download Dashboard Report (PDF)",
                    pdf_data,
                    "data_quality_dashboard.pdf",
                    "application/pdf",
                    key='download-pdf'
                )
            except Exception as e:
                st.sidebar.error(f"Error generating PDF: {str(e)}")
            
            # Display overall statistics
            st.subheader("Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total DP/Analytics Tickets", quality_results['total_dp_analytics_tickets'])
            
            with col2:
                st.metric("Unique Data Quality Issues", quality_results['unique_data_quality_issues'])
            
            # Display issues by POD with export
            st.subheader("Issues by POD")
            if quality_results['issues_by_pod']:
                pod_issues = pd.DataFrame(list(quality_results['issues_by_pod'].items()),
                                        columns=['POD', 'Count']).sort_values('Count', ascending=False)
                
                # Create and display chart
                fig_pods = px.bar(
                    pod_issues,
                    x='POD',
                    y='Count',
                    title='Data Quality Issues by POD',
                    labels={'POD': 'POD Name', 'Count': 'Number of Issues'}
                )
                fig_pods.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_pods, use_container_width=True)
                
                # Export options
                st.download_button(
                    "Download POD Data (CSV)",
                    pod_issues.to_csv(index=False).encode('utf-8'),
                    "pod_issues.csv",
                    "text/csv",
                    key='download-pod-csv'
                )
            
            # Display issues over time with export
            if quality_results['time_series_data'] is not None:
                st.subheader("Issues Over Time")
                time_data = quality_results['time_series_data']
                
                fig_time = px.line(
                    time_data,
                    x='Month',
                    y='Issue Count',
                    title='Data Quality Issues by Month',
                    labels={'Month': 'Month', 'Issue Count': 'Number of Issues'}
                )
                fig_time.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Number of Issues",
                    showlegend=True
                )
                st.plotly_chart(fig_time, use_container_width=True)
                
                # Export options
                st.download_button(
                    "Download Time Series Data (CSV)",
                    time_data.to_csv(index=False).encode('utf-8'),
                    "issues_over_time.csv",
                    "text/csv",
                    key='download-time-csv'
                )
            
            # Display common terms with export
            st.subheader("Common Data Quality Terms")
            if quality_results['common_terms']:
                terms_df = pd.DataFrame(list(quality_results['common_terms'].items()),
                                      columns=['Term', 'Count']).sort_values('Count', ascending=False)
                st.dataframe(terms_df)
                
                st.download_button(
                    "Download Terms Data (CSV)",
                    terms_df.to_csv(index=False).encode('utf-8'),
                    "data_quality_terms.csv",
                    "text/csv",
                    key='download-terms-csv'
                )
            
            # Display sample tickets with export
            st.subheader("Sample Tickets with Data Quality Issues")
            if quality_results['sample_tickets']:
                sample_df = pd.DataFrame(quality_results['sample_tickets'])
                st.dataframe(sample_df)
                
                st.download_button(
                    "Download Sample Tickets (CSV)",
                    sample_df.to_csv(index=False).encode('utf-8'),
                    "sample_tickets.csv",
                    "text/csv",
                    key='download-samples-csv'
                )
            
            # Export full dataset
            if not quality_results['raw_data'].empty:
                st.subheader("Export Full Dataset")
                
                # Create a copy of the data and convert timezone-aware datetime to naive
                export_df = quality_results['raw_data'].copy()
                datetime_columns = export_df.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64[ns]']).columns
                for col in datetime_columns:
                    export_df[col] = export_df[col].dt.tz_localize(None)
                
                # Create Excel file in memory
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    export_df.to_excel(writer, index=False, sheet_name='Data Quality Issues')
                
                st.download_button(
                    "Download Complete Analysis (Excel)",
                    output.getvalue(),
                    "data_quality_analysis.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key='download-full-excel'
                )
            
        except Exception as e:
            st.error(f"Error analyzing the file: {str(e)}")
            st.error("Please ensure the required columns are present and contain valid data.")
    
    else:
        st.info("Please upload a CSV or Excel file to begin the analysis.")
        st.markdown("""
        This tool analyzes data quality issues in Data Platform and Analytics tickets,
        looking for patterns related to:
        - Data corruption
        - Data inconsistency
        - Data invalidity
        - Data mismatches
        - GDPR issues
        - Data accuracy
        - Wrong data
        - Sync issues
        And more...
        """)

if __name__ == "__main__":
    main() 
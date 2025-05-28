"""
Dashboard Manager for Support Ticket Analytics.

This module contains functions to create and manage visualization dashboards.
"""

import streamlit as st
import pandas as pd
import numpy as np
import traceback
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.ui.app_components import debug
from src.visualization.app_visualizations import (
    create_ticket_volume_chart,
    create_resolution_time_chart,
    create_wordcloud,
    create_priority_distribution_chart,
    create_product_area_chart,
    create_csat_distribution_chart,
    create_root_cause_chart,
    create_root_cause_product_heatmap,
    create_first_response_time_chart,
    create_priority_time_chart,
    create_status_time_chart
)
from utils.ai_analysis import AIAnalyzer
from utils.time_analysis import calculate_first_response_time, calculate_sla_breaches

def display_visualization_dashboard(df: pd.DataFrame) -> None:
    """
    Display the main visualization dashboard.
    
    Args:
        df: DataFrame containing support ticket data
    """
    st.header("Visualizations")
    
    # Create tabs for different visualizations
    tabs = st.tabs([
        "Volume", "Resolution Time", "Priority", "Product Areas", "CSAT"
    ])
    
    # Volume tab
    with tabs[0]:
        st.subheader("Ticket Volume Over Time")
        time_unit = st.radio(
            "Time Aggregation",
            options=["day", "week", "month"],
            index=2,  # Default to month
            horizontal=True,
            key="overview_time_unit"
        )
        volume_chart = create_ticket_volume_chart(df, time_unit)
        st.plotly_chart(volume_chart, use_container_width=True, key="overview_volume_chart")
    
    # Resolution Time tab
    with tabs[1]:
        st.subheader("Resolution Time Analysis")
        resolution_chart = create_resolution_time_chart(df)
        st.plotly_chart(resolution_chart, use_container_width=True, key="overview_resolution_chart")
    
    # Priority tab
    with tabs[2]:
        st.subheader("Priority Distribution")
        priority_chart = create_priority_distribution_chart(df)
        st.plotly_chart(priority_chart, use_container_width=True, key="overview_priority_chart")
    
    # Product Areas tab
    with tabs[3]:
        st.subheader("Product Area Distribution")
        product_chart = create_product_area_chart(df)
        st.plotly_chart(product_chart, use_container_width=True, key="overview_product_chart")
    
    # CSAT tab
    with tabs[4]:
        st.subheader("Customer Satisfaction")
        if 'CSAT' in df.columns and not df.dropna(subset=['CSAT']).empty:
            csat_chart = create_csat_distribution_chart(df)
            st.plotly_chart(csat_chart, use_container_width=True, key="overview_csat_chart")
        else:
            st.info("No CSAT data available.")


def display_detailed_analysis(df: pd.DataFrame, enable_ai_analysis: bool = False, enable_pii_processing: bool = False) -> None:
    """
    Display detailed analysis of the support ticket data.
    
    Args:
        df: DataFrame containing support ticket data
        enable_ai_analysis: Whether to enable AI analysis
        enable_pii_processing: Whether to enable PII processing
    """
    st.header("Detailed Analysis")
    
    if df.empty:
        st.warning("No data available for analysis.")
        return
    
    # Add Case_Type__c if not present
    if 'Case_Type__c' not in df.columns:
        df['Case_Type__c'] = 'Unknown'
    
    # Create tabs for different analysis sections - removed duplicate chart tabs
    tabs = st.tabs([
        "⏳ Resolution Analysis", 
        "🗃️ Categorization", 
        "📝 Text Analysis",
        "🔧 Service Requests",
        "🧠 AI Insights"
    ])
    
    # Filter out Service Requests for main analysis
    main_analysis_df = df[df['Case_Type__c'] != 'Service Request'].copy()
    service_requests_df = df[df['Case_Type__c'] == 'Service Request'].copy()
    
    # Resolution Analysis Tab (enhanced version, not duplicating basic charts)
    with tabs[0]:
        if main_analysis_df.empty:
            st.warning("No regular support tickets found (excluding Service Requests).")
        else:
            # Add Priority Time Analysis (this is unique to detailed analysis)
            if 'history_df' in st.session_state and not st.session_state.history_df.empty:
                st.subheader("Priority Time Analysis (Closed Tickets Only)")
                priority_time_chart = create_priority_time_chart(main_analysis_df, st.session_state.history_df)
                st.plotly_chart(priority_time_chart, use_container_width=True, key="detailed_priority_time_chart")
                
                st.subheader("Status Time Analysis")
                status_time_chart = create_status_time_chart(main_analysis_df, st.session_state.history_df)
                st.plotly_chart(status_time_chart, use_container_width=True, key="detailed_status_time_chart")
            else:
                st.info("Case history data not available. Please enable 'Include History Data' in the sidebar and reload data to see Priority and Status Time Analysis.")
            
            # First Response Time Analysis
            st.subheader("First Response Time Analysis (Excluding Service Requests)")
            
            try:
                # Calculate first response times
                response_hours, stats = calculate_first_response_time(main_analysis_df, allow_synthetic=False)
                
                # Create and display first response time chart
                first_response_chart = create_first_response_time_chart(main_analysis_df, response_hours)
                st.plotly_chart(first_response_chart, use_container_width=True, key="detailed_response_time_chart")
                
                # Display SLA statistics if we have valid data
                if stats['valid_records'] > 0:
                    # Create columns for metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Response Time", f"{response_hours.mean():.2f} hours")
                    with col2:
                        st.metric("Median Response Time", f"{response_hours.median():.2f} hours")
                    with col3:
                        within_24h = (response_hours <= 24).mean() * 100
                        st.metric("Within 24h SLA", f"{within_24h:.1f}%")
                    
                    # Show SLA breach statistics
                    st.subheader("SLA Compliance (Excluding Service Requests)")
                    # Prioritize Internal_Priority__c if available
                    priority_col = next((col for col in ['Internal_Priority__c', 'Highest Priority', 'Highest_Priority', 'Priority'] 
                                        if col in main_analysis_df.columns), None)
                    if priority_col:
                        summary_stats = calculate_sla_breaches(response_hours, main_analysis_df[priority_col])
                        st.dataframe(summary_stats)
                else:
                    st.warning("No valid first response time data available for analysis.")
            except Exception as e:
                st.error(f"Error analyzing first response time: {str(e)}")
                if st.session_state.get('debug_mode', False):
                    st.exception(e)
                debug(f"Error in first response time analysis: {str(e)}", 
                     {'traceback': traceback.format_exc()}, category="error")
    
    # Categorization Tab
    with tabs[1]:
        if main_analysis_df.empty:
            st.warning("No regular support tickets found (excluding Service Requests).")
        else:
            st.subheader("Product Area Analysis (Excluding Service Requests)")
            
            if 'Product_Area__c' in main_analysis_df.columns:
                product_chart = create_product_area_chart(main_analysis_df)
                st.plotly_chart(product_chart, use_container_width=True, key="main_product_chart")
            else:
                st.info("Product area information not available in the data.")
            
            # Root Cause Analysis
            st.subheader("Root Cause Analysis (Excluding Service Requests)")
            
            if 'RCA__c' in main_analysis_df.columns:
                root_cause_chart = create_root_cause_chart(main_analysis_df)
                st.plotly_chart(root_cause_chart, use_container_width=True, key="main_root_cause_chart")
                
                # Add root cause by product area heatmap if both fields exist
                if 'Product_Area__c' in main_analysis_df.columns:
                    st.subheader("Root Cause by Product Area (Excluding Service Requests)")
                    heatmap = create_root_cause_product_heatmap(main_analysis_df)
                    st.plotly_chart(heatmap, use_container_width=True, key="main_heatmap_chart")
            else:
                st.info("Root cause information not available in the data.")
                
            # Status distribution
            st.subheader("Status Distribution (Excluding Service Requests)")
            if 'Status' in main_analysis_df.columns:
                # Use Plotly Express for a simple pie chart
                status_counts = main_analysis_df['Status'].value_counts().reset_index()
                status_counts.columns = ['Status', 'Count']
                
                fig = px.pie(
                    status_counts, 
                    values='Count', 
                    names='Status',
                    title="Ticket Status Distribution (Excluding Service Requests)",
                    color_discrete_sequence=px.colors.sequential.Viridis,
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True, key="main_status_pie_chart")
            else:
                st.info("Status information not available in the data.")
    
    # Text Analysis Tab
    with tabs[2]:
        if main_analysis_df.empty:
            st.warning("No regular support tickets found (excluding Service Requests).")
        else:
            st.subheader("Text Analysis (Excluding Service Requests)")
            
            # Select text column for analysis
            text_column = st.selectbox(
                "Select text field to analyze",
                options=['Subject', 'Description', 'Comments'],
                index=0,
                key="main_text_column"
            )
            
            # Process text if PII protection is enabled
            analysis_df = main_analysis_df.copy()
            if enable_pii_processing and text_column in analysis_df.columns:
                analysis_df, _ = st.session_state.pii_handler.process_dataframe(
                    analysis_df,
                    [text_column]
                )
                st.info("🔒 PII Protection is enabled. Text is masked for privacy.")
            
            # Create and display wordcloud
            if text_column in analysis_df.columns:
                st.subheader(f"Word Cloud: {text_column} (Excluding Service Requests)")
                wordcloud_fig = create_wordcloud(analysis_df, text_column)
                st.pyplot(wordcloud_fig)
            else:
                st.warning(f"{text_column} column not found in the data.")
    
    # Service Requests Tab
    with tabs[3]:
        if service_requests_df.empty:
            st.warning("No Service Request tickets found in the data.")
        else:
            st.markdown("## Service Request Analysis")
            st.info(f"Found {len(service_requests_df)} service request tickets in the data.")
            
            # Service Request Volume Analysis
            st.subheader("Service Request Volume")
            
            # Time unit selector for service requests
            sr_time_unit = st.radio(
                "Time Aggregation",
                options=["day", "week", "month"],
                index=2,  # Default to month
                horizontal=True,
                key="sr_time_unit"
            )
            
            # Ticket volume chart for service requests
            sr_volume_chart = create_ticket_volume_chart(service_requests_df, sr_time_unit)
            st.plotly_chart(sr_volume_chart, use_container_width=True, key="sr_volume_chart")
            
            # Service Request Resolution Time Analysis
            st.subheader("Service Request Resolution Time")
            
            # Calculate resolution time statistics
            closed_sr = service_requests_df[service_requests_df['Status'] == 'Closed'].copy()
            if not closed_sr.empty and 'Resolution Time (Days)' in closed_sr.columns:
                # Convert to numeric and ensure we have valid data
                closed_sr['Resolution Time (Days)'] = pd.to_numeric(closed_sr['Resolution Time (Days)'], errors='coerce')
                valid_resolution_data = closed_sr['Resolution Time (Days)'].dropna()
                
                if len(valid_resolution_data) > 0:
                    # Basic statistics
                    avg_resolution = valid_resolution_data.mean()
                    median_resolution = valid_resolution_data.median()
                    
                    # Display metrics
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Average Resolution Time", f"{avg_resolution:.1f} days")
                    with cols[1]:
                        st.metric("Median Resolution Time", f"{median_resolution:.1f} days")
                    with cols[2]:
                        pct_resolved_7d = (valid_resolution_data <= 7).mean() * 100
                        st.metric("Resolved Within 7 Days", f"{pct_resolved_7d:.1f}%")
                    
                    # Resolution time distribution
                    import plotly.figure_factory as ff
                    
                    try:
                        # Ensure we have a list of numeric values as a numpy array
                        resolution_data = valid_resolution_data.values
                        
                        if len(resolution_data) > 1:  # Need at least 2 data points for distribution
                            fig = ff.create_distplot(
                                [resolution_data], 
                                ['Service Requests'],
                                bin_size=1,
                                curve_type='normal',
                                show_rug=True
                            )
                            fig.update_layout(
                                title="Resolution Time Distribution for Service Requests",
                                xaxis_title="Resolution Time (Days)",
                                yaxis_title="Density",
                                showlegend=True
                            )
                            st.plotly_chart(fig, use_container_width=True, key="sr_dist_chart")
                        else:
                            st.warning("Not enough data points to create a distribution plot.")
                    except Exception as plot_error:
                        st.error(f"Could not create distribution plot: {str(plot_error)}")
                        
                        # Fallback to a simpler histogram
                        fig = px.histogram(
                            x=valid_resolution_data,
                            labels={"x": "Resolution Time (Days)"},
                            title="Resolution Time Histogram for Service Requests"
                        )
                        st.plotly_chart(fig, use_container_width=True, key="sr_hist_chart")
                    
                    # Compare with non-service requests
                    closed_main = main_analysis_df[main_analysis_df['Status'] == 'Closed'].copy()
                    if not closed_main.empty and 'Resolution Time (Days)' in closed_main.columns:
                        # Ensure numeric data for comparison
                        closed_main['Resolution Time (Days)'] = pd.to_numeric(closed_main['Resolution Time (Days)'], errors='coerce')
                        valid_main_data = closed_main['Resolution Time (Days)'].dropna()
                        
                        if len(valid_main_data) > 0:
                            # Create comparison chart
                            import plotly.graph_objects as go
                            
                            fig = go.Figure()
                            
                            # Add box plots
                            fig.add_trace(go.Box(
                                y=valid_resolution_data,
                                name='Service Requests',
                                boxmean=True,
                                marker_color='#2C6E49'
                            ))
                            
                            fig.add_trace(go.Box(
                                y=valid_main_data,
                                name='Support Tickets',
                                boxmean=True,
                                marker_color='#4D908E'
                            ))
                            
                            fig.update_layout(
                                title="Resolution Time Comparison",
                                yaxis_title="Resolution Time (Days)",
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key="sr_comparison_chart")
                            
                            # Calculate statistical comparison
                            sr_mean = valid_resolution_data.mean()
                            main_mean = valid_main_data.mean()
                            
                            if main_mean > 0:  # Avoid division by zero
                                diff_percent = ((sr_mean - main_mean) / main_mean) * 100
                                
                                st.info(
                                    f"On average, service requests take "
                                    f"{'longer' if diff_percent > 0 else 'less time'} to resolve "
                                    f"than regular support tickets by {abs(diff_percent):.1f}%."
                                )
                else:
                    st.warning("No valid resolution time data available for service requests.")
            else:
                st.warning("No closed service requests with resolution time data available.")
            
            # Priority Change Analysis for Service Requests
            st.subheader("Priority Change Analysis (Service Requests)")
            
            if 'history_df' in st.session_state and not st.session_state.history_df.empty:
                # Filter history data for Service Requests only
                sr_case_ids = service_requests_df['Id'].unique()
                
                # Check which column name is used for the case ID in history data
                case_id_column = None
                for col_name in ['ParentId', 'CaseId']:
                    if col_name in st.session_state.history_df.columns:
                        case_id_column = col_name
                        break
                
                if case_id_column is None:
                    st.error("Case ID column not found in history data. Expected 'ParentId' or 'CaseId'.")
                    sr_history_df = pd.DataFrame()
                else:
                    sr_history_df = st.session_state.history_df[st.session_state.history_df[case_id_column].isin(sr_case_ids)].copy()
                
                if not sr_history_df.empty:
                    sr_closed = service_requests_df[service_requests_df['Status'] == 'Closed'].copy()
                    if not sr_closed.empty:
                        priority_time_chart = create_priority_time_chart(sr_closed, sr_history_df)
                        st.plotly_chart(priority_time_chart, use_container_width=True, key="sr_priority_time_chart")
                        
                        # Compare with non-service requests
                        st.subheader("Priority Time Comparison")
                        main_closed = main_analysis_df[main_analysis_df['Status'] == 'Closed'].copy()
                        if not main_closed.empty:
                            # Create comparison visualization
                            import plotly.graph_objects as go
                            
                            # We need to calculate the time spent in each priority for both datasets
                            # This would require analyzing the history data for both types
                            # For simplicity, we'll just display the charts side by side
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Service Requests")
                                priority_sr_chart = create_priority_time_chart(sr_closed, sr_history_df)
                                st.plotly_chart(priority_sr_chart, use_container_width=True, key="sr_priority_chart_col1")
                            
                            with col2:
                                st.subheader("Support Tickets")
                                main_case_ids = main_closed['Id'].unique()
                                main_history_df = st.session_state.history_df[st.session_state.history_df[case_id_column].isin(main_case_ids)].copy()
                                priority_main_chart = create_priority_time_chart(main_closed, main_history_df)
                                st.plotly_chart(priority_main_chart, use_container_width=True, key="main_priority_chart_col2")
                    else:
                        st.info("No closed service requests available for priority change analysis.")
                else:
                    st.info("No history data available for service requests.")
            else:
                st.info("Case history data not available. Please enable 'Include History Data' in the sidebar and reload data to see Priority Change Analysis.")
            
            # Status Duration Analysis for Service Requests
            st.subheader("Status Duration Analysis (Service Requests)")
            
            if 'history_df' in st.session_state and not st.session_state.history_df.empty:
                # Filter history data for Service Requests only
                sr_case_ids = service_requests_df['Id'].unique()
                
                # Check which column name is used for the case ID in history data (reuse from above if already determined)
                if not 'case_id_column' in locals() or case_id_column is None:
                    case_id_column = None
                    for col_name in ['ParentId', 'CaseId']:
                        if col_name in st.session_state.history_df.columns:
                            case_id_column = col_name
                            break
                
                if case_id_column is None:
                    st.error("Case ID column not found in history data. Expected 'ParentId' or 'CaseId'.")
                    sr_history_df = pd.DataFrame()
                else:
                    sr_history_df = st.session_state.history_df[st.session_state.history_df[case_id_column].isin(sr_case_ids)].copy()
                
                if not sr_history_df.empty:
                    status_time_chart = create_status_time_chart(service_requests_df, sr_history_df)
                    st.plotly_chart(status_time_chart, use_container_width=True, key="sr_status_time_chart")
                    
                    # Identify bottlenecks
                    try:
                        # Calculate average time in each status
                        status_times = {}
                        status_field = 'Status' if 'Status' in sr_history_df.columns else 'Field'
                        old_value_field = 'OldValue' if 'OldValue' in sr_history_df.columns else 'Old_Value'
                        new_value_field = 'NewValue' if 'NewValue' in sr_history_df.columns else 'New_Value'
                        
                        status_changes = sr_history_df[
                            (sr_history_df[status_field] == 'Status') & 
                            (sr_history_df[old_value_field].notna()) & 
                            (sr_history_df[new_value_field].notna())
                        ].copy()
                        
                        if not status_changes.empty:
                            # Group by case to calculate time in each status
                            for case_id in sr_case_ids:
                                case_changes = status_changes[status_changes[case_id_column] == case_id].sort_values('CreatedDate')
                                if len(case_changes) > 1:
                                    for i in range(len(case_changes) - 1):
                                        current_status = case_changes.iloc[i][new_value_field]
                                        current_time = case_changes.iloc[i]['CreatedDate']
                                        next_time = case_changes.iloc[i+1]['CreatedDate']
                                        
                                        try:
                                            # Ensure timestamps are datetime objects, not strings
                                            if isinstance(current_time, str):
                                                current_time = pd.to_datetime(current_time)
                                            if isinstance(next_time, str):
                                                next_time = pd.to_datetime(next_time)
                                            
                                            duration_hours = (next_time - current_time).total_seconds() / 3600
                                            
                                            if current_status not in status_times:
                                                status_times[current_status] = []
                                            
                                            status_times[current_status].append(duration_hours)
                                        except Exception as e:
                                            st.warning(f"Error processing time data for status '{current_status}': {str(e)}")
                                            continue
                            
                            # Calculate averages and identify bottlenecks
                            bottlenecks = []
                            for status, times in status_times.items():
                                if times:
                                    avg_time = sum(times) / len(times)
                                    bottlenecks.append({
                                        'Status': status,
                                        'Average Hours': avg_time,
                                        'Average Days': avg_time / 24,
                                        'Count': len(times)
                                    })
                            
                            if bottlenecks:
                                # Sort by average time (descending)
                                bottlenecks = sorted(bottlenecks, key=lambda x: x['Average Hours'], reverse=True)
                                
                                # Display bottleneck analysis
                                st.subheader("Bottleneck Analysis")
                                bottleneck_df = pd.DataFrame(bottlenecks)
                                st.dataframe(bottleneck_df)
                                
                                # Create a visualization of the bottlenecks
                                fig = px.bar(
                                    bottleneck_df, 
                                    x='Status', 
                                    y='Average Days',
                                    color='Average Days',
                                    color_continuous_scale='Viridis',
                                    title="Average Time Spent in Each Status (Service Requests)"
                                )
                                fig.update_layout(
                                    xaxis_title="Status",
                                    yaxis_title="Average Time (Days)"
                                )
                                st.plotly_chart(fig, use_container_width=True, key="sr_bottleneck_chart")
                    except Exception as e:
                        st.error(f"Error analyzing status durations: {str(e)}")
                        debug(f"Error analyzing status durations: {str(e)}", 
                             {'traceback': traceback.format_exc()}, category="error")
                else:
                    st.info("No history data available for service requests.")
            else:
                st.info("Case history data not available. Please enable 'Include History Data' in the sidebar and reload data to see Status Duration Analysis.")
    
    # AI Insights Tab
    with tabs[4]:
        st.subheader("AI Analysis Insights")
        
        if enable_ai_analysis:
            # Initialize OpenAI client if not already done
            if 'openai_client' not in st.session_state:
                from openai import OpenAI
                st.session_state.openai_client = OpenAI()
            
            # Initialize AI analyzer if not already done
            if 'ai_analyzer' not in st.session_state:
                st.session_state.ai_analyzer = AIAnalyzer(st.session_state.openai_client)
            
            # Show AI analysis options
            analysis_type = st.radio(
                "Select Analysis Type",
                ["Ticket Categorization", "Pattern Detection", "Root Cause Analysis", "Trend Analysis"],
                horizontal=True
            )
            
            # Add option to include or exclude service requests
            include_sr = st.checkbox("Include Service Requests in Analysis", value=False)
            
            # Select dataset based on user choice
            analysis_dataset = df if include_sr else main_analysis_df
            
            if analysis_dataset.empty:
                st.warning("No data available for selected filters.")
                return
            
            # Run AI analysis
            with st.spinner("Running AI analysis..."):
                try:
                    # Process data for AI
                    from utils.text_processing import prepare_text_for_ai
                    
                    # Prepare a sample of data for AI processing
                    sample_size = min(100, len(analysis_dataset))  # Limit to 100 records for performance
                    sample_df = analysis_dataset.sample(n=sample_size) if len(analysis_dataset) > sample_size else analysis_dataset.copy()
                    
                    # Clean and prepare text
                    if enable_pii_processing:
                        sample_df, _ = st.session_state.pii_handler.process_dataframe(
                            sample_df,
                            ['Subject', 'Description', 'Comments']
                        )
                    
                    # Format data for AI
                    formatted_data = []
                    for _, row in sample_df.iterrows():
                        ticket_data = {
                            "case_number": row.get('Case Number', 'Unknown'),
                            "subject": row.get('Subject', ''),
                            "status": row.get('Status', 'Unknown'),
                            "priority": row.get('Priority', 'Unknown'),
                            "product_area": row.get('Product_Area__c', ''),
                            "case_type": row.get('Case_Type__c', ''),
                            "created_date": row.get('Created Date', '').strftime('%Y-%m-%d') if pd.notnull(row.get('Created Date', '')) else '',
                            "resolution_time": row.get('Resolution Time (Days)', None)
                        }
                        formatted_data.append(ticket_data)
                    
                    # Run analysis based on selected type
                    if analysis_type == "Ticket Categorization":
                        result = st.session_state.ai_analyzer.analyze_ticket_categories(formatted_data)
                    elif analysis_type == "Pattern Detection":
                        result = st.session_state.ai_analyzer.detect_patterns(formatted_data)
                    elif analysis_type == "Root Cause Analysis":
                        result = st.session_state.ai_analyzer.analyze_root_causes(formatted_data)
                    else:  # Trend Analysis
                        result = st.session_state.ai_analyzer.analyze_trends(formatted_data)
                    
                    # Display results
                    if result and 'analysis' in result:
                        st.markdown("### AI Analysis Results")
                        
                        # Main analysis
                        st.markdown(result['analysis'])
                        
                        # Recommendations if available
                        if 'recommendations' in result and result['recommendations']:
                            st.markdown("### Recommendations")
                            for i, rec in enumerate(result['recommendations'], 1):
                                st.markdown(f"{i}. {rec}")
                        
                        # Additional visualizations if available
                        if 'data' in result and result['data']:
                            st.markdown("### Data Insights")
                            
                            # Create visualization based on data
                            try:
                                # Try to create a simple bar chart if data is suitable
                                if isinstance(result['data'], dict) and all(isinstance(v, (int, float)) for v in result['data'].values()):
                                    # Bar chart for category counts or similar
                                    categories = list(result['data'].keys())
                                    values = list(result['data'].values())
                                    
                                    fig = px.bar(
                                        x=categories, 
                                        y=values,
                                        labels={'x': 'Category', 'y': 'Count'},
                                        title="AI Analysis Results",
                                        color=values,
                                        color_continuous_scale='viridis'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    # Just display the data as JSON
                                    st.json(result['data'])
                            except Exception as viz_error:
                                st.error(f"Error creating visualization: {str(viz_error)}")
                                st.json(result['data'])
                    else:
                        st.error("AI analysis returned no results.")
                
                except Exception as e:
                    st.error(f"Error running AI analysis: {str(e)}")
                    if st.session_state.get('debug_mode', False):
                        st.exception(e)
                    debug(f"Error in AI analysis: {str(e)}", 
                         {'traceback': traceback.format_exc()}, category="error")
        else:
            # Prompt to enable AI analysis
            st.info("""
            AI analysis is currently disabled. 
            
            Enable the "AI Analysis" option in the sidebar to get AI-powered insights about your support tickets.
            
            The AI analysis can help you:
            - Identify common themes and patterns
            - Detect root causes of issues
            - Analyze trends over time
            - Categorize tickets by underlying problems
            """) 
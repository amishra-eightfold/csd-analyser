"""Detailed visualization functions for the support ticket analysis dashboard."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
import os

# Import common helpers
from utils.visualization_helpers import truncate_string
from utils.time_analysis import calculate_first_response_time, calculate_sla_breaches
from config.logging_config import get_logger

# Initialize logger
logger = get_logger('detailed_viz')

# Custom color palettes
VIRIDIS_PALETTE = ["#440154", "#3B528B", "#21918C", "#5EC962", "#FDE725"]  # Viridis colors
AQUA_PALETTE = ["#E0F7FA", "#80DEEA", "#26C6DA", "#00ACC1", "#00838F", "#006064"]   # Material Cyan/Aqua
CSAT_PALETTE = AQUA_PALETTE  # Aqua palette for CSAT
PRIORITY_COLORS = {
    'P1': VIRIDIS_PALETTE[0],
    'P2': VIRIDIS_PALETTE[1],
    'P3': VIRIDIS_PALETTE[2],
    'P4': VIRIDIS_PALETTE[3]
}
ROOT_CAUSE_PALETTE = VIRIDIS_PALETTE
HEATMAP_PALETTE = "viridis"  # Viridis colorscale for heatmaps

def debug(message, data=None, category="app"):
    """Log debug information to the logger."""
    if hasattr(st.session_state, 'debug_logger'):
        st.session_state.debug_logger.log(message, data, category)
    
    # Log to file logger
    logger = get_logger(category)
    if data is not None:
        # Convert data to string if needed for logging
        if isinstance(data, dict):
            try:
                import json
                logger.info(f"{message} - {json.dumps(data)}")
            except:
                logger.info(f"{message} - {str(data)}")
        else:
            logger.info(f"{message} - {str(data)}")
    else:
        logger.info(message)

def display_detailed_analysis(df: pd.DataFrame, enable_ai_analysis: bool = False, enable_pii_processing: bool = False) -> None:
    """Display detailed analysis of support tickets.
    
    Args:
        df: DataFrame containing the support ticket data
        enable_ai_analysis: Whether to enable AI-powered analysis
        enable_pii_processing: Whether to enable PII protection during analysis
    """
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
            display_ai_analysis(df, enable_pii_processing)

        # Root Cause Analysis
        display_root_cause_analysis(df)
        
        # Resolution Time Analysis
        display_resolution_time_analysis(df)
        
        # CSAT Analysis
        display_csat_analysis(df)
        
        # First Response Time Analysis
        display_first_response_analysis(df)
        
    except Exception as e:
        error_msg = f"Error in detailed analysis: {str(e)}"
        st.error(error_msg)
        debug(error_msg, {'traceback': traceback.format_exc()}, category="error")
        logger.error(error_msg, exc_info=True)
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.exception(e)

def display_pattern_evolution(df: pd.DataFrame) -> None:
    """Display pattern evolution analysis.
    
    Args:
        df: DataFrame containing the support ticket data
    """
    try:
        # Validate required columns
        required_columns = ['Created Date', 'Resolution Time (Days)', 'CSAT', 'Root Cause', 'Priority']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.warning(f"Missing required columns for pattern analysis: {', '.join(missing_columns)}")
            return

        # Ensure data is not empty
        if df.empty:
            st.warning("No data available for pattern analysis.")
            return
            
        # Convert and validate numeric columns
        df['Resolution Time (Days)'] = pd.to_numeric(df['Resolution Time (Days)'], errors='coerce')
        df['CSAT'] = pd.to_numeric(df['CSAT'], errors='coerce')

        # Ensure Created Date is datetime
        df['Created Date'] = pd.to_datetime(df['Created Date'], errors='coerce')

        # Remove rows with invalid dates
        df = df.dropna(subset=['Created Date'])

        if len(df) == 0:
            st.warning("No valid data available after cleaning for pattern analysis.")
            return

        # Get analysis results using the visualizer
        results = st.session_state.pattern_visualizer.analyze_pattern_evolution(df)
        
        if not results or not isinstance(results, dict):
            st.warning("Pattern analysis produced no results.")
            return
            
        # Display visualizations if they exist
        if 'root_cause_evolution' in results:
            st.plotly_chart(results['root_cause_evolution'])
        
        if 'resolution_time_evolution' in results:
            st.plotly_chart(results['resolution_time_evolution'])
        
        if 'csat_evolution' in results:
            st.plotly_chart(results['csat_evolution'])
        
        # Display pattern statistics if they exist
        if 'pattern_stats' in results:
            st.write("### Pattern Analysis Summary")
            col1, col2 = st.columns(2)
            
            stats = results['pattern_stats']
            with col1:
                st.write("Overall Statistics:")
                st.write(f"- Total Tickets: {stats.get('total_tickets', 'N/A')}")
                st.write(f"- Unique Root Causes: {stats.get('unique_root_causes', 'N/A')}")
                if 'avg_resolution_time' in stats and pd.notna(stats['avg_resolution_time']):
                    st.write(f"- Average Resolution Time: {stats['avg_resolution_time']:.1f} days")
                else:
                    st.write("- Average Resolution Time: N/A")
            
            with col2:
                st.write("Trend Analysis:")
                trend_summary = stats.get('trend_summary', {})
                st.write(f"- Resolution Time Trend: {trend_summary.get('resolution_time', 'N/A')}")
                st.write(f"- CSAT Trend: {trend_summary.get('csat', 'N/A')}")
                if 'avg_csat' in stats and pd.notna(stats['avg_csat']):
                    st.write(f"- Average CSAT: {stats['avg_csat']:.2f}")
                else:
                    st.write("- Average CSAT: N/A")
        else:
            st.warning("No pattern statistics available.")
            
    except Exception as e:
        st.error(f"Error in pattern evolution analysis: {str(e)}")
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.exception(e)

def display_ai_analysis(df: pd.DataFrame, enable_pii_processing: bool = False) -> None:
    """Display AI-powered analysis of support tickets.
    
    Args:
        df: DataFrame containing the support ticket data
        enable_pii_processing: Whether to enable PII protection
    """
    from utils.ai_analysis import AIAnalyzer
    import os
    from openai import OpenAI
    
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
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                st.write("Generating AI insights...")
                st.write(f"Data shape: {analysis_df.shape}")
                st.write("Columns:", list(analysis_df.columns))
            
            insights = analyzer.analyze_tickets(analysis_df)
            
            # Display insights
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
                        metadata_display = [
                            f"- Analysis Timestamp: {meta.get('analysis_timestamp', 'Not available')}",
                            f"- Tickets Analyzed: {meta.get('tickets_analyzed', 0)} of {meta.get('total_tickets', 0)}",
                            f"- Chunks Processed: {meta.get('chunks_processed', 0)}",
                            f"- Patterns Detected: {meta.get('patterns_detected', 0)}"
                        ]
                        
                        if 'pattern_insights_generated' in meta:
                            metadata_display.append(f"- Pattern Insights Generated: {meta['pattern_insights_generated']}")
                            
                        st.markdown("\n".join(metadata_display))
            else:
                st.error("Failed to generate AI insights. Please check the logs for more information.")
                if 'error' in insights:
                    st.error(f"Error: {insights['error']}")
                    if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                        st.write("Debug information:")
                        st.write(insights)
        except Exception as e:
            st.error("Error generating AI insights. Please try again or contact support if the issue persists.")
            if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
                st.exception(e)
                st.write("Debug information:")
                st.write({
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                })

def display_root_cause_analysis(df: pd.DataFrame) -> None:
    """Display root cause analysis visualizations.
    
    Args:
        df: DataFrame containing the support ticket data
    """
    st.subheader("Root Cause Analysis")
    debug("Starting root cause analysis")
    
    try:
        # Filter for closed tickets
        closed_statuses = ['Closed', 'Resolved', 'Completed']
        closed_tickets_df = df[df['Status'].isin(closed_statuses)].copy()
        
        if closed_tickets_df.empty:
            st.warning("No closed tickets available for root cause analysis.")
            debug("No closed tickets available", category="warning")
            return
        
        root_cause_counts = closed_tickets_df['Root Cause'].value_counts()
        if not root_cause_counts.empty:
            fig_root_cause = go.Figure(data=[
                go.Bar(
                    x=root_cause_counts.index,
                    y=root_cause_counts.values,
                    marker_color=ROOT_CAUSE_PALETTE[0]
                )
            ])
            
            fig_root_cause.update_layout(
                title='Root Cause Distribution (Closed Tickets Only)',
                xaxis_title='Root Cause',
                yaxis_title='Number of Tickets',
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_root_cause)
            
            debug("Root cause analysis completed", {
                'total_closed_tickets': len(closed_tickets_df),
                'total_root_causes': len(root_cause_counts),
                'distribution': root_cause_counts.to_dict()
            })
            
            # Root cause by product area heatmap
            root_cause_product = pd.crosstab(
                closed_tickets_df['Root Cause'],
                closed_tickets_df['Product Area']
            )
            
            # Create a container for the heatmap
            with st.container():
                plt.figure(figsize=(12, 8))
                sns.heatmap(root_cause_product, annot=True, fmt='d', cmap=HEATMAP_PALETTE,
                           cbar_kws={'label': 'Ticket Count'})
                plt.title('Root Causes by Product Area (Closed Tickets Only)')
                plt.tight_layout()
                st.pyplot(plt)
                plt.close()
            
            # Resolution time by root cause
            resolution_by_root = closed_tickets_df.copy()
            resolution_by_root['Resolution_Time_Days'] = (
                resolution_by_root['Closed Date'] - resolution_by_root['Created Date']
            ).dt.total_seconds() / (24 * 3600)
            
            # Create a container for the resolution time plot
            with st.container():
                plt.figure(figsize=(14, 8))
                sns.boxplot(data=resolution_by_root, x='Root Cause', y='Resolution_Time_Days',
                           hue='Root Cause', palette=ROOT_CAUSE_PALETTE, legend=False)
                plt.title('Resolution Time by Root Cause (Closed Tickets Only)')
                plt.xlabel('Root Cause')
                plt.ylabel('Resolution Time (Days)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(plt)
                plt.close()
            
            debug("Root cause visualizations completed", {
                'total_closed_tickets': len(closed_tickets_df),
                'product_areas': len(root_cause_product.columns),
                'root_causes': len(root_cause_product.index),
                'avg_resolution_time': resolution_by_root['Resolution_Time_Days'].mean()
            })
        else:
            st.warning("No root cause data available for analysis.")
            debug("No root cause data available", category="warning")
    
    except Exception as e:
        st.error(f"Error in root cause analysis: {str(e)}")
        debug(f"Error in root cause analysis: {str(e)}", category="error")
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.exception(e)

def display_resolution_time_analysis(df: pd.DataFrame) -> None:
    """Display resolution time analysis visualizations.
    
    Args:
        df: DataFrame containing the support ticket data
    """
    st.subheader("Resolution Time Analysis")
    debug("Starting resolution time analysis")
    
    # Filter out Service Requests
    analysis_df = df[df['Case_Type__c'] != 'Service Request'].copy()
    
    # Calculate resolution time in days
    analysis_df['resolution_time_days'] = (
        analysis_df['Closed Date'] - analysis_df['Created Date']
    ).dt.total_seconds() / (24 * 3600)
    
    # Filter out tickets with invalid resolution times
    valid_resolution_df = analysis_df[
        (analysis_df['resolution_time_days'].notna()) & 
        (analysis_df['resolution_time_days'] > 0) &
        (analysis_df['Highest_Priority'].notna()) & 
        (~analysis_df['Highest_Priority'].isin(['Unspecified', '', ' ', None]))
    ]
    
    if len(valid_resolution_df) > 0:
        # Create box plot
        fig_resolution = go.Figure()
        
        # Get all unique priorities and sort them
        all_priorities = sorted(valid_resolution_df['Highest_Priority'].unique())
        
        for priority in all_priorities:
            priority_data = valid_resolution_df[valid_resolution_df['Highest_Priority'] == priority]
            if len(priority_data) > 0:
                fig_resolution.add_trace(go.Box(
                    y=priority_data['resolution_time_days'],
                    name=f'Priority {priority}',
                    marker_color=PRIORITY_COLORS.get(priority, VIRIDIS_PALETTE[0]),
                    boxpoints='outliers'
                ))
        
        fig_resolution.update_layout(
            title='Resolution Time Distribution by Priority',
            yaxis_title='Resolution Time (Days)',
            showlegend=True,
            boxmode='group'
        )
        
        st.plotly_chart(fig_resolution)
        
        # Display summary statistics
        st.write("### Resolution Time Summary")
        summary_stats = valid_resolution_df.groupby('Highest_Priority').agg({
            'resolution_time_days': ['count', 'mean', 'median']
        }).round(2)
        summary_stats.columns = ['Count', 'Mean Days', 'Median Days']
        st.write(summary_stats)
        
        debug("Resolution time analysis completed", {
            'total_cases': len(analysis_df),
            'valid_cases': len(valid_resolution_df),
            'priority_distribution': valid_resolution_df['Highest_Priority'].value_counts().to_dict()
        })
    else:
        st.warning("No valid resolution time data available for analysis.")
        debug("No valid resolution time data", category="warning")

def display_csat_analysis(df: pd.DataFrame) -> None:
    """Display customer satisfaction analysis visualizations.
    
    Args:
        df: DataFrame containing the support ticket data
    """
    st.subheader("Customer Satisfaction Analysis")
    debug("Starting CSAT analysis")
    
    valid_csat = df[df['CSAT'].notna()]
    if len(valid_csat) > 0:
        # Group by month and calculate CSAT metrics
        valid_csat['Month'] = valid_csat['Created Date'].dt.to_period('M')
        monthly_stats = valid_csat.groupby('Month').agg({
            'CSAT': ['mean', 'count']
        }).reset_index()
        
        # Flatten column names
        monthly_stats.columns = ['Month', 'Average CSAT', 'Response Count']
        
        # Convert Month to datetime and format as month name
        monthly_stats['Month'] = pd.PeriodIndex(monthly_stats['Month']).to_timestamp()
        monthly_stats = monthly_stats.sort_values('Month')
        monthly_stats['Month_Display'] = monthly_stats['Month'].dt.strftime('%b %Y')
        
        # Create figure with secondary Y-axis
        fig_csat = go.Figure()
        
        # Add CSAT bars
        fig_csat.add_trace(
            go.Bar(
                x=monthly_stats['Month_Display'],
                y=monthly_stats['Average CSAT'],
                name='Average CSAT',
                marker_color=CSAT_PALETTE[2],
                yaxis='y'
            )
        )
        
        # Add response count line
        fig_csat.add_trace(
            go.Scatter(
                x=monthly_stats['Month_Display'],
                y=monthly_stats['Response Count'],
                name='Response Count',
                marker_color=CSAT_PALETTE[4],
                yaxis='y2',
                mode='lines+markers'
            )
        )
        
        # Update layout for dual axes
        fig_csat.update_layout(
            title='Monthly CSAT Trends',
            xaxis_title='Month',
            yaxis_title='Average CSAT Score',
            yaxis2=dict(
                title='Number of Responses',
                overlaying='y',
                side='right'
            ),
            showlegend=True,
            xaxis_tickangle=-45,
            barmode='group',
            height=500,
            xaxis=dict(
                type='category',
                tickmode='array',
                ticktext=monthly_stats['Month_Display'],
                tickvals=monthly_stats['Month_Display']
            )
        )
        
        st.plotly_chart(fig_csat)
        
        # Display CSAT statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average CSAT", f"{valid_csat['CSAT'].mean():.2f}")
        with col2:
            st.metric("Median CSAT", f"{valid_csat['CSAT'].median():.2f}")
        with col3:
            response_rate = (len(valid_csat) / len(df)) * 100
            st.metric("Response Rate", f"{response_rate:.1f}%")
        
        debug("CSAT analysis completed", {
            'total_responses': len(valid_csat),
            'response_rate': response_rate,
            'avg_csat': valid_csat['CSAT'].mean()
        })
    else:
        st.warning("No CSAT data available for analysis.")
        debug("No CSAT data available", category="warning")

def display_first_response_analysis(df: pd.DataFrame) -> None:
    """Display first response time analysis visualizations.
    
    Args:
        df: DataFrame containing the support ticket data
    """
    st.subheader("First Response Time Analysis")
    debug("Starting first response time analysis")
    
    # Calculate response times
    response_hours, stats = calculate_first_response_time(df)
    
    if stats['valid_records'] > 0:
        # Create box plot for response times by priority
        fig_response = go.Figure()
        
        # Sort priorities in the correct order (P0 to P3)
        priorities = sorted(df['Highest_Priority'].unique())
        
        for priority in priorities:
            priority_mask = df['Highest_Priority'] == priority
            priority_data = response_hours[priority_mask].dropna()
            
            if len(priority_data) > 0:
                fig_response.add_trace(go.Box(
                    y=priority_data,
                    name=f'Priority {priority}',
                    marker_color=PRIORITY_COLORS.get(priority, VIRIDIS_PALETTE[0]),
                    boxpoints='outliers'
                ))
        
        fig_response.update_layout(
            title='First Response Time Distribution by Priority',
            yaxis_title='Response Time (Hours)',
            showlegend=True,
            boxmode='group'
        )
        
        st.plotly_chart(fig_response)
        
        # Display summary statistics
        st.write("### First Response Time Summary")
        
        # Calculate and display breach statistics
        summary_stats = calculate_sla_breaches(response_hours, df['Highest_Priority'])
        st.write(summary_stats)
        
        # Display SLA thresholds for reference
        st.info("First Response Time SLA Thresholds:\n" + "\n".join([
            "- P0: 1 hour",
            "- P1: 24 hours",
            "- P2: 48 hours",
            "- P3: No SLA"
        ]))
        
        # Display validation statistics if there were any issues
        if stats['invalid_records'] > 0:
            st.warning(
                f"⚠️ {stats['invalid_records']} records were excluded due to invalid response times. "
                f"This represents {(stats['invalid_records']/stats['total_records']*100):.1f}% of total records."
            )
            if stats['error_details']:
                with st.expander("See validation details"):
                    for detail in stats['error_details']:
                        st.write(f"- {detail}")
        
        debug("First response time analysis completed", {
            'total_tickets': stats['total_records'],
            'valid_tickets': stats['valid_records'],
            'invalid_tickets': stats['invalid_records']
        })
    else:
        st.warning("No valid first response time data available for analysis.")
        debug("No valid first response time data", category="warning") 
"""Dashboards module for CSD Analyzer."""

from typing import Dict, List, Any, Optional
import pandas as pd
import streamlit as st
import logging
from .charts import ChartGenerator
from ..analysis.ticket_analysis import TicketAnalyzer
from ..analysis.pattern_analysis import PatternAnalyzer

logger = logging.getLogger(__name__)

class DashboardGenerator:
    """Generates interactive dashboards for support data analysis."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize dashboard generator.
        
        Args:
            df (pd.DataFrame): DataFrame containing support ticket data
        """
        self.df = df
        self.chart_gen = ChartGenerator(df)
        self.ticket_analyzer = TicketAnalyzer(df)
        self.pattern_analyzer = PatternAnalyzer(df)
        
    def display_overview_dashboard(self):
        """Display overview dashboard with key metrics and trends."""
        try:
            st.title("Support Overview Dashboard")
            
            # Get basic metrics
            metrics = self.ticket_analyzer.get_basic_metrics()
            
            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Tickets",
                    f"{metrics['total_tickets']:,}",
                    help="Total number of support tickets"
                )
                
            with col2:
                st.metric(
                    "Avg Resolution Time",
                    f"{metrics['avg_resolution_time']:.1f} days",
                    help="Average time to resolve tickets"
                )
                
            with col3:
                st.metric(
                    "CSAT Score",
                    f"{metrics['avg_csat']:.2f}",
                    help="Average customer satisfaction score"
                )
                
            with col4:
                st.metric(
                    "Escalation Rate",
                    f"{metrics['escalation_rate']:.1%}",
                    help="Percentage of tickets escalated"
                )
            
            # Display charts
            st.subheader("Ticket Volume Trend")
            st.plotly_chart(
                self.chart_gen.create_ticket_volume_chart(),
                use_container_width=True
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Priority Distribution")
                st.plotly_chart(
                    self.chart_gen.create_priority_distribution_chart(),
                    use_container_width=True
                )
                
            with col2:
                st.subheader("Resolution Time by Priority")
                st.plotly_chart(
                    self.chart_gen.create_resolution_time_chart(),
                    use_container_width=True
                )
                
        except Exception as e:
            logger.error(f"Failed to display overview dashboard: {str(e)}")
            st.error("Error displaying overview dashboard")
            
    def display_pattern_dashboard(self):
        """Display pattern analysis dashboard."""
        try:
            st.title("Pattern Analysis Dashboard")
            
            # Get time patterns
            patterns = self.pattern_analyzer.analyze_time_patterns()
            
            # Display time patterns
            st.subheader("Temporal Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Daily Distribution")
                daily = patterns.get('daily', {})
                if daily:
                    st.write(f"Peak Hours: {', '.join(map(str, daily['peak_hours']))}")
                    st.write(f"Average Tickets per Hour: {daily['avg_tickets_per_hour']:.1f}")
                    
            with col2:
                st.write("Weekly Distribution")
                weekly = patterns.get('weekly', {})
                if weekly:
                    st.write(f"Busiest Days: {', '.join(weekly['busy_days'])}")
                    st.write(f"Average Tickets per Day: {weekly['avg_tickets_per_day']:.1f}")
            
            # Display correlation matrix
            st.subheader("Metric Correlations")
            st.plotly_chart(
                self.chart_gen.create_correlation_matrix(),
                use_container_width=True
            )
            
            # Display anomalies
            st.subheader("Anomaly Detection")
            anomalies = self.pattern_analyzer.identify_anomalies()
            
            if anomalies:
                for anomaly in anomalies:
                    if anomaly['type'] == 'resolution_time':
                        st.warning(
                            f"Unusual resolution time detected for case {anomaly['case_number']}: "
                            f"{anomaly['value']:.1f} days (z-score: {anomaly['z_score']:.2f})"
                        )
                    else:  # volume anomaly
                        st.warning(
                            f"Unusual ticket volume on {anomaly['date'].strftime('%Y-%m-%d')}: "
                            f"{anomaly['count']} tickets (expected: {anomaly['expected']:.1f})"
                        )
            else:
                st.info("No significant anomalies detected")
                
            # Display issue clusters
            st.subheader("Issue Clusters")
            clusters = self.pattern_analyzer.identify_issue_clusters()
            
            if clusters:
                for cluster in clusters:
                    with st.expander(f"Cluster {cluster['cluster_id']} ({cluster['size']} tickets)"):
                        st.write("Common Keywords:", ", ".join(cluster['keywords']))
                        st.write(f"Average Resolution Time: {cluster['avg_resolution_time']:.1f} days")
                        st.write(f"Average CSAT: {cluster['avg_csat']:.2f}")
                        
                        if st.checkbox(f"Show tickets in cluster {cluster['cluster_id']}"):
                            st.dataframe(
                                self.df[self.df['CaseNumber'].isin(cluster['tickets'])][
                                    ['CaseNumber', 'Subject', 'Status', 'Priority']
                                ]
                            )
            else:
                st.info("No significant issue clusters identified")
                
        except Exception as e:
            logger.error(f"Failed to display pattern dashboard: {str(e)}")
            st.error("Error displaying pattern dashboard")
            
    def display_product_dashboard(self):
        """Display product area analysis dashboard."""
        try:
            st.title("Product Analysis Dashboard")
            
            # Add metric selector
            metric = st.selectbox(
                "Select Analysis Metric",
                ["count", "resolution_time", "csat"],
                format_func=lambda x: {
                    "count": "Ticket Volume",
                    "resolution_time": "Resolution Time",
                    "csat": "CSAT Score"
                }[x]
            )
            
            # Display product area chart
            st.plotly_chart(
                self.chart_gen.create_product_area_chart(metric=metric),
                use_container_width=True
            )
            
            # Get patterns by product area
            patterns = self.pattern_analyzer.identify_patterns()
            
            if patterns:
                # Display product area correlations
                st.subheader("Product Area Insights")
                
                for area, metrics in patterns.get('correlations', {}).items():
                    with st.expander(f"Analysis for {area}"):
                        for metric_name, value in metrics.items():
                            st.write(f"{metric_name}: {value:.2f}")
                            
                # Display feature analysis
                st.subheader("Feature Analysis")
                features = patterns.get('features', {})
                
                if features:
                    feature_df = pd.DataFrame(
                        features.items(),
                        columns=['Feature', 'Ticket Count']
                    ).sort_values('Ticket Count', ascending=False)
                    
                    st.dataframe(
                        feature_df,
                        use_container_width=True
                    )
                    
                # Display root cause analysis
                st.subheader("Root Cause Analysis")
                root_causes = patterns.get('root_causes', {})
                
                if root_causes:
                    cause_df = pd.DataFrame(
                        root_causes.items(),
                        columns=['Root Cause', 'Ticket Count']
                    ).sort_values('Ticket Count', ascending=False)
                    
                    st.dataframe(
                        cause_df,
                        use_container_width=True
                    )
                    
        except Exception as e:
            logger.error(f"Failed to display product dashboard: {str(e)}")
            st.error("Error displaying product dashboard")
            
    def display_csat_dashboard(self):
        """Display CSAT analysis dashboard."""
        try:
            st.title("Customer Satisfaction Dashboard")
            
            # Display CSAT trend
            st.subheader("CSAT Score Trend")
            
            # Add time grouping selector
            grouping = st.selectbox(
                "Select Time Grouping",
                ["D", "W", "M"],
                format_func=lambda x: {
                    "D": "Daily",
                    "W": "Weekly",
                    "M": "Monthly"
                }[x]
            )
            
            st.plotly_chart(
                self.chart_gen.create_csat_trend_chart(grouping=grouping),
                use_container_width=True
            )
            
            # Display CSAT by different dimensions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("CSAT by Priority")
                st.plotly_chart(
                    self.chart_gen.create_resolution_time_chart(
                        by='Priority',
                        title='Average CSAT by Priority'
                    ),
                    use_container_width=True
                )
                
            with col2:
                st.subheader("CSAT by Product Area")
                st.plotly_chart(
                    self.chart_gen.create_product_area_chart(
                        metric='csat',
                        title='Average CSAT by Product Area'
                    ),
                    use_container_width=True
                )
                
            # Display CSAT statistics
            st.subheader("CSAT Statistics")
            
            stats_df = pd.DataFrame({
                'Metric': [
                    'Average CSAT',
                    'Median CSAT',
                    'CSAT Standard Deviation',
                    'Percentage of High Satisfaction (>4)',
                    'Percentage of Low Satisfaction (<2)'
                ],
                'Value': [
                    f"{self.df['CSAT'].mean():.2f}",
                    f"{self.df['CSAT'].median():.2f}",
                    f"{self.df['CSAT'].std():.2f}",
                    f"{(self.df['CSAT'] > 4).mean():.1%}",
                    f"{(self.df['CSAT'] < 2).mean():.1%}"
                ]
            })
            
            st.dataframe(
                stats_df,
                use_container_width=True
            )
            
        except Exception as e:
            logger.error(f"Failed to display CSAT dashboard: {str(e)}")
            st.error("Error displaying CSAT dashboard")

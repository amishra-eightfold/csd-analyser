"""Display module for CSD Analyzer analysis results."""
from typing import Dict, Any, List
import pandas as pd
from ..visualization.salesforce_visualizer import SalesforceVisualizer

def display_detailed_analysis(analysis_results: Dict[str, Any], cases_df: pd.DataFrame) -> None:
    """Display detailed analysis results with visualizations.
    
    Args:
        analysis_results: Dictionary containing analysis results from the AI
        cases_df: DataFrame containing the case data for visualization
    """
    # Initialize visualizer
    visualizer = SalesforceVisualizer(cases_df)
    
    # Display executive summary
    if 'executive_summary' in analysis_results:
        print("\n=== Executive Summary ===")
        print(analysis_results['executive_summary'])
    
    # Display key metrics and trends
    if 'key_metrics' in analysis_results:
        print("\n=== Key Metrics and Trends ===")
        for metric, value in analysis_results['key_metrics'].items():
            print(f"{metric}: {value}")
    
    # Display visualizations
    print("\n=== Visual Analysis ===")
    
    # Case volume over time
    visualizer.plot_case_volume(title="Case Volume Trends")
    
    # Priority distribution
    visualizer.plot_priority_distribution(title="Case Priority Distribution")
    
    # CSAT distribution if available
    if 'CSAT__c' in cases_df.columns:
        visualizer.plot_csat_distribution(title="Customer Satisfaction Scores")
    
    # Response times if available
    if 'First_Response_Time__c' in cases_df.columns:
        visualizer.plot_response_times(title="First Response Time Distribution")
    
    # Product area distribution
    visualizer.plot_product_areas(title="Cases by Product Area")
    
    # Display recommendations
    if 'recommendations' in analysis_results:
        print("\n=== Recommendations ===")
        for i, rec in enumerate(analysis_results['recommendations'], 1):
            print(f"{i}. {rec}")
    
    # Display risk analysis
    if 'risk_analysis' in analysis_results:
        print("\n=== Risk Analysis ===")
        for risk in analysis_results['risk_analysis']:
            print(f"- {risk}")
    
    # Display action items
    if 'action_items' in analysis_results:
        print("\n=== Action Items ===")
        for i, item in enumerate(analysis_results['action_items'], 1):
            print(f"{i}. {item}") 
"""AI-powered insights generation for CSD Analyzer."""
from typing import Dict, Any, List
import pandas as pd
from ..core.openai_client import openai
from utils.token_manager import TokenManager
from utils.pattern_recognition import PatternRecognizer

def generate_ai_insights(
    cases_df: pd.DataFrame,
    comments_df: pd.DataFrame = None,
    emails_df: pd.DataFrame = None,
    include_patterns: bool = True,
    include_trends: bool = True
) -> Dict[str, Any]:
    """Generate AI-powered insights from case data.
    
    Args:
        cases_df: DataFrame containing case data
        comments_df: Optional DataFrame containing case comments
        emails_df: Optional DataFrame containing email messages
        include_patterns: Whether to include pattern analysis
        include_trends: Whether to include trend analysis
    
    Returns:
        Dictionary containing various insights and analyses
    """
    # Initialize components
    token_manager = TokenManager()
    pattern_recognizer = PatternRecognizer()
    
    # Prepare data for analysis
    analysis_data = {
        'cases': cases_df.to_dict('records'),
        'comments': comments_df.to_dict('records') if comments_df is not None else [],
        'emails': emails_df.to_dict('records') if emails_df is not None else []
    }
    
    # Extract metadata context
    context = token_manager.extract_metadata_context(analysis_data['cases'])
    
    # Create chunks for analysis
    chunks = token_manager.create_chunks(analysis_data['cases'])
    
    # Initialize results
    insights = {
        'executive_summary': {
            'key_findings': [],
            'metrics': {}
        },
        'pattern_insights': {
            'recurring_issues': [],
            'confidence_levels': {
                'high_confidence': [],
                'medium_confidence': [],
                'low_confidence': []
            }
        },
        'trend_analysis': {
            'pattern_evolution': [],
            'seasonal_patterns': [],
            'anomalies': []
        },
        'recommendations': [],
        'customer_impact_analysis': '',
        'next_steps': [],
        'metadata': {
            'version': '1.0',
            'analysis_coverage': {
                'cases': len(cases_df),
                'comments': len(comments_df) if comments_df is not None else 0,
                'emails': len(emails_df) if emails_df is not None else 0
            }
        }
    }
    
    try:
        # Process each chunk
        for chunk in chunks:
            # Generate insights for the chunk
            chunk_insights = openai.analyze_chunk(chunk['items'], chunk['context'])
            
            # Update executive summary
            if 'key_findings' in chunk_insights:
                insights['executive_summary']['key_findings'].extend(chunk_insights['key_findings'])
            
            # Update pattern insights
            if include_patterns and 'patterns' in chunk_insights:
                for pattern in chunk_insights['patterns']:
                    confidence = pattern.get('confidence', 'medium_confidence')
                    insights['pattern_insights']['confidence_levels'][confidence].append(pattern)
            
            # Update recommendations
            if 'recommendations' in chunk_insights:
                insights['recommendations'].extend(chunk_insights['recommendations'])
        
        # Perform pattern recognition if requested
        if include_patterns:
            pattern_results = pattern_recognizer.analyze_patterns(cases_df)
            insights['pattern_insights']['recurring_issues'] = pattern_results['recurring_issues']
            insights['pattern_insights'].update(pattern_results.get('additional_insights', {}))
        
        # Perform trend analysis if requested
        if include_trends:
            trend_results = pattern_recognizer.analyze_trends(cases_df)
            insights['trend_analysis'].update(trend_results)
        
        # Generate customer impact analysis
        impact_analysis = openai.analyze_customer_impact(
            cases_df.to_dict('records'),
            context.get('customer_stats', {})
        )
        insights['customer_impact_analysis'] = impact_analysis.get('impact_summary', '')
        insights['next_steps'] = impact_analysis.get('recommended_actions', [])
        
        # Update metadata
        insights['metadata']['analysis_timestamp'] = pd.Timestamp.now().isoformat()
        insights['metadata']['analysis_status'] = 'success'
        
    except Exception as e:
        # Handle errors gracefully
        insights['metadata']['analysis_status'] = 'error'
        insights['metadata']['error_message'] = str(e)
        insights['executive_summary']['key_findings'] = [f"Analysis failed: {str(e)}"]
    
    return insights 
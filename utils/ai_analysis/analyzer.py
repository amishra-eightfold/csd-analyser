"""AI Analyzer for support ticket analysis."""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json
from openai import OpenAI
from collections import defaultdict
import traceback

from ..token_manager import TokenManager, convert_value_for_json
from ..text_processing import prepare_text_for_ai
from ..pattern_recognition import PatternRecognizer, PatternAnalyzer
from config.logging_config import get_logger

from .context_manager import ContextManager
from .openai_logger import OpenAILogger
from .utils import preprocess_text_for_ai, convert_confidence_to_float, calculate_ticket_importance, prepare_patterns_dict

class AIAnalyzer:
    """
    Enhanced AI analysis for support tickets.
    
    This class leverages OpenAI APIs to analyze support ticket data,
    identify patterns, and generate insights.
    """
    
    def __init__(self, client: OpenAI) -> None:
        """
        Initialize the AI analyzer.
        
        Args:
            client: OpenAI client instance
        """
        self.client = client
        self.token_manager = TokenManager()
        self.context_manager = ContextManager()
        self.pattern_detector = PatternRecognizer()
        self.pattern_analyzer = PatternAnalyzer()
        self.logger = get_logger('app')
        self.openai_logger = OpenAILogger()
        
        # Initialize pattern tracking
        self.detected_patterns = []
        self.pattern_confidences = defaultdict(float)
        self.pattern_frequencies = defaultdict(int)
        
    def _track_pattern(self, pattern: str, confidence: float = 0.0) -> None:
        """
        Track a detected pattern with its confidence score.
        
        Args:
            pattern: The detected pattern string
            confidence: Confidence score (0-1) for the pattern
        """
        if pattern not in self.detected_patterns:
            self.detected_patterns.append(pattern)
        self.pattern_confidences[pattern] = max(self.pattern_confidences[pattern], confidence)
        self.pattern_frequencies[pattern] += 1
        
    def _get_pattern_summary(self) -> Dict[str, Any]:
        """
        Get summary of tracked patterns with confidence levels.
        
        Returns:
            Dict: Summary of patterns with confidence levels
        """
        return {
            'patterns': self.detected_patterns,
            'confidences': dict(self.pattern_confidences),
            'frequencies': dict(self.pattern_frequencies),
            'total_patterns': len(self.detected_patterns),
            'high_confidence': [p for p, c in self.pattern_confidences.items() if c >= 0.8],
            'medium_confidence': [p for p, c in self.pattern_confidences.items() if 0.5 <= c < 0.8],
            'low_confidence': [p for p, c in self.pattern_confidences.items() if c < 0.5]
        }

    def _prepare_chunk_prompt(self, chunk_data: List[Dict], context: Dict[str, Any]) -> str:
        """
        Prepare a concise prompt for chunk analysis.
        
        Args:
            chunk_data: List of ticket data dictionaries
            context: Context information for analysis
            
        Returns:
            str: Formatted prompt for OpenAI
        """
        # Convert and trim context data
        serializable_context = convert_value_for_json(context)
        # Keep only essential context
        trimmed_context = {
            'current_patterns': serializable_context.get('current_patterns', {}),
            'previous_insights': serializable_context.get('previous_insights', [])[-2:]  # Keep only last 2 insights
        }
        
        # Convert and trim chunk data
        serializable_chunk_data = []
        for ticket in chunk_data:
            # Keep only essential ticket data
            serializable_chunk_data.append({
                'case_number': ticket.get('case_number', ''),
                'subject': ticket.get('subject', '')[:100],  # Truncate long subjects
                'priority': ticket.get('priority', ''),
                'status': ticket.get('status', ''),
                'product_area': ticket.get('product_area', ''),
                'resolution_time': ticket.get('resolution_time', 'N/A')
            })
        
        # Create the prompt
        prompt = f"""
Analyze this batch of support tickets and identify patterns, themes, and insights.

Ticket Data: {json.dumps(serializable_chunk_data, indent=2)}

Context from Previous Analysis: {json.dumps(trimmed_context, indent=2)}

For each pattern you identify, include the following:
1. Pattern: [pattern description]
2. Supporting Cases: [list relevant case numbers]
3. Confidence: [high/medium/low]
4. Impact: [description of the impact]

Also provide:
- Overall assessment of themes in this batch
- Common product areas or features involved
- Priority distribution insights
- Resolution time insights

Format your response as a JSON with the following structure:
{{
  "patterns": [
    {{
      "pattern": "string",
      "cases": ["case1", "case2"],
      "confidence": "high/medium/low",
      "impact": "string"
    }}
  ],
  "themes": ["string"],
  "product_insights": "string",
  "priority_insights": "string",
  "resolution_insights": "string",
  "overall_assessment": "string"
}}
"""
        return prompt

    def _analyze_chunk(self, chunk_data: List[Dict], context: Dict) -> Dict[str, Any]:
        """
        Analyze a chunk of ticket data.
        
        Args:
            chunk_data: List of ticket data dictionaries to analyze
            context: Context information for analysis
            
        Returns:
            Dict: Analysis results for the chunk
        """
        if not chunk_data:
            return {
                "patterns": [],
                "themes": [],
                "product_insights": "No data provided",
                "priority_insights": "No data provided",
                "resolution_insights": "No data provided",
                "overall_assessment": "No data provided"
            }
        
        try:
            # Prepare prompt
            prompt = self._prepare_chunk_prompt(chunk_data, context)
            
            # Call OpenAI API
            request_data = {
                "model": "gpt-4-turbo",
                "messages": [
                    {"role": "system", "content": "You are an expert support ticket analyzer. Your task is to identify patterns and insights from support ticket data."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"}
            }
            
            # Log the request
            self.logger.info(f"Sending chunk analysis request for {len(chunk_data)} tickets")
            
            response = self.client.chat.completions.create(**request_data)
            
            # Process response
            try:
                response_content = response.choices[0].message.content
                analysis_result = json.loads(response_content)
                
                # Log the interaction
                self.openai_logger.log_interaction(
                    request_data,
                    {"content": response_content},
                    "chunk_analysis"
                )
                
                # Track patterns
                for pattern_item in analysis_result.get("patterns", []):
                    pattern_name = pattern_item.get("pattern", "")
                    if pattern_name:
                        confidence = convert_confidence_to_float(pattern_item.get("confidence", "medium"))
                        self._track_pattern(pattern_name, confidence)
                
                return analysis_result
            except json.JSONDecodeError:
                self.logger.error("Failed to parse JSON response from OpenAI")
                # Return a minimal result if parsing fails
                return {
                    "patterns": [],
                    "themes": [],
                    "product_insights": "Error in analysis",
                    "priority_insights": "Error in analysis",
                    "resolution_insights": "Error in analysis", 
                    "overall_assessment": "Error in analysis: Invalid JSON response"
                }
        except Exception as e:
            self.logger.error(f"Error in chunk analysis: {str(e)}")
            # Return a minimal result on error
            return {
                "patterns": [],
                "themes": [],
                "product_insights": "Error in analysis",
                "priority_insights": "Error in analysis",
                "resolution_insights": "Error in analysis",
                "overall_assessment": f"Error in analysis: {str(e)}"
            }

    def _prepare_final_prompt(self, all_insights: List[Dict], summary_stats: Dict, pattern_analysis: Dict) -> str:
        """
        Prepare prompt for final analysis summarization.
        
        Args:
            all_insights: List of insights from chunk analyses
            summary_stats: Statistical summary of the data
            pattern_analysis: Pattern analysis results
            
        Returns:
            str: Formatted prompt for OpenAI
        """
        # Prepare patterns section with structured format
        patterns_data = prepare_patterns_dict(
            self.detected_patterns,
            self.pattern_confidences,
            self.pattern_frequencies
        )
        
        # Sort patterns by importance (confidence * frequency)
        sorted_patterns = sorted(
            patterns_data.items(), 
            key=lambda x: x[1]['importance'], 
            reverse=True
        )
        
        # Take top 10 patterns to avoid token limits
        top_patterns = dict(sorted_patterns[:10])
        
        # Prepare the prompt
        prompt = f"""
Analyze these support ticket insights and provide a comprehensive summary.

Summary Statistics:
{json.dumps(summary_stats, indent=2)}

Top Patterns Detected:
{json.dumps(top_patterns, indent=2)}

Pattern Analysis:
{json.dumps(pattern_analysis, indent=2)}

Based on the data and patterns above, provide:

1. A concise summary of key findings
2. Major patterns identified across tickets
3. Recommendations for improving customer support
4. Product areas that need attention
5. Any notable trends in resolution times or CSAT

Format your response as a JSON with the following structure:
{{
  "analysis": "string", // Main analysis text
  "key_patterns": ["string"], // List of most significant patterns
  "recommendations": ["string"], // List of specific recommendations
  "problem_areas": ["string"], // List of product areas needing attention
  "trends": ["string"], // List of notable trends
  "data": {{}} // Any structured data to support your findings
}}
"""
        return prompt

    def analyze_tickets(self, analysis_df: pd.DataFrame, max_tickets: int = 100) -> Dict[str, Any]:
        """
        Analyze support tickets to generate insights.
        
        Args:
            analysis_df: DataFrame containing ticket data
            max_tickets: Maximum number of tickets to analyze
            
        Returns:
            Dict: Analysis results with insights and recommendations
        """
        if analysis_df.empty:
            return {
                "analysis": "No data available for analysis",
                "key_patterns": [],
                "recommendations": [],
                "problem_areas": [],
                "trends": []
            }
        
        try:
            self.logger.info(f"Starting AI analysis with {len(analysis_df)} tickets")
            
            # Reset pattern tracking for new analysis
            self.detected_patterns = []
            self.pattern_confidences.clear()
            self.pattern_frequencies.clear()
            
            # Limit number of tickets for analysis
            if len(analysis_df) > max_tickets:
                # Sample tickets with priority to more important tickets
                ticket_importances = []
                
                for _, row in analysis_df.iterrows():
                    ticket_data = {
                        'priority': row.get('Priority', ''),
                        'created_date': str(row.get('Created Date', '')),
                        'csat': row.get('CSAT', None)
                    }
                    importance = calculate_ticket_importance(ticket_data)
                    ticket_importances.append(importance)
                
                # Sample with higher probability for important tickets
                sample_probs = np.array(ticket_importances) / sum(ticket_importances)
                sample_indices = np.random.choice(
                    len(analysis_df), 
                    size=max_tickets, 
                    replace=False, 
                    p=sample_probs
                )
                analysis_df = analysis_df.iloc[sample_indices].copy()
            
            # Format data for AI analysis
            formatted_data = []
            for _, row in analysis_df.iterrows():
                # Clean and format text fields
                subject = preprocess_text_for_ai(row.get('Subject', ''))
                description = preprocess_text_for_ai(row.get('Description', ''))
                
                ticket_data = {
                    "case_number": row.get('Case Number', 'Unknown'),
                    "subject": subject,
                    "description": description[:500],  # Limit description length
                    "status": row.get('Status', 'Unknown'),
                    "priority": row.get('Priority', 'Unknown'),
                    "product_area": row.get('Product Area', ''),
                    "created_date": row.get('Created Date', '').strftime('%Y-%m-%d') if pd.notnull(row.get('Created Date', '')) else '',
                    "resolution_time": row.get('Resolution Time (Days)', None)
                }
                formatted_data.append(ticket_data)
            
            # Split into chunks for analysis
            chunk_size = 20  # Analyze 20 tickets at a time
            chunks = [formatted_data[i:i+chunk_size] for i in range(0, len(formatted_data), chunk_size)]
            
            self.logger.info(f"Split data into {len(chunks)} chunks for analysis")
            
            # Analyze each chunk
            all_insights = []
            
            for i, chunk in enumerate(chunks):
                self.logger.info(f"Analyzing chunk {i+1}/{len(chunks)}")
                
                # Get current context
                current_context = {
                    'current_patterns': self._get_pattern_summary(),
                    'previous_insights': all_insights
                }
                
                # Analyze chunk
                chunk_result = self._analyze_chunk(chunk, current_context)
                all_insights.append(chunk_result)
                
                # Update context manager
                if 'patterns' in chunk_result:
                    pattern_dict = {p.get('pattern', ''): 1 for p in chunk_result.get('patterns', [])}
                    self.context_manager.update_patterns(pattern_dict)
                
                self.context_manager.add_insight(chunk_result)
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_stats(analysis_df)
            
            # Analyze patterns
            pattern_analysis = self.pattern_analyzer.analyze_patterns(analysis_df)
            
            # Create final summary prompt
            final_prompt = self._prepare_final_prompt(all_insights, summary_stats, pattern_analysis)
            
            # Get final summary from OpenAI
            request_data = {
                "model": "gpt-4-turbo",
                "messages": [
                    {"role": "system", "content": "You are an expert support ticket analyzer. Provide a comprehensive summary of support ticket analysis."},
                    {"role": "user", "content": final_prompt}
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"}
            }
            
            response = self.client.chat.completions.create(**request_data)
            
            # Process final response
            try:
                response_content = response.choices[0].message.content
                final_result = json.loads(response_content)
                
                # Log the interaction
                self.openai_logger.log_interaction(
                    request_data,
                    {"content": response_content},
                    "final_analysis"
                )
                
                return final_result
            except json.JSONDecodeError:
                self.logger.error("Failed to parse final analysis JSON response")
                return {
                    "analysis": "Analysis completed but encountered an error formatting the results.",
                    "key_patterns": self.detected_patterns[:5],
                    "recommendations": ["Contact support team for assistance with analysis."],
                    "problem_areas": [],
                    "trends": []
                }
                
        except Exception as e:
            self.logger.error(f"Error in AI analysis: {str(e)}")
            # Return error details
            return {
                "analysis": f"Error during analysis: {str(e)}",
                "key_patterns": [],
                "recommendations": ["Check logs for error details and try again with fewer tickets."],
                "problem_areas": [],
                "trends": []
            }
    
    def _calculate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary statistics for the dataset.
        
        Args:
            df: DataFrame containing ticket data
            
        Returns:
            Dict: Summary statistics
        """
        stats = {}
        
        try:
            # Total tickets
            stats['total_tickets'] = len(df)
            
            # Status distribution
            if 'Status' in df.columns:
                status_counts = df['Status'].value_counts().to_dict()
                stats['status_distribution'] = status_counts
                stats['open_tickets'] = status_counts.get('Open', 0)
                stats['closed_tickets'] = status_counts.get('Closed', 0)
            
            # Priority distribution
            if 'Priority' in df.columns:
                stats['priority_distribution'] = df['Priority'].value_counts().to_dict()
            
            # Product area distribution
            if 'Product Area' in df.columns:
                # Limit to top 5 areas to avoid too much data
                top_areas = df['Product Area'].value_counts().head(5).to_dict()
                stats['product_areas'] = top_areas
            
            # Resolution time stats
            if 'Resolution Time (Days)' in df.columns:
                resolution_times = df['Resolution Time (Days)'].dropna()
                if len(resolution_times) > 0:
                    stats['avg_resolution_time'] = float(resolution_times.mean())
                    stats['median_resolution_time'] = float(resolution_times.median())
                    stats['min_resolution_time'] = float(resolution_times.min())
                    stats['max_resolution_time'] = float(resolution_times.max())
            
            # CSAT stats
            if 'CSAT' in df.columns:
                csat_values = df['CSAT'].dropna()
                if len(csat_values) > 0:
                    stats['avg_csat'] = float(csat_values.mean())
                    stats['csat_distribution'] = {
                        str(i): int((csat_values == i).sum()) 
                        for i in range(1, 6)
                    }
            
            # Date range
            if 'Created Date' in df.columns:
                created_dates = pd.to_datetime(df['Created Date']).dropna()
                if len(created_dates) > 0:
                    stats['first_date'] = created_dates.min().strftime('%Y-%m-%d')
                    stats['last_date'] = created_dates.max().strftime('%Y-%m-%d')
                    stats['date_range_days'] = (created_dates.max() - created_dates.min()).days
        
        except Exception as e:
            self.logger.error(f"Error calculating summary stats: {str(e)}")
            stats['error'] = str(e)
        
        return stats
    
    # Public API methods for specific analysis types
    
    def analyze_ticket_categories(self, tickets: List[Dict]) -> Dict[str, Any]:
        """
        Analyze tickets to identify categories and groupings.
        
        Args:
            tickets: List of ticket data dictionaries
            
        Returns:
            Dict: Analysis of ticket categories
        """
        prompt = """
Analyze these support tickets and categorize them into logical groupings.
Identify common themes, issues, and feature requests.

For each category identified, provide:
1. Category name
2. Description of the category
3. Count/percentage of tickets in this category
4. Common keywords or phrases

Also suggest improvements to ticket categorization for the support team.
"""
        return self._run_analysis(tickets, prompt, "categorization")
    
    def detect_patterns(self, tickets: List[Dict]) -> Dict[str, Any]:
        """
        Detect patterns in ticket data.
        
        Args:
            tickets: List of ticket data dictionaries
            
        Returns:
            Dict: Pattern detection results
        """
        prompt = """
Analyze these support tickets and identify recurring patterns.
Look for common workflows, user behaviors, or system issues that appear repeatedly.

For each pattern identified, provide:
1. Pattern name
2. Description of the pattern
3. Supporting evidence from tickets
4. Impact on users and support team
5. Potential root causes

Also suggest ways to address the most impactful patterns.
"""
        return self._run_analysis(tickets, prompt, "pattern_detection")
    
    def analyze_root_causes(self, tickets: List[Dict]) -> Dict[str, Any]:
        """
        Analyze root causes of support issues.
        
        Args:
            tickets: List of ticket data dictionaries
            
        Returns:
            Dict: Root cause analysis results
        """
        prompt = """
Perform a root cause analysis on these support tickets.
Look beyond the symptoms to identify underlying issues.

For each root cause identified, provide:
1. Root cause name
2. Description of the root cause
3. Affected tickets
4. Impact severity
5. Recommendations to address the root cause

Also provide a high-level summary of the most critical root causes.
"""
        return self._run_analysis(tickets, prompt, "root_cause")
    
    def analyze_trends(self, tickets: List[Dict]) -> Dict[str, Any]:
        """
        Analyze trends over time in the ticket data.
        
        Args:
            tickets: List of ticket data dictionaries
            
        Returns:
            Dict: Trend analysis results
        """
        prompt = """
Analyze trends in these support tickets.
Look for patterns over time, changes in issue types, and evolving user needs.

Provide:
1. Major trends identified
2. Changes in ticket volume or type
3. Emerging issues
4. Declining issues
5. Seasonal patterns (if any)

Also suggest ways the support team can prepare for future trends.
"""
        return self._run_analysis(tickets, prompt, "trend_analysis")
    
    def _run_analysis(self, tickets: List[Dict], prompt: str, analysis_type: str) -> Dict[str, Any]:
        """
        Run a specific type of analysis.
        
        Args:
            tickets: List of ticket data dictionaries
            prompt: Prompt for the analysis
            analysis_type: Type of analysis being performed
            
        Returns:
            Dict: Analysis results
        """
        if not tickets:
            return {
                "analysis": "No data provided for analysis",
                "recommendations": []
            }
        
        try:
            # Format tickets for the prompt
            tickets_json = json.dumps(tickets[:50], indent=2)  # Limit to 50 tickets to avoid token limits
            
            full_prompt = f"{prompt}\n\nTicket Data:\n{tickets_json}"
            
            # Call OpenAI API
            request_data = {
                "model": "gpt-4-turbo",
                "messages": [
                    {"role": "system", "content": "You are an expert support ticket analyzer."},
                    {"role": "user", "content": full_prompt}
                ],
                "temperature": 0.3
            }
            
            response = self.client.chat.completions.create(**request_data)
            
            response_content = response.choices[0].message.content
            
            # Log the interaction
            self.openai_logger.log_interaction(
                request_data,
                {"content": response_content},
                analysis_type
            )
            
            # Extract recommendations from the response
            lines = response_content.split('\n')
            recommendations = []
            
            for i, line in enumerate(lines):
                if "recommendation" in line.lower() or "suggest" in line.lower():
                    # Try to extract recommendations that follow this line
                    for j in range(i+1, min(i+6, len(lines))):
                        if lines[j].strip().startswith('-') or lines[j].strip().startswith('*') or lines[j].strip()[0:2].isdigit():
                            recommendations.append(lines[j].strip().lstrip('-*').strip())
            
            # Structure the response
            result = {
                "analysis": response_content,
                "recommendations": recommendations,
                "analysis_type": analysis_type,
                "data": {}  # Add structured data if available
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in {analysis_type} analysis: {str(e)}")
            return {
                "analysis": f"Error during {analysis_type} analysis: {str(e)}",
                "recommendations": ["Check logs for error details."],
                "analysis_type": analysis_type
            } 
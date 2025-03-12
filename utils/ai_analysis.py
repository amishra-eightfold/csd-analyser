"""Advanced AI analysis utilities for support ticket analysis."""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from openai import OpenAI
import logging
from collections import defaultdict
from .token_manager import TokenManager, convert_value_for_json
from .text_processing import prepare_text_for_ai, clean_text
from .pattern_recognition import PatternDetector, PatternAnalyzer

class ContextManager:
    """Manages context across multiple analysis chunks."""
    
    def __init__(self):
        self.previous_insights = []
        self.global_patterns = defaultdict(int)
        self.priority_context = defaultdict(list)
        self.temporal_context = defaultdict(list)
        self.pattern_insights = []  # Add storage for pattern insights
        
    def add_insight(self, insight: Dict[str, Any]):
        """Add an insight to the context."""
        self.previous_insights.append(insight)
        if len(self.previous_insights) > 5:  # Keep last 5 insights
            self.previous_insights.pop(0)
            
    def update_patterns(self, patterns: Dict[str, int]):
        """Update global pattern frequencies."""
        for pattern, freq in patterns.items():
            self.global_patterns[pattern] += freq
            
    def add_priority_context(self, priority: str, data: Dict[str, Any]):
        """Add context for a specific priority level."""
        self.priority_context[priority].append(data)
        if len(self.priority_context[priority]) > 3:  # Keep last 3 entries
            self.priority_context[priority].pop(0)
            
    def add_temporal_context(self, time_period: str, data: Dict[str, Any]):
        """Add temporal context for trend analysis."""
        self.temporal_context[time_period].append(data)
        
    def add_pattern_insight(self, insight: Dict[str, Any]):
        """Add pattern recognition insight."""
        self.pattern_insights.append(insight)
        if len(self.pattern_insights) > 10:  # Keep last 10 pattern insights
            self.pattern_insights.pop(0)
        
    def get_summary_context(self) -> Dict[str, Any]:
        """Get consolidated context for summary generation."""
        return {
            "previous_insights": self.previous_insights,
            "global_patterns": dict(self.global_patterns),
            "priority_context": dict(self.priority_context),
            "temporal_context": dict(self.temporal_context),
            "pattern_insights": self.pattern_insights
        }

class AIAnalyzer:
    """Enhanced AI analysis for support tickets."""
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.token_manager = TokenManager()
        self.context_manager = ContextManager()
        self.pattern_detector = PatternDetector()
        self.pattern_analyzer = PatternAnalyzer()
        self.logger = logging.getLogger(__name__)
        
    def _prepare_chunk_prompt(self, chunk_data: List[Dict], context: Dict[str, Any]) -> str:
        """Prepare a detailed prompt for chunk analysis."""
        prompt = f"""Analyze these support tickets in the context of previous findings and detected patterns.

Previous Context:
{json.dumps(context, indent=2)}

Current Tickets to Analyze:
{json.dumps(chunk_data, indent=2)}

Consider:
1. How these tickets relate to previously identified patterns
2. Any new patterns or trends emerging
3. Priority distribution and its implications
4. Resolution times and their relationship to ticket complexity
5. Customer impact and satisfaction indicators
6. Relationship to detected pattern insights

Provide your analysis in this JSON format:
{{
    "chunk_summary": {{
        "main_findings": "Key insights from this chunk",
        "relation_to_previous": "How these findings relate to previous chunks",
        "new_patterns": ["List of new patterns identified"],
        "pattern_validation": "How this chunk validates or challenges detected patterns"
    }},
    "detailed_analysis": {{
        "priority_insights": {{
            "distribution": "Analysis of priority distribution",
            "trends": "Identified trends in priority levels",
            "recommendations": ["Priority-based recommendations"]
        }},
        "resolution_insights": {{
            "patterns": "Patterns in resolution times",
            "bottlenecks": ["Identified bottlenecks"],
            "improvements": ["Suggested improvements"]
        }},
        "customer_impact": {{
            "satisfaction_indicators": ["CSAT-related insights"],
            "risk_factors": ["Identified risk factors"],
            "improvement_areas": ["Areas needing attention"]
        }},
        "pattern_correlation": {{
            "matches": ["Patterns that match previous findings"],
            "deviations": ["Patterns that deviate from previous findings"],
            "new_insights": ["New pattern-related insights"]
        }}
    }},
    "patterns": [
        {{"pattern": "Pattern Description", "frequency": "Frequency Description", "impact": "Impact Assessment", "confidence": "Confidence Level"}}
    ],
    "recommendations": [
        {{"title": "Recommendation Title", "description": "Detailed Description", "priority": "High/Medium/Low", "impact": "Expected Impact", "effort": "Implementation Effort"}}
    ]
}}"""
        return prompt

    def _analyze_chunk(self, chunk_data: List[Dict], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a chunk of tickets with context preservation."""
        try:
            # Optimize context to fit within token limits
            max_context_tokens = int(self.token_manager.chunk_tokens * 0.3)  # Reserve 30% for context
            optimized_context = self.token_manager.optimize_context(context, max_context_tokens)
            
            # Prepare prompt
            prompt = self._prepare_chunk_prompt(chunk_data, optimized_context)
            
            # Check total tokens and truncate if necessary
            total_tokens = self.token_manager.count_tokens(prompt)
            if total_tokens > self.token_manager.available_tokens:
                self.logger.warning(f"Prompt too long ({total_tokens} tokens). Truncating...")
                # Truncate chunk data while preserving high-priority items
                truncated_data = []
                current_tokens = self.token_manager.count_tokens(self._prepare_chunk_prompt([], optimized_context))
                
                # Sort items by priority score
                sorted_items = sorted(chunk_data, 
                                   key=lambda x: self.token_manager.calculate_priority_score(x),
                                   reverse=True)
                
                for item in sorted_items:
                    item_prompt = self._prepare_chunk_prompt(truncated_data + [item], optimized_context)
                    item_tokens = self.token_manager.count_tokens(item_prompt)
                    
                    if item_tokens <= self.token_manager.available_tokens:
                        truncated_data.append(item)
                        current_tokens = item_tokens
                    else:
                        # Try to add a truncated version of the item
                        truncated_item = self.token_manager.truncate_item(item)
                        truncated_prompt = self._prepare_chunk_prompt(truncated_data + [truncated_item], optimized_context)
                        if self.token_manager.count_tokens(truncated_prompt) <= self.token_manager.available_tokens:
                            truncated_data.append(truncated_item)
                
                chunk_data = truncated_data
                prompt = self._prepare_chunk_prompt(chunk_data, optimized_context)
            
            # Get token info for logging
            token_info = self.token_manager.get_token_info((chunk_data, optimized_context), "chunk_analysis")
            self.logger.info(f"Analyzing chunk with {token_info.total_tokens} tokens")
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert support ticket analyst focusing on detailed pattern analysis and context-aware insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={ "type": "json_object" }
            )
            
            chunk_insights = json.loads(response.choices[0].message.content.strip())
            self.context_manager.add_insight(chunk_insights)
            
            if 'patterns' in chunk_insights:
                pattern_freq = {p['pattern']: 1 for p in chunk_insights['patterns']}
                self.context_manager.update_patterns(pattern_freq)
                
            return chunk_insights
            
        except Exception as e:
            self.logger.error(f"Error in chunk analysis: {str(e)}")
            return {
                "chunk_summary": {"main_findings": "Error in analysis", "relation_to_previous": "N/A", "new_patterns": []},
                "detailed_analysis": {},
                "patterns": [],
                "recommendations": []
            }

    def _prepare_final_prompt(self, all_insights: List[Dict], summary_stats: Dict, pattern_analysis: Dict) -> str:
        """Prepare prompt for final analysis with comprehensive context."""
        context = self.context_manager.get_summary_context()
        
        prompt = f"""Based on the complete analysis of support tickets and detected patterns, provide a comprehensive summary.

Context and Statistics:
{json.dumps(summary_stats, indent=2)}

Pattern Analysis Results:
{json.dumps(pattern_analysis, indent=2)}

Accumulated Insights:
{json.dumps(context, indent=2)}

Previous Chunk Analyses:
{json.dumps(all_insights, indent=2)}

Provide a detailed analysis in this JSON format:
{{
    "executive_summary": {{
        "key_findings": ["Most important insights"],
        "critical_patterns": ["Most significant patterns"],
        "risk_areas": ["Areas requiring immediate attention"]
    }},
    "trend_analysis": {{
        "volume_trends": "Analysis of ticket volume trends",
        "priority_shifts": "Changes in priority distribution",
        "resolution_patterns": "Patterns in resolution times",
        "pattern_evolution": "How patterns have evolved over time"
    }},
    "pattern_insights": {{
        "recurring_issues": ["Identified recurring problems"],
        "root_causes": ["Common root causes"],
        "correlation_patterns": ["Related issues or dependencies"],
        "confidence_levels": {{
            "high_confidence": ["Patterns with strong evidence"],
            "medium_confidence": ["Patterns with moderate evidence"],
            "low_confidence": ["Patterns requiring more validation"]
        }}
    }},
    "customer_impact_analysis": {{
        "satisfaction_trends": "CSAT trend analysis",
        "pain_points": ["Identified customer pain points"],
        "improvement_opportunities": ["Areas for improving customer experience"],
        "pattern_impact": "How identified patterns affect customer satisfaction"
    }},
    "recommendations": [
        {{
            "title": "Recommendation Title",
            "description": "Detailed description",
            "priority": "High/Medium/Low",
            "impact": "Expected impact",
            "effort": "Implementation effort",
            "timeline": "Suggested timeline",
            "pattern_correlation": "Related patterns"
        }}
    ],
    "next_steps": ["Prioritized list of actions to take"]
}}"""
        return prompt

    def analyze_tickets(self, df: pd.DataFrame, chunk_size: int = 3) -> Dict[str, Any]:
        """Perform enhanced analysis of support tickets with context preservation and pattern recognition."""
        try:
            # Prepare data
            analysis_df = prepare_text_for_ai(df.copy())
            total_tickets = len(analysis_df)
            max_tickets = min(30, total_tickets)  # Reduced from 50 to 30 for better token management
            
            # Run pattern detection
            patterns = self.pattern_detector.detect_patterns(analysis_df)
            pattern_insights = self.pattern_analyzer.analyze_patterns(patterns)
            
            # Add pattern insights to context
            for insight_type, insights in pattern_insights.items():
                if isinstance(insights, list):
                    for insight in insights:
                        self.context_manager.add_pattern_insight(insight)
            
            # Calculate summary statistics
            summary_stats = {
                'total_tickets': total_tickets,
                'date_range': {
                    'start': analysis_df['Created Date'].min().strftime('%Y-%m-%d'),
                    'end': analysis_df['Created Date'].max().strftime('%Y-%m-%d')
                },
                'priority_distribution': analysis_df['Priority'].value_counts().to_dict(),
                'product_areas': analysis_df['Product Area'].value_counts().to_dict(),
                'avg_resolution_time': analysis_df['Resolution Time (Days)'].mean()
            }
            
            # Process in chunks using TokenManager
            all_insights = []
            processed_tickets = 0
            
            # Create chunks using TokenManager
            tickets_data = []
            for _, case in analysis_df.iloc[:max_tickets].iterrows():
                case_dict = {}
                for col, val in case.items():
                    case_dict[col] = convert_value_for_json(val)
                tickets_data.append(case_dict)
            
            chunks = self.token_manager.create_chunks(tickets_data)
            
            # Process each chunk
            for chunk in chunks:
                try:
                    chunk_insights = self._analyze_chunk(chunk['items'], chunk['context'])
                    all_insights.append(chunk_insights)
                    processed_tickets += len(chunk['items'])
                    
                    self.logger.info(f"Processed chunk {chunk['context']['chunk_position']} of {chunk['context']['total_chunks']}")
                except Exception as chunk_error:
                    self.logger.error(f"Error processing chunk: {str(chunk_error)}")
                    continue
            
            if not all_insights:
                raise Exception("No insights generated from any chunks")
            
            # Prepare final analysis with optimized context
            final_context = {
                'summary_stats': summary_stats,
                'pattern_analysis': pattern_insights,
                'processed_chunks': len(all_insights),
                'processed_tickets': processed_tickets
            }
            
            # Optimize final prompt context
            max_context_tokens = int(self.token_manager.chunk_tokens * 0.4)  # Reserve 40% for final context
            optimized_context = self.token_manager.optimize_context(final_context, max_context_tokens)
            
            # Generate final analysis
            final_prompt = self._prepare_final_prompt(all_insights, optimized_context, pattern_insights)
            
            # Check final prompt tokens
            if self.token_manager.count_tokens(final_prompt) > self.token_manager.available_tokens:
                self.logger.warning("Final prompt too long, optimizing...")
                # Reduce insights to most important ones
                all_insights = all_insights[-3:]  # Keep only last 3 chunks
                final_prompt = self._prepare_final_prompt(all_insights, optimized_context, pattern_insights)
            
            final_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert support ticket analyst providing comprehensive analysis and actionable insights."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={ "type": "json_object" }
            )
            
            final_insights = json.loads(final_response.choices[0].message.content.strip())
            
            # Add metadata
            final_insights['metadata'] = {
                'analysis_timestamp': datetime.now().isoformat(),
                'tickets_analyzed': processed_tickets,
                'total_tickets': total_tickets,
                'chunks_processed': len(all_insights),
                'patterns_detected': len(patterns.get('issue_clusters', {})),
                'pattern_insights_generated': pattern_insights['summary']['total_insights']
            }
            
            return final_insights
            
        except Exception as e:
            self.logger.error(f"Error in ticket analysis: {str(e)}")
            return {
                'error': str(e),
                'executive_summary': {
                    'key_findings': ['Analysis failed'],
                    'critical_patterns': [],
                    'risk_areas': []
                },
                'recommendations': [],
                'next_steps': ['Retry analysis with smaller dataset']
            } 
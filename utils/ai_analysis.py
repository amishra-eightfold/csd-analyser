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
from .text_processing import prepare_text_for_ai, clean_text, get_technical_stopwords, remove_stopwords
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
        
        # Initialize pattern tracking
        self.detected_patterns = []
        self.pattern_confidences = defaultdict(float)
        self.pattern_frequencies = defaultdict(int)
        
    def _track_pattern(self, pattern: str, confidence: float = 0.0):
        """Track a detected pattern with its confidence score."""
        if pattern not in self.detected_patterns:
            self.detected_patterns.append(pattern)
        self.pattern_confidences[pattern] = max(self.pattern_confidences[pattern], confidence)
        self.pattern_frequencies[pattern] += 1
        
    def _get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of tracked patterns with confidence levels."""
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

    def _calculate_ticket_importance(self, ticket: Dict[str, Any]) -> float:
        """Calculate importance score for a ticket to prioritize analysis.
        
        Higher scores indicate higher importance based on:
        - Ticket priority (P1 > P2 > P3)
        - Escalation status (escalated tickets are more important)
        - Resolution time (longer resolution times may indicate complex issues)
        - CSAT score (lower scores indicate potential issues)
        - Root cause (tickets with specified root causes provide better insights)
        - Closed status (closed tickets have complete information)
        
        Returns:
            float: Importance score between 0 and 100
        """
        score = 50.0  # Base score
        
        # Priority factor (25 points max)
        priority_map = {'P1': 25, 'P2': 20, 'P3': 15, 'P4': 10}
        priority = str(ticket.get('Priority', '')).upper()
        score += priority_map.get(priority, 0)
        
        # Escalation factor (10 points)
        if ticket.get('IsEscalated') is True:
            score += 10
        
        # Resolution time factor (5 points max)
        resolution_time = ticket.get('Resolution Time (Days)', 0)
        if resolution_time and resolution_time > 0:
            # Normalize resolution time (up to 5 points for longer resolution times)
            score += min(5, resolution_time / 5)
        
        # CSAT factor (5 points max - lower scores get more points)
        csat = ticket.get('CSAT', None)
        if csat is not None and isinstance(csat, (int, float)) and 1 <= csat <= 5:
            # Invert CSAT score to prioritize low satisfaction
            score += (6 - csat)
        
        # Root cause factor (5 points)
        root_cause = str(ticket.get('Root Cause', '')).strip()
        if root_cause and root_cause.lower() not in ('not specified', 'none', 'unknown', 'n/a'):
            score += 5
        
        # Status factor (5 points for closed tickets with complete data)
        status = str(ticket.get('Status', '')).lower()
        if status in ('closed', 'resolved', 'completed'):
            score += 5
            
        return min(100, score)  # Cap at 100

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
                
                # Sort items by importance score
                sorted_items = sorted(chunk_data, 
                                   key=lambda x: self._calculate_ticket_importance(x),
                                   reverse=True)
                
                for item in sorted_items:
                    # First, try to add a compact version of the item
                    compact_item = self._create_compact_ticket(item)
                    compact_prompt = self._prepare_chunk_prompt(truncated_data + [compact_item], optimized_context)
                    compact_tokens = self.token_manager.count_tokens(compact_prompt)
                    
                    if compact_tokens <= self.token_manager.available_tokens:
                        # Can add the compact version
                        truncated_data.append(compact_item)
                        current_tokens = compact_tokens
                    else:
                        # Can't add even the compact version, try ultra compact
                        ultra_compact = self._create_ultra_compact_ticket(item)
                        ultra_prompt = self._prepare_chunk_prompt(truncated_data + [ultra_compact], optimized_context)
                        if self.token_manager.count_tokens(ultra_prompt) <= self.token_manager.available_tokens:
                            truncated_data.append(ultra_compact)
                
                chunk_data = truncated_data
                prompt = self._prepare_chunk_prompt(chunk_data, optimized_context)
                
                self.logger.info(f"Truncated chunk from {len(sorted_items)} to {len(chunk_data)} tickets")
            
            # Get token info for logging
            token_info = self.token_manager.get_token_info((chunk_data, optimized_context), "chunk_analysis")
            self.logger.info(f"Analyzing chunk with {token_info.total_tokens} tokens ({len(chunk_data)} tickets)")
            
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
    
    def _create_compact_ticket(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Create a more compact version of a ticket to optimize token usage.
        
        Keeps essential fields and truncates longer text fields.
        """
        # Define essential fields to keep
        essential_fields = {
            'Id', 'CaseNumber', 'Subject', 'Status', 'Priority', 
            'Root Cause', 'Product Area', 'Product Feature', 'Created Date', 
            'Closed Date', 'Resolution Time (Days)', 'CSAT', 'IsEscalated'
        }
        
        # Create compact version with only essential fields
        compact = {k: v for k, v in ticket.items() if k in essential_fields}
        
        # Truncate description to conserve tokens
        if 'Description' in ticket and ticket['Description']:
            # Extract first 100 words or fewer
            words = str(ticket['Description']).split()
            if len(words) > 100:
                compact['Description'] = ' '.join(words[:100]) + '...'
            else:
                compact['Description'] = ticket['Description']
        
        return compact
    
    def _create_ultra_compact_ticket(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Create an ultra compact version of a ticket when tokens are very limited.
        
        Keeps only the most critical fields needed for analysis.
        """
        # Define minimal fields for ultra compact mode
        minimal_fields = {
            'CaseNumber', 'Subject', 'Status', 'Priority', 
            'Root Cause', 'Product Area', 'Resolution Time (Days)', 'CSAT'
        }
        
        # Create ultra compact version
        ultra_compact = {k: v for k, v in ticket.items() if k in minimal_fields}
        
        # Truncate subject to conserve tokens
        if 'Subject' in ultra_compact and ultra_compact['Subject']:
            words = str(ultra_compact['Subject']).split()
            if len(words) > 15:
                ultra_compact['Subject'] = ' '.join(words[:15]) + '...'
        
        return ultra_compact

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

    def analyze_tickets(self, df: pd.DataFrame, chunk_size: int = 5) -> Dict[str, Any]:
        """Perform enhanced analysis of support tickets with context preservation and pattern recognition."""
        try:
            # Reset pattern tracking for new analysis
            self.detected_patterns = []
            self.pattern_confidences.clear()
            self.pattern_frequencies.clear()
            
            # Pre-process data with stopword removal and PII handling
            analysis_df = prepare_text_for_ai(df.copy(), remove_stops=True)
            total_tickets = len(analysis_df)
            
            # Increase max tickets based on available tokens
            available_token_capacity = self.token_manager.available_tokens * 0.8  # Use 80% of available tokens
            avg_tokens_per_ticket = 500  # Conservative estimate
            max_possible_tickets = int(available_token_capacity / avg_tokens_per_ticket)
            max_tickets = min(max_possible_tickets, total_tickets)
            
            self.logger.info(f"Processing up to {max_tickets} tickets out of {total_tickets} total")
            
            # Run initial pattern detection on full dataset
            patterns = self.pattern_detector.detect_patterns(analysis_df)
            pattern_insights = self.pattern_analyzer.analyze_patterns(patterns)
            
            # Track detected patterns
            if 'issue_clusters' in patterns:
                for cluster in patterns['issue_clusters']:
                    pattern_name = cluster.get('pattern', '')
                    confidence = cluster.get('confidence', 0.0)
                    if pattern_name:
                        self._track_pattern(pattern_name, confidence)
            
            # Add pattern insights to context
            for insight_type, insights in pattern_insights.items():
                if isinstance(insights, list):
                    for insight in insights:
                        self.context_manager.add_pattern_insight(insight)
                        if 'pattern' in insight:
                            self._track_pattern(insight['pattern'], insight.get('confidence', 0.5))
            
            # Calculate summary statistics
            summary_stats = {
                'total_tickets': total_tickets,
                'date_range': {
                    'start': analysis_df['Created Date'].min().strftime('%Y-%m-%d'),
                    'end': analysis_df['Created Date'].max().strftime('%Y-%m-%d')
                },
                'priority_distribution': analysis_df['Priority'].value_counts().to_dict(),
                'product_areas': analysis_df['Product Area'].value_counts().to_dict(),
                'avg_resolution_time': analysis_df['Resolution Time (Days)'].mean(),
                'pattern_summary': self._get_pattern_summary()
            }
            
            # Process in chunks using TokenManager
            all_insights = []
            processed_tickets = 0
            
            # Prepare tickets data with enhanced preprocessing
            tickets_data = []
            for _, case in analysis_df.iloc[:max_tickets].iterrows():
                case_dict = {}
                for col, val in case.items():
                    # Apply additional preprocessing for text fields
                    if col in ['Subject', 'Description', 'Comments']:
                        val = clean_text(str(val))  # Clean text
                        if len(val.split()) > 5:  # Only remove stopwords from longer text
                            val = remove_stopwords(val)
                    case_dict[col] = convert_value_for_json(val)
                tickets_data.append(case_dict)
            
            # Calculate importance scores for tickets
            for ticket in tickets_data:
                ticket['_importance_score'] = self._calculate_ticket_importance(ticket)
            
            # Sort tickets by importance score
            sorted_tickets = sorted(tickets_data, key=lambda x: x.get('_importance_score', 0), reverse=True)
            
            # Create more efficient chunks
            chunks = []
            # First, ensure high-priority tickets are included
            high_priority_tickets = [t for t in sorted_tickets if t.get('_importance_score', 0) >= 75]
            remaining_tickets = [t for t in sorted_tickets if t.get('_importance_score', 0) < 75]
            
            # Process high-priority tickets first
            if high_priority_tickets:
                for i in range(0, len(high_priority_tickets), chunk_size):
                    chunk_tickets = high_priority_tickets[i:i+chunk_size]
                    chunk_context = {
                        'chunk_position': len(chunks) + 1,
                        'chunk_type': 'high_priority',
                        'importance_range': f"{min([t.get('_importance_score', 0) for t in chunk_tickets]):.1f}-{max([t.get('_importance_score', 0) for t in chunk_tickets]):.1f}",
                        'pattern_context': self._get_pattern_summary()
                    }
                    chunks.append({
                        'items': chunk_tickets,
                        'context': chunk_context
                    })
            
            # Process remaining tickets
            for i in range(0, len(remaining_tickets), chunk_size):
                chunk_tickets = remaining_tickets[i:i+chunk_size]
                chunk_context = {
                    'chunk_position': len(chunks) + 1,
                    'chunk_type': 'standard',
                    'importance_range': f"{min([t.get('_importance_score', 0) for t in chunk_tickets]):.1f}-{max([t.get('_importance_score', 0) for t in chunk_tickets]):.1f}",
                    'pattern_context': self._get_pattern_summary()
                }
                chunks.append({
                    'items': chunk_tickets,
                    'context': chunk_context
                })
            
            # Update total chunks info
            for i, chunk in enumerate(chunks):
                chunk['context']['total_chunks'] = len(chunks)
            
            self.logger.info(f"Created {len(chunks)} chunks for analysis from {len(sorted_tickets)} tickets")
            
            # Process each chunk with enhanced pattern detection
            for chunk in chunks:
                try:
                    # Remove importance score before analysis
                    for ticket in chunk['items']:
                        if '_importance_score' in ticket:
                            del ticket['_importance_score']
                    
                    # Add current pattern context
                    chunk['context']['current_patterns'] = self._get_pattern_summary()
                    
                    # Analyze chunk
                    chunk_insights = self._analyze_chunk(chunk['items'], chunk['context'])
                    
                    # Extract and track new patterns from chunk insights
                    if 'patterns' in chunk_insights:
                        for pattern in chunk_insights['patterns']:
                            pattern_name = pattern.get('pattern', '')
                            confidence = float(pattern.get('confidence', '0.5').split()[0])  # Handle "0.8 High" format
                            if pattern_name:
                                self._track_pattern(pattern_name, confidence)
                    
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
                'processed_tickets': processed_tickets,
                'pattern_summary': self._get_pattern_summary()
            }
            
            # Generate final analysis
            final_prompt = self._prepare_final_prompt(all_insights, final_context, pattern_insights)
            
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
                'patterns_detected': len(self.detected_patterns),
                'pattern_insights_generated': len([p for p in self.detected_patterns if self.pattern_confidences[p] >= 0.5])
            }
            
            # Log pattern analysis details
            self.logger.info("Pattern analysis details", {
                'patterns_detected': len(self.detected_patterns),
                'high_confidence_patterns': len([p for p in self.detected_patterns if self.pattern_confidences[p] >= 0.8]),
                'medium_confidence_patterns': len([p for p in self.detected_patterns if 0.5 <= self.pattern_confidences[p] < 0.8]),
                'low_confidence_patterns': len([p for p in self.detected_patterns if self.pattern_confidences[p] < 0.5])
            })
            
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
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
            } 
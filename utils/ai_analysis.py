"""Advanced AI analysis utilities for support ticket analysis."""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from openai import OpenAI
from collections import defaultdict
from .token_manager import TokenManager, convert_value_for_json
from .text_processing import prepare_text_for_ai, clean_text, get_technical_stopwords, remove_stopwords
from .pattern_recognition import PatternRecognizer, PatternAnalyzer
from config.logging_config import get_logger, log_error
import traceback
import os
from pathlib import Path

class OpenAILogger:
    """Logger for OpenAI requests and responses."""
    
    def __init__(self, log_dir: str = "logs/openai"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger('openai')
    
    def log_interaction(self, request_data: Dict, response_data: Dict, interaction_type: str):
        """Log OpenAI interaction to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{interaction_type}_{timestamp}.json"
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "interaction_type": interaction_type,
            "request": request_data,
            "response": response_data
        }
        
        try:
            with open(self.log_dir / filename, 'w') as f:
                json.dump(log_data, f, indent=2)
            self.logger.info(f"Logged OpenAI interaction to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to log OpenAI interaction: {str(e)}")

def preprocess_text_for_ai(text: str) -> str:
    """Enhanced text preprocessing for AI analysis."""
    if not text or not isinstance(text, str):
        return ""
        
    # Basic cleaning
    text = clean_text(text)
    
    # Remove technical stopwords
    tech_stopwords = get_technical_stopwords()
    
    # Split into sentences for better processing
    sentences = text.split('. ')
    processed_sentences = []
    
    for sentence in sentences:
        # Skip very short or empty sentences
        if len(sentence.split()) < 3:
            continue
            
        # Remove stopwords and technical jargon
        words = sentence.split()
        words = [w for w in words if w.lower() not in tech_stopwords]
        
        # Skip sentences that lost too much meaning
        if len(words) < 3:
            continue
            
        processed_sentences.append(' '.join(words))
    
    return '. '.join(processed_sentences)

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
        self.pattern_detector = PatternRecognizer()
        self.pattern_analyzer = PatternAnalyzer()
        self.logger = get_logger('app')
        self.openai_logger = OpenAILogger()
        
        # Initialize pattern tracking with lists instead of sets
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

    def _convert_confidence_to_float(self, confidence_str: str) -> float:
        """Convert string confidence level to float value.
        
        Args:
            confidence_str: String confidence level ('High', 'Medium', 'Low', etc.)
            
        Returns:
            float: Confidence value between 0 and 1
        """
        # Remove any trailing commas or whitespace
        confidence_str = confidence_str.strip().rstrip(',').lower()
        
        # Map confidence levels to float values
        confidence_map = {
            'high': 0.9,
            'medium': 0.6,
            'low': 0.3,
            'very high': 1.0,
            'very low': 0.1
        }
        
        try:
            # First try to convert directly to float if it's a number
            return float(confidence_str)
        except ValueError:
            # If it's a string confidence level, use the mapping
            return confidence_map.get(confidence_str, 0.5)  # Default to 0.5 if unknown

    def _prepare_chunk_prompt(self, chunk_data: List[Dict], context: Dict[str, Any]) -> str:
        """Prepare a concise prompt for chunk analysis."""
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
            # Keep only essential fields
            trimmed_ticket = {
                'Id': ticket.get('Id'),
                'Subject': ticket.get('Subject'),
                'Priority': ticket.get('Priority'),
                'Status': ticket.get('Status'),
                'Resolution Time (Days)': ticket.get('Resolution Time (Days)'),
                'CSAT': ticket.get('CSAT'),
                'Root Cause': ticket.get('Root Cause')
            }
            serializable_chunk_data.append(convert_value_for_json(trimmed_ticket))
        
        prompt = f"""Analyze these support tickets considering previous patterns.

Context:
{json.dumps(trimmed_context, indent=2)}

Tickets:
{json.dumps(serializable_chunk_data, indent=2)}

Provide analysis in JSON format:
{{
    "chunk_summary": {{
        "main_findings": "key insights",
        "new_patterns": ["new patterns"]
    }},
    "patterns": [
        {{"pattern": "pattern", "confidence": "high/medium/low"}}
    ],
    "recommendations": [
        {{
            "title": "title",
            "description": "detailed description",
            "priority": "priority",
            "impact": "impact"
        }}
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
        
        if not isinstance(ticket, dict):
            return score
        
        # Priority factor (25 points max)
        priority_map = {'P1': 25, 'P2': 20, 'P3': 15, 'P4': 10}
        priority = str(ticket.get('Priority', '')).upper()
        score += priority_map.get(priority, 0)
        
        # Escalation factor (10 points)
        if ticket.get('IsEscalated') is True:
            score += 10
        
        # Resolution time factor (5 points max)
        resolution_time = ticket.get('Resolution Time (Days)')
        if isinstance(resolution_time, (int, float)) and resolution_time > 0:
            # Normalize resolution time (up to 5 points for longer resolution times)
            score += min(5, resolution_time / 5)
        
        # CSAT factor (5 points max - lower scores get more points)
        csat = ticket.get('CSAT')
        if isinstance(csat, (int, float)) and 1 <= csat <= 5:
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

    def _analyze_chunk(self, chunk_data: List[Dict], context: Dict) -> Dict[str, Any]:
        """Analyze a chunk of data using OpenAI with enhanced preprocessing."""
        try:
            # Preprocess text fields in chunk data
            processed_chunk_data = []
            for ticket in chunk_data:
                processed_ticket = ticket.copy()
                for field in ['Subject', 'Description', 'Comments']:
                    if field in ticket:
                        processed_ticket[field] = preprocess_text_for_ai(ticket[field])
                processed_chunk_data.append(processed_ticket)
            
            # Prepare the prompt with context
            prompt = self._prepare_chunk_prompt(processed_chunk_data, context)
            
            # Prepare request data for logging
            request_data = {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are an expert support ticket analyst providing insights and pattern recognition. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 2000
            }
            
            # Call OpenAI API
            response = self.client.chat.completions.create(**request_data)
            
            # Prepare response data for logging
            response_data = {
                "choices": [{"message": {"content": choice.message.content}} for choice in response.choices],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if hasattr(response, 'usage') else None
            }
            
            # Log the interaction
            self.openai_logger.log_interaction(request_data, response_data, "chunk_analysis")
            
            # Extract and validate response
            if not response or not response.choices:
                raise ValueError("Empty response from OpenAI API")
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty content in OpenAI response")
            
            # Parse and validate the response
            try:
                insights = json.loads(content)
                if not isinstance(insights, dict):
                    raise ValueError("Response is not a valid JSON object")
                return insights
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse OpenAI response: {str(e)}")
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error analyzing chunk: {error_msg}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'error': error_msg,
                'details': 'Error during chunk analysis',
                'context': {
                    'chunk_size': len(chunk_data),
                    'context_size': len(context) if context else 0
                }
            }

    def _prepare_final_prompt(self, all_insights: List[Dict], summary_stats: Dict, pattern_analysis: Dict) -> str:
        """Prepare concise prompt for final analysis."""
        # Trim insights to most recent and relevant ones
        recent_insights = all_insights[-3:]  # Keep only last 3 chunks
        
        # Trim summary stats to essential metrics
        essential_stats = {
            'ticket_volume': summary_stats.get('ticket_volume', {}),
            'resolution_metrics': summary_stats.get('resolution_metrics', {}),
            'customer_satisfaction': summary_stats.get('customer_satisfaction', {})
        }
        
        # Get only high confidence patterns
        pattern_summary = self._get_pattern_summary()
        high_confidence_patterns = pattern_summary.get('high_confidence', [])
        
        prompt = f"""Analyze support tickets based on these insights.

Stats:
{json.dumps(essential_stats, indent=2)}

High Confidence Patterns:
{json.dumps(high_confidence_patterns, indent=2)}

Recent Insights:
{json.dumps(recent_insights, indent=2)}

Provide analysis in JSON format:
{{
    "executive_summary": {{
        "key_findings": ["findings"],
        "critical_patterns": ["patterns"]
    }},
    "pattern_insights": {{
        "recurring_issues": ["issues"],
        "confidence_levels": {{
            "high_confidence": ["patterns"],
            "medium_confidence": ["patterns"]
        }}
    }},
    "recommendations": [
        {{
            "title": "title",
            "description": "detailed description",
            "priority": "priority",
            "impact": "impact",
            "effort": "implementation effort"
        }}
    ],
    "next_steps": ["steps"]
}}"""
        return prompt

    def analyze_tickets(self, analysis_df: pd.DataFrame, max_tickets: int = 100) -> Dict[str, Any]:
        """
        Analyze support tickets with enhanced error handling, validation, and debug logging.
        
        Args:
            analysis_df: DataFrame containing ticket data
            max_tickets: Maximum number of tickets to analyze
            
        Returns:
            Dict containing analysis results or error information
        """
        try:
            # Initialize debug logging
            self.logger.info("Starting ticket analysis", extra={
                'total_tickets': len(analysis_df),
                'max_tickets': max_tickets,
                'columns': list(analysis_df.columns)
            })
            
            if analysis_df.empty:
                self.logger.error("Empty dataset provided for analysis")
                return {
                    'error': 'Empty dataset',
                    'details': 'No tickets available for analysis'
                }
            
            # Initialize pattern tracking with lists
            self.detected_patterns = []
            self.pattern_confidences = defaultdict(float)
            self.pattern_frequencies = defaultdict(int)
            
            # Calculate total tickets and validate data
            total_tickets = len(analysis_df)
            if total_tickets == 0:
                self.logger.error("No valid tickets found in dataset")
                return {
                    'error': 'No valid tickets',
                    'details': 'Dataset contains no valid tickets for analysis'
                }
            
            # Initialize pattern recognition metrics
            pattern_recognition_metrics = {
                'total_patterns_found': 0,
                'high_confidence_patterns': 0,
                'medium_confidence_patterns': 0,
                'low_confidence_patterns': 0,
                'pattern_recognition_start_time': datetime.now()
            }
            
            # Get pattern insights with validation and logging
            self.logger.info("Starting pattern recognition")
            try:
                pattern_insights = self.pattern_detector.analyze_patterns(analysis_df)
                if not pattern_insights or not isinstance(pattern_insights, dict):
                    self.logger.warning("Pattern recognition failed or returned invalid results", extra={
                        'pattern_insights_type': type(pattern_insights).__name__,
                        'is_empty': not pattern_insights
                    })
                    pattern_insights = {}
                else:
                    self.logger.info("Pattern recognition completed successfully", extra={
                        'patterns_found': len(pattern_insights.get('patterns', [])),
                        'confidence_levels': pattern_insights.get('confidence_distribution', {})
                    })
            except Exception as pattern_error:
                self.logger.error(f"Error in pattern recognition: {str(pattern_error)}", 
                                extra={'traceback': traceback.format_exc()})
                pattern_insights = {}
            
            # Process in chunks using TokenManager
            all_insights = []
            processed_tickets = 0
            chunk_processing_metrics = {
                'total_chunks': 0,
                'successful_chunks': 0,
                'failed_chunks': 0,
                'total_tokens_used': 0
            }
            
            # Prepare tickets data with enhanced preprocessing
            self.logger.info("Preparing ticket data for analysis")
            tickets_data = []
            for _, case in analysis_df.iloc[:max_tickets].iterrows():
                case_dict = {}
                for col, val in case.items():
                    # Apply additional preprocessing for text fields
                    if col in ['Subject', 'Description', 'Comments']:
                        val = clean_text(str(val))
                        if len(val.split()) > 5:
                            val = remove_stopwords(val)
                    case_dict[col] = convert_value_for_json(val)
                tickets_data.append(case_dict)
            
            self.logger.info(f"Prepared {len(tickets_data)} tickets for analysis")
            
            # Create chunks with TokenManager
            chunks = self.token_manager.create_chunks(
                tickets_data,
                size=3,
                include_context=True
            )
            
            chunk_processing_metrics['total_chunks'] = len(chunks)
            self.logger.info(f"Created {len(chunks)} chunks for processing")
            
            # Process each chunk with enhanced error handling and logging
            for chunk_index, chunk in enumerate(chunks, 1):
                try:
                    self.logger.info(f"Processing chunk {chunk_index}/{len(chunks)}", extra={
                        'chunk_size': len(chunk['items']),
                        'context_size': len(chunk.get('context', {}))
                    })
                    
                    # Remove importance score before analysis
                    for ticket in chunk['items']:
                        if '_importance_score' in ticket:
                            del ticket['_importance_score']
                    
                    # Add current pattern context
                    chunk['context']['current_patterns'] = self._get_pattern_summary()
                    
                    # Analyze chunk
                    chunk_insights = self._analyze_chunk(chunk['items'], chunk['context'])
                    
                    # Check for errors in chunk analysis
                    if 'error' in chunk_insights:
                        chunk_processing_metrics['failed_chunks'] += 1
                        self.logger.error(f"Chunk {chunk_index} analysis error: {chunk_insights['error']}")
                        if chunk_insights.get('details') == 'API quota exceeded or invalid key':
                            return chunk_insights
                        continue
                    
                    # Extract and track patterns
                    if 'patterns' in chunk_insights:
                        for pattern in chunk_insights['patterns']:
                            pattern_name = pattern.get('pattern', '')
                            confidence_str = str(pattern.get('confidence', '0.5'))
                            if pattern_name:
                                confidence = self._convert_confidence_to_float(confidence_str)
                                self._track_pattern(pattern_name, confidence)
                                pattern_recognition_metrics['total_patterns_found'] += 1
                                
                                # Update confidence metrics
                                if confidence >= 0.8:
                                    pattern_recognition_metrics['high_confidence_patterns'] += 1
                                elif confidence >= 0.5:
                                    pattern_recognition_metrics['medium_confidence_patterns'] += 1
                                else:
                                    pattern_recognition_metrics['low_confidence_patterns'] += 1
                    
                    all_insights.append(chunk_insights)
                    processed_tickets += len(chunk['items'])
                    chunk_processing_metrics['successful_chunks'] += 1
                    
                    self.logger.info(f"Successfully processed chunk {chunk_index}", extra={
                        'patterns_found': len(chunk_insights.get('patterns', [])),
                        'processed_tickets': processed_tickets
                    })
                    
                except Exception as chunk_error:
                    chunk_processing_metrics['failed_chunks'] += 1
                    self.logger.error(f"Error processing chunk {chunk_index}: {str(chunk_error)}", 
                                    extra={'traceback': traceback.format_exc()})
                    continue
            
            if not all_insights:
                self.logger.error("No valid insights generated from any chunks")
                return {
                    'error': 'Analysis failed',
                    'details': 'No valid insights generated from any chunks',
                    'context': {
                        'total_tickets': total_tickets,
                        'processed_tickets': processed_tickets,
                        'chunk_metrics': chunk_processing_metrics
                    }
                }
            
            # Prepare final analysis
            try:
                pattern_recognition_metrics['pattern_recognition_duration'] = \
                    (datetime.now() - pattern_recognition_metrics['pattern_recognition_start_time']).total_seconds()
                
                final_context = {
                    'summary_stats': self._calculate_summary_stats(analysis_df),
                    'pattern_analysis': pattern_insights,
                    'processed_chunks': len(all_insights),
                    'processed_tickets': processed_tickets,
                    'pattern_summary': self._get_pattern_summary()
                }
                
                self.logger.info("Generating final analysis", extra={
                    'context_size': len(final_context),
                    'total_insights': len(all_insights)
                })
                
                # Generate final analysis
                final_prompt = self._prepare_final_prompt(all_insights, final_context, pattern_insights)
                final_response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert support ticket analyst providing comprehensive analysis and actionable insights. Always respond with valid JSON."},
                        {"role": "user", "content": final_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                
                # Validate and return final insights
                if not final_response or not final_response.choices:
                    raise ValueError("Empty response from OpenAI API")
                
                final_response_content = final_response.choices[0].message.content.strip()
                try:
                    final_insights = json.loads(final_response_content)
                    if not isinstance(final_insights, dict):
                        raise ValueError("Final response is not a dictionary")
                    
                    # Add detailed metadata
                    final_insights['metadata'] = {
                        'analysis_timestamp': datetime.now().isoformat(),
                        'tickets_analyzed': processed_tickets,
                        'total_tickets': total_tickets,
                        'chunks_processed': chunk_processing_metrics['total_chunks'],
                        'successful_chunks': chunk_processing_metrics['successful_chunks'],
                        'failed_chunks': chunk_processing_metrics['failed_chunks'],
                        'patterns_detected': pattern_recognition_metrics['total_patterns_found'],
                        'pattern_confidence_distribution': {
                            'high': pattern_recognition_metrics['high_confidence_patterns'],
                            'medium': pattern_recognition_metrics['medium_confidence_patterns'],
                            'low': pattern_recognition_metrics['low_confidence_patterns']
                        },
                        'pattern_recognition_duration': pattern_recognition_metrics['pattern_recognition_duration'],
                        'analysis_success_rate': (chunk_processing_metrics['successful_chunks'] / 
                                                chunk_processing_metrics['total_chunks'] * 100) if chunk_processing_metrics['total_chunks'] > 0 else 0
                    }
                    
                    self.logger.info("Analysis completed successfully", extra={
                        'metadata': final_insights['metadata']
                    })
                    
                    return final_insights
                    
                except json.JSONDecodeError as je:
                    raise ValueError(f"Failed to parse final response: {str(je)}")
                
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"Error in final analysis: {error_msg}", 
                                extra={'traceback': traceback.format_exc()})
                
                return {
                    'error': error_msg,
                    'details': 'Error during final analysis',
                    'context': {
                        'total_tickets': total_tickets,
                        'processed_tickets': processed_tickets,
                        'chunks_processed': len(all_insights),
                        'chunk_metrics': chunk_processing_metrics,
                        'pattern_metrics': pattern_recognition_metrics
                    }
                }
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error in ticket analysis: {error_msg}", 
                            extra={'traceback': traceback.format_exc()})
            
            return {
                'error': error_msg,
                'details': 'Error during ticket analysis',
                'context': {
                    'total_tickets': len(analysis_df) if analysis_df is not None else 0
                }
            }

    def _calculate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics from the ticket data.
        
        Args:
            df: DataFrame containing ticket data
            
        Returns:
            Dict containing summary statistics
        """
        try:
            stats = {
                'ticket_volume': {
                    'total': len(df),
                    'by_priority': df['Priority'].value_counts().to_dict() if 'Priority' in df.columns else {},
                    'by_status': df['Status'].value_counts().to_dict() if 'Status' in df.columns else {}
                },
                'resolution_metrics': {
                    'avg_resolution_time': df['Resolution Time (Days)'].mean() if 'Resolution Time (Days)' in df.columns else None,
                    'median_resolution_time': df['Resolution Time (Days)'].median() if 'Resolution Time (Days)' in df.columns else None
                },
                'customer_satisfaction': {
                    'avg_csat': df['CSAT'].mean() if 'CSAT' in df.columns else None,
                    'csat_distribution': df['CSAT'].value_counts().to_dict() if 'CSAT' in df.columns else {}
                },
                'product_metrics': {
                    'product_areas': df['Product Area'].value_counts().to_dict() if 'Product Area' in df.columns else {},
                    'features': df['Product Feature'].value_counts().to_dict() if 'Product Feature' in df.columns else {}
                },
                'escalation_metrics': {
                    'escalation_rate': (df['IsEscalated'].sum() / len(df) * 100) if 'IsEscalated' in df.columns else None,
                    'escalated_count': df['IsEscalated'].sum() if 'IsEscalated' in df.columns else None
                },
                'root_cause_analysis': {
                    'root_causes': df['Root Cause'].value_counts().to_dict() if 'Root Cause' in df.columns else {},
                    'unspecified_count': len(df[df['Root Cause'].isin(['Not Specified', 'Unknown', None])]) if 'Root Cause' in df.columns else None
                }
            }
            
            # Convert numpy types to Python native types for JSON serialization
            stats = convert_value_for_json(stats)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating summary stats: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'error': str(e),
                'details': 'Error calculating summary statistics'
            } 
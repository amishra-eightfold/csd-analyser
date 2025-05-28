"""OpenAI integration module for CSD Analyzer."""

from typing import Dict, List, Any, Optional, Union
import logging
import time
import traceback
from openai import OpenAI, APIError, RateLimitError
from pathlib import Path
import toml
from .config import config
from config.logging_config import setup_openai_logger, log_openai_request, log_openai_response, log_openai_error

class OpenAIClient:
    """Manages OpenAI API interactions for generating insights."""
    
    def __init__(self):
        """Initialize OpenAI client with API key from secrets.toml."""
        self._client = None
        self._max_retries = 3
        self._base_delay = 1  # Base delay for exponential backoff
        self._max_delay = 30  # Maximum delay between retries
        self.logger = setup_openai_logger()
        
        try:
            # Try to get API key from environment variable first
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            
            # If not found in environment, try secrets.toml
            if not api_key:
                try:
                    import streamlit as st
                    api_key = st.secrets.get("OPENAI_API_KEY", None)
                except:
                    # Fallback to reading secrets.toml directly
                    project_root = Path(__file__).parent.parent.parent
                    secrets_path = project_root / '.streamlit' / 'secrets.toml'
                    
                    if secrets_path.exists():
                        secrets = toml.load(secrets_path)
                        api_key = secrets.get('OPENAI_API_KEY')
            
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables or secrets.toml")
            
            # Initialize the client
            self._client = OpenAI(api_key=api_key)
            
            # Validate the API key by making a test call
            self._validate_api_key()
            
            self.logger.info("Successfully initialized OpenAI client with API key from secrets.toml")
            
        except Exception as e:
            log_openai_error(self.logger, e, {
                'stage': 'initialization',
                'secrets_path': str(secrets_path)
            })
            raise
    
    def _validate_api_key(self) -> None:
        """Validate the API key by making a test call."""
        try:
            start_time = time.time()
            
            # Log the test request
            log_openai_request(self.logger, 'chat.completions.create', {
                'model': 'gpt-3.5-turbo',
                'max_tokens': 1,
                'messages': '[REDACTED]'  # Don't log actual messages
            })
            
            # Make a minimal API call to validate the key
            response = self._client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            
            # Log the successful response
            processing_time = time.time() - start_time
            log_openai_response(self.logger, response, processing_time)
            
        except Exception as e:
            log_openai_error(self.logger, e, {
                'stage': 'api_key_validation'
            })
            raise ValueError(f"Invalid API key: {str(e)}")
    
    def _handle_api_error(self, e: Exception, attempt: int) -> None:
        """Handle API errors with appropriate logging and delay."""
        context = {
            'attempt': attempt,
            'max_retries': self._max_retries
        }
        
        if isinstance(e, RateLimitError):
            delay = min(self._base_delay * (2 ** (attempt - 1)), self._max_delay)
            context['delay'] = delay
            context['error_type'] = 'rate_limit'
            log_openai_error(self.logger, e, context)
            time.sleep(delay)
        elif isinstance(e, APIError):
            if "insufficient_quota" in str(e):
                context['error_type'] = 'quota_exceeded'
                log_openai_error(self.logger, e, context)
                raise ValueError("OpenAI API quota exceeded")
            delay = min(self._base_delay * (2 ** (attempt - 1)), self._max_delay)
            context['delay'] = delay
            context['error_type'] = 'api_error'
            log_openai_error(self.logger, e, context)
            time.sleep(delay)
        else:
            context['error_type'] = 'unknown'
            log_openai_error(self.logger, e, context)
            raise e
        
    def generate_insights(self, 
                         prompt: str, 
                         system_message: Optional[str] = None,
                         max_tokens: Optional[int] = None,
                         temperature: Optional[float] = None) -> Union[str, Dict[str, Any]]:
        """
        Generate insights using OpenAI's chat completion with retry logic.
        
        Args:
            prompt (str): User prompt for analysis
            system_message (Optional[str]): System message to guide the model
            max_tokens (Optional[int]): Maximum tokens for response
            temperature (Optional[float]): Temperature for response generation
            
        Returns:
            Union[str, Dict[str, Any]]: Generated insights or error response
            
        Raises:
            Exception: If API call fails after all retries
        """
        if not self._client:
            error_msg = 'OpenAI client not initialized'
            self.logger.error(error_msg)
            return {
                'error': error_msg,
                'details': 'Client initialization failed. Check logs for details.'
            }
            
        messages = []
        if system_message:
            messages.append({
                "role": "system",
                "content": system_message
            })
            
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        for attempt in range(1, self._max_retries + 1):
            try:
                start_time = time.time()
                
                # Log the request
                request_params = {
                    'model': config.openai_model,
                    'max_tokens': max_tokens or config.openai_max_tokens,
                    'temperature': temperature or config.openai_temperature,
                    'response_format': {"type": "json_object"},
                    'messages': '[REDACTED]'  # Don't log actual messages
                }
                log_openai_request(self.logger, 'chat.completions.create', request_params)
                
                # Make the API call
                response = self._client.chat.completions.create(
                    model=config.openai_model,
                    messages=messages,
                    max_tokens=max_tokens or config.openai_max_tokens,
                    temperature=temperature or config.openai_temperature,
                    response_format={"type": "json_object"}
                )
                
                # Log the successful response
                processing_time = time.time() - start_time
                log_openai_response(self.logger, response, processing_time)
                
                return response.choices[0].message.content
                
            except Exception as e:
                context = {
                    'attempt': attempt,
                    'max_retries': self._max_retries,
                    'model': config.openai_model,
                    'max_tokens': max_tokens or config.openai_max_tokens
                }
                log_openai_error(self.logger, e, context)
                
                if attempt < self._max_retries:
                    try:
                        self._handle_api_error(e, attempt)
                        continue
                    except ValueError as ve:
                        return {
                            'error': str(ve),
                            'details': 'API quota exceeded or invalid key'
                        }
                else:
                    return {
                        'error': str(e),
                        'details': 'Failed after all retry attempts'
                    }
            
    def analyze_support_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze support ticket data and generate insights with enhanced error handling.
        
        Args:
            data (Dict[str, Any]): Support data including metrics and patterns
            
        Returns:
            Dict[str, Any]: Dictionary containing various insights or error information
        """
        if not self._client:
            return {
                'error': 'OpenAI client not initialized',
                'details': 'Client initialization failed. Check logs for details.'
            }
            
        insights = {
            'error': None,
            'executive_summary': None,
            'recommendations': None,
            'risk_analysis': None
        }
        
        try:
            # Generate executive summary
            exec_prompt = f"""
            Analyze the following support ticket data and provide an executive summary:
            
            Key Metrics:
            - Total Tickets: {data.get('total_tickets', 0)}
            - Average Resolution Time: {data.get('avg_resolution_time', 0):.1f} days
            - CSAT Score: {data.get('avg_csat', 0):.2f}
            - Escalation Rate: {data.get('escalation_rate', 0):.1%}
            
            Top Issues:
            {data.get('top_issues', '')}
            
            Trend Analysis:
            {data.get('trend_analysis', '')}
            """
            
            exec_response = self.generate_insights(
                exec_prompt,
                system_message="You are a Support Operations Analyst providing insights on customer support data.",
                temperature=0.3
            )
            
            if isinstance(exec_response, dict) and 'error' in exec_response:
                return exec_response
            insights['executive_summary'] = exec_response
            
            # Generate recommendations
            rec_prompt = f"""
            Based on the following support data, provide specific recommendations for improvement:
            
            Pain Points:
            {data.get('pain_points', '')}
            
            Customer Feedback:
            {data.get('customer_feedback', '')}
            
            Product Areas:
            {data.get('product_areas', '')}
            """
            
            rec_response = self.generate_insights(
                rec_prompt,
                system_message="You are a Support Strategy Consultant providing actionable recommendations.",
                temperature=0.5
            )
            
            if isinstance(rec_response, dict) and 'error' in rec_response:
                return rec_response
            insights['recommendations'] = rec_response
            
            # Generate risk analysis
            risk_prompt = f"""
            Analyze potential risks based on the following support patterns:
            
            Escalation Patterns:
            {data.get('escalation_patterns', '')}
            
            Critical Issues:
            {data.get('critical_issues', '')}
            
            Response Times:
            {data.get('response_times', '')}
            """
            
            risk_response = self.generate_insights(
                risk_prompt,
                system_message="You are a Risk Analysis Expert identifying potential support risks.",
                temperature=0.4
            )
            
            if isinstance(risk_response, dict) and 'error' in risk_response:
                return risk_response
            insights['risk_analysis'] = risk_response
            
            return insights
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to analyze support data: {error_msg}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'error': error_msg,
                'details': 'Error during support data analysis',
                'executive_summary': "Error generating executive summary.",
                'recommendations': "Error generating recommendations.",
                'risk_analysis': "Error generating risk analysis."
            }

# Global OpenAI client instance (lazy initialization)
openai = None

def get_openai_client():
    """Get or create the global OpenAI client instance."""
    global openai
    if openai is None:
        openai = OpenAIClient()
    return openai

"""OpenAI integration module for CSD Analyzer."""

from typing import Dict, List, Any, Optional
import logging
from openai import OpenAI
from .config import config

logger = logging.getLogger(__name__)

class OpenAIClient:
    """Manages OpenAI API interactions for generating insights."""
    
    def __init__(self):
        """Initialize OpenAI client."""
        self._client = OpenAI(api_key=config.openai_api_key)
        
    def generate_insights(self, 
                         prompt: str, 
                         system_message: Optional[str] = None,
                         max_tokens: Optional[int] = None,
                         temperature: Optional[float] = None) -> str:
        """
        Generate insights using OpenAI's chat completion.
        
        Args:
            prompt (str): User prompt for analysis
            system_message (Optional[str]): System message to guide the model
            max_tokens (Optional[int]): Maximum tokens for response
            temperature (Optional[float]): Temperature for response generation
            
        Returns:
            str: Generated insights
            
        Raises:
            Exception: If API call fails
        """
        try:
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
            
            response = self._client.chat.completions.create(
                model=config.openai_model,
                messages=messages,
                max_tokens=max_tokens or config.openai_max_tokens,
                temperature=temperature or config.openai_temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {str(e)}")
            raise
            
    def analyze_support_data(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Analyze support ticket data and generate insights.
        
        Args:
            data (Dict[str, Any]): Support data including metrics and patterns
            
        Returns:
            Dict[str, str]: Dictionary containing various insights
        """
        try:
            insights = {}
            
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
            
            insights['executive_summary'] = self.generate_insights(
                exec_prompt,
                system_message="You are a Support Operations Analyst providing insights on customer support data.",
                temperature=0.3
            )
            
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
            
            insights['recommendations'] = self.generate_insights(
                rec_prompt,
                system_message="You are a Support Strategy Consultant providing actionable recommendations.",
                temperature=0.5
            )
            
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
            
            insights['risk_analysis'] = self.generate_insights(
                risk_prompt,
                system_message="You are a Risk Analysis Expert identifying potential support risks.",
                temperature=0.4
            )
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to analyze support data: {str(e)}")
            return {
                'executive_summary': "Error generating executive summary.",
                'recommendations': "Error generating recommendations.",
                'risk_analysis': "Error generating risk analysis."
            }

# Create global OpenAI client instance
openai = OpenAIClient()

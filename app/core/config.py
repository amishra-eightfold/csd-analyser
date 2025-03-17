"""Core configuration settings for the CSD Analyzer application."""

import os
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Custom color palettes for different visualizations
VIRIDIS_PALETTE = ["#440154", "#3B528B", "#21918C", "#5EC962", "#FDE725"]
AQUA_PALETTE = ["#E0F7FA", "#80DEEA", "#26C6DA", "#00ACC1", "#00838F", "#006064"]
PRIORITY_COLORS = {
    'P1': VIRIDIS_PALETTE[0],
    'P2': VIRIDIS_PALETTE[1],
    'P3': VIRIDIS_PALETTE[2],
    'P4': VIRIDIS_PALETTE[3]
}

# Define custom color palettes for each visualization type
VOLUME_PALETTE = [AQUA_PALETTE[2], AQUA_PALETTE[4]]  # Two distinct colors for Created/Closed
PRIORITY_PALETTE = VIRIDIS_PALETTE  # Viridis for priority levels
CSAT_PALETTE = AQUA_PALETTE  # Aqua palette for CSAT
HEATMAP_PALETTE = "Viridis"  # Viridis colorscale for heatmaps
ROOT_CAUSE_PALETTE = VIRIDIS_PALETTE

# Performance settings
QUERY_TIMEOUT = 30  # seconds
VISUALIZATION_TIMEOUT = 5  # seconds
EXPORT_TIMEOUT = 60  # seconds
DETAILED_DATA_TIMEOUT = 45  # seconds
AI_ANALYSIS_TIMEOUT = 120  # seconds
PII_PROCESSING_TIMEOUT = 5  # seconds per MB

# OpenAI settings
OPENAI_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 2000
TEMPERATURE = 0.7

class Config:
    """Configuration management for the application."""
    
    def __init__(self):
        """Initialize configuration with environment variables and defaults."""
        self.debug_mode = os.getenv('DEBUG_MODE', 'False').lower() in ('true', '1', 't')
        self.environment = os.getenv('ENVIRONMENT', 'development')
        
        # Default date range (last 30 days)
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=30)
        
        # API keys and credentials
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.salesforce_username = os.getenv('SALESFORCE_USERNAME')
        self.salesforce_password = os.getenv('SALESFORCE_PASSWORD')
        self.salesforce_security_token = os.getenv('SALESFORCE_SECURITY_TOKEN')
        
        # Feature flags
        self.enable_ai_analysis = False
        self.enable_pii_processing = False
        self.enable_detailed_analysis = True
        
        # OpenAI settings
        self.openai_model = OPENAI_MODEL
        self.openai_max_tokens = MAX_TOKENS
        self.openai_temperature = TEMPERATURE
        
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == 'production'
    
    def get_api_timeouts(self) -> Dict[str, int]:
        """Get API timeout settings."""
        return {
            'query': QUERY_TIMEOUT,
            'visualization': VISUALIZATION_TIMEOUT,
            'export': EXPORT_TIMEOUT,
            'detailed_data': DETAILED_DATA_TIMEOUT,
            'ai_analysis': AI_ANALYSIS_TIMEOUT,
            'pii_processing': PII_PROCESSING_TIMEOUT
        }
    
    def get_visualization_colors(self) -> Dict[str, Any]:
        """Get visualization color settings."""
        return {
            'priority_colors': PRIORITY_COLORS,
            'volume_palette': VOLUME_PALETTE,
            'priority_palette': PRIORITY_PALETTE,
            'csat_palette': CSAT_PALETTE,
            'heatmap_palette': HEATMAP_PALETTE,
            'root_cause_palette': ROOT_CAUSE_PALETTE,
            'viridis_palette': VIRIDIS_PALETTE
        }
    
    def validate(self) -> List[str]:
        """
        Validate configuration settings.
        
        Returns:
            List[str]: List of validation errors, empty if all valid
        """
        errors = []
        
        # Check required API keys
        if not self.openai_api_key and self.enable_ai_analysis:
            errors.append("OpenAI API key is required for AI analysis")
            
        # Check Salesforce credentials
        if not all([self.salesforce_username, self.salesforce_password, self.salesforce_security_token]):
            errors.append("Salesforce credentials are incomplete")
            
        return errors

# Create global config instance
config = Config()

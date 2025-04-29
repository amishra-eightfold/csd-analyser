"""Centralized logging configuration for the CSD Analyzer application."""

import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional
import traceback

# Create logs directory if it doesn't exist
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Define log formats
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'

# Define log files
LOG_FILES = {
    'app': 'app_debug.log',
    'api': 'api_debug.log',
    'error': 'error_debug.log',
    'token': 'token_usage.log',
    'performance': 'token_performance.log',
    'cost': 'token_cost.log',
    'pii': 'pii_audit.log'
}

class LogConfig:
    """Manages logging configuration for the application."""
    
    def __init__(self):
        """Initialize logging configuration."""
        self.loggers: Dict[str, logging.Logger] = {}
        self.setup_logging()
        
    def setup_logging(self):
        """Set up all loggers with appropriate handlers and formatters."""
        # Create formatters
        default_formatter = logging.Formatter(DEFAULT_FORMAT)
        detailed_formatter = logging.Formatter(DETAILED_FORMAT)
        
        # Configure root logger
        logging.basicConfig(level=logging.INFO)
        
        # Set up each logger
        for logger_name, log_file in LOG_FILES.items():
            # Create logger
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)
            
            # Create file handler
            file_handler = RotatingFileHandler(
                log_dir / log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            
            # Set formatter based on logger type
            if logger_name in ['error', 'api']:
                file_handler.setFormatter(detailed_formatter)
            else:
                file_handler.setFormatter(default_formatter)
            
            # Add handler to logger
            logger.addHandler(file_handler)
            
            # Store logger reference
            self.loggers[logger_name] = logger
    
    def get_logger(self, name: str) -> Optional[logging.Logger]:
        """Get a logger by name."""
        return self.loggers.get(name)
    
    def log_error(self, error: Exception, context: Optional[Dict] = None):
        """
        Log an error with context to the error logger.
        
        Args:
            error (Exception): The error that occurred
            context (Optional[Dict]): Additional context about the error
        """
        error_logger = self.get_logger('error')
        if error_logger:
            error_message = f"Error: {str(error)}"
            if context:
                error_message += f"\nContext: {context}"
            error_logger.error(error_message, exc_info=True)

# Create global instance
log_config = LogConfig()

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger by name. If the logger doesn't exist, returns the default app logger.
    
    Args:
        name (str): Name of the logger to retrieve
        
    Returns:
        logging.Logger: The requested logger or default app logger
    """
    logger = log_config.get_logger(name)
    if not logger:
        logger = log_config.get_logger('app')
    return logger

def log_error(error: Exception, context: Optional[Dict] = None):
    """
    Convenience function to log an error with context.
    
    Args:
        error (Exception): The error that occurred
        context (Optional[Dict]): Additional context about the error
    """
    log_config.log_error(error, context)

def setup_openai_logger():
    """Setup dedicated logger for OpenAI API interactions."""
    logger = logging.getLogger('openai')
    logger.setLevel(logging.DEBUG)
    
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Setup debug log file for OpenAI
    debug_file = log_dir / 'openai_debug.log'
    debug_handler = RotatingFileHandler(
        debug_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    debug_handler.setFormatter(debug_formatter)
    
    # Setup error log file for OpenAI
    error_file = log_dir / 'openai_error.log'
    error_handler = RotatingFileHandler(
        error_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s\nTraceback: %(traceback)s'
    )
    error_handler.setFormatter(error_formatter)
    
    # Add handlers to logger
    logger.addHandler(debug_handler)
    logger.addHandler(error_handler)
    
    return logger

def log_openai_request(logger, endpoint: str, params: dict):
    """Log OpenAI API request details."""
    logger.debug(
        "OpenAI API Request",
        extra={
            'endpoint': endpoint,
            'parameters': {
                k: v for k, v in params.items() 
                if k not in ('api_key', 'messages')  # Exclude sensitive data
            }
        }
    )

def log_openai_response(logger, response: dict, processing_time: float):
    """Log OpenAI API response details."""
    logger.debug(
        "OpenAI API Response",
        extra={
            'processing_time': f"{processing_time:.2f}s",
            'response_metadata': {
                'model': response.get('model'),
                'usage': response.get('usage'),
                'finish_reason': response.get('choices', [{}])[0].get('finish_reason')
            }
        }
    )

def log_openai_error(logger, error: Exception, context: dict = None):
    """Log OpenAI API errors with context."""
    logger.error(
        f"OpenAI API Error: {str(error)}",
        extra={
            'error_type': type(error).__name__,
            'error_details': str(error),
            'context': context or {},
            'traceback': traceback.format_exc()
        }
    ) 
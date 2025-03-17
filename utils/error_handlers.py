"""Error handling utilities for the CSD Analyzer application."""

import functools
import traceback
import logging
import streamlit as st
from typing import Callable, Any, Optional, Dict, Type
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ErrorHandler:
    """Base class for error handling functionality."""
    
    @staticmethod
    def log_error(error: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Log error details with context.
        
        Args:
            error (Exception): The error that occurred
            context (Optional[Dict[str, Any]]): Additional context about the error
        """
        error_details = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        }
        
        if context:
            error_details.update(context)
        
        # Log to debug logger if available
        if hasattr(st.session_state, 'debug_logger'):
            st.session_state.debug_logger.log(
                f"Error occurred: {error_details['error_type']}",
                error_details,
                category="error"
            )
        else:
            logger.error(f"Error occurred: {error_details}")
    
    @staticmethod
    def handle_error(error: Exception, 
                    show_traceback: bool = False,
                    error_message: Optional[str] = None):
        """
        Handle error by displaying appropriate message to user.
        
        Args:
            error (Exception): The error that occurred
            show_traceback (bool): Whether to show traceback in UI
            error_message (Optional[str]): Custom error message to display
        """
        if error_message:
            st.error(error_message)
        else:
            st.error(f"An error occurred: {str(error)}")
            
        if show_traceback and hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.error(f"Traceback: {traceback.format_exc()}")
            
        # Log error details
        ErrorHandler.log_error(error, {
            'show_traceback': show_traceback,
            'custom_message': error_message
        })

def handle_errors(error_types: Optional[Type[Exception]] = Exception,
                 show_traceback: bool = False,
                 custom_message: Optional[str] = None):
    """
    Decorator for handling errors in functions.
    
    Args:
        error_types (Optional[Type[Exception]]): Types of errors to catch
        show_traceback (bool): Whether to show traceback in UI
        custom_message (Optional[str]): Custom error message to display
        
    Returns:
        callable: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                # Log the error
                ErrorHandler.log_error(e, {
                    'function': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                })
                
                # Handle the error
                ErrorHandler.handle_error(
                    e,
                    show_traceback=show_traceback,
                    error_message=custom_message
                )
                
                # Return None to indicate error
                return None
        return wrapper
    return decorator

def track_progress(total_steps: int):
    """
    Decorator for tracking progress of long-running functions.
    
    Args:
        total_steps (int): Total number of steps in the operation
        
    Returns:
        callable: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create progress bar
            progress_bar = st.progress(0)
            step = 0
            
            def update_progress():
                nonlocal step
                step += 1
                progress = min(step / total_steps, 1.0)
                progress_bar.progress(progress)
            
            # Add progress_callback to kwargs
            kwargs['progress_callback'] = update_progress
            
            try:
                result = func(*args, **kwargs)
                progress_bar.progress(1.0)
                return result
            except Exception as e:
                ErrorHandler.log_error(e, {
                    'function': func.__name__,
                    'progress_step': step,
                    'total_steps': total_steps
                })
                raise
            finally:
                # Clean up progress bar
                progress_bar.empty()
        return wrapper
    return decorator

def retry_on_error(max_retries: int = 3, 
                  retry_delay: float = 1.0,
                  error_types: Optional[Type[Exception]] = Exception):
    """
    Decorator for retrying functions on error.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        retry_delay (float): Delay between retries in seconds
        error_types (Optional[Type[Exception]]): Types of errors to retry on
        
    Returns:
        callable: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except error_types as e:
                    if attempt == max_retries - 1:
                        # Log final error
                        ErrorHandler.log_error(e, {
                            'function': func.__name__,
                            'attempt': attempt + 1,
                            'max_retries': max_retries
                        })
                        raise
                    
                    # Log retry attempt
                    logger.warning(f"Retry attempt {attempt + 1} of {max_retries} for {func.__name__}")
                    time.sleep(retry_delay)
        return wrapper
    return decorator 
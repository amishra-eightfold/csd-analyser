"""
Advanced AI analysis utilities for support ticket analysis.

This module is maintained for backward compatibility. New code should
import from the utils.ai_analysis package directly.
"""

# Re-export all public classes and functions from the new package
from .ai_analysis.analyzer import AIAnalyzer
from .ai_analysis.context_manager import ContextManager
from .ai_analysis.openai_logger import OpenAILogger
from .ai_analysis.utils import preprocess_text_for_ai 
"""CSD Analyzer package."""

__version__ = '1.0.0'

from .data import fetch_data
from .core.openai_client import openai
from .core import salesforce
from .analysis.display import display_detailed_analysis
from .analysis.ai_insights import generate_ai_insights

__all__ = ['fetch_data', 'salesforce', 'openai', 'display_detailed_analysis', 'generate_ai_insights'] 
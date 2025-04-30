"""Utility functions for AI analysis."""

from typing import Dict, List, Any, Optional
import json
from ..text_processing import clean_text, get_technical_stopwords

def preprocess_text_for_ai(text: str) -> str:
    """
    Enhanced text preprocessing for AI analysis.
    
    Args:
        text: Input text to process
        
    Returns:
        str: Processed text ready for AI analysis
    """
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

def convert_confidence_to_float(confidence_str: str) -> float:
    """
    Convert string confidence level to float value.
    
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

def calculate_ticket_importance(ticket: Dict[str, Any]) -> float:
    """
    Calculate importance score for a ticket based on its attributes.
    
    Args:
        ticket: Dictionary containing ticket information
        
    Returns:
        float: Importance score between 0 and 1
    """
    importance = 0.5  # Default importance
    
    # Add weight based on priority
    priority = ticket.get('priority', '').upper()
    if priority == 'P0':
        importance += 0.4
    elif priority == 'P1':
        importance += 0.3
    elif priority == 'P2':
        importance += 0.2
    elif priority == 'P3':
        importance += 0.1
    
    # Recent tickets slightly more important
    created_date = ticket.get('created_date', '')
    if created_date and '2023' in created_date:
        importance += 0.1
    
    # CSAT influence
    csat = ticket.get('csat')
    if csat is not None:
        try:
            csat_value = float(csat)
            # Lower CSAT means higher importance (issues to learn from)
            if csat_value <= 2:
                importance += 0.2
            elif csat_value <= 3:
                importance += 0.1
        except (ValueError, TypeError):
            pass
    
    # Cap at 1.0
    return min(importance, 1.0)

def prepare_patterns_dict(patterns: List[str], confidences: Dict[str, float], 
                         frequencies: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
    """
    Prepare a structured dictionary of patterns with metadata.
    
    Args:
        patterns: List of pattern strings
        confidences: Dictionary mapping patterns to confidence values
        frequencies: Dictionary mapping patterns to frequency counts
        
    Returns:
        Dict: Structured patterns dictionary with metadata
    """
    result = {}
    
    for pattern in patterns:
        result[pattern] = {
            'confidence': confidences.get(pattern, 0.5),
            'frequency': frequencies.get(pattern, 1),
            'importance': confidences.get(pattern, 0.5) * frequencies.get(pattern, 1)
        }
    
    return result 
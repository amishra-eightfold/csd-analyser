"""Context management for AI analysis."""

from typing import Dict, List, Any
from collections import defaultdict

class ContextManager:
    """
    Manages context across multiple analysis chunks.
    
    This class is responsible for storing and retrieving context information
    during the analysis of support tickets, allowing for patterns and insights
    to be tracked across multiple processing chunks.
    """
    
    def __init__(self) -> None:
        """Initialize the context manager with empty context structures."""
        self.previous_insights = []
        self.global_patterns = defaultdict(int)
        self.priority_context = defaultdict(list)
        self.temporal_context = defaultdict(list)
        self.pattern_insights = []  # Add storage for pattern insights
        
    def add_insight(self, insight: Dict[str, Any]) -> None:
        """
        Add an insight to the context.
        
        Args:
            insight: Dictionary containing insight information
        """
        self.previous_insights.append(insight)
        if len(self.previous_insights) > 5:  # Keep last 5 insights
            self.previous_insights.pop(0)
            
    def update_patterns(self, patterns: Dict[str, int]) -> None:
        """
        Update global pattern frequencies.
        
        Args:
            patterns: Dictionary of pattern names to frequencies
        """
        for pattern, freq in patterns.items():
            self.global_patterns[pattern] += freq
            
    def add_priority_context(self, priority: str, data: Dict[str, Any]) -> None:
        """
        Add context for a specific priority level.
        
        Args:
            priority: The priority level (e.g., 'P0', 'P1')
            data: Dictionary containing context data for this priority
        """
        self.priority_context[priority].append(data)
        if len(self.priority_context[priority]) > 3:  # Keep last 3 entries
            self.priority_context[priority].pop(0)
            
    def add_temporal_context(self, time_period: str, data: Dict[str, Any]) -> None:
        """
        Add temporal context for trend analysis.
        
        Args:
            time_period: String representing the time period
            data: Dictionary containing context data for this time period
        """
        self.temporal_context[time_period].append(data)
        
    def add_pattern_insight(self, insight: Dict[str, Any]) -> None:
        """
        Add pattern recognition insight.
        
        Args:
            insight: Dictionary containing pattern insight information
        """
        self.pattern_insights.append(insight)
        if len(self.pattern_insights) > 10:  # Keep last 10 pattern insights
            self.pattern_insights.pop(0)
        
    def get_summary_context(self) -> Dict[str, Any]:
        """
        Get consolidated context for summary generation.
        
        Returns:
            Dict: Consolidated context information
        """
        return {
            "previous_insights": self.previous_insights,
            "global_patterns": dict(self.global_patterns),
            "priority_context": dict(self.priority_context),
            "temporal_context": dict(self.temporal_context),
            "pattern_insights": self.pattern_insights
        } 
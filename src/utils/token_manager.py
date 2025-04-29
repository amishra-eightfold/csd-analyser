"""Token management utilities for handling large text processing."""

import logging
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from datetime import datetime
import math
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('token_usage')

class TokenManager:
    """Manages token usage and chunk creation for large text processing."""
    
    def __init__(self, token_limit: int = 4096, model: str = "gpt-3.5-turbo"):
        """
        Initialize the token manager.
        
        Args:
            token_limit: Maximum tokens per request
            model: The model being used for processing
        """
        self.token_limit = token_limit
        self.model = model
        self.total_tokens_processed = 0
        self.request_count = 0
        self.available_tokens = token_limit - 1200  # Reserve tokens for system messages
        
        logger.info(f"Initialized TokenManager with model {model}\n"
                   f"Token limit: {token_limit}\n"
                   f"Available tokens: {self.available_tokens}")

    def create_chunks(self, items: List[Dict]) -> List[Tuple[List[Dict], Dict]]:
        """Create chunks of items with preserved context."""
        if not items:
            return []

        chunks = []
        current_chunk = []
        current_tokens = 0
        total_chunks = math.ceil(len(items) / self.chunk_size)

        # Calculate global stats once
        global_stats = self._calculate_global_stats(items)

        for i, item in enumerate(items):
            item_tokens = self._estimate_item_tokens(item)
            
            if current_tokens + item_tokens > self.available_tokens or len(current_chunk) >= self.chunk_size:
                if current_chunk:  # Only append if we have items
                    context = {
                        'global_stats': global_stats,
                        'chunk_position': len(chunks) + 1,
                        'total_chunks': total_chunks,
                        'chunk_stats': self._calculate_chunk_stats(current_chunk)
                    }
                    chunks.append((current_chunk, context))
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(item)
            current_tokens += item_tokens

        # Don't forget the last chunk
        if current_chunk:
            context = {
                'global_stats': global_stats,
                'chunk_position': len(chunks) + 1,
                'total_chunks': total_chunks,
                'chunk_stats': self._calculate_chunk_stats(current_chunk)
            }
            chunks.append((current_chunk, context))

        return chunks

    def _calculate_global_stats(self, items: List[Dict]) -> Dict:
        """Calculate global statistics for all items."""
        df = pd.DataFrame(items)
        
        stats = {
            'total_items': len(items),
            'date_range': None  # Initialize to None
        }
        
        # Calculate date range if CreatedDate exists
        if 'CreatedDate' in df.columns:
            created_dates = pd.to_datetime(df['CreatedDate'])
            stats['date_range'] = {
                'start': created_dates.min().isoformat(),
                'end': created_dates.max().isoformat()
            }
        
        # Calculate distributions for various fields
        for field in ['Priority', 'Product_Area__c', 'Product_Feature__c', 'RCA__c']:
            if field in df.columns:
                dist_name = field.replace('__c', '').lower() + '_distribution'
                stats[dist_name] = df[field].value_counts().to_dict()
                stats[f'unique_{field.replace("__c", "").lower()}s'] = len(stats[dist_name])
        
        return stats

    def _calculate_chunk_stats(self, chunk: List[Dict]) -> Dict:
        """Calculate statistics for a specific chunk."""
        return {
            'chunk_size': len(chunk),
            'avg_priority': self._calculate_avg_priority(chunk)
        }

    def _calculate_avg_priority(self, items: List[Dict]) -> float:
        """Calculate average priority score for items."""
        priority_map = {'P0': 4, 'P1': 3, 'P2': 2, 'P3': 1}
        priorities = [item.get('Priority') for item in items if item.get('Priority') in priority_map]
        if not priorities:
            return 0.0
        return sum(priority_map[p] for p in priorities) / len(priorities)

    def _estimate_item_tokens(self, item: Dict) -> int:
        """Estimate the number of tokens in an item."""
        # Simple estimation based on character count
        text = json.dumps(item)
        return len(text) // 4  # Rough estimate of tokens based on characters

    def get_token_info(self, chunk_data: Tuple[List[Dict], Dict]) -> Dict:
        """Get token usage information for a chunk."""
        items, context = chunk_data
        
        # Calculate token counts
        original_tokens = sum(self._estimate_item_tokens(item) for item in items)
        context_tokens = self._estimate_item_tokens(context)
        total_tokens = original_tokens + context_tokens
        
        # Update global counters
        self.total_tokens_processed += total_tokens
        self.request_count += 1
        
        return {
            'original_tokens': original_tokens,
            'context_tokens': context_tokens,
            'total_tokens': total_tokens,
            'total_processed': self.total_tokens_processed,
            'request_count': self.request_count
        }
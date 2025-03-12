import tiktoken
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import pandas as pd

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle non-serializable objects."""
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super().default(obj)

# Set up logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create separate loggers for different aspects
token_logger = logging.getLogger('token_usage')
performance_logger = logging.getLogger('token_performance')
cost_logger = logging.getLogger('token_cost')

# Set log levels
token_logger.setLevel(logging.INFO)
performance_logger.setLevel(logging.INFO)
cost_logger.setLevel(logging.INFO)

# Create handlers for each logger
handlers = {}
for name in ['token_usage', 'token_performance', 'token_cost']:
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, f'{name}.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    console_handler = logging.StreamHandler()
    
    # Create formatter
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    # Store handlers
    handlers[name] = (file_handler, console_handler)
    
    # Get logger and add handlers
    logger = logging.getLogger(name)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

@dataclass
class TokenInfo:
    """Class to store token usage information."""
    total_tokens: int
    content_tokens: int
    metadata_tokens: int
    context_tokens: int = 0
    cost_estimate: float = 0.0  # Cost in USD
    processing_time: float = 0.0  # Processing time in seconds
    compression_ratio: float = 1.0  # Ratio of compressed to original tokens
    chunk_size: int = 0  # Number of items in the chunk

@dataclass
class TokenUsageStats:
    """Class to store token usage statistics."""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_tokens_per_request: float = 0.0
    avg_cost_per_request: float = 0.0
    peak_tokens: int = 0
    peak_cost: float = 0.0
    compression_stats: Dict[str, float] = None
    performance_stats: Dict[str, float] = None

def convert_value_for_json(value: Any) -> Any:
    """Convert a value to a JSON-serializable format."""
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    elif isinstance(value, (int, float, str, bool, type(None))):
        return value
    elif isinstance(value, (list, tuple)):
        return [convert_value_for_json(item) for item in value]
    elif isinstance(value, dict):
        return {k: convert_value_for_json(v) for k, v in value.items()}
    else:
        return str(value)

class TokenManager:
    """Manages token usage and chunking with enhanced context preservation."""
    
    # Model token limits
    MODEL_LIMITS = {
        'gpt-3.5-turbo': 4096,
        'gpt-4': 8192,
        'gpt-4-32k': 32768
    }
    
    # Reserve tokens for system message and response
    SYSTEM_TOKENS = 200
    RESPONSE_TOKENS = 1000
    
    # Token costs per 1K tokens (in USD)
    TOKEN_COSTS = {
        'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-4-32k': {'input': 0.06, 'output': 0.12}
    }
    
    def __init__(self, model_name: str = 'gpt-3.5-turbo'):
        """Initialize TokenManager with model settings."""
        self.model_name = model_name
        self.model_limit = self.MODEL_LIMITS.get(model_name, 4096)
        self.available_tokens = self.model_limit - self.SYSTEM_TOKENS - self.RESPONSE_TOKENS
        self.chunk_tokens = int(self.available_tokens * 0.7)  # Leave room for context
        
        # Initialize token tracking
        self.total_tokens_processed = 0
        self.total_cost = 0.0
        self.request_count = 0
        self.peak_tokens = 0
        self.peak_cost = 0.0
        
        # Initialize performance tracking
        self.compression_ratios = []
        self.processing_times = []
        
        # Initialize encoder
        self.encoder = tiktoken.encoding_for_model(model_name)
        
        # Initialize JSON encoder
        self.json_encoder = CustomJSONEncoder()
        
        # Get loggers
        self.token_logger = logging.getLogger('token_usage')
        self.performance_logger = logging.getLogger('token_performance')
        self.cost_logger = logging.getLogger('token_cost')
        
        # Log initialization
        self.token_logger.info(
            f"Initialized TokenManager with model {model_name}\n"
            f"Token limit: {self.model_limit}\n"
            f"Available tokens: {self.available_tokens}"
        )
    
    def get_usage_stats(self) -> TokenUsageStats:
        """Get current token usage statistics."""
        stats = TokenUsageStats(
            total_requests=self.request_count,
            total_tokens=self.total_tokens_processed,
            total_cost=self.total_cost,
            avg_tokens_per_request=self.total_tokens_processed / max(1, self.request_count),
            avg_cost_per_request=self.total_cost / max(1, self.request_count),
            peak_tokens=self.peak_tokens,
            peak_cost=self.peak_cost,
            compression_stats={
                'avg_ratio': np.mean(self.compression_ratios) if self.compression_ratios else 1.0,
                'min_ratio': min(self.compression_ratios) if self.compression_ratios else 1.0,
                'max_ratio': max(self.compression_ratios) if self.compression_ratios else 1.0
            },
            performance_stats={
                'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0.0,
                'min_processing_time': min(self.processing_times) if self.processing_times else 0.0,
                'max_processing_time': max(self.processing_times) if self.processing_times else 0.0
            }
        )
        
        # Log statistics
        self.token_logger.info(
            f"Token Usage Statistics:\n"
            f"Total Requests: {stats.total_requests}\n"
            f"Total Tokens: {stats.total_tokens}\n"
            f"Average Tokens per Request: {stats.avg_tokens_per_request:.2f}"
        )
        
        self.cost_logger.info(
            f"Cost Statistics:\n"
            f"Total Cost: ${stats.total_cost:.4f}\n"
            f"Average Cost per Request: ${stats.avg_cost_per_request:.4f}\n"
            f"Peak Cost: ${stats.peak_cost:.4f}"
        )
        
        self.performance_logger.info(
            f"Performance Statistics:\n"
            f"Average Compression Ratio: {stats.compression_stats['avg_ratio']:.2f}\n"
            f"Average Processing Time: {stats.performance_stats['avg_processing_time']:.3f}s"
        )
        
        return stats
    
    def log_token_usage(self, token_info: TokenInfo, context: str = "") -> None:
        """Log token usage information with enhanced metrics."""
        # Update counters
        self.total_tokens_processed += token_info.total_tokens
        self.request_count += 1
        self.peak_tokens = max(self.peak_tokens, token_info.total_tokens)
        
        # Calculate and update costs
        cost = self.calculate_cost(token_info.total_tokens)
        self.total_cost += cost
        self.peak_cost = max(self.peak_cost, cost)
        
        # Update performance metrics
        if token_info.processing_time > 0:
            self.processing_times.append(token_info.processing_time)
        if token_info.compression_ratio != 1.0:
            self.compression_ratios.append(token_info.compression_ratio)
        
        # Log detailed information
        self.token_logger.info(
            f"Token Usage - Context: {context}\n"
            f"Total Tokens: {token_info.total_tokens}\n"
            f"Content Tokens: {token_info.content_tokens}\n"
            f"Metadata Tokens: {token_info.metadata_tokens}\n"
            f"Context Tokens: {token_info.context_tokens}\n"
            f"Chunk Size: {token_info.chunk_size} items"
        )
        
        self.cost_logger.info(
            f"Cost Analysis - Context: {context}\n"
            f"Cost Estimate: ${cost:.4f}\n"
            f"Cumulative Cost: ${self.total_cost:.4f}\n"
            f"Average Cost per Request: ${self.total_cost/self.request_count:.4f}"
        )
        
        self.performance_logger.info(
            f"Performance Metrics - Context: {context}\n"
            f"Processing Time: {token_info.processing_time:.3f}s\n"
            f"Compression Ratio: {token_info.compression_ratio:.2f}\n"
            f"Token Utilization: {(token_info.total_tokens/self.model_limit)*100:.1f}%"
        )
    
    def get_token_info(self, chunk_with_context: Tuple[List[Dict[str, Any]], Dict[str, Any]], context: str = "") -> TokenInfo:
        """Get token usage information for a chunk with enhanced metrics."""
        import time
        start_time = time.time()
        
        chunk, context_data = chunk_with_context
        
        # Calculate token counts using custom JSON encoder
        content_tokens = sum(self.count_tokens(self.json_dumps(item)) for item in chunk)
        context_tokens = self.count_tokens(self.json_dumps(context_data))
        total_tokens = content_tokens + context_tokens + self.SYSTEM_TOKENS
        
        # Calculate compression ratio if content was compressed
        original_tokens = content_tokens
        if hasattr(chunk, 'original_token_count'):
            original_tokens = chunk.original_token_count
        compression_ratio = content_tokens / max(1, original_tokens)
        
        # Create token info with enhanced metrics
        token_info = TokenInfo(
            total_tokens=total_tokens,
            content_tokens=content_tokens,
            metadata_tokens=context_tokens,
            context_tokens=self.SYSTEM_TOKENS,
            cost_estimate=self.calculate_cost(total_tokens),
            processing_time=time.time() - start_time,
            compression_ratio=compression_ratio,
            chunk_size=len(chunk)
        )
        
        # Log the information
        self.log_token_usage(token_info, context)
        
        return token_info

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if not text:
            return 0
        return len(self.encoder.encode(str(text)))
    
    def json_dumps(self, obj: Any) -> str:
        """Serialize object to JSON string using custom encoder."""
        return json.dumps(obj, cls=CustomJSONEncoder)
    
    def calculate_priority_score(self, item: Dict[str, Any]) -> float:
        """Calculate priority score for context preservation."""
        score = 0.0
        
        # Priority based on ticket status
        priority_weights = {'P0': 1.0, 'P1': 0.8, 'P2': 0.6, 'P3': 0.4}
        priority = item.get('Internal_Priority__c', 'P3')
        score += priority_weights.get(priority, 0.2)
        
        # Recency factor (more recent items get higher priority)
        if 'CreatedDate' in item:
            try:
                created_date = datetime.strptime(item['CreatedDate'], '%Y-%m-%d %H:%M:%S')
                days_old = (datetime.now() - created_date).days
                recency_score = max(0, 1 - (days_old / 365))  # Normalize to 1 year
                score += recency_score * 0.5
            except (ValueError, TypeError):
                pass
        
        # Content richness factor
        description = str(item.get('Description', ''))
        content_length_score = min(len(description) / 1000, 1.0)  # Normalize to 1000 chars
        score += content_length_score * 0.3
        
        return score
    
    def extract_metadata_context(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract relevant metadata context from items."""
        metadata = {
            'product_areas': set(),
            'features': set(),
            'priorities': set(),
            'root_causes': set(),
            'date_range': {'start': None, 'end': None},
            'key_relationships': {}
        }
        
        for item in items:
            # Collect unique values
            if 'Product_Area__c' in item:
                metadata['product_areas'].add(item['Product_Area__c'])
            if 'Product_Feature__c' in item:
                metadata['features'].add(item['Product_Feature__c'])
            if 'Internal_Priority__c' in item:
                metadata['priorities'].add(item['Internal_Priority__c'])
            if 'RCA__c' in item:
                metadata['root_causes'].add(item['RCA__c'])
            
            # Track date range
            if 'CreatedDate' in item:
                try:
                    date = datetime.strptime(item['CreatedDate'], '%Y-%m-%d %H:%M:%S')
                    if metadata['date_range']['start'] is None or date < metadata['date_range']['start']:
                        metadata['date_range']['start'] = date
                    if metadata['date_range']['end'] is None or date > metadata['date_range']['end']:
                        metadata['date_range']['end'] = date
                except (ValueError, TypeError):
                    pass
            
            # Track relationships
            if 'Id' in item:
                metadata['key_relationships'][item['Id']] = {
                    'product_area': item.get('Product_Area__c'),
                    'feature': item.get('Product_Feature__c'),
                    'priority': item.get('Internal_Priority__c')
                }
        
        # Convert sets to lists for JSON serialization
        metadata['product_areas'] = list(metadata['product_areas'])
        metadata['features'] = list(metadata['features'])
        metadata['priorities'] = list(metadata['priorities'])
        metadata['root_causes'] = list(metadata['root_causes'])
        
        # Convert datetime objects to strings
        if metadata['date_range']['start']:
            metadata['date_range']['start'] = metadata['date_range']['start'].strftime('%Y-%m-%d %H:%M:%S')
        if metadata['date_range']['end']:
            metadata['date_range']['end'] = metadata['date_range']['end'].strftime('%Y-%m-%d %H:%M:%S')
        
        return metadata
    
    def create_semantic_groups(self, items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group items based on semantic relationships."""
        groups = []
        processed = set()
        
        for item in items:
            if item.get('Id') in processed:
                continue
                
            group = [item]
            processed.add(item.get('Id'))
            
            # Find related items based on product area, feature, and root cause
            for other in items:
                if other.get('Id') in processed:
                    continue
                    
                is_related = (
                    other.get('Product_Area__c') == item.get('Product_Area__c') or
                    other.get('Product_Feature__c') == item.get('Product_Feature__c') or
                    other.get('RCA__c') == item.get('RCA__c')
                )
                
                if is_related:
                    group.append(other)
                    processed.add(other.get('Id'))
            
            groups.append(group)
        
        return groups
    
    def create_chunks(self, items: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Create optimized chunks from items while preserving context.
        
        Args:
            items: List of dictionaries containing item data
            context: Optional context dictionary to include with each chunk
            
        Returns:
            List of chunks with preserved context
        """
        if not items:
            return []
        
        # Calculate global stats
        global_stats = {
            'total_items': len(items),
            'unique_product_areas': len(set(item.get('Product_Area__c', '') for item in items)),
            'unique_features': len(set(item.get('Product_Feature__c', '') for item in items)),
            'unique_priorities': len(set(item.get('Internal_Priority__c', '') for item in items)),
            'unique_root_causes': len(set(item.get('RCA__c', '') for item in items)),
            'date_range': {
                'start': min(item.get('CreatedDate', '') for item in items),
                'end': max(item.get('CreatedDate', '') for item in items)
            }
        }
        
        # Sort items by priority and date
        sorted_items = sorted(
            items,
            key=lambda x: (
                self._get_priority_score(x.get('Internal_Priority__c', '')),
                x.get('CreatedDate', '')
            ),
            reverse=True
        )
        
        # Initialize chunks
        chunks = []
        current_chunk = []
        current_chunk_tokens = 0
        
        # Process each item
        for item in sorted_items:
            # Calculate item tokens including context
            item_context = {
                'global_stats': global_stats,
                'chunk_position': len(chunks) + 1,
                'total_chunks': '[to be updated]',
                'chunk_stats': self._get_chunk_stats(current_chunk + [item])
            }
            
            # Merge with provided context
            if context:
                item_context.update(context)
            
            # Calculate total tokens for item with context
            total_tokens = self.count_tokens(self.json_dumps({
                'items': current_chunk + [item],
                'context': item_context
            }))
            
            if total_tokens <= self.chunk_tokens:
                # Item fits in current chunk
                current_chunk.append(item)
                current_chunk_tokens = total_tokens
            else:
                if not current_chunk:
                    # Single item is too large, truncate it
                    truncated_item = self.truncate_item(item)
                    if truncated_item:
                        current_chunk.append(truncated_item)
                
                # Finalize current chunk if not empty
                if current_chunk:
                    chunks.append({
                        'items': current_chunk,
                        'context': {
                            'global_stats': global_stats,
                            'chunk_position': len(chunks) + 1,
                            'total_chunks': '[to be updated]',
                            'chunk_stats': self._get_chunk_stats(current_chunk)
                        }
                    })
                
                # Start new chunk
                current_chunk = []
                current_chunk_tokens = 0
                
                # Try to add current item to new chunk
                if item not in current_chunk:  # Skip if it was truncated and added above
                    total_tokens = self.count_tokens(self.json_dumps({
                        'items': [item],
                        'context': item_context
                    }))
                    
                    if total_tokens <= self.chunk_tokens:
                        current_chunk.append(item)
                        current_chunk_tokens = total_tokens
                    else:
                        # Truncate item if too large
                        truncated_item = self.truncate_item(item)
                        if truncated_item:
                            current_chunk.append(truncated_item)
                            current_chunk_tokens = self.count_tokens(self.json_dumps({
                                'items': current_chunk,
                                'context': item_context
                            }))
        
        # Add final chunk if not empty
        if current_chunk:
            chunks.append({
                'items': current_chunk,
                'context': {
                    'global_stats': global_stats,
                    'chunk_position': len(chunks) + 1,
                    'total_chunks': '[to be updated]',
                    'chunk_stats': self._get_chunk_stats(current_chunk)
                }
            })
        
        # Update total chunks
        for chunk in chunks:
            chunk['context']['total_chunks'] = len(chunks)
        
        # Log chunking metrics
        self.token_logger.info(
            f"Chunking Metrics:\n"
            f"Total items: {len(items)}\n"
            f"Total chunks: {len(chunks)}\n"
            f"Average items per chunk: {len(items) / len(chunks) if chunks else 0:.2f}\n"
            f"Max chunk tokens: {self.chunk_tokens}"
        )
        
        return chunks
    
    def _get_priority_score(self, priority: str) -> int:
        """Get numeric score for priority sorting."""
        priority_scores = {'P0': 4, 'P1': 3, 'P2': 2, 'P3': 1}
        return priority_scores.get(priority, 0)
    
    def _get_chunk_stats(self, chunk: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics for a specific chunk."""
        return {
            'size': len(chunk),
            'priorities': list(set(item.get('Internal_Priority__c') for item in chunk if item.get('Internal_Priority__c'))),
            'product_areas': list(set(item.get('Product_Area__c') for item in chunk if item.get('Product_Area__c'))),
            'features': list(set(item.get('Product_Feature__c') for item in chunk if item.get('Product_Feature__c')))
        }
    
    def split_large_group(self, group: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Split a large group into smaller chunks that fit within token limits."""
        result = []
        current_chunk = []
        current_tokens = 0
        
        for item in group:
            item_tokens = self.count_tokens(self.json_dumps(item))
            
            # If the item itself is too large, truncate it
            if item_tokens > self.chunk_tokens:
                truncated_item = self.truncate_item(item)
                item_tokens = self.count_tokens(self.json_dumps(truncated_item))
                if current_tokens + item_tokens <= self.chunk_tokens:
                    current_chunk.append(truncated_item)
                    current_tokens += item_tokens
                else:
                    if current_chunk:
                        result.append(current_chunk)
                    current_chunk = [truncated_item]
                    current_tokens = item_tokens
            else:
                if current_tokens + item_tokens <= self.chunk_tokens:
                    current_chunk.append(item)
                    current_tokens += item_tokens
                else:
                    if current_chunk:
                        result.append(current_chunk)
                    current_chunk = [item]
                    current_tokens = item_tokens
        
        if current_chunk:
            result.append(current_chunk)
        
        return result
    
    def truncate_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Truncate large text fields in an item to fit within token limits."""
        truncated = item.copy()
        text_fields = ['Description', 'Subject', 'Comments']
        
        for field in text_fields:
            if field in truncated and isinstance(truncated[field], str):
                field_tokens = self.count_tokens(truncated[field])
                if field_tokens > self.chunk_tokens // 2:  # Allow half the chunk size for a single field
                    truncated[field] = self.truncate_text(truncated[field], self.chunk_tokens // 2)
        
        return truncated
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens while preserving meaning."""
        if self.count_tokens(text) <= max_tokens:
            return text
        
        # Split into sentences and gradually build up until we hit the token limit
        sentences = text.split('. ')
        result = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence + '. ')
            if current_tokens + sentence_tokens <= max_tokens:
                result.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        truncated = '. '.join(result)
        if not truncated.endswith('.'):
            truncated += '.'
        
        return truncated + ' [truncated...]'
    
    def optimize_context(self, context: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """Optimize context to fit within max_tokens."""
        if self.count_tokens(self.json_dumps(context)) <= max_tokens:
            return context
        
        # Prioritize certain context fields
        priority_fields = ['metadata', 'summary', 'key_patterns']
        optimized = {}
        
        # Add high-priority fields first
        for field in priority_fields:
            if field in context:
                optimized[field] = context[field]
                if self.count_tokens(self.json_dumps(optimized)) > max_tokens:
                    # If we exceed the limit, start removing fields from the end
                    while optimized and self.count_tokens(self.json_dumps(optimized)) > max_tokens:
                        optimized.popitem()
                    break
        
        return optimized
    
    def calculate_cost(self, token_count: int, is_output: bool = False) -> float:
        """Calculate cost for token usage."""
        pricing = self.TOKEN_COSTS.get(self.model_name, self.TOKEN_COSTS['gpt-3.5-turbo'])
        rate = pricing['output'] if is_output else pricing['input']
        return (token_count / 1000) * rate 
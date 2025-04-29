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
    """Custom JSON encoder to handle pandas Timestamp objects."""
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
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
    original_tokens: int = 0

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
    """Manages token usage and chunking for OpenAI API calls."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.encoder = tiktoken.encoding_for_model(model)
        self.max_tokens = 4096  # Default for gpt-3.5-turbo
        self.chunk_tokens = int(self.max_tokens * 0.8)  # Use 80% of max tokens
        self.available_tokens = self.max_tokens
        
    def create_chunks(self, 
                     data: List[Dict], 
                     size: int = 3,
                     include_context: bool = True,
                     overlap: int = 1) -> List[Dict]:
        """
        Create chunks of data suitable for API processing.
        
        Args:
            data: List of dictionaries containing the data to chunk
            size: Number of items per chunk (default: 3)
            include_context: Whether to include context information (default: True)
            overlap: Number of items to overlap between chunks (default: 1)
            
        Returns:
            List of dictionaries containing chunked data with context
        """
        if not data:
            return []
            
        if size < 1:
            raise ValueError("Chunk size must be at least 1")
            
        if overlap >= size:
            raise ValueError("Overlap must be less than chunk size")
            
        chunks = []
        i = 0
        
        while i < len(data):
            # Calculate chunk boundaries
            chunk_end = min(i + size, len(data))
            chunk_data = data[i:chunk_end]
            
            # Create chunk with context
            chunk = {
                'items': chunk_data,
                'context': {
                    'chunk_position': len(chunks) + 1,
                    'total_items': len(data),
                    'chunk_size': len(chunk_data),
                    'start_index': i,
                    'end_index': chunk_end - 1,
                    'global_stats': self.extract_metadata_context(data) if include_context else {}
                } if include_context else {}
            }
            
            # Add chunk metadata
            if include_context:
                chunk['context'].update({
                    'has_previous': i > 0,
                    'has_next': chunk_end < len(data),
                    'total_chunks': (len(data) + size - 1) // size
                })
            
            chunks.append(chunk)
            
            # Move to next chunk, considering overlap
            i = chunk_end - overlap if chunk_end < len(data) else chunk_end
            
        return chunks

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
            f"Token Utilization: {(token_info.total_tokens/self.max_tokens)*100:.1f}%"
        )
    
    def get_token_info(self, chunk: Dict[str, Any], context: str = "") -> TokenInfo:
        """Get token usage information for a chunk with enhanced metrics."""
        import time
        start_time = time.time()

        # Handle both dictionary and tuple formats for backward compatibility
        if isinstance(chunk, tuple):
            items, context_data = chunk
        else:
            items = chunk['items']
            context_data = chunk['context']

        # Calculate content tokens
        content_tokens = sum(self.count_tokens(json.dumps(item, cls=CustomJSONEncoder)) for item in items)
        
        # Calculate original tokens if available
        original_tokens = sum(item.get('original_token_count', 0) for item in items)
        
        # Calculate context tokens
        context_tokens = self.count_tokens(json.dumps(context_data, cls=CustomJSONEncoder))
        
        # Calculate metadata tokens (e.g., field names, structure)
        metadata_tokens = self.count_tokens(json.dumps({
            'field_names': list(items[0].keys()) if items else [],
            'structure_info': {
                'num_items': len(items),
                'context_type': type(context_data).__name__
            }
        }, cls=CustomJSONEncoder))
        
        # Calculate total tokens
        total_tokens = content_tokens + context_tokens + metadata_tokens
        
        # Get processing time
        processing_time = time.time() - start_time
        
        return TokenInfo(
            chunk_size=len(items),
            content_tokens=content_tokens,
            context_tokens=context_tokens,
            total_tokens=total_tokens,
            original_tokens=original_tokens,
            processing_time=processing_time,
            metadata_tokens=metadata_tokens
        )

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
            'product_areas': [],
            'features': [],
            'priorities': [],
            'root_causes': [],
            'date_range': {'start': None, 'end': None},
            'key_relationships': {},
            'global_stats': {
                'total_items': len(items),
                'priority_distribution': {},
                'product_area_distribution': {},
                'feature_distribution': {},
                'unique_product_areas': 0,
                'unique_features': 0,
                'unique_priorities': 0,
                'unique_root_causes': 0
            }
        }
        
        # Use sets temporarily for collecting unique values
        unique_product_areas = set()
        unique_features = set()
        unique_priorities = set()
        unique_root_causes = set()
        
        for item in items:
            # Collect unique values using sets
            if 'Product_Area__c' in item:
                unique_product_areas.add(item['Product_Area__c'])
                metadata['global_stats']['product_area_distribution'][item['Product_Area__c']] = \
                    metadata['global_stats']['product_area_distribution'].get(item['Product_Area__c'], 0) + 1
            if 'Product_Feature__c' in item:
                unique_features.add(item['Product_Feature__c'])
                metadata['global_stats']['feature_distribution'][item['Product_Feature__c']] = \
                    metadata['global_stats']['feature_distribution'].get(item['Product_Feature__c'], 0) + 1
            if 'Internal_Priority__c' in item:
                unique_priorities.add(item['Internal_Priority__c'])
                metadata['global_stats']['priority_distribution'][item['Internal_Priority__c']] = \
                    metadata['global_stats']['priority_distribution'].get(item['Internal_Priority__c'], 0) + 1
            if 'RCA__c' in item:
                unique_root_causes.add(item['RCA__c'])
            
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
        
        # Convert sets to sorted lists for consistent output
        metadata['product_areas'] = sorted(list(unique_product_areas))
        metadata['features'] = sorted(list(unique_features))
        metadata['priorities'] = sorted(list(unique_priorities))
        metadata['root_causes'] = sorted(list(unique_root_causes))
        
        # Update unique counts in global stats
        metadata['global_stats']['unique_product_areas'] = len(metadata['product_areas'])
        metadata['global_stats']['unique_features'] = len(metadata['features'])
        metadata['global_stats']['unique_priorities'] = len(metadata['priorities'])
        metadata['global_stats']['unique_root_causes'] = len(metadata['root_causes'])
        
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
    
    def _calculate_priority_score(self, item: Dict[str, Any]) -> float:
        """Calculate priority score for an item."""
        score = 0.0

        # Base priority score
        priority_map = {'P0': 4.0, 'P1': 3.0, 'P2': 2.0, 'P3': 1.0}
        priority = item.get('Internal_Priority__c')
        score += priority_map.get(priority, 0.0)

        # Add score for escalated cases
        if item.get('IsEscalated'):
            score += 2.0

        # Add score for customer satisfaction
        csat = item.get('CSAT__c')
        if csat is not None:
            score += max(0, 5 - csat)  # Lower CSAT means higher priority

        # Add score for response time
        response_time = item.get('First_Response_Time__c')
        if response_time is not None:
            score += min(response_time / 24, 2.0)  # Cap at 2 points for response time

        return score

    def _get_priority_distribution(self, items: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get priority distribution for a list of items."""
        distribution = {}
        for item in items:
            priority = item.get('Internal_Priority__c')
            if priority:
                distribution[priority] = distribution.get(priority, 0) + 1
        return distribution

    def optimize_chunk_content(self, chunk: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """Optimize chunk content while preserving essential information."""
        if not chunk:
            return []
            
        # Define essential fields that must be preserved
        essential_fields = {'Id', 'Internal_Priority__c', 'Subject', 'CreatedDate'}
        
        # Calculate current token usage
        current_tokens = sum(self.count_tokens(json.dumps(item)) for item in chunk)
        
        # If already within limit, return as is
        if current_tokens <= max_tokens:
            return chunk
            
        optimized_chunk = []
        for item in chunk:
            # Start with essential fields
            optimized_item = {k: item[k] for k in essential_fields if k in item}
            
            # Add other fields if space allows
            remaining_fields = set(item.keys()) - essential_fields
            for field in remaining_fields:
                temp_item = optimized_item.copy()
                temp_item[field] = item[field]
                
                # Check if adding this field would exceed token limit
                temp_tokens = sum(self.count_tokens(json.dumps(i)) for i in optimized_chunk)
                temp_tokens += self.count_tokens(json.dumps(temp_item))
                
                if temp_tokens <= max_tokens:
                    optimized_item = temp_item
                else:
                    break
            
            optimized_chunk.append(optimized_item)
            
            # Check if we've exceeded max tokens
            current_tokens = sum(self.count_tokens(json.dumps(i)) for i in optimized_chunk)
            if current_tokens > max_tokens:
                # Remove last item and stop
                optimized_chunk.pop()
                break
        
        return optimized_chunk

    def calculate_cost(self, token_count: int, is_output: bool = False) -> float:
        """Calculate cost for token usage."""
        pricing = self.TOKEN_COSTS.get(self.model_name, self.TOKEN_COSTS['gpt-3.5-turbo'])
        rate = pricing['output'] if is_output else pricing['input']
        return (token_count / 1000) * rate 

    def optimize_context(self, context: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """Optimize context to fit within token limits."""
        if not context:
            return {}

        # Create a copy of the context to avoid modifying the original
        optimized = context.copy()

        # Calculate current token count
        current_tokens = self.count_tokens(json.dumps(optimized, cls=CustomJSONEncoder))

        # If already within limit, return as is
        if current_tokens <= max_tokens:
            return optimized

        # Define priority order for context fields
        priority_fields = [
            'previous_insights',
            'global_stats',
            'key_relationships',
            'product_areas',
            'features',
            'priorities',
            'root_causes',
            'date_range'
        ]

        # Remove fields in reverse priority order until within token limit
        for field in reversed(priority_fields):
            if field in optimized:
                temp = optimized.copy()
                del temp[field]
                new_tokens = self.count_tokens(json.dumps(temp, cls=CustomJSONEncoder))
                if new_tokens <= max_tokens:
                    return temp
                optimized = temp

        # If still over limit, return minimal context
        return {'previous_insights': context.get('previous_insights', [])} 
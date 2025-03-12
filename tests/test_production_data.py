import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from utils.token_manager import TokenManager

class TestProductionTokenManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and token manager."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        
        # Initialize token manager
        cls.token_manager = TokenManager(model_name='gpt-3.5-turbo')
        
        # Create sample production-like data
        num_cases = 200
        current_date = datetime.now()
        
        # Create test data
        data = {
            'Id': [f'case_{i}' for i in range(num_cases)],
            'Subject': [f'Test case {i} subject' for i in range(num_cases)],
            'Description': [f'Detailed description for case {i}. This is a longer text that includes technical details and steps to reproduce.' for i in range(num_cases)],
            'Status': np.random.choice(['New', 'In Progress', 'Resolved', 'Closed'], num_cases),
            'Internal_Priority__c': np.random.choice(['P0', 'P1', 'P2', 'P3'], num_cases, p=[0.1, 0.2, 0.3, 0.4]),
            'Product_Area__c': np.random.choice(['Frontend', 'Backend', 'Database', 'API'], num_cases),
            'Product_Feature__c': np.random.choice(['Authentication', 'Data Processing', 'UI', 'Integration'], num_cases),
            'RCA__c': np.random.choice(['Bug', 'Feature Request', 'Configuration', 'Documentation'], num_cases),
            'CreatedDate': [current_date - timedelta(days=i) for i in range(num_cases)],
            'CSAT__c': np.random.choice([1, 2, 3, 4, 5], num_cases)
        }
        
        cls.test_data = pd.DataFrame(data)
        cls.logger.info(f"Created test dataset with {num_cases} cases")

    def test_token_counting_with_production_data(self):
        """Test token counting for large text fields."""
        self.logger.info("Starting token counting test")
        
        # Test counting tokens for a single case
        case = self.test_data.iloc[0].to_dict()
        tokens = self.token_manager.count_tokens(self.token_manager.json_dumps(case))
        self.assertGreater(tokens, 0)
        self.logger.info(f"Single case token count: {tokens}")
        
        # Test counting tokens for combined fields
        description = case['Description']
        subject = case['Subject']
        combined_tokens = self.token_manager.count_tokens(f"{subject}\n{description}")
        self.assertGreater(combined_tokens, 0)
        self.logger.info(f"Combined fields token count: {combined_tokens}")

    def test_chunking_with_production_data(self):
        """Test chunking functionality with production data."""
        self.logger.info("Starting chunking test")
        
        # Convert DataFrame rows to list of dictionaries
        cases = self.test_data.to_dict('records')
        
        # Create chunks
        chunks = self.token_manager.create_chunks(cases)
        
        # Verify chunks
        self.assertGreater(len(chunks), 0)
        for chunk, context in chunks:
            # Verify chunk size
            chunk_tokens = self.token_manager.count_tokens(self.token_manager.json_dumps(chunk))
            self.assertLessEqual(chunk_tokens, self.token_manager.chunk_tokens)
            
            # Log chunk information
            self.logger.info(f"Chunk size: {len(chunk)} items, {chunk_tokens} tokens")

    def test_context_preservation(self):
        """Test context preservation in production scenarios."""
        self.logger.info("Starting context preservation test")
        
        # Convert DataFrame rows to list of dictionaries
        cases = self.test_data.to_dict('records')
        
        # Create chunks with context
        chunks = self.token_manager.create_chunks(cases)
        
        # Verify chunk structure and context
        for chunk in chunks:
            self.assertIsInstance(chunk, dict)
            self.assertIn('items', chunk)
            self.assertIn('context', chunk)
            self.assertIsInstance(chunk['items'], list)
            self.assertIsInstance(chunk['context'], dict)
            
            # Verify context contains required fields
            context = chunk['context']
            self.assertIn('global_stats', context)
            self.assertIn('chunk_position', context)
            self.assertIn('total_chunks', context)
            self.assertIn('chunk_stats', context)
            
            # Verify global stats
            global_stats = context['global_stats']
            self.assertIn('total_items', global_stats)
            self.assertIn('unique_product_areas', global_stats)
            self.assertIn('unique_features', global_stats)
            self.assertIn('unique_priorities', global_stats)
            self.assertIn('unique_root_causes', global_stats)
            self.assertIn('date_range', global_stats)

    def test_token_usage_monitoring(self):
        """Test token usage monitoring and logging."""
        self.logger.info("Starting token usage monitoring test")
        
        # Reset token usage counters
        self.token_manager.total_tokens_processed = 0
        self.token_manager.request_count = 0
        
        # Process some test data
        cases = self.test_data.iloc[:10].to_dict('records')
        chunks = self.token_manager.create_chunks(cases)
        
        for chunk, context in chunks:
            # Get token info for the chunk
            token_info = self.token_manager.get_token_info((chunk, context))
            
            # Verify token tracking
            self.assertGreater(token_info.total_tokens, 0)
            self.assertGreater(token_info.content_tokens, 0)
            self.assertGreater(token_info.metadata_tokens, 0)
            
            # Log token usage
            self.logger.info(
                f"Chunk token usage - Total: {token_info.total_tokens}, "
                f"Content: {token_info.content_tokens}, "
                f"Metadata: {token_info.metadata_tokens}"
            )

    def test_ai_insights_with_production_data(self):
        """Test AI insights generation with production data."""
        self.logger.info("Starting AI insights test")
        
        # Prepare test data
        cases = self.test_data.iloc[:5].to_dict('records')
        chunks = self.token_manager.create_chunks(cases)
        
        for chunk in chunks:
            # Verify chunk structure
            self.assertIsInstance(chunk, dict)
            self.assertIn('items', chunk)
            self.assertIn('context', chunk)
            self.assertIsInstance(chunk['items'], list)
            
            # Verify items in chunk
            for item in chunk['items']:
                self.assertIsInstance(item, dict)
                self.assertIn('Id', item)
                self.assertIn('Subject', item)
                self.assertIn('Description', item)
                self.assertIn('Status', item)
                self.assertIn('Internal_Priority__c', item)
                self.assertIn('Product_Area__c', item)
            
            # Verify context
            context = chunk['context']
            self.assertIsInstance(context, dict)
            self.assertIn('global_stats', context)
            self.assertIn('chunk_position', context)
            self.assertIn('total_chunks', context)
            self.assertIn('chunk_stats', context)

if __name__ == '__main__':
    unittest.main() 
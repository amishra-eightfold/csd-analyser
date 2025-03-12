import unittest
from datetime import datetime, timedelta
import json
from utils.token_manager import TokenManager, TokenInfo

class TestTokenManager(unittest.TestCase):
    def setUp(self):
        self.token_manager = TokenManager('gpt-3.5-turbo')
        
        # Create sample test data
        self.test_data = [
            {
                'Id': '1',
                'Subject': 'API Integration Issue',
                'Description': 'Customer is experiencing timeout issues with the API integration.',
                'Internal_Priority__c': 'P1',
                'Product_Area__c': 'API',
                'Product_Feature__c': 'Integration',
                'RCA__c': 'Configuration',
                'CreatedDate': (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'Id': '2',
                'Subject': 'Dashboard Loading Slow',
                'Description': 'Dashboard performance degraded after recent update.',
                'Internal_Priority__c': 'P2',
                'Product_Area__c': 'Dashboard',
                'Product_Feature__c': 'Performance',
                'RCA__c': 'Performance',
                'CreatedDate': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'Id': '3',
                'Subject': 'API Authentication Failed',
                'Description': 'Authentication tokens not being refreshed properly.',
                'Internal_Priority__c': 'P1',
                'Product_Area__c': 'API',
                'Product_Feature__c': 'Authentication',
                'RCA__c': 'Configuration',
                'CreatedDate': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
    
    def test_token_counting(self):
        """Test basic token counting functionality."""
        text = "This is a test string"
        token_count = self.token_manager.count_tokens(text)
        self.assertGreater(token_count, 0)
        
    def test_priority_scoring(self):
        """Test priority score calculation."""
        # Test P1 priority with recent date
        recent_item = self.test_data[2]  # Using the most recent item
        recent_score = self.token_manager.calculate_priority_score(recent_item)
        
        # Test P2 priority with older date
        older_item = self.test_data[1]  # Using an older item
        older_score = self.token_manager.calculate_priority_score(older_item)
        
        # Recent P1 should score higher than older P2
        self.assertGreater(recent_score, older_score)
    
    def test_metadata_extraction(self):
        """Test metadata context extraction."""
        metadata = self.token_manager.extract_metadata_context(self.test_data)
        
        # Verify all expected fields are present
        self.assertIn('product_areas', metadata)
        self.assertIn('features', metadata)
        self.assertIn('priorities', metadata)
        self.assertIn('root_causes', metadata)
        self.assertIn('date_range', metadata)
        
        # Verify content
        self.assertEqual(len(metadata['product_areas']), 2)  # API and Dashboard
        self.assertIn('API', metadata['product_areas'])
        self.assertEqual(len(metadata['priorities']), 2)  # P1 and P2
    
    def test_semantic_grouping(self):
        """Test semantic grouping of related items."""
        groups = self.token_manager.create_semantic_groups(self.test_data)
        
        # Verify API-related tickets are grouped together
        api_group = None
        for group in groups:
            if any(item['Product_Area__c'] == 'API' for item in group):
                api_group = group
                break
        
        self.assertIsNotNone(api_group)
        self.assertEqual(len(api_group), 2)  # Should contain both API tickets
    
    def test_chunk_creation(self):
        """Test chunk creation with context preservation."""
        chunks = self.token_manager.create_chunks(self.test_data)
        
        # Verify chunks structure
        self.assertGreater(len(chunks), 0)
        for chunk, context in chunks:
            # Verify chunk content
            self.assertIsInstance(chunk, list)
            self.assertGreater(len(chunk), 0)
            
            # Verify context
            self.assertIn('global_stats', context)
            self.assertIn('chunk_position', context)
            self.assertIn('total_chunks', context)
            
            # Verify token limits
            token_info = self.token_manager.get_token_info((chunk, context))
            self.assertLessEqual(token_info.total_tokens, self.token_manager.model_limit)
    
    def test_content_optimization(self):
        """Test content optimization while preserving essential information."""
        # Create a large chunk that exceeds token limit
        large_chunk = self.test_data * 10  # Multiply data to create larger dataset
        max_tokens = 1000  # Set a small token limit for testing
        
        optimized_chunk = self.token_manager.optimize_chunk_content(large_chunk, max_tokens)
        
        # Verify optimization
        original_tokens = sum(self.token_manager.count_tokens(json.dumps(item)) for item in large_chunk)
        optimized_tokens = sum(self.token_manager.count_tokens(json.dumps(item)) for item in optimized_chunk)
        
        self.assertLess(optimized_tokens, original_tokens)
        self.assertLessEqual(optimized_tokens, max_tokens)
        
        # Verify essential information is preserved
        for item in optimized_chunk:
            self.assertIn('Id', item)
            self.assertIn('Internal_Priority__c', item)
    
    def test_token_info(self):
        """Test token information calculation."""
        chunk = self.test_data[:2]
        context = self.token_manager.extract_metadata_context(chunk)
        
        token_info = self.token_manager.get_token_info((chunk, context))
        
        self.assertIsInstance(token_info, TokenInfo)
        self.assertGreater(token_info.total_tokens, 0)
        self.assertGreater(token_info.content_tokens, 0)
        self.assertGreater(token_info.metadata_tokens, 0)

if __name__ == '__main__':
    unittest.main() 
#!/usr/bin/env python3
"""
Test script to validate OpenAI credentials and connection.
This script checks if the OpenAI API key is valid and the service is accessible.
"""

import os
import sys
import logging
from datetime import datetime
from openai import OpenAI, OpenAIError
import argparse
import toml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('openai_test.log')
    ]
)
logger = logging.getLogger(__name__)

def get_api_key_from_secrets() -> str:
    """
    Get OpenAI API key from Streamlit secrets.toml file.
    
    Returns:
        str: API key if found, None otherwise
    """
    try:
        # Get the project root directory (assuming we're in tests/)
        project_root = Path(__file__).parent.parent
        secrets_path = project_root / '.streamlit' / 'secrets.toml'
        
        if not secrets_path.exists():
            logger.error(f"Secrets file not found at {secrets_path}")
            return None
            
        # Read the secrets file
        secrets = toml.load(secrets_path)
        
        # Get the API key
        api_key = secrets.get('OPENAI_API_KEY')
        if not api_key:
            logger.error("OPENAI_API_KEY not found in secrets.toml")
            return None
            
        return api_key
        
    except Exception as e:
        logger.error(f"Error reading secrets file: {str(e)}")
        return None

def test_openai_connection(api_key: str = None) -> bool:
    """
    Test OpenAI connection and API key validity.
    
    Args:
        api_key: Optional API key. If not provided, will look for OPENAI_API_KEY in secrets.toml or environment.
        
    Returns:
        bool: True if connection is successful, False otherwise.
    """
    try:
        # Get API key from parameter, secrets, or environment
        api_key = api_key or get_api_key_from_secrets() or os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("No OpenAI API key found in arguments, secrets.toml, or environment variables")
            return False
            
        logger.info("Testing OpenAI connection...")
        
        # Initialize client
        client = OpenAI(api_key=api_key)
        
        # Try a simple API call
        start_time = datetime.now()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test message."}],
            max_tokens=10
        )
        end_time = datetime.now()
        
        # Calculate response time
        response_time = (end_time - start_time).total_seconds()
        
        # Log success
        logger.info(f"Connection successful! Response time: {response_time:.2f} seconds")
        logger.info(f"Model responded with: {response.choices[0].message.content}")
        
        # Additional API information
        logger.info("Checking API information...")
        models = client.models.list()
        logger.info(f"Available models: {len(models.data)}")
        
        return True
        
    except OpenAIError as e:
        logger.error(f"OpenAI API Error: {str(e)}")
        if "Incorrect API key" in str(e):
            logger.error("The provided API key appears to be invalid")
        elif "Rate limit" in str(e):
            logger.error("Rate limit exceeded. Please try again later")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False

def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description='Test OpenAI API connection')
    parser.add_argument('--api-key', help='OpenAI API key (optional, can use secrets.toml or OPENAI_API_KEY env var)')
    args = parser.parse_args()
    
    # Test connection
    success = test_openai_connection(args.api_key)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 
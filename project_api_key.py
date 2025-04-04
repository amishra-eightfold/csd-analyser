#!/usr/bin/env python3
"""
This module provides a solution for using project-scoped API keys with the OpenAI client.
It sets up the correct environment variables for you and provides direct API access if needed.
"""

import os
import streamlit as st
from dotenv import load_dotenv
import sys
import requests
import json
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("project_api_key")

def setup_project_api_key():
    """
    Set up the environment to work with a project-scoped API key.
    This handles project-scoped API keys differently to make them work with OpenAI client.
    """
    # First load from .env file
    load_dotenv()
    
    # Get the API key from environment or Streamlit secrets
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", None)
        except:
            pass
    
    if not api_key:
        logger.warning("No OpenAI API key found.")
        return False
    
    # Check if it's a project-scoped key
    if not api_key.startswith('sk-proj-'):
        logger.info("API key is not a project-scoped key, no special handling required.")
        # Set it directly in the environment
        os.environ['OPENAI_API_KEY'] = api_key
        return True
    
    logger.info(f"Detected project-scoped API key: {api_key[:12]}...{api_key[-4:]}")
    
    # For project-scoped keys, we need special handling
    
    # 1. Try to modify the global openai module directly (if it's already imported)
    try:
        import openai
        # Set the API key on the module
        openai.api_key = api_key
        logger.info("Set API key directly on openai module")
    except ImportError:
        logger.info("OpenAI module not yet imported")
    
    # 2. Set environment variables for different versions of the OpenAI client
    os.environ['OPENAI_API_KEY'] = api_key
    os.environ['OPENAI_KEY'] = api_key
    
    # 3. Set configuration to help with project API keys
    os.environ['OPENAI_API_TYPE'] = 'open_ai'
    os.environ['OPENAI_API_VERSION'] = '2023-05-15'
    os.environ['OPENAI_API_BASE'] = 'https://api.openai.com/v1'
    
    # 4. Unset any proxy variables that might interfere
    if 'http_proxy' in os.environ:
        del os.environ['http_proxy']
    if 'https_proxy' in os.environ:
        del os.environ['https_proxy']
    
    # 5. Set up global monkey patch for the OpenAI client
    try:
        monkey_patch_openai_for_project_keys()
    except Exception as e:
        logger.warning(f"Failed to monkey patch OpenAI: {e}")
    
    return True

def monkey_patch_openai_for_project_keys():
    """
    Monkey patch the OpenAI client to handle project API keys correctly.
    """
    import openai
    from functools import wraps
    
    # Store the original client creation function
    original_openai_client = openai.OpenAI
    
    # Create a new wrapper function
    @wraps(original_openai_client)
    def patched_openai_client(*args, **kwargs):
        # If api_key is in kwargs, use it directly
        if 'api_key' in kwargs:
            return original_openai_client(*args, **kwargs)
        
        # Otherwise, try to get it from environment
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            kwargs['api_key'] = api_key
            
        # Create and return the original client
        return original_openai_client(*args, **kwargs)
    
    # Replace the original function with our patched version
    openai.OpenAI = patched_openai_client
    logger.info("Successfully monkey patched OpenAI client")

def direct_chat_completion(model, messages, temperature=0.5, max_tokens=2000):
    """
    Fallback method to directly call the OpenAI API without using the client.
    This works around issues with project-scoped API keys.
    """
    api_key = os.environ.get('OPENAI_API_KEY') or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise ValueError("No API key available")
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

def test_openai_connection():
    """Test if the OpenAI connection works with the configured environment."""
    try:
        import openai
        client = openai.OpenAI()
        
        # Try listing models (a simple API call)
        response = client.models.list()
        logger.info(f"Successfully connected! Found {len(response.data)} models.")
        logger.info(f"First few models: {', '.join([m.id for m in response.data[:3]])}")
        return True
    except Exception as e:
        logger.error(f"Error connecting to OpenAI: {e}")
        
        # Try a direct API call as fallback
        try:
            logger.info("Trying direct API call...")
            api_key = os.environ.get('OPENAI_API_KEY') or st.secrets.get("OPENAI_API_KEY", None)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=headers
            )
            
            if response.status_code == 200:
                models = response.json()
                logger.info(f"Direct API call successful! Found {len(models['data'])} models.")
                return True
            else:
                logger.error(f"Direct API call failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error with direct API call: {e}")
            return False
    
# For direct testing
if __name__ == "__main__":
    if setup_project_api_key():
        logger.info("API key setup complete.")
        
        if test_openai_connection():
            logger.info("✅ Successfully connected to OpenAI API!")
            
            # Try a chat completion
            try:
                logger.info("Testing chat completion...")
                
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'Hello, API is working!'"}
                ]
                
                try:
                    # Try with regular client
                    import openai
                    client = openai.OpenAI()
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        temperature=0,
                        max_tokens=50
                    )
                    logger.info(f"Response: {response.choices[0].message.content}")
                except Exception as e:
                    logger.warning(f"Standard client failed: {e}")
                    
                    # Try with direct API call
                    logger.info("Trying direct API call...")
                    response = direct_chat_completion(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        temperature=0,
                        max_tokens=50
                    )
                    logger.info(f"Direct API response: {response['choices'][0]['message']['content']}")
                
                logger.info("✅ Chat completion successful!")
            except Exception as e:
                logger.error(f"❌ Chat completion failed: {e}")
        else:
            logger.error("❌ Failed to connect to OpenAI API.")
    else:
        logger.error("❌ API key setup failed.") 
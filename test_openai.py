#!/usr/bin/env python3
"""
Test script to verify OpenAI integration is working correctly.
This script tests:
1. OpenAI package installation
2. API key configuration
3. Basic API functionality
"""

import os
import sys
import json
from dotenv import load_dotenv

def main():
    print("OpenAI Integration Test")
    print("======================\n")
    
    # Step 1: Check if OpenAI package is installed
    print("Step 1: Checking OpenAI package installation...")
    try:
        import openai
        print("✅ OpenAI package is installed (version: {})".format(openai.__version__))
    except ImportError:
        print("❌ OpenAI package is not installed. Please run 'pip install openai'")
        sys.exit(1)
    
    # Step 2: Check if API key is configured
    print("\nStep 2: Checking API key configuration...")
    
    # Try to load from .env file
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        # Try to load from Streamlit secrets (if available)
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENAI_API_KEY", None)
            print("Attempting to load API key from Streamlit secrets...")
        except:
            pass
    
    if not api_key:
        print("❌ OpenAI API key not found in environment variables or Streamlit secrets")
        print("Please set the OPENAI_API_KEY environment variable or add it to .streamlit/secrets.toml")
        sys.exit(1)
    
    # Mask the API key for security
    masked_key = api_key[:4] + "..." + api_key[-4:]
    print(f"✅ OpenAI API key found: {masked_key}")
    
    # Step 3: Test API functionality
    print("\nStep 3: Testing API functionality...")
    client = openai.OpenAI(api_key=api_key)
    
    try:
        # Make a simple API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a smaller model for testing
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Return a JSON object with a 'status' field set to 'success' and a 'message' field with 'OpenAI API is working correctly'."}
            ],
            temperature=0,
            max_tokens=100
        )
        
        # Get the response content
        content = response.choices[0].message.content
        print(f"API Response: {content}")
        
        # Try to parse as JSON
        try:
            json_response = json.loads(content)
            if json_response.get("status") == "success":
                print("✅ API call successful and returned expected response")
            else:
                print("⚠️ API call successful but returned unexpected response")
        except json.JSONDecodeError:
            print("⚠️ API call successful but response is not valid JSON")
            print("This might still be okay for the application")
        
    except Exception as e:
        print(f"❌ API call failed: {str(e)}")
        sys.exit(1)
    
    print("\nSummary:")
    print("✅ OpenAI package is installed")
    print(f"✅ API key is configured: {masked_key}")
    print("✅ API functionality is working")
    print("\nThe OpenAI integration appears to be working correctly!")

if __name__ == "__main__":
    main() 
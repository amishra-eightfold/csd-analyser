#!/usr/bin/env python3
"""
Test script to verify our project API key setup.
"""

import os
import sys
from project_api_key import setup_project_api_key, test_openai_connection

def main():
    print("Testing Project API Key Setup")
    print("============================\n")
    
    # Set up the environment for the project API key
    if setup_project_api_key():
        print("✅ API key environment setup successful")
    else:
        print("❌ API key setup failed")
        sys.exit(1)
    
    # Test OpenAI connection
    print("\nTesting OpenAI connection...")
    if test_openai_connection():
        print("✅ OpenAI connection test succeeded")
    else:
        print("❌ OpenAI connection test failed")
        sys.exit(1)
    
    # Test chat completion
    print("\nTesting chat completion...")
    try:
        import openai
        client = openai.OpenAI()  # No need to pass API key - it's in the environment
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Return a JSON object with a 'status' field set to 'success'."}
            ],
            temperature=0,
            max_tokens=100
        )
        
        content = response.choices[0].message.content
        print(f"Response: {content}")
        print("✅ Chat completion succeeded")
        
    except Exception as e:
        print(f"❌ Chat completion failed: {str(e)}")
        sys.exit(1)
    
    print("\nAll tests passed! 🎉")
    print("Your project API key is working correctly.")

if __name__ == "__main__":
    main() 
import os
import sys
import toml
import openai
from pathlib import Path
import streamlit as st

def test_openai_connection():
    """Test OpenAI connectivity using both TOML and environment configurations."""
    results = {
        'toml_key_exists': False,
        'env_key_exists': False,
        'toml_key_works': False,
        'env_key_works': False,
        'errors': []
    }
    
    # Test function to check if a key works
    def test_key(api_key):
        try:
            client = openai.OpenAI(api_key=api_key)
            # Try a simple completion to test connectivity
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello, this is a test message."}],
                max_tokens=10
            )
            return True, None
        except Exception as e:
            return False, str(e)

    print("\n=== OpenAI Connectivity Test ===\n")

    # Check TOML configuration
    try:
        secrets_path = Path('.streamlit/secrets.toml')
        if secrets_path.exists():
            config = toml.load(secrets_path)
            toml_key = config.get('OPENAI_API_KEY')
            if toml_key:
                results['toml_key_exists'] = True
                print("✓ Found OpenAI API key in secrets.toml")
                
                # Test TOML key functionality
                works, error = test_key(toml_key)
                results['toml_key_works'] = works
                if works:
                    print("✓ TOML API key is working correctly")
                else:
                    print("✗ TOML API key error:", error)
                    results['errors'].append(f"TOML key error: {error}")
            else:
                print("✗ No OpenAI API key found in secrets.toml")
        else:
            print("✗ secrets.toml file not found")
    except Exception as e:
        print("✗ Error reading secrets.toml:", str(e))
        results['errors'].append(f"TOML reading error: {str(e)}")

    # Check environment variable
    env_key = os.getenv('OPENAI_API_KEY')
    if env_key:
        results['env_key_exists'] = True
        print("\n✓ Found OpenAI API key in environment variables")
        
        # Test environment key functionality
        works, error = test_key(env_key)
        results['env_key_works'] = works
        if works:
            print("✓ Environment API key is working correctly")
        else:
            print("✗ Environment API key error:", error)
            results['errors'].append(f"Environment key error: {error}")
    else:
        print("\n✗ No OpenAI API key found in environment variables")

    # Summary
    print("\n=== Test Summary ===")
    print(f"TOML Configuration: {'✓' if results['toml_key_works'] else '✗'}")
    print(f"Environment Configuration: {'✓' if results['env_key_works'] else '✗'}")
    
    if not (results['toml_key_works'] or results['env_key_works']):
        print("\n⚠️  No working OpenAI API key found!")
        print("Please ensure either:")
        print("1. Add OPENAI_API_KEY to .streamlit/secrets.toml")
        print("2. Set OPENAI_API_KEY environment variable")
    
    if results['errors']:
        print("\nErrors encountered:")
        for error in results['errors']:
            print(f"- {error}")

    return results

if __name__ == "__main__":
    test_openai_connection() 
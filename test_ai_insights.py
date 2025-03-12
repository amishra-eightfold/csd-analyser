#!/usr/bin/env python3
"""
Test script to verify the AI insights generation functionality.
This script mimics the generate_ai_insights function from the main application.
"""

import os
import sys
import json
import re
import pandas as pd
from dotenv import load_dotenv

def main():
    print("AI Insights Generation Test")
    print("==========================\n")
    
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
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_api_key:
        # Try to load from Streamlit secrets (if available)
        try:
            import streamlit as st
            openai_api_key = st.secrets.get("OPENAI_API_KEY", None)
            print("Attempting to load API key from Streamlit secrets...")
        except:
            pass
    
    if not openai_api_key:
        print("❌ OpenAI API key not found in environment variables or Streamlit secrets")
        print("Please set the OPENAI_API_KEY environment variable or add it to .streamlit/secrets.toml")
        sys.exit(1)
    
    # Mask the API key for security
    masked_key = openai_api_key[:4] + "..." + openai_api_key[-4:]
    print(f"✅ OpenAI API key found: {masked_key}")
    
    # Step 3: Create sample data
    print("\nStep 3: Creating sample data...")
    
    # Create a sample dataframe similar to what we'd have in the application
    cases_data = {
        'Id': ['case1', 'case2', 'case3', 'case4', 'case5'],
        'CaseNumber': ['C-001', 'C-002', 'C-003', 'C-004', 'C-005'],
        'Subject': [
            'Configuration issue with API integration',
            'Dashboard not loading correctly',
            'Error when importing data',
            'User cannot access admin panel',
            'Performance degradation after update'
        ],
        'Description': [
            'Customer is experiencing issues with the API integration. The configuration seems incorrect.',
            'The dashboard is not loading properly. It shows a blank screen.',
            'When importing data from CSV, the system throws an error about invalid format.',
            'Admin user cannot access the admin panel after password reset.',
            'System performance has degraded after the latest update.'
        ],
        'Status': ['Open', 'Closed', 'Open', 'Closed', 'Open'],
        'Internal_Priority__c': ['High', 'Medium', 'High', 'Low', 'Critical'],
        'Product_Area__c': ['API', 'UI', 'Data Import', 'Authentication', 'Performance'],
        'Product_Feature__c': ['Integration', 'Dashboard', 'Import Tool', 'Admin Panel', 'Core Engine'],
        'RCA__c': ['Configuration', 'Bug', 'User Error', 'Bug', 'Design Limitation'],
        'IMPL_Phase__c': ['Implementation', 'Post-Launch', 'Implementation', 'Post-Launch', 'Onboarding']
    }
    
    cases_df = pd.DataFrame(cases_data)
    print("✅ Sample data created with {} cases".format(len(cases_df)))
    
    # Step 4: Test AI insights generation
    print("\nStep 4: Testing AI insights generation...")
    
    # Set up OpenAI client
    client = openai.OpenAI(api_key=openai_api_key)
    
    # Create a summary of the data (similar to the app)
    case_summary = {
        "total_cases": len(cases_df),
        "product_areas": cases_df['Product_Area__c'].value_counts().to_dict(),
        "priorities": cases_df['Internal_Priority__c'].value_counts().to_dict(),
        "statuses": cases_df['Status'].value_counts().to_dict(),
        "root_causes": cases_df['RCA__c'].value_counts().to_dict(),
        "implementation_phases": cases_df['IMPL_Phase__c'].value_counts().to_dict()
    }
    
    # Sample of case subjects and descriptions
    sample_columns = ['Subject', 'Description', 'RCA__c', 'Status', 'IMPL_Phase__c']
    case_samples = cases_df[sample_columns].to_dict('records')
    
    # Prepare the prompt (same as in the app)
    prompt = f"""
    Analyze the following support ticket data and provide insights:
    
    Summary Statistics:
    {json.dumps(case_summary, indent=2)}
    
    Sample Cases:
    {json.dumps(case_samples, indent=2)}
    
    Please provide:
    1. A summary of key insights from the data
    2. Patterns or trends you identify in the tickets
    3. Recommendations for improving customer support
    
    Format your response as JSON with the following structure:
    {{
        "summary": "Overall summary of insights",
        "patterns": [
            {{"title": "Pattern 1", "description": "Description of pattern 1"}},
            {{"title": "Pattern 2", "description": "Description of pattern 2"}}
        ],
        "recommendations": [
            "Recommendation 1",
            "Recommendation 2"
        ]
    }}
    """
    
    try:
        # Call OpenAI API
        print("Calling OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4",  # Using GPT-4 as in the app
            messages=[
                {"role": "system", "content": "You are an expert support ticket analyst. Analyze the provided data and extract meaningful insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=2000
        )
        
        # Extract and parse the response
        ai_response = response.choices[0].message.content
        print("\nAPI Response received. First 100 characters:")
        print(ai_response[:100] + "..." if len(ai_response) > 100 else ai_response)
        
        # Extract JSON from the response
        json_match = re.search(r'({.*})', ai_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                insights = json.loads(json_str)
                print("\n✅ Successfully parsed JSON response")
                
                # Verify the structure
                if all(key in insights for key in ['summary', 'patterns', 'recommendations']):
                    print("✅ Response has the expected structure")
                    
                    # Print a summary of the insights
                    print("\nInsights Summary:")
                    print(f"- Summary: {insights['summary'][:100]}...")
                    print(f"- Patterns: {len(insights['patterns'])} identified")
                    print(f"- Recommendations: {len(insights['recommendations'])} provided")
                    
                    # Save the full insights to a file for inspection
                    with open('ai_insights_test_result.json', 'w') as f:
                        json.dump(insights, f, indent=2)
                    print("\nFull insights saved to 'ai_insights_test_result.json'")
                    
                else:
                    print("❌ Response is missing expected keys")
                    print(f"Expected: ['summary', 'patterns', 'recommendations']")
                    print(f"Found: {list(insights.keys())}")
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing JSON response: {str(e)}")
                with open('ai_response_error.txt', 'w') as f:
                    f.write(ai_response)
                print("Raw response saved to 'ai_response_error.txt'")
        else:
            print("❌ Could not find JSON in the response")
            with open('ai_response_error.txt', 'w') as f:
                f.write(ai_response)
            print("Raw response saved to 'ai_response_error.txt'")
            
    except Exception as e:
        print(f"❌ Error calling OpenAI API: {str(e)}")
        sys.exit(1)
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 
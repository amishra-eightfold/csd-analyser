#!/bin/bash

# Test OpenAI Connection Script
# This script runs the OpenAI connection test and displays the results

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
YELLOW='\033[1;33m'
BLUE='\033[0;34m'

echo -e "${YELLOW}OpenAI Connection Test${NC}"
echo "=============================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check if required Python packages are installed
echo "Checking required packages..."
python3 -c "import toml" 2>/dev/null || {
    echo -e "${RED}Error: Python package 'toml' is not installed${NC}"
    echo "Please install it using: pip install toml"
    exit 1
}

# Check if the test script exists
if [ ! -f "tests/test_openai_connection.py" ]; then
    echo -e "${RED}Error: Test script not found at tests/test_openai_connection.py${NC}"
    exit 1
fi

# Check if secrets.toml exists
if [ ! -f ".streamlit/secrets.toml" ]; then
    echo -e "${YELLOW}Warning: secrets.toml not found at .streamlit/secrets.toml${NC}"
    echo "The script will try to use environment variable or provided API key instead."
fi

# Check if API key is provided as argument
if [ $# -eq 1 ]; then
    echo -e "${BLUE}Using provided API key${NC}"
    python3 tests/test_openai_connection.py --api-key "$1"
else
    # Check if secrets.toml exists and contains OPENAI_API_KEY
    if [ -f ".streamlit/secrets.toml" ] && grep -q "OPENAI_API_KEY" ".streamlit/secrets.toml"; then
        echo -e "${BLUE}Using API key from secrets.toml${NC}"
        python3 tests/test_openai_connection.py
    else
        # Check if OPENAI_API_KEY is set in environment
        if [ -z "${OPENAI_API_KEY}" ]; then
            echo -e "${RED}Error: No API key found${NC}"
            echo "Please either:"
            echo "1. Add OPENAI_API_KEY to .streamlit/secrets.toml"
            echo "2. Set OPENAI_API_KEY environment variable"
            echo "3. Pass API key as argument: ./test_openai.sh YOUR_API_KEY"
            exit 1
        else
            echo -e "${BLUE}Using OPENAI_API_KEY from environment${NC}"
            python3 tests/test_openai_connection.py
        fi
    fi
fi

# Check the exit status
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Test completed successfully!${NC}"
else
    echo -e "${RED}Test failed. Check the logs for details.${NC}"
    echo "Log file: openai_test.log"
fi 
# CSD Analyser

A Streamlit application for analyzing customer support ticket data from Salesforce.

## Features

- Interactive data visualization
- Customer support ticket analysis
- CSAT trend analysis
- Response time analysis
- Root cause analysis
- Pattern evolution analysis
- AI-powered insights
- Export capabilities (Excel, PowerPoint, CSV)
- PII protection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/csd-analyser.git
cd csd-analyser
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

## Configuration

1. Set up your environment variables:
```bash
export SALESFORCE_USERNAME="your_username"
export SALESFORCE_PASSWORD="your_password"
export SALESFORCE_SECURITY_TOKEN="your_token"
export OPENAI_API_KEY="your_api_key"
```

2. Or create a `.env` file:
```
SALESFORCE_USERNAME=your_username
SALESFORCE_PASSWORD=your_password
SALESFORCE_SECURITY_TOKEN=your_token
OPENAI_API_KEY=your_api_key
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Select customers and date range to analyze

4. Use the various analysis features:
   - Ticket volume analysis
   - Response time analysis
   - CSAT analysis
   - Root cause analysis
   - Pattern evolution
   - AI insights

5. Export your analysis in various formats:
   - Excel reports
   - PowerPoint presentations
   - CSV data exports

## Development

1. Install development dependencies:
```bash
pip install -e ".[test]"
```

2. Run tests:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Required Data Format

Your input file should contain the following columns:
- `Id`: Case ID
- `CaseNumber`: Case number
- `Account.Account_Name__c`: Account name
- `Group_Id__c`: Customer Group ID
- `Subject`: Case subject
- `Description`: Case description
- `Product_Area__c`: Product area
- `Product_Feature__c`: Product feature
- `POD_Name__c`: POD name
- `CreatedDate`: Date when the ticket was created
- `ClosedDate`: Date when the ticket was closed
- `Case_Type__c`: Type of case
- `Age_days__c`: Age of the case in days
- `IsEscalated`: Whether the case is escalated
- `CSAT__c`: Customer satisfaction score
- `Internal_Priority__c`: Internal priority level

## License

MIT License

## Secrets Management

This application uses a `.streamlit/secrets.toml` file for storing sensitive information like API keys and database credentials. To protect your secrets:

1. The `.streamlit/secrets.toml` file and `.env` files are excluded from Git via the `.gitignore` file.
2. Never commit sensitive information directly in code files.
3. If you need to share your code, make sure these files are not included.

### Setting up your secrets

To set up your secrets:

1. Create a `.streamlit/secrets.toml` file in the project root (if not already present)
2. Add your sensitive information in the following format:

```toml
[salesforce]
username = "your_username"
password = "your_password"
security_token = "your_security_token"
domain = "your_domain"

[openai]
api_key = "your_openai_api_key"
```

The application will use these variables for authentication with external services.

# Checking if secrets are properly handled

## Privacy and PII Protection

The application includes robust privacy protection features to ensure sensitive information is handled securely:

### PII Detection and Removal

The system automatically detects and removes the following types of Personally Identifiable Information (PII):
- Email addresses
- Phone numbers (multiple formats including international)
- URLs and IP addresses
- Credit card numbers
- Social security numbers
- Names (with common patterns and titles)
- Passwords
- Dates of birth

### Privacy Features

1. **Pre-processing Pipeline**: All text data is processed through a PII removal pipeline before any analysis or storage.
2. **AI Analysis Protection**: Data is automatically sanitized before being sent to external AI services (e.g., OpenAI).
3. **Standardized Placeholders**: PII is replaced with standardized placeholders (e.g., [EMAIL], [PHONE]) to maintain context while protecting privacy.
4. **Multi-format Support**: PII removal works across various data formats:
   - String data
   - Structured data (DataFrames)
   - Nested data structures (dictionaries, lists)

### Implementation

The PII protection is implemented through two main components:
1. `remove_pii(text)`: Core function for detecting and removing PII from text data
2. `prepare_text_for_ai(data)`: High-level function that handles different data types and structures

### Best Practices

- Always use the PII removal functions before sharing or analyzing sensitive data
- Regularly audit and update PII detection patterns
- Monitor and log PII removal statistics (without logging the PII itself)
- Review and validate PII removal effectiveness periodically

# Customer Support Data Analyzer

A Streamlit-based dashboard for analyzing customer support ticket data. This tool provides comprehensive visualizations and insights from support ticket data, including trends, patterns, and key metrics.

## Features

- Google Authentication for secure access
- Interactive data filtering and visualization
- Support for CSV and Excel input files
- Comprehensive analysis including:
  - Overview statistics
  - Account analysis
  - Monthly trends
  - Priority and escalation analysis
  - Product area analysis
  - Root cause analysis
- Group ID prediction for missing values
- Product feature prediction
- Export capabilities:
  - Excel export of filtered data
  - PowerPoint presentation with all visualizations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/amishra-eightfold/csd-analyser.git
cd csd-analyser
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up Google OAuth credentials:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Navigate to "APIs & Services" > "Credentials"
   - Create an OAuth 2.0 Client ID (Web application)
   - Add `http://localhost:8501/callback` as an authorized redirect URI
   - For Streamlit Cloud deployment, also add `https://csd-analyser.streamlit.app/callback` as an authorized redirect URI
   - Create a `.streamlit/secrets.toml` file in the project root with the following content:
   ```toml
   # Google OAuth credentials
   GOOGLE_CLIENT_ID = "your_client_id_here"
   GOOGLE_CLIENT_SECRET = "your_client_secret_here"
   REDIRECT_URI = "http://localhost:8501/callback"  # For local development
   # REDIRECT_URI = "https://csd-analyser.streamlit.app/callback"  # For Streamlit Cloud
   ALLOWED_DOMAIN = "eightfold.ai"
   ```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to http://localhost:8501

3. Sign in with your Google account (only emails from the allowed domain can access the app)

4. Upload your data file (CSV or Excel) containing support ticket information

5. Use the sidebar filters to analyze specific data segments

6. Export your analysis as Excel or PowerPoint presentation

## Deployment to Streamlit Cloud

1. Push your code to GitHub:
```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

2. Create a new branch for Streamlit deployment:
```bash
git checkout -b dynamic-query-seaborn-llm
git push origin dynamic-query-seaborn-llm
```

3. Set up your app on Streamlit Cloud:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository and the `dynamic-query-seaborn-llm` branch
   - Set the app URL to `https://csd-analyser.streamlit.app/`
   - Set the main file path to `app.py`
   - Click "Deploy"

4. Configure Streamlit Cloud secrets:
   - In your Streamlit Cloud dashboard, navigate to your app settings
   - Under "Secrets", add the same secrets as in your local `.streamlit/secrets.toml` file
   - Make sure to set `REDIRECT_URI` to `https://csd-analyser.streamlit.app/callback`

5. Your app should now be accessible at `https://csd-analyser.streamlit.app/`

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

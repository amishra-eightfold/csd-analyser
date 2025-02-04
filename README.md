# Customer Support Data Analyzer

A Streamlit-based dashboard for analyzing customer support ticket data. This tool provides comprehensive visualizations and insights from support ticket data, including trends, patterns, and key metrics.

## Features

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
git clone https://github.com/YOUR_USERNAME/csd-analyser.git
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

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to http://localhost:8501

3. Upload your data file (CSV or Excel) containing support ticket information

4. Use the sidebar filters to analyze specific data segments

5. Export your analysis as Excel or PowerPoint presentation

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
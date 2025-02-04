import pandas as pd

# Read the Excel file
df = pd.read_excel('/Users/ldominc/dev/vscode/cases-125.xlsx')

# Print column names
print("Column names:")
print(df.columns.tolist())

# Print first few rows
print("\nFirst few rows:")
print(df.head()) 
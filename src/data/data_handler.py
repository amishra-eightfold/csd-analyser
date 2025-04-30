"""Data handling functions for the support ticket analysis dashboard."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
import os
import streamlit as st
from config.logging_config import get_logger

# Initialize logger
logger = get_logger('data')

def debug(message, data=None, category="data"):
    """Log debug information to the logger."""
    if hasattr(st.session_state, 'debug_logger'):
        st.session_state.debug_logger.log(message, data, category)
    
    # Log to file logger
    logger = get_logger(category)
    if data is not None:
        # Convert data to string if needed for logging
        if isinstance(data, dict):
            try:
                import json
                logger.info(f"{message} - {json.dumps(data)}")
            except:
                logger.info(f"{message} - {str(data)}")
        else:
            logger.info(f"{message} - {str(data)}")
    else:
        logger.info(message)

def load_data(start_date: datetime.date, end_date: datetime.date, 
              selected_customer: str = "All Customers") -> pd.DataFrame:
    """Load support ticket data based on date range and customer selection.
    
    Args:
        start_date: Start date for filtering tickets
        end_date: End date for filtering tickets
        selected_customer: Customer name to filter by, or "All Customers"
        
    Returns:
        DataFrame containing the filtered support ticket data
    """
    try:
        debug("Loading data", {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'customer': selected_customer
        })
        
        # Check if data is already in session state
        if 'data' not in st.session_state:
            # Load sample data or connect to data source
            data_path = os.path.join('data', 'sample', 'support_tickets.csv')
            
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, parse_dates=['Created Date', 'Closed Date'])
                debug(f"Data loaded from {data_path}", {'rows': len(df)})
            else:
                # If sample data doesn't exist, create synthetic data
                debug("Sample data not found, generating synthetic data")
                df = generate_sample_data(2000)
                
            # Store in session state
            st.session_state.data = df
            debug("Data stored in session state", {'rows': len(df)})
            
            # Extract and store available customers
            if 'Account_Name' in df.columns:
                customers = sorted(df['Account_Name'].unique().tolist())
                st.session_state.available_customers = customers
                debug("Available customers stored", {'count': len(customers)})
                
        else:
            # Use cached data from session state
            df = st.session_state.data
            debug("Using cached data from session state", {'rows': len(df)})
            
        # Filter data by date range
        mask = (df['Created Date'].dt.date >= start_date) & (df['Created Date'].dt.date <= end_date)
        filtered_df = df[mask].copy()
        
        # Filter by customer if specified
        if selected_customer != "All Customers" and 'Account_Name' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Account_Name'] == selected_customer]
            
        # Add derived columns for analysis
        filtered_df = preprocess_data(filtered_df)
        
        debug("Data filtered successfully", {
            'original_rows': len(df),
            'filtered_rows': len(filtered_df),
            'date_range_applied': f"{start_date} to {end_date}",
            'customer_filter_applied': selected_customer != "All Customers"
        })
        
        return filtered_df
        
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        logger.error(error_msg, exc_info=True)
        debug(error_msg, {'traceback': traceback.format_exc()}, category="error")
        # Return empty DataFrame
        return pd.DataFrame()

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for analysis.
    
    Args:
        df: DataFrame containing the raw support ticket data
        
    Returns:
        DataFrame with additional derived columns
    """
    try:
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Calculate Resolution Time in days if not already present
        if 'Resolution Time (Days)' not in result_df.columns and 'Closed Date' in result_df.columns:
            # Only calculate for tickets that have been closed
            closed_mask = ~result_df['Closed Date'].isna()
            
            # Calculate time difference in days
            result_df.loc[closed_mask, 'Resolution Time (Days)'] = (
                (result_df.loc[closed_mask, 'Closed Date'] - result_df.loc[closed_mask, 'Created Date'])
                .dt.total_seconds() / (24 * 3600)
            ).round(1)
            
            debug("Added Resolution Time column", {
                'non_null_values': closed_mask.sum(),
                'mean_resolution_time': result_df['Resolution Time (Days)'].mean()
            })
        
        # Extract month and year for trend analysis
        result_df['Month_Year'] = result_df['Created Date'].dt.to_period('M').astype(str)
        
        # Add day of week for pattern analysis
        result_df['Day_of_Week'] = result_df['Created Date'].dt.day_name()
        
        # Add week number for weekly reporting
        result_df['Week_Number'] = result_df['Created Date'].dt.isocalendar().week
        
        # Add ticket age for open tickets
        result_df['Ticket_Age_Days'] = np.where(
            result_df['Status'] != 'Closed',
            (datetime.now() - result_df['Created Date']).dt.total_seconds() / (24 * 3600),
            result_df.get('Resolution Time (Days)', 0)
        )
        
        # Convert categorical columns to categories for memory efficiency
        for col in ['Status', 'Priority', 'Product Area', 'Account_Name', 'Month_Year', 'Day_of_Week']:
            if col in result_df.columns:
                result_df[col] = result_df[col].astype('category')
        
        # Log preprocessing completion
        debug("Data preprocessing completed", {
            'rows': len(result_df),
            'columns': list(result_df.columns)
        })
        
        return result_df
        
    except Exception as e:
        error_msg = f"Error preprocessing data: {str(e)}"
        logger.error(error_msg, exc_info=True)
        debug(error_msg, {'traceback': traceback.format_exc()}, category="error")
        # Return original DataFrame if preprocessing fails
        return df

def generate_sample_data(num_records: int = 1000) -> pd.DataFrame:
    """Generate synthetic support ticket data for testing.
    
    Args:
        num_records: Number of synthetic records to generate
        
    Returns:
        DataFrame containing synthetic support ticket data
    """
    try:
        debug(f"Generating {num_records} synthetic records")
        
        # Create date range for tickets
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Last year of data
        
        # Random creation dates
        created_dates = [
            start_date + timedelta(days=np.random.randint(0, 365))
            for _ in range(num_records)
        ]
        
        # Sample account names
        account_names = ["Acme Corp", "TechGiant", "DataSystems", "CloudNine", 
                        "FutureTech", "InnovateCo", "MegaSoft", "SmartSolutions"]
        
        # Sample product areas
        product_areas = ["Authentication", "API", "Dashboard", "Reporting", 
                        "Integration", "Mobile App", "Admin Panel", "Data Import"]
        
        # Sample priorities
        priorities = ["Low", "Medium", "High", "Critical"]
        priority_weights = [0.3, 0.4, 0.2, 0.1]  # More medium priority tickets
        
        # Sample statuses
        statuses = ["Open", "In Progress", "Pending", "Closed"]
        status_weights = [0.1, 0.2, 0.1, 0.6]  # More closed tickets
        
        # Generate ticket numbers
        case_numbers = [f"CASE-{np.random.randint(10000, 99999)}" for _ in range(num_records)]
        
        # Generate subjects
        subject_prefixes = ["Error when", "Issue with", "Problem in", "Cannot access", 
                          "Failed to", "Bug in", "Help with", "Question about"]
        
        subject_actions = ["logging in", "saving data", "generating report", "connecting to API", 
                         "updating profile", "exporting data", "configuring settings", "using dashboard"]
        
        subjects = [
            f"{np.random.choice(subject_prefixes)} {np.random.choice(subject_actions)}"
            for _ in range(num_records)
        ]
        
        # Generate data
        data = {
            'Case Number': case_numbers,
            'Subject': subjects,
            'Status': np.random.choice(statuses, size=num_records, p=status_weights),
            'Priority': np.random.choice(priorities, size=num_records, p=priority_weights),
            'Created Date': created_dates,
            'Account_Name': np.random.choice(account_names, size=num_records),
            'Product Area': np.random.choice(product_areas, size=num_records)
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add closed dates for closed tickets
        closed_mask = df['Status'] == 'Closed'
        
        # Resolution time follows a lognormal distribution
        # Log-normal parameters: mean=3 days, but with some outliers
        resolution_days = np.random.lognormal(mean=1.0, sigma=1.0, size=closed_mask.sum())
        resolution_days = np.clip(resolution_days, 0.1, 60)  # Clip to reasonable range
        
        # Calculate closed dates
        df.loc[closed_mask, 'Closed Date'] = [
            created + timedelta(days=days)
            for created, days in zip(df.loc[closed_mask, 'Created Date'], resolution_days)
        ]
        
        # NaN for non-closed tickets
        df.loc[~closed_mask, 'Closed Date'] = pd.NaT
        
        # Add resolution time column
        df.loc[closed_mask, 'Resolution Time (Days)'] = resolution_days
        
        # Add CSAT scores for some closed tickets (about 40%)
        csat_mask = (df['Status'] == 'Closed') & (np.random.random(size=num_records) < 0.4)
        
        # CSAT scores - bias towards higher scores (more 4s and 5s)
        csat_options = [1, 2, 3, 4, 5]
        csat_weights = [0.05, 0.1, 0.2, 0.35, 0.3]
        df.loc[csat_mask, 'CSAT'] = np.random.choice(csat_options, size=csat_mask.sum(), p=csat_weights)
        
        debug("Sample data generated successfully", {'rows': len(df)})
        return df
        
    except Exception as e:
        error_msg = f"Error generating sample data: {str(e)}"
        logger.error(error_msg, exc_info=True)
        debug(error_msg, {'traceback': traceback.format_exc()}, category="error")
        # Return empty DataFrame if generation fails
        return pd.DataFrame()


def export_data(df: pd.DataFrame, export_format: str = "CSV") -> Tuple[bool, Optional[str], Optional[bytes]]:
    """Export the data to the specified format.
    
    Args:
        df: DataFrame to export
        export_format: Format to export (CSV, Excel, or JSON)
        
    Returns:
        Tuple of (success, filename, file_data)
    """
    try:
        if df.empty:
            debug("Cannot export: DataFrame is empty")
            return False, None, None
            
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Process for each format
        if export_format == "CSV":
            filename = f"support_tickets_{timestamp}.csv"
            buffer = df.to_csv(index=False).encode('utf-8')
            
        elif export_format == "Excel":
            filename = f"support_tickets_{timestamp}.xlsx"
            # Use BytesIO to create in-memory Excel file
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Support Tickets')
            buffer = buffer.getvalue()
            
        elif export_format == "JSON":
            filename = f"support_tickets_{timestamp}.json"
            buffer = df.to_json(orient='records', date_format='iso').encode('utf-8')
            
        else:
            debug(f"Invalid export format: {export_format}")
            return False, None, None
            
        debug(f"Data exported successfully", {
            'format': export_format,
            'filename': filename,
            'rows': len(df)
        })
        
        return True, filename, buffer
        
    except Exception as e:
        error_msg = f"Error exporting data: {str(e)}"
        logger.error(error_msg, exc_info=True)
        debug(error_msg, {'traceback': traceback.format_exc()}, category="error")
        return False, None, None 
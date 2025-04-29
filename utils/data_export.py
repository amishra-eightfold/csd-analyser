"""Utilities for exporting analyzed ticket data."""

import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import zipfile
from io import BytesIO
from config.logging_config import get_logger

logger = get_logger('app')

def get_available_exports(export_dir: str = "exports") -> List[Dict[str, Any]]:
    """
    Get information about available exports.
    
    Args:
        export_dir: Directory containing exports
        
    Returns:
        List of dictionaries containing export information
    """
    try:
        export_path = Path(export_dir)
        if not export_path.exists():
            return []
            
        exports = []
        for customer_dir in export_path.iterdir():
            if not customer_dir.is_dir():
                continue
                
            customer_files = list(customer_dir.glob("*_tickets_*"))
            if not customer_files:
                continue
                
            # Get timestamp from first file
            timestamp_str = customer_files[0].stem.split("_tickets_")[-1]
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except ValueError:
                timestamp = None
                
            export_info = {
                'customer_name': customer_dir.name,
                'timestamp': timestamp,
                'timestamp_str': timestamp_str,
                'files': [
                    {
                        'name': f.name,
                        'path': f,
                        'size': f.stat().st_size,
                        'type': f.suffix[1:].upper()
                    }
                    for f in customer_files
                ]
            }
            exports.append(export_info)
            
        # Sort by timestamp, most recent first
        exports.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min, reverse=True)
        return exports
        
    except Exception as e:
        logger.error(f"Error getting available exports: {str(e)}")
        return []

def create_customer_export_zip(customer_name: str, timestamp_str: str, export_dir: str = "exports") -> Tuple[BytesIO, str]:
    """
    Create a zip file containing all export files for a customer.
    
    Args:
        customer_name: Name of the customer
        timestamp_str: Timestamp string to identify the export
        export_dir: Directory containing exports
        
    Returns:
        Tuple of (BytesIO containing zip file, filename)
    """
    try:
        export_path = Path(export_dir)
        customer_dir = export_path / customer_name
        if not customer_dir.exists():
            raise FileNotFoundError(f"No exports found for customer {customer_name}")
            
        # Create zip file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add all files for this customer and timestamp
            for file_path in customer_dir.glob(f"*_tickets_{timestamp_str}*"):
                zip_file.write(file_path, file_path.name)
                
        zip_buffer.seek(0)
        filename = f"{customer_name}_export_{timestamp_str}.zip"
        
        return zip_buffer, filename
        
    except Exception as e:
        logger.error(f"Error creating zip for customer {customer_name}: {str(e)}")
        raise

def ensure_export_directory(export_dir: str = "exports") -> Path:
    """
    Ensure the export directory exists and has proper structure.
    Creates a README file if it doesn't exist.
    
    Args:
        export_dir: Base directory for exports
        
    Returns:
        Path object pointing to the export directory
    """
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    
    # Create README file if it doesn't exist
    readme_path = export_path / "README.md"
    if not readme_path.exists():
        readme_content = """# Support Ticket Analysis Exports

This directory contains exported support ticket analysis data organized by customer.

## Directory Structure
```
exports/
├── README.md
├── Customer1/
│   ├── customer1_tickets_20240401_123456_details.csv    # Ticket details
│   ├── customer1_tickets_20240401_123456_history.csv    # Priority history
│   └── customer1_tickets_20240401_123456_emails.csv     # Email messages
└── Customer2/
    └── ...
```

## File Types
- `*_details.csv`: Contains main ticket information including highest priority
- `*_history.csv`: Contains priority change history for tickets
- `*_emails.csv`: Contains email messages related to tickets

## Timestamps
Files are named with timestamps in the format: YYYYMMDD_HHMMSS

## Download
Use the Exports section in the sidebar to download individual files or complete customer data as ZIP archives.
"""
        readme_path.write_text(readme_content)
        
    return export_path

def export_customer_ticket_details(
    tickets_df: pd.DataFrame,
    history_df: pd.DataFrame,
    email_messages: Dict[str, List[Dict[str, Any]]],
    highest_priorities: Dict[str, str],
    export_dir: str = "exports"
) -> None:
    """
    Export ticket details per customer with priority history and email messages.
    
    Args:
        tickets_df: DataFrame containing ticket data from Salesforce
        history_df: DataFrame containing ticket history data
        email_messages: Dictionary mapping ticket IDs to their email messages
        highest_priorities: Dictionary mapping ticket IDs to their highest priority
        export_dir: Directory to save the exported files
    """
    try:
        # Ensure export directory exists and has proper structure
        export_path = ensure_export_directory(export_dir)
        
        # Create timestamp for export
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log DataFrame info for debugging
        logger.info("Starting export with data frames:", extra={
            'tickets_columns': list(tickets_df.columns),
            'history_columns': list(history_df.columns) if not history_df.empty else [],
            'total_tickets': len(tickets_df),
            'total_history_records': len(history_df)
        })
        
        # Normalize history DataFrame column names if they exist
        if not history_df.empty:
            column_mapping = {
                'Case ID': 'CaseId',
                'CaseID': 'CaseId',
                'Case_ID': 'CaseId',
                'caseid': 'CaseId'
            }
            history_df = history_df.rename(columns=column_mapping)
            
            if 'CaseId' not in history_df.columns:
                logger.error("Could not find case ID column in history DataFrame", extra={
                    'available_columns': list(history_df.columns)
                })
                # Create empty CaseId column to prevent errors
                history_df['CaseId'] = None
        
        # Group tickets by customer
        for customer, customer_tickets in tickets_df.groupby('Account_Name'):
            try:
                logger.debug(f"Processing customer: {customer}", extra={
                    'customer_tickets': len(customer_tickets),
                    'customer_name': customer
                })
                
                # Clean customer name for filename
                customer_name = "".join(c for c in customer if c.isalnum() or c in (' ', '-', '_')).strip()
                if not customer_name:
                    customer_name = "unknown_customer"
                
                # Create customer export directory
                customer_dir = export_path / customer_name
                customer_dir.mkdir(exist_ok=True)
                
                # Get ticket IDs for this customer
                customer_ticket_ids = customer_tickets['Id'].tolist()
                logger.debug(f"Found {len(customer_ticket_ids)} tickets for customer", extra={
                    'customer': customer,
                    'ticket_count': len(customer_ticket_ids)
                })
                
                # Add highest priority to tickets
                customer_tickets = customer_tickets.copy()
                customer_tickets['Highest Priority'] = customer_tickets['Id'].map(
                    lambda x: highest_priorities.get(x, customer_tickets.loc[customer_tickets['Id'] == x, 'Priority'].iloc[0] if len(customer_tickets.loc[customer_tickets['Id'] == x]) > 0 else 'Unknown')
                )
                
                # Get history records for customer tickets - with error handling
                try:
                    if not history_df.empty and 'CaseId' in history_df.columns:
                        customer_history = history_df[history_df['CaseId'].isin(customer_ticket_ids)].copy()
                        logger.debug(f"Found {len(customer_history)} history records", extra={
                            'customer': customer,
                            'history_count': len(customer_history)
                        })
                    else:
                        customer_history = pd.DataFrame()
                        logger.warning(f"No history data available for customer {customer}")
                except Exception as history_error:
                    logger.error(f"Error processing history for customer {customer}: {str(history_error)}")
                    customer_history = pd.DataFrame()
                
                # Get email messages for customer tickets
                customer_emails = {}
                for ticket_id in customer_ticket_ids:
                    if ticket_id in email_messages:
                        customer_emails[ticket_id] = email_messages[ticket_id]
                
                # Create email messages DataFrame
                email_rows = []
                for ticket_id, messages in customer_emails.items():
                    for msg in messages:
                        email_rows.append({
                            'CaseId': ticket_id,
                            'Timestamp': msg.get('timestamp', ''),
                            'From': msg.get('from', ''),
                            'To': msg.get('to', ''),
                            'Subject': msg.get('subject', ''),
                            'Body': msg.get('body', ''),
                            'Direction': msg.get('direction', '')  # inbound/outbound
                        })
                emails_df = pd.DataFrame(email_rows)
                
                # Export files
                export_filename = f"{customer_name}_tickets_{timestamp}"
                
                # Export main ticket data
                customer_tickets.to_csv(
                    customer_dir / f"{export_filename}_details.csv",
                    index=False,
                    encoding='utf-8'
                )
                
                # Export history data
                if not customer_history.empty:
                    customer_history.to_csv(
                        customer_dir / f"{export_filename}_history.csv",
                        index=False,
                        encoding='utf-8'
                    )
                
                # Export email messages
                if not emails_df.empty:
                    emails_df.to_csv(
                        customer_dir / f"{export_filename}_emails.csv",
                        index=False,
                        encoding='utf-8'
                    )
                
                logger.info(
                    f"Exported data for customer: {customer}",
                    extra={
                        'customer': customer,
                        'tickets': len(customer_tickets),
                        'history_records': len(customer_history),
                        'email_messages': len(emails_df)
                    }
                )
                
            except Exception as customer_error:
                logger.error(
                    f"Error exporting data for customer {customer}: {str(customer_error)}",
                    extra={'customer': customer}
                )
                continue
        
        logger.info(
            "Completed exporting customer ticket details",
            extra={'total_customers': len(tickets_df['Account_Name'].unique())}
        )
        
    except Exception as e:
        logger.error(f"Error in export_customer_ticket_details: {str(e)}")
        raise 
"""Data fetching module for CSD Analyzer."""

from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
from .core.salesforce import salesforce

def fetch_data(
    customers: List[str],
    start_date: datetime,
    end_date: datetime,
    include_comments: bool = True,
    include_emails: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data from Salesforce for analysis.

    Args:
        customers: List of customer names to fetch data for
        start_date: Start date for data fetch
        end_date: End date for data fetch
        include_comments: Whether to fetch case comments
        include_emails: Whether to fetch email messages

    Returns:
        Dictionary containing DataFrames for cases, comments, and emails
    """
    if not salesforce.is_connected:
        if not salesforce.connect():
            raise ConnectionError("Failed to connect to Salesforce")

    # Fetch cases
    cases_df = salesforce.get_support_tickets(customers, start_date, end_date)
    
    # Initialize result dictionary
    result = {'cases': cases_df}

    # Fetch comments if requested
    if include_comments and not cases_df.empty:
        case_ids = "'" + "','".join(cases_df['Id'].tolist()) + "'"
        comment_query = f"""
            SELECT Id, ParentId, CommentBody, CreatedDate
            FROM CaseComment
            WHERE ParentId IN ({case_ids})
            ORDER BY CreatedDate ASC
        """
        try:
            comments = salesforce.execute_query(comment_query)
            result['comments'] = pd.DataFrame(comments)
        except Exception:
            result['comments'] = pd.DataFrame()

    # Fetch emails if requested
    if include_emails and not cases_df.empty:
        email_query = f"""
            SELECT Id, ParentId, Subject, TextBody, CreatedDate
            FROM EmailMessage
            WHERE ParentId IN ({case_ids})
            ORDER BY CreatedDate ASC
        """
        try:
            emails = salesforce.execute_query(email_query)
            result['emails'] = pd.DataFrame(emails)
        except Exception:
            result['emails'] = pd.DataFrame()

    return result 
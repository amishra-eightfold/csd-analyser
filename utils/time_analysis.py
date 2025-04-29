"""Time analysis utilities for support ticket data."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_first_response_time(
    df: pd.DataFrame,
    created_date_col: str = 'Created Date',
    first_response_col: str = 'First Response Time'
) -> Tuple[pd.Series, dict]:
    """
    Calculate first response time in hours between case creation and first customer email.
    
    Args:
        df: DataFrame containing the ticket data
        created_date_col: Name of the column containing case creation timestamp
        first_response_col: Name of the column containing first response timestamp
        
    Returns:
        Tuple containing:
        - Series with response times in hours
        - Dictionary with validation statistics
    """
    try:
        stats = {
            'total_records': len(df),
            'valid_records': 0,
            'invalid_records': 0,
            'error_details': []
        }
        
        # Ensure required columns exist
        if not all(col in df.columns for col in [created_date_col, first_response_col]):
            missing_cols = [col for col in [created_date_col, first_response_col] if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Initialize response time series
        response_hours = pd.Series(index=df.index, dtype='float64')
        
        # Convert timestamps to datetime if needed
        created_dates = pd.to_datetime(df[created_date_col], utc=True)
        response_dates = pd.to_datetime(df[first_response_col], utc=True)
        
        # Calculate time difference
        valid_mask = created_dates.notna() & response_dates.notna()
        if valid_mask.any():
            time_diff = response_dates[valid_mask] - created_dates[valid_mask]
            response_hours[valid_mask] = time_diff.dt.total_seconds() / 3600
            
            # Filter out negative or extremely large values (more than 30 days)
            valid_range_mask = (response_hours >= 0) & (response_hours <= 720)  # 720 hours = 30 days
            response_hours[~valid_range_mask] = np.nan
            
            stats['valid_records'] = valid_range_mask.sum()
            stats['invalid_records'] = (~valid_range_mask).sum()
            
            if (~valid_range_mask).any():
                stats['error_details'].append(
                    f"Found {(~valid_range_mask).sum()} records with invalid response times "
                    "(negative or > 30 days)"
                )
        
        # Log validation results
        logger.info(
            f"First response time calculation completed. "
            f"Valid records: {stats['valid_records']}, "
            f"Invalid records: {stats['invalid_records']}"
        )
        
        return response_hours, stats
        
    except Exception as e:
        logger.error(f"Error calculating first response time: {str(e)}", exc_info=True)
        raise

def calculate_sla_breaches(
    response_hours: pd.Series,
    priority_series: pd.Series,
    sla_thresholds: dict = None
) -> pd.DataFrame:
    """
    Calculate SLA breach statistics for first response times.
    
    Args:
        response_hours: Series containing response times in hours
        priority_series: Series containing priority levels
        sla_thresholds: Dictionary mapping priority levels to SLA thresholds in hours
                       Defaults to P0: 1h, P1: 24h, P2: 48h
                       
    Returns:
        DataFrame with breach statistics per priority level
    """
    if sla_thresholds is None:
        sla_thresholds = {
            'P0': 1,     # 1 hour
            'P1': 24,    # 24 hours
            'P2': 48,    # 48 hours
            # P3 has no SLA
        }
    
    try:
        breach_stats = []
        
        for priority in sorted(priority_series.unique()):
            if pd.isna(priority) or priority in ['Not Set', 'Unknown', '', None]:
                continue
                
            priority_mask = priority_series == priority
            priority_times = response_hours[priority_mask].dropna()
            
            if len(priority_times) == 0:
                continue
                
            stats = {
                'Priority': priority,
                'Count': len(priority_times),
                'Mean Hours': priority_times.mean(),
                'Median Hours': priority_times.median(),
                '90th Percentile': priority_times.quantile(0.9)
            }
            
            # Calculate breach percentage if priority has SLA
            if priority in sla_thresholds:
                threshold = sla_thresholds[priority]
                breached = priority_times[priority_times > threshold]
                stats['SLA Breach %'] = (len(breached) / len(priority_times)) * 100
            else:
                stats['SLA Breach %'] = None
                
            breach_stats.append(stats)
        
        return pd.DataFrame(breach_stats)
        
    except Exception as e:
        logger.error(f"Error calculating SLA breaches: {str(e)}", exc_info=True)
        raise 
"""Data validation utilities for the CSD Analyzer application."""

import pandas as pd
from typing import List, Dict, Union, Optional
from functools import wraps

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate if a DataFrame has all required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (List[str]): List of required column names
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    return all(col in df.columns for col in required_columns)

def validate_date_range(df: pd.DataFrame, 
                       start_date: pd.Timestamp, 
                       end_date: pd.Timestamp, 
                       date_column: str = 'CreatedDate') -> bool:
    """
    Validate if DataFrame contains data within the specified date range.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        start_date (pd.Timestamp): Start date for validation
        end_date (pd.Timestamp): End date for validation
        date_column (str): Name of the date column to check
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    if date_column not in df.columns:
        return False
        
    df_dates = pd.to_datetime(df[date_column])
    return not df_dates[(df_dates >= start_date) & (df_dates <= end_date)].empty

def validate_numeric_range(df: pd.DataFrame, 
                         column: str, 
                         min_value: float, 
                         max_value: float) -> bool:
    """
    Validate if numeric values in a column fall within the specified range.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        column (str): Column name to check
        min_value (float): Minimum allowed value
        max_value (float): Maximum allowed value
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    if column not in df.columns:
        return False
        
    numeric_values = pd.to_numeric(df[column], errors='coerce')
    return numeric_values.between(min_value, max_value).all()

class DataFrameValidator:
    """Class for validating DataFrame properties and contents."""
    
    @staticmethod
    def validate_schema(df: pd.DataFrame, schema: Dict[str, str]) -> bool:
        """
        Validate DataFrame schema against expected data types.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            schema (Dict[str, str]): Dictionary mapping column names to expected dtypes
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        if not all(col in df.columns for col in schema.keys()):
            return False
            
        for col, dtype in schema.items():
            if str(df[col].dtype) != dtype:
                return False
        return True
    
    @staticmethod
    def validate_no_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> bool:
        """
        Check if DataFrame contains duplicate rows.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            subset (Optional[List[str]]): List of columns to check for duplicates
            
        Returns:
            bool: True if no duplicates found, False otherwise
        """
        return not df.duplicated(subset=subset).any()
    
    @staticmethod
    def validate_no_nulls(df: pd.DataFrame, columns: Optional[List[str]] = None) -> bool:
        """
        Check if DataFrame contains null values in specified columns.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            columns (Optional[List[str]]): List of columns to check for nulls
            
        Returns:
            bool: True if no nulls found, False otherwise
        """
        if columns is None:
            columns = df.columns
        return not df[columns].isnull().any().any()

def validate_dataframe_decorator(required_columns: List[str]):
    """
    Decorator for validating DataFrame inputs to functions.
    
    Args:
        required_columns (List[str]): List of required column names
        
    Returns:
        callable: Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find DataFrame argument
            df = None
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    df = arg
                    break
            for value in kwargs.values():
                if isinstance(value, pd.DataFrame):
                    df = value
                    break
            
            if df is None:
                raise ValueError("No DataFrame argument found")
                
            if not validate_dataframe(df, required_columns):
                raise ValueError(f"DataFrame missing required columns: {required_columns}")
                
            return func(*args, **kwargs)
        return wrapper
    return decorator 
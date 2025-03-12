"""Base class for DataFrame operations."""

import pandas as pd
from typing import List, Dict, Optional, Union, Any
from utils.data_validation import validate_dataframe, validate_date_range
from utils.error_handlers import handle_errors
import numpy as np

class BaseDataProcessor:
    """Base class for processing DataFrames with common operations."""
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initialize the data processor.
        
        Args:
            df (Optional[pd.DataFrame]): Initial DataFrame to process
        """
        self.df = df.copy() if df is not None else None
        
    def set_data(self, df: pd.DataFrame):
        """
        Set the DataFrame to process.
        
        Args:
            df (pd.DataFrame): DataFrame to process
        """
        self.df = df.copy()
        
    @handle_errors(custom_message="Error cleaning data")
    def clean_data(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Clean the DataFrame by handling missing values and data types.
        
        Args:
            columns (Optional[List[str]]): Specific columns to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if self.df is None:
            raise ValueError("No DataFrame set")
            
        df = self.df.copy()
        columns = columns or df.columns
        
        for col in columns:
            if col not in df.columns:
                continue
                
            # Handle missing values based on data type
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].fillna(False)
            else:
                df[col] = df[col].fillna('')
                
        return df
    
    @handle_errors(custom_message="Error filtering data")
    def filter_data(self, 
                   conditions: Dict[str, Any],
                   combine: str = 'and') -> pd.DataFrame:
        """
        Filter DataFrame based on conditions.
        
        Args:
            conditions (Dict[str, Any]): Dictionary of column-value pairs for filtering
            combine (str): How to combine conditions ('and' or 'or')
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        if self.df is None:
            raise ValueError("No DataFrame set")
            
        df = self.df.copy()
        mask = None
        
        for column, value in conditions.items():
            if column not in df.columns:
                continue
                
            current_mask = df[column] == value
            
            if mask is None:
                mask = current_mask
            elif combine.lower() == 'and':
                mask = mask & current_mask
            else:
                mask = mask | current_mask
                
        return df[mask] if mask is not None else df
    
    @handle_errors(custom_message="Error aggregating data")
    def aggregate_data(self,
                      group_by: Union[str, List[str]],
                      agg_columns: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Aggregate DataFrame by specified columns.
        
        Args:
            group_by (Union[str, List[str]]): Column(s) to group by
            agg_columns (Dict[str, List[str]]): Dictionary of column-aggregation pairs
            
        Returns:
            pd.DataFrame: Aggregated DataFrame
        """
        if self.df is None:
            raise ValueError("No DataFrame set")
            
        df = self.df.copy()
        return df.groupby(group_by).agg(agg_columns).reset_index()
    
    @handle_errors(custom_message="Error transforming data")
    def transform_column(self,
                        column: str,
                        transformation: str,
                        **kwargs) -> pd.DataFrame:
        """
        Apply transformation to a column.
        
        Args:
            column (str): Column to transform
            transformation (str): Type of transformation
            **kwargs: Additional arguments for transformation
            
        Returns:
            pd.DataFrame: DataFrame with transformed column
        """
        if self.df is None:
            raise ValueError("No DataFrame set")
            
        df = self.df.copy()
        
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")
            
        if transformation == 'normalize':
            df[f"{column}_normalized"] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        elif transformation == 'standardize':
            df[f"{column}_standardized"] = (df[column] - df[column].mean()) / df[column].std()
        elif transformation == 'log':
            df[f"{column}_log"] = np.log1p(df[column])
        elif transformation == 'bin':
            bins = kwargs.get('bins', 10)
            df[f"{column}_binned"] = pd.qcut(df[column], bins, labels=False)
        else:
            raise ValueError(f"Unsupported transformation: {transformation}")
            
        return df
    
    @handle_errors(custom_message="Error calculating statistics")
    def calculate_statistics(self, 
                           columns: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate basic statistics for specified columns.
        
        Args:
            columns (Optional[List[str]]): Columns to analyze
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of statistics by column
        """
        if self.df is None:
            raise ValueError("No DataFrame set")
            
        df = self.df.copy()
        columns = columns or df.select_dtypes(include=['number']).columns
        
        stats = {}
        for col in columns:
            if col not in df.columns:
                continue
                
            if pd.api.types.is_numeric_dtype(df[col]):
                stats[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
                
        return stats 
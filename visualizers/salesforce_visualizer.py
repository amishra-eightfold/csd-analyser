"""Salesforce data visualizer for the CSD Analyzer application."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, List, Any
from base.visualizer import BaseVisualizer
from utils.error_handlers import handle_errors
from config.visualization_config import (
    VOLUME_PALETTE, PRIORITY_PALETTE, CSAT_PALETTE,
    HEATMAP_PALETTE, ROOT_CAUSE_PALETTE,
    TIME_SERIES_SETTINGS, BAR_PLOT_SETTINGS,
    BOX_PLOT_SETTINGS, HEATMAP_SETTINGS
)

class SalesforceVisualizer(BaseVisualizer):
    """Class for creating visualizations from Salesforce data."""
    
    @handle_errors(custom_message="Error creating case volume plot")
    def plot_case_volume(self,
                        df: pd.DataFrame,
                        title: str = "Case Volume Over Time") -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a case volume over time plot.
        
        Args:
            df (pd.DataFrame): Cases DataFrame
            title (str): Plot title
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        # Prepare data
        daily_created = df.groupby(df['CreatedDate'].dt.date).size()
        daily_closed = df.groupby(df['ClosedDate'].dt.date).size()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot lines with custom settings
        ax.plot(daily_created.index, daily_created.values,
                label='Created',
                color=VOLUME_PALETTE[0],
                **TIME_SERIES_SETTINGS)
        ax.plot(daily_closed.index, daily_closed.values,
                label='Closed',
                color=VOLUME_PALETTE[1],
                **TIME_SERIES_SETTINGS)
        
        # Style plot
        ax.set_title(title, pad=20)
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Cases")
        ax.legend()
        plt.tight_layout()
        
        return fig, ax
    
    @handle_errors(custom_message="Error creating priority distribution plot")
    def plot_priority_distribution(self,
                                 df: pd.DataFrame,
                                 title: str = "Case Priority Distribution") -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a priority distribution plot.
        
        Args:
            df (pd.DataFrame): Cases DataFrame
            title (str): Plot title
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        # Prepare data
        priority_counts = df['Internal_Priority__c'].value_counts()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bars with custom settings
        sns.barplot(x=priority_counts.index,
                   y=priority_counts.values,
                   palette=PRIORITY_PALETTE,
                   **BAR_PLOT_SETTINGS)
        
        # Style plot
        ax.set_title(title, pad=20)
        ax.set_xlabel("Priority")
        ax.set_ylabel("Number of Cases")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig, ax
    
    @handle_errors(custom_message="Error creating CSAT distribution plot")
    def plot_csat_distribution(self,
                             df: pd.DataFrame,
                             title: str = "CSAT Score Distribution") -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a CSAT score distribution plot.
        
        Args:
            df (pd.DataFrame): Cases DataFrame
            title (str): Plot title
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        # Prepare data
        csat_scores = df['CSAT__c'].dropna()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram with custom settings
        sns.histplot(data=csat_scores,
                    bins=5,
                    color=CSAT_PALETTE[2],
                    **BAR_PLOT_SETTINGS)
        
        # Style plot
        ax.set_title(title, pad=20)
        ax.set_xlabel("CSAT Score")
        ax.set_ylabel("Number of Responses")
        plt.tight_layout()
        
        return fig, ax
    
    @handle_errors(custom_message="Error creating response time plot")
    def plot_response_times(self,
                          df: pd.DataFrame,
                          title: str = "First Response Time Distribution") -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a response time distribution plot.
        
        Args:
            df (pd.DataFrame): Cases DataFrame
            title (str): Plot title
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        # Prepare data
        response_times = df['First_Response_Time__c'].dropna()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot box plot with custom settings
        sns.boxplot(y=response_times,
                   color=PRIORITY_PALETTE[2],
                   **BOX_PLOT_SETTINGS)
        
        # Style plot
        ax.set_title(title, pad=20)
        ax.set_ylabel("Hours")
        plt.tight_layout()
        
        return fig, ax
    
    @handle_errors(custom_message="Error creating product area plot")
    def plot_product_areas(self,
                          df: pd.DataFrame,
                          title: str = "Cases by Product Area") -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a product area distribution plot.
        
        Args:
            df (pd.DataFrame): Cases DataFrame
            title (str): Plot title
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        # Prepare data
        area_counts = df['Product_Area__c'].value_counts()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot bars with custom settings
        sns.barplot(x=area_counts.values,
                   y=area_counts.index,
                   palette=ROOT_CAUSE_PALETTE,
                   **BAR_PLOT_SETTINGS)
        
        # Style plot
        ax.set_title(title, pad=20)
        ax.set_xlabel("Number of Cases")
        ax.set_ylabel("Product Area")
        plt.tight_layout()
        
        return fig, ax
    
    @handle_errors(custom_message="Error creating correlation heatmap")
    def plot_correlation_matrix(self,
                              df: pd.DataFrame,
                              numeric_columns: Optional[List[str]] = None,
                              title: str = "Correlation Matrix") -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a correlation matrix heatmap.
        
        Args:
            df (pd.DataFrame): Cases DataFrame
            numeric_columns (Optional[List[str]]): List of numeric columns to include
            title (str): Plot title
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        # Prepare data
        if numeric_columns:
            corr_matrix = df[numeric_columns].corr()
        else:
            corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap with custom settings
        sns.heatmap(corr_matrix,
                   cmap=HEATMAP_PALETTE,
                   **HEATMAP_SETTINGS,
                   ax=ax)
        
        # Style plot
        ax.set_title(title, pad=20)
        plt.tight_layout()
        
        return fig, ax 
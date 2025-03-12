"""Base class for visualization operations."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, List, Union
from utils.visualization_helpers import PlotStyle
from utils.error_handlers import handle_errors

class BaseVisualizer:
    """Base class for creating visualizations with common styling and setup."""
    
    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        """
        Initialize the visualizer.
        
        Args:
            style (str): Matplotlib style to use
        """
        self.style = style
        plt.style.use(self.style)
        sns.set_theme(style="whitegrid")
        
    @handle_errors(custom_message="Error creating time series plot")
    def plot_time_series(self,
                        df: pd.DataFrame,
                        date_column: str,
                        value_column: str,
                        group_column: Optional[str] = None,
                        title: str = "",
                        xlabel: str = "Date",
                        ylabel: str = "Value",
                        figsize: Tuple[int, int] = (12, 6)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a time series plot.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data
            date_column (str): Name of the date column
            value_column (str): Name of the value column
            group_column (Optional[str]): Name of the grouping column
            title (str): Plot title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if group_column:
            for group in df[group_column].unique():
                group_data = df[df[group_column] == group]
                ax.plot(group_data[date_column], 
                       group_data[value_column], 
                       marker='o', 
                       label=group)
            ax.legend()
        else:
            ax.plot(df[date_column], df[value_column], marker='o')
            
        PlotStyle.apply_common_styling(ax, title, xlabel, ylabel)
        return fig, ax
    
    @handle_errors(custom_message="Error creating distribution plot")
    def plot_distribution(self,
                         df: pd.DataFrame,
                         column: str,
                         plot_type: str = 'histogram',
                         title: str = "",
                         xlabel: str = "",
                         ylabel: str = "Count",
                         figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a distribution plot.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data
            column (str): Column to plot
            plot_type (str): Type of plot ('histogram' or 'kde')
            title (str): Plot title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if plot_type == 'histogram':
            sns.histplot(data=df, x=column, ax=ax)
        elif plot_type == 'kde':
            sns.kdeplot(data=df, x=column, ax=ax)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
            
        PlotStyle.apply_common_styling(ax, title, xlabel, ylabel)
        return fig, ax
    
    @handle_errors(custom_message="Error creating correlation plot")
    def plot_correlation(self,
                        df: pd.DataFrame,
                        columns: Optional[List[str]] = None,
                        method: str = 'pearson',
                        title: str = "Correlation Matrix",
                        figsize: Tuple[int, int] = (10, 8)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a correlation matrix plot.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data
            columns (Optional[List[str]]): Columns to include
            method (str): Correlation method ('pearson' or 'spearman')
            title (str): Plot title
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        if columns:
            correlation_matrix = df[columns].corr(method=method)
        else:
            correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr(method=method)
            
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='YlGnBu', 
                   center=0,
                   ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        return fig, ax
    
    @handle_errors(custom_message="Error creating box plot")
    def plot_box(self,
                 df: pd.DataFrame,
                 x_column: str,
                 y_column: str,
                 title: str = "",
                 xlabel: str = "",
                 ylabel: str = "",
                 figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a box plot.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data
            x_column (str): Column for x-axis categories
            y_column (str): Column for y-axis values
            title (str): Plot title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(data=df, x=x_column, y=y_column, hue=x_column, legend=False)
        PlotStyle.apply_common_styling(ax, title, xlabel, ylabel, rotate_xlabels=True)
        return fig, ax
    
    @handle_errors(custom_message="Error creating bar plot")
    def plot_bar(self,
                 df: pd.DataFrame,
                 x_column: str,
                 y_column: str,
                 title: str = "",
                 xlabel: str = "",
                 ylabel: str = "",
                 figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a bar plot.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data
            x_column (str): Column for x-axis categories
            y_column (str): Column for y-axis values
            title (str): Plot title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=df, x=x_column, y=y_column)
        PlotStyle.apply_common_styling(ax, title, xlabel, ylabel, rotate_xlabels=True)
        return fig, ax 
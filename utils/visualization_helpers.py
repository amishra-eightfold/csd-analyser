"""Visualization helper utilities for the CSD Analyzer application."""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, List, Union
import pandas as pd

# Define color palettes
BLUES_PALETTE = ["#E3F2FD", "#90CAF9", "#42A5F5", "#1E88E5", "#1565C0", "#0D47A1"]
AQUA_PALETTE = ["#E0F7FA", "#80DEEA", "#26C6DA", "#00ACC1", "#00838F", "#006064"]
PURPLE_PALETTE = ["#F3E5F5", "#CE93D8", "#AB47BC", "#8E24AA", "#6A1B9A", "#4A148C"]

class PlotStyle:
    """Class for managing plot styles and themes."""
    
    @staticmethod
    def set_default_style():
        """Set default plot style for consistency."""
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_theme(style="whitegrid")
    
    @staticmethod
    def set_figure_size(width: float = 10, height: float = 6) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create figure and axes with specified size.
        
        Args:
            width (float): Figure width
            height (float): Figure height
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        fig, ax = plt.subplots(figsize=(width, height))
        return fig, ax

    @staticmethod
    def apply_common_styling(ax: plt.Axes, 
                           title: str, 
                           xlabel: str, 
                           ylabel: str,
                           rotate_xlabels: bool = False):
        """
        Apply common styling to plot axes.
        
        Args:
            ax (plt.Axes): Axes object to style
            title (str): Plot title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            rotate_xlabels (bool): Whether to rotate x-axis labels
        """
        ax.set_title(title, pad=20)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if rotate_xlabels:
            plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

class TimeSeriesPlotter:
    """Class for creating time series visualizations."""
    
    @staticmethod
    def plot_trend(df: pd.DataFrame,
                  date_column: str,
                  value_column: str,
                  group_column: Optional[str] = None,
                  title: str = "",
                  xlabel: str = "Date",
                  ylabel: str = "Value") -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a time series trend plot.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data
            date_column (str): Name of the date column
            value_column (str): Name of the value column
            group_column (Optional[str]): Name of the grouping column
            title (str): Plot title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        fig, ax = PlotStyle.set_figure_size(12, 6)
        
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

class DistributionPlotter:
    """Class for creating distribution visualizations."""
    
    @staticmethod
    def plot_boxplot(df: pd.DataFrame,
                    x_column: str,
                    y_column: str,
                    title: str = "",
                    xlabel: str = "",
                    ylabel: str = "") -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a box plot.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data
            x_column (str): Name of the x-axis column
            y_column (str): Name of the y-axis column
            title (str): Plot title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        fig, ax = PlotStyle.set_figure_size(10, 6)
        sns.boxplot(data=df, x=x_column, y=y_column, hue=x_column, legend=False)
        PlotStyle.apply_common_styling(ax, title, xlabel, ylabel, rotate_xlabels=True)
        return fig, ax
    
    @staticmethod
    def plot_histogram(df: pd.DataFrame,
                      column: str,
                      bins: int = 20,
                      title: str = "",
                      xlabel: str = "",
                      ylabel: str = "Count") -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a histogram.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data
            column (str): Name of the column to plot
            bins (int): Number of bins
            title (str): Plot title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        fig, ax = PlotStyle.set_figure_size(10, 6)
        sns.histplot(data=df, x=column, bins=bins)
        PlotStyle.apply_common_styling(ax, title, xlabel, ylabel)
        return fig, ax

class HeatmapPlotter:
    """Class for creating heatmap visualizations."""
    
    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               title: str = "Correlation Heatmap") -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a correlation heatmap.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data
            columns (Optional[List[str]]): List of columns to include
            title (str): Plot title
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        if columns:
            correlation_matrix = df[columns].corr()
        else:
            correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
            
        fig, ax = PlotStyle.set_figure_size(10, 8)
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='YlGnBu', 
                   center=0,
                   ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        return fig, ax 
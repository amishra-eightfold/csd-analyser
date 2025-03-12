"""Unit tests for SalesforceVisualizer."""

import pytest
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from visualizers.salesforce_visualizer import SalesforceVisualizer

@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test."""
    yield
    plt.close('all')

def test_init():
    """Test visualizer initialization."""
    visualizer = SalesforceVisualizer()
    assert visualizer is not None
    assert visualizer.style == "seaborn-v0_8-whitegrid"
    assert len(visualizer.colors) == 15
    assert visualizer.fig_size == (12, 6)

def test_plot_case_volume(sample_case_data):
    """Test case volume plot creation."""
    visualizer = SalesforceVisualizer()
    
    # Create plot
    fig, ax = visualizer.plot_case_volume(sample_case_data)
    
    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Case Volume Over Time"
    assert ax.get_xlabel() == "Date"
    assert ax.get_ylabel() == "Number of Cases"
    
    # Check legend entries
    legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "Created" in legend_texts
    assert "Closed" in legend_texts
    assert len(ax.get_legend().get_texts()) == 2  # Created and Closed

def test_plot_priority_distribution(sample_case_data):
    """Test priority distribution plot creation."""
    visualizer = SalesforceVisualizer()
    
    # Create plot
    fig, ax = visualizer.plot_priority_distribution(sample_case_data)
    
    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Case Priority Distribution"
    assert ax.get_xlabel() == "Priority"
    assert ax.get_ylabel() == "Number of Cases"
    
    # Check data presence
    assert len(ax.patches) > 0  # Should have bars

def test_plot_csat_distribution(sample_case_data):
    """Test CSAT distribution plot creation."""
    visualizer = SalesforceVisualizer()
    
    # Create plot
    fig, ax = visualizer.plot_csat_distribution(sample_case_data)
    
    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "CSAT Score Distribution"
    assert ax.get_xlabel() == "CSAT Score"
    assert ax.get_ylabel() == "Number of Cases"
    
    # Check data presence
    assert len(ax.patches) > 0  # Should have bars

def test_plot_response_times(sample_case_data):
    """Test response time plot creation."""
    visualizer = SalesforceVisualizer()
    
    # Create plot
    fig, ax = visualizer.plot_response_times(sample_case_data)
    
    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "First Response Time Distribution"
    assert ax.get_ylabel() == "Hours"

def test_plot_product_areas(sample_case_data):
    """Test product area plot creation."""
    visualizer = SalesforceVisualizer()
    
    # Create plot
    fig, ax = visualizer.plot_product_areas(sample_case_data)
    
    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Cases by Product Area"
    assert ax.get_xlabel() == "Number of Cases"
    assert ax.get_ylabel() == "Product Area"
    
    # Check data presence
    assert len(ax.patches) > 0  # Should have bars

def test_plot_correlation_matrix(sample_case_data):
    """Test correlation matrix plot creation."""
    visualizer = SalesforceVisualizer()
    
    # Create plot with specific numeric columns
    numeric_columns = ['First_Response_Time__c', 'CSAT__c']
    fig, ax = visualizer.plot_correlation_matrix(
        sample_case_data,
        numeric_columns=numeric_columns
    )
    
    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Correlation Matrix"
    
    # Check heatmap presence
    assert len(ax.collections) > 0  # Should have heatmap

def test_error_handling():
    """Test error handling in the visualizer."""
    visualizer = SalesforceVisualizer()
    
    # Test with invalid DataFrame
    with pytest.raises(ValueError):
        visualizer.plot_case_volume(None)
    
    # Test with empty DataFrame
    with pytest.raises(ValueError):
        visualizer.plot_case_volume(pd.DataFrame())
    
    # Test with missing required columns
    df = pd.DataFrame({'A': [1, 2, 3]})
    with pytest.raises(ValueError):
        visualizer.plot_case_volume(df)

def test_custom_styling(sample_case_data):
    """Test custom styling options."""
    visualizer = SalesforceVisualizer(style="seaborn-v0_8-darkgrid")
    assert visualizer.style == "seaborn-v0_8-darkgrid"
    
    # Create plot with custom title
    fig, ax = visualizer.plot_case_volume(sample_case_data, title="Custom Title")
    
    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Custom Title"
    
    # Check figure size
    assert fig.get_size_inches().tolist() == [12, 6]  # Default size for case volume plot 
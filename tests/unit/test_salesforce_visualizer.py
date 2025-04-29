"""
Unit tests for the SalesforceVisualizer class.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from src.visualization.salesforce_visualizer import SalesforceVisualizer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='D')
    data = {
        'Created Date': dates,
        'Closed Date': dates + timedelta(days=2),
        'First Response Time': dates + timedelta(hours=4),
        'Priority': np.random.choice(['P1', 'P2', 'P3', 'P4'], size=len(dates)),
        'CSAT': np.random.uniform(1, 5, size=len(dates)),
        'Product Area': np.random.choice(['Area1', 'Area2'], size=len(dates)),
        'Product Feature': np.random.choice(['Feature1', 'Feature2'], size=len(dates))
    }
    return pd.DataFrame(data)

@pytest.fixture
def visualizer():
    """Create a SalesforceVisualizer instance."""
    return SalesforceVisualizer()

def test_create_ticket_volume_chart(visualizer, sample_data):
    """Test ticket volume chart creation."""
    fig = visualizer.create_ticket_volume_chart(sample_data)
    assert fig is not None
    assert len(fig.data) == 2  # Should have two traces (Created and Closed)
    assert fig.data[0].name == 'Created'
    assert fig.data[1].name == 'Closed'

def test_create_response_time_chart(visualizer, sample_data):
    """Test response time chart creation."""
    fig = visualizer.create_response_time_chart(sample_data)
    assert fig is not None
    assert len(fig.data) == len(sample_data['Priority'].unique())
    for trace in fig.data:
        assert trace.type == 'box'

def test_create_csat_chart(visualizer, sample_data):
    """Test CSAT chart creation."""
    fig = visualizer.create_csat_chart(sample_data)
    assert fig is not None
    assert len(fig.data) == 1  # Should have one trace
    assert fig.data[0].mode == 'lines+markers'

def test_create_product_heatmap(visualizer, sample_data):
    """Test product heatmap creation."""
    fig = visualizer.create_product_heatmap(sample_data)
    assert fig is not None
    assert fig.data[0].type == 'heatmap'

def test_color_palettes(visualizer):
    """Test color palette initialization."""
    assert len(visualizer.VIRIDIS_PALETTE) == 5
    assert len(visualizer.AQUA_PALETTE) == 6
    assert len(visualizer.PRIORITY_COLORS) == 4
    assert 'P1' in visualizer.PRIORITY_COLORS
    assert 'P4' in visualizer.PRIORITY_COLORS

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

@pytest.fixture
def sample_case_data():
    """Create sample case data for testing."""
    data = {
        'Id': ['case1', 'case2'],
        'CaseNumber': ['00001', '00002'],
        'Subject': ['Test Case 1', 'Test Case 2'],
        'Description': ['Test Description 1', 'Test Description 2'],
        'Priority': ['P1', 'P2'],
        'CreatedDate': ['2025-03-28', '2025-03-30'],
        'ClosedDate': ['2025-03-29', None],
        'Status': ['Closed', 'Open'],
        'Internal_Priority__c': ['High', 'Medium'],
        'Product_Area__c': ['Frontend', 'Backend'],
        'Product_Feature__c': ['UI', 'API'],
        'RCA__c': ['Bug', 'Configuration'],
        'First_Response_Time__c': [1.5, None],
        'CSAT__c': [4.0, None],
        'IsEscalated': [True, False],
        'Resolution_Time_Days__c': [3.0, None]
    }
    return pd.DataFrame(data)

def test_init():
    """Test visualizer initialization."""
    visualizer = SalesforceVisualizer()
    assert visualizer is not None
    assert visualizer.style == "seaborn-v0_8-whitegrid"
    assert visualizer.VIRIDIS_PALETTE is not None
    assert visualizer.AQUA_PALETTE is not None

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
    assert ax.get_xlabel() == "Product Area"
    assert ax.get_ylabel() == "Number of Cases"

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

def test_error_handling():
    """Test error handling in the visualizer."""
    visualizer = SalesforceVisualizer()

    # Test with invalid DataFrame
    with pytest.raises(ValueError):
        visualizer.plot_case_volume(None)

    # Test with empty DataFrame
    with pytest.raises(ValueError):
        visualizer.plot_case_volume(pd.DataFrame())

def test_custom_styling(sample_case_data):
    """Test custom styling options."""
    visualizer = SalesforceVisualizer(style="seaborn-v0_8-darkgrid")
    assert visualizer.style == "seaborn-v0_8-darkgrid"

    # Create plot
    fig, ax = visualizer.plot_case_volume(sample_case_data)
    assert fig is not None
    assert ax is not None

def test_create_ticket_volume_chart(sample_case_data):
    """Test ticket volume chart creation."""
    visualizer = SalesforceVisualizer()
    fig, ax = visualizer.create_ticket_volume_chart(sample_case_data)
    assert fig is not None
    assert ax is not None

def test_create_response_time_chart(sample_case_data):
    """Test response time chart creation."""
    visualizer = SalesforceVisualizer()
    fig, ax = visualizer.create_response_time_chart(sample_case_data)
    assert fig is not None
    assert ax is not None

def test_create_csat_chart(sample_case_data):
    """Test CSAT chart creation."""
    visualizer = SalesforceVisualizer()
    fig, ax = visualizer.create_csat_chart(sample_case_data)
    assert fig is not None
    assert ax is not None

def test_create_product_heatmap(sample_case_data):
    """Test product heatmap creation."""
    visualizer = SalesforceVisualizer()
    fig, ax = visualizer.create_product_heatmap(sample_case_data)
    assert fig is not None
    assert ax is not None

def test_color_palettes():
    """Test color palette initialization."""
    visualizer = SalesforceVisualizer()
    assert visualizer.VIRIDIS_PALETTE is not None
    assert visualizer.AQUA_PALETTE is not None 
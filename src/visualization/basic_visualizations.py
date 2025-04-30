"""Basic visualization functions for the support ticket analysis dashboard."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import traceback
from typing import List, Dict, Any
import logging

# Import common helpers
from utils.visualization_helpers import truncate_string
from config.logging_config import get_logger

# Initialize logger
logger = get_logger('visualizations')

# Custom color palettes
VOLUME_PALETTE = ["#26C6DA", "#00838F"]  # Two distinct colors for Created/Closed
AQUA_PALETTE = ["#E0F7FA", "#80DEEA", "#26C6DA", "#00ACC1", "#00838F", "#006064"]  # Material Cyan/Aqua

def debug(message, data=None, category="app"):
    """Log debug information to the logger."""
    if hasattr(st.session_state, 'debug_logger'):
        st.session_state.debug_logger.log(message, data, category)
    
    # Log to file logger
    logger = get_logger(category)
    if data is not None:
        # Convert data to string if needed for logging
        if isinstance(data, dict):
            try:
                import json
                logger.info(f"{message} - {json.dumps(data)}")
            except:
                logger.info(f"{message} - {str(data)}")
        else:
            logger.info(f"{message} - {str(data)}")
    else:
        logger.info(message)

def display_visualizations(df: pd.DataFrame, customers: List[str]) -> None:
    """Display basic visualizations using the dataset.
    
    Args:
        df: DataFrame containing the support ticket data
        customers: List of selected customer names
    """
    try:
        # 1. Ticket Volume by Customer (Bar Chart)
        st.subheader("Ticket Distribution")
        debug("Generating ticket distribution visualization")
        
        # Create a mapping of truncated names to ticket counts
        ticket_counts = df.groupby('Account_Name', dropna=False).size().reset_index(name='Ticket_Count')
        ticket_counts['Truncated_Name'] = ticket_counts['Account_Name'].apply(lambda x: truncate_string(x, 20))
        
        fig_counts = go.Figure(data=[
            go.Bar(
                x=ticket_counts['Truncated_Name'],
                y=ticket_counts['Ticket_Count'],
                text=ticket_counts['Ticket_Count'],
                textposition='auto',
                marker_color=AQUA_PALETTE[2],
                hovertext=ticket_counts['Account_Name']  # Show full name on hover
            )
        ])
        
        fig_counts.update_layout(
            title='Ticket Count by Customer',
            xaxis_title='Customer',
            yaxis_title='Number of Tickets',
            showlegend=False,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_counts, key='ticket_distribution')
        debug("Ticket distribution visualization completed")
        
        # 2. Monthly Ticket Trends (Bar Chart)
        st.subheader("Monthly Ticket Trends")
        debug("Generating monthly trends visualization")
        
        df['Month'] = df['Created Date'].dt.to_period('M')
        df['Month_Closed'] = df['Closed Date'].dt.to_period('M')
        
        monthly_created = df.groupby('Month').size().reset_index(name='Created')
        monthly_created['Month'] = monthly_created['Month'].astype(str)
        
        monthly_closed = df.groupby('Month_Closed').size().reset_index(name='Closed')
        monthly_closed['Month_Closed'] = monthly_closed['Month_Closed'].astype(str)
        
        monthly_trends = pd.merge(
            monthly_created, 
            monthly_closed.rename(columns={'Month_Closed': 'Month'}), 
            on='Month', 
            how='outer'
        ).fillna(0)
        
        monthly_trends['Month'] = pd.to_datetime(monthly_trends['Month'])
        monthly_trends = monthly_trends.sort_values('Month')
        monthly_trends['Month'] = monthly_trends['Month'].dt.strftime('%b')
        
        debug("Monthly trends data prepared", {
            'months_available': len(monthly_trends),
            'date_range': f"{monthly_trends['Month'].iloc[0]} to {monthly_trends['Month'].iloc[-1]}"
        })
        
        fig_trends = go.Figure()
        
        fig_trends.add_trace(go.Bar(
            name='Created',
            x=monthly_trends['Month'],
            y=monthly_trends['Created'],
            marker_color=VOLUME_PALETTE[0]
        ))
        
        fig_trends.add_trace(go.Bar(
            name='Closed',
            x=monthly_trends['Month'],
            y=monthly_trends['Closed'],
            marker_color=VOLUME_PALETTE[1]
        ))
        
        fig_trends.update_layout(
            title='Monthly Ticket Volume',
            xaxis_title='Month',
            yaxis_title='Number of Tickets',
            barmode='group',
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_trends, key='monthly_trends')
        debug("Monthly trends visualization completed")
        
    except Exception as e:
        error_msg = f"Error in basic visualizations: {str(e)}"
        st.error(error_msg)
        debug(error_msg, {'traceback': traceback.format_exc()}, category="error")
        logger.error(error_msg, exc_info=True)
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.exception(e) 
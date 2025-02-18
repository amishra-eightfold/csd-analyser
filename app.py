import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from io import BytesIO
from pptx import Presentation
from pptx.util import Inches, Pt
import base64
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Customer Support Ticket Analysis",
    page_icon="📊",
    layout="wide"
)

# Add title and file uploader
st.title("Customer Support Ticket Analysis Dashboard")
st.markdown("Upload your data file (CSV or Excel)")

# Define required columns based on Case-28_01_2025.csv
REQUIRED_COLUMNS = [
    'Id', 'CaseNumber', 'Account.Account_Name__c', 'Group_Id__c', 
    'Subject', 'Description', 'Product_Area__c', 'Product_Feature__c',
    'POD_Name__c', 'CreatedDate', 'ClosedDate', 'Case_Type__c',
    'Age_days__c', 'IsEscalated', 'CSAT__c', 'Internal_Priority__c'
]

def truncate_account_name(name, max_length=15):
    """Helper function to truncate account names."""
    if isinstance(name, str) and len(name) > max_length:
        return name[:max_length] + '...'
    return name

def generate_wordcloud(text_data, title):
    """Generate a word cloud from the given text data."""
    if not isinstance(text_data, str):
        text_data = ' '.join(str(x) for x in text_data if pd.notna(x))
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        collocations=False,
        stopwords=set(['nan', 'none', 'null', 'unspecified'])
    ).generate(text_data)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    
    return fig

def generate_powerpoint(filtered_df, active_accounts, avg_csat, escalation_rate):
    """Generate PowerPoint presentation with charts and statistics."""
    try:
        # Create presentation
        prs = Presentation()
        
        # Set slide width and height (16:9 aspect ratio)
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
        
        # Title slide
        title_slide_layout = prs.slide_layouts[0]
        slide = prs.slides.add_slide(title_slide_layout)
        title = slide.shapes.title
        subtitle = slide.placeholders[1]
        title.text = "Customer Support Ticket Analysis"
        subtitle.text = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Overview Statistics slide
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        title = slide.shapes.title
        title.text = "Overview Statistics"
        content = slide.placeholders[1]
        
        # Add statistics as bullet points
        stats_text = (
            f"• Total Cases: {len(filtered_df)}\n"
            f"• Active Accounts: {active_accounts}\n"
            f"• Product Areas: {filtered_df['Product_Area__c'].nunique()}\n"
            f"• Average CSAT: {avg_csat:.2f}\n"
            f"• Escalation Rate: {escalation_rate:.1f}%"
        )
        content.text = stats_text
        
        # Account Analysis slides
        # Top Accounts slide
        slide = prs.slides.add_slide(prs.slide_layouts[5])  # Blank layout
        title = slide.shapes.title
        title.text = "Top Accounts by Case Volume"
        
        # Save account volume chart
        account_cases = filtered_df.groupby('Account.Account_Name__c').size().reset_index(name='count')
        account_cases = account_cases.sort_values('count', ascending=True).tail(10)
        account_cases['Account.Account_Name__c'] = account_cases['Account.Account_Name__c'].apply(truncate_account_name)
        fig_account = px.bar(
            account_cases,
            x='count',
            y='Account.Account_Name__c',
            title='Top 10 Accounts by Case Volume',
            orientation='h',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_account.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        img_bytes = fig_account.to_image(format="png", width=1000, height=600, scale=2)
        img_stream = BytesIO(img_bytes)
        slide.shapes.add_picture(img_stream, Inches(1), Inches(1.5), width=Inches(11))

        # CSAT by Account slide
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        title = slide.shapes.title
        title.text = "CSAT by Account"
        
        csat_data = filtered_df[filtered_df['CSAT__c'] != 0]
        if not csat_data.empty:
            csat_by_account = csat_data.groupby('Account.Account_Name__c')['CSAT__c'].mean().reset_index()
            csat_by_account['Account.Account_Name__c'] = csat_by_account['Account.Account_Name__c'].apply(truncate_account_name)
            fig_csat = px.bar(
                csat_by_account,
                x='Account.Account_Name__c',
                y='CSAT__c',
                title='Average CSAT Score by Account',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_csat.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False,
                xaxis_tickangle=-45
            )
            img_bytes = fig_csat.to_image(format="png", width=1000, height=600, scale=2)
            img_stream = BytesIO(img_bytes)
            slide.shapes.add_picture(img_stream, Inches(1), Inches(1.5), width=Inches(11))

        # Monthly Trends slides
        # Volume Trends slide
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        title = slide.shapes.title
        title.text = "Monthly Ticket Volume Trends"
        
        filtered_df['Month'] = filtered_df['CreatedDate'].dt.to_period('M')
        monthly_volume = filtered_df.groupby(['Month', 'Account.Account_Name__c']).size().reset_index(name='count')
        monthly_volume['Month'] = monthly_volume['Month'].astype(str)
        
        fig_monthly_volume = px.line(
            monthly_volume,
            x='Month',
            y='count',
            color='Account.Account_Name__c',
            title='Monthly Ticket Volume by Account',
            labels={'count': 'Number of Tickets', 'Month': 'Month', 'Account.Account_Name__c': 'Account'}
        )
        fig_monthly_volume.update_layout(
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        img_bytes = fig_monthly_volume.to_image(format="png", width=1000, height=600, scale=2)
        img_stream = BytesIO(img_bytes)
        slide.shapes.add_picture(img_stream, Inches(1), Inches(1.5), width=Inches(11))

        # Monthly CSAT Trends slide
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        title = slide.shapes.title
        title.text = "Monthly CSAT Trends"
        
        monthly_csat = filtered_df[filtered_df['CSAT__c'] != 0].groupby(['Month', 'Account.Account_Name__c']).agg({
            'CSAT__c': ['mean', 'count']
        }).reset_index()
        monthly_csat.columns = ['Month', 'Account.Account_Name__c', 'Average CSAT', 'CSAT Count']
        monthly_csat['Month'] = monthly_csat['Month'].astype(str)
        
        fig_monthly_csat = go.Figure()
        
        # Add traces for Average CSAT (left y-axis)
        for account in monthly_csat['Account.Account_Name__c'].unique():
            account_data = monthly_csat[monthly_csat['Account.Account_Name__c'] == account]
            fig_monthly_csat.add_trace(
                go.Scatter(
                    x=account_data['Month'],
                    y=account_data['Average CSAT'],
                    name=f"{account} (Avg)",
                    mode='lines+markers',
                    line=dict(dash='solid')
                )
            )
        
        # Add traces for CSAT Count (right y-axis)
        for account in monthly_csat['Account.Account_Name__c'].unique():
            account_data = monthly_csat[monthly_csat['Account.Account_Name__c'] == account]
            fig_monthly_csat.add_trace(
                go.Scatter(
                    x=account_data['Month'],
                    y=account_data['CSAT Count'],
                    name=f"{account} (Count)",
                    mode='lines+markers',
                    line=dict(dash='dot'),
                    yaxis='y2'
                )
            )
        
        fig_monthly_csat.update_layout(
            title='Monthly CSAT Metrics by Account',
            xaxis=dict(title='Month', tickangle=-45),
            yaxis=dict(title='Average CSAT', side='left'),
            yaxis2=dict(title='CSAT Count', side='right', overlaying='y'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        img_bytes = fig_monthly_csat.to_image(format="png", width=1000, height=600, scale=2)
        img_stream = BytesIO(img_bytes)
        slide.shapes.add_picture(img_stream, Inches(1), Inches(1.5), width=Inches(11))

        # Priority Analysis slides
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        title = slide.shapes.title
        title.text = "Priority Distribution"
        
        # Cases by Priority chart
        priority_dist = filtered_df[filtered_df['Internal_Priority__c'] != 'Unspecified'].copy()
        priority_dist = priority_dist.groupby('Internal_Priority__c').size().reset_index(name='count')
        priority_dist = priority_dist.sort_values(
            by='Internal_Priority__c',
            key=lambda x: x.map(lambda y: int(y[1:]) if y.startswith('P') and y[1:].isdigit() else 999)
        )
        
        fig_priority = px.pie(
            priority_dist,
            values='count',
            names='Internal_Priority__c',
            title='Case Distribution by Priority',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_priority.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        img_bytes = fig_priority.to_image(format="png", width=1000, height=600, scale=2)
        img_stream = BytesIO(img_bytes)
        slide.shapes.add_picture(img_stream, Inches(1), Inches(1.5), width=Inches(11))

        # Escalation Analysis slide
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        title = slide.shapes.title
        title.text = "Escalation Analysis"
        
        escalation_by_priority = filtered_df[filtered_df['Internal_Priority__c'] != 'Unspecified'].copy()
        escalation_by_priority = escalation_by_priority.groupby('Internal_Priority__c')['IsEscalated'].mean().mul(100).reset_index(name='escalation_rate')
        escalation_by_priority = escalation_by_priority.sort_values(
            by='Internal_Priority__c',
            key=lambda x: x.map(lambda y: int(y[1:]) if y.startswith('P') and y[1:].isdigit() else 999)
        )
        
        fig_escalation = px.bar(
            escalation_by_priority,
            x='Internal_Priority__c',
            y='escalation_rate',
            title='Escalation Rate by Priority (%)',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_escalation.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="Priority",
            yaxis_title="Escalation Rate (%)",
            showlegend=False
        )
        img_bytes = fig_escalation.to_image(format="png", width=1000, height=600, scale=2)
        img_stream = BytesIO(img_bytes)
        slide.shapes.add_picture(img_stream, Inches(1), Inches(1.5), width=Inches(11))

        # Product Analysis slides
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        title = slide.shapes.title
        title.text = "Product Analysis"
        
        # Cases by Product Area
        area_cases = filtered_df.groupby('Product_Area__c').size().reset_index(name='count')
        fig_area = px.pie(
            area_cases,
            values='count',
            names='Product_Area__c',
            title='Case Distribution by Product Area',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_area.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        img_bytes = fig_area.to_image(format="png", width=1000, height=600, scale=2)
        img_stream = BytesIO(img_bytes)
        slide.shapes.add_picture(img_stream, Inches(1), Inches(1.5), width=Inches(11))

        # Feature Distribution slide
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        title = slide.shapes.title
        title.text = "Feature Distribution"
        
        feature_area = filtered_df.groupby(['Product_Area__c', 'Product_Feature__c']).size().reset_index(name='count')
        fig_feature_area = px.treemap(
            feature_area,
            path=['Product_Area__c', 'Product_Feature__c'],
            values='count',
            title='Feature Distribution by Product Area',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_feature_area.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        img_bytes = fig_feature_area.to_image(format="png", width=1000, height=600, scale=2)
        img_stream = BytesIO(img_bytes)
        slide.shapes.add_picture(img_stream, Inches(1), Inches(1.5), width=Inches(11))

        # Add RCA Analysis slides after Product Analysis
        # RCA Distribution slide
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        title = slide.shapes.title
        title.text = "Root Cause Analysis"
        
        rca_dist = filtered_df[filtered_df['RCA__c'] != 'Unspecified'].groupby('RCA__c').size().reset_index(name='count')
        rca_dist = rca_dist.sort_values('count', ascending=True)
        
        fig_rca = px.bar(
            rca_dist,
            x='count',
            y='RCA__c',
            title='Distribution of Root Causes',
            orientation='h',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_rca.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            yaxis_title="Root Cause",
            xaxis_title="Number of Cases"
        )
        img_bytes = fig_rca.to_image(format="png", width=1000, height=600, scale=2)
        img_stream = BytesIO(img_bytes)
        slide.shapes.add_picture(img_stream, Inches(1), Inches(1.5), width=Inches(11))

        # RCA by Priority slide
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        title = slide.shapes.title
        title.text = "Root Causes by Priority"
        
        rca_priority = filtered_df[
            (filtered_df['RCA__c'] != 'Unspecified') & 
            (filtered_df['Internal_Priority__c'] != 'Unspecified')
        ].groupby(['Internal_Priority__c', 'RCA__c']).size().reset_index(name='count')
        
        rca_priority = rca_priority.sort_values(
            by='Internal_Priority__c',
            key=lambda x: x.map(lambda y: int(y[1:]) if y.startswith('P') and y[1:].isdigit() else 999)
        )
        
        fig_rca_priority = px.bar(
            rca_priority,
            x='Internal_Priority__c',
            y='count',
            color='RCA__c',
            title='Root Causes by Priority',
            barmode='stack',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_rca_priority.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="Priority",
            yaxis_title="Number of Cases",
            legend_title="Root Cause",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        img_bytes = fig_rca_priority.to_image(format="png", width=1000, height=600, scale=2)
        img_stream = BytesIO(img_bytes)
        slide.shapes.add_picture(img_stream, Inches(1), Inches(1.5), width=Inches(11))

        # Add Text Analysis slides if requested
        if st.session_state.get('include_wordclouds', False):
            # Subject Word Cloud slide
            slide = prs.slides.add_slide(prs.slide_layouts[5])
            title = slide.shapes.title
            title.text = "Text Analysis - Common Terms"
            
            fig_subject = generate_wordcloud(filtered_df['Subject'], 'Common Terms in Case Subjects')
            img_stream = BytesIO()
            fig_subject.savefig(img_stream, format='png', bbox_inches='tight', dpi=300)
            plt.close(fig_subject)
            img_stream.seek(0)
            slide.shapes.add_picture(img_stream, Inches(1), Inches(1.5), width=Inches(11))
            
            # Description Word Cloud slide
            slide = prs.slides.add_slide(prs.slide_layouts[5])
            title = slide.shapes.title
            title.text = "Text Analysis - Case Descriptions"
            
            fig_desc = generate_wordcloud(filtered_df['Description'], 'Common Terms in Case Descriptions')
            img_stream = BytesIO()
            fig_desc.savefig(img_stream, format='png', bbox_inches='tight', dpi=300)
            plt.close(fig_desc)
            img_stream.seek(0)
            slide.shapes.add_picture(img_stream, Inches(1), Inches(1.5), width=Inches(11))
            
            # Product Information Word Clouds slide
            slide = prs.slides.add_slide(prs.slide_layouts[5])
            title = slide.shapes.title
            title.text = "Product Information Analysis"
            
            # Create a 2x2 grid of smaller word clouds
            fields = ['POD_Name__c', 'Product_Area__c', 'Product_Feature__c']
            titles = ['POD Names', 'Product Areas', 'Product Features']
            
            for i, (field, subtitle) in enumerate(zip(fields, titles)):
                fig = generate_wordcloud(filtered_df[field], subtitle)
                img_stream = BytesIO()
                fig.savefig(img_stream, format='png', bbox_inches='tight', dpi=300)
                plt.close(fig)
                img_stream.seek(0)
                
                # Calculate position for 2x2 grid
                left = Inches(1 if i % 2 == 0 else 7)
                top = Inches(1.5 if i < 2 else 4)
                slide.shapes.add_picture(img_stream, left, top, width=Inches(5))
        
        # Save presentation
        pptx_output = BytesIO()
        prs.save(pptx_output)
        return pptx_output.getvalue()
    except Exception as e:
        raise Exception(f"Error generating PowerPoint: {str(e)}")

def display_visualizations(filtered_df):
    """Display all visualizations for the filtered data."""
    # Display basic statistics
    st.header("Overview Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Cases", len(filtered_df))

    with col2:
        # Calculate active accounts (accounts with at least 5 tickets)
        account_ticket_counts = filtered_df['Account.Account_Name__c'].value_counts()
        active_accounts_series = account_ticket_counts[account_ticket_counts >= 5]
        active_accounts_series = active_accounts_series[active_accounts_series.index != 'Unspecified']
        active_accounts = len(active_accounts_series)
        st.metric("Active Accounts", active_accounts)

    with col3:
        st.metric("Product Areas", filtered_df['Product_Area__c'].nunique())

    with col4:
        avg_csat = filtered_df[filtered_df['CSAT__c'] != 0]['CSAT__c'].mean()
        st.metric("Avg CSAT", f"{avg_csat:.2f}" if not pd.isna(avg_csat) else "N/A")

    with col5:
        escalation_rate = filtered_df['IsEscalated'].mean() * 100
        st.metric("Escalation Rate", f"{escalation_rate:.1f}%")

    # Account Analysis
    st.header("Account Analysis")
    
    # Top Accounts chart
    account_cases = filtered_df.groupby('Account.Account_Name__c').size().reset_index(name='count')
    account_cases = account_cases.sort_values('count', ascending=True).tail(10)
    account_cases['Account.Account_Name__c'] = account_cases['Account.Account_Name__c'].apply(truncate_account_name)

    fig_account = px.bar(
        account_cases,
        x='count',
        y='Account.Account_Name__c',
        title='Top 10 Accounts by Case Volume',
        orientation='h',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_account.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False
    )
    st.plotly_chart(fig_account, use_container_width=True)

    # CSAT by Account chart
    csat_data = filtered_df[filtered_df['CSAT__c'] != 0]
    if not csat_data.empty:
        csat_by_account = csat_data.groupby('Account.Account_Name__c')['CSAT__c'].mean().reset_index()
        csat_by_account['Account.Account_Name__c'] = csat_by_account['Account.Account_Name__c'].apply(truncate_account_name)
        
        fig_csat = px.bar(
            csat_by_account,
            x='Account.Account_Name__c',
            y='CSAT__c',
            title='Average CSAT Score by Account',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_csat.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_csat, use_container_width=True)

    # Monthly Trends Analysis
    st.header("Monthly Trends")

    # Prepare monthly data
    filtered_df['Month'] = filtered_df['CreatedDate'].dt.to_period('M')
    monthly_volume = filtered_df.groupby(['Month', 'Account.Account_Name__c']).size().reset_index(name='count')
    monthly_volume['Month'] = monthly_volume['Month'].astype(str)
    
    # Monthly Volume Trends
    fig_monthly_volume = px.line(
        monthly_volume,
        x='Month',
        y='count',
        color='Account.Account_Name__c',
        title='Monthly Ticket Volume by Account',
        labels={'count': 'Number of Tickets', 'Month': 'Month', 'Account.Account_Name__c': 'Account'}
    )
    fig_monthly_volume.update_layout(
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig_monthly_volume, use_container_width=True)

    # Monthly CSAT Trends
    monthly_csat = filtered_df[filtered_df['CSAT__c'] != 0].groupby(['Month', 'Account.Account_Name__c']).agg({
        'CSAT__c': ['mean', 'count']
    }).reset_index()
    monthly_csat.columns = ['Month', 'Account.Account_Name__c', 'Average CSAT', 'CSAT Count']
    monthly_csat['Month'] = monthly_csat['Month'].astype(str)
    
    fig_monthly_csat = go.Figure()
    
    # Add traces for Average CSAT (left y-axis)
    for account in monthly_csat['Account.Account_Name__c'].unique():
        account_data = monthly_csat[monthly_csat['Account.Account_Name__c'] == account]
        fig_monthly_csat.add_trace(
            go.Scatter(
                x=account_data['Month'],
                y=account_data['Average CSAT'],
                name=f"{account} (Avg)",
                mode='lines+markers',
                line=dict(dash='solid')
            )
        )
    
    # Add traces for CSAT Count (right y-axis)
    for account in monthly_csat['Account.Account_Name__c'].unique():
        account_data = monthly_csat[monthly_csat['Account.Account_Name__c'] == account]
        fig_monthly_csat.add_trace(
            go.Scatter(
                x=account_data['Month'],
                y=account_data['CSAT Count'],
                name=f"{account} (Count)",
                mode='lines+markers',
                line=dict(dash='dot'),
                yaxis='y2'
            )
        )
    
    fig_monthly_csat.update_layout(
        title='Monthly CSAT Metrics by Account',
        xaxis=dict(title='Month', tickangle=-45),
        yaxis=dict(title='Average CSAT', side='left'),
        yaxis2=dict(title='CSAT Count', side='right', overlaying='y'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig_monthly_csat, use_container_width=True)

    # Priority and Escalation Analysis
    st.header("Priority and Escalation Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Cases by Priority chart
        priority_dist = filtered_df[filtered_df['Internal_Priority__c'] != 'Unspecified'].copy()
        priority_dist = priority_dist.groupby('Internal_Priority__c').size().reset_index(name='count')
        priority_dist = priority_dist.sort_values(
            by='Internal_Priority__c',
            key=lambda x: x.map(lambda y: int(y[1:]) if y.startswith('P') and y[1:].isdigit() else 999)
        )
        
        fig_priority = px.pie(
            priority_dist,
            values='count',
            names='Internal_Priority__c',
            title='Case Distribution by Priority',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_priority.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_priority, use_container_width=True)
    
    with col2:
        # Escalation Rate by Priority chart
        escalation_by_priority = filtered_df[filtered_df['Internal_Priority__c'] != 'Unspecified'].copy()
        escalation_by_priority = escalation_by_priority.groupby('Internal_Priority__c')['IsEscalated'].mean().mul(100).reset_index(name='escalation_rate')
        escalation_by_priority = escalation_by_priority.sort_values(
            by='Internal_Priority__c',
            key=lambda x: x.map(lambda y: int(y[1:]) if y.startswith('P') and y[1:].isdigit() else 999)
        )
        
        fig_escalation = px.bar(
            escalation_by_priority,
            x='Internal_Priority__c',
            y='escalation_rate',
            title='Escalation Rate by Priority (%)',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_escalation.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="Priority",
            yaxis_title="Escalation Rate (%)",
            showlegend=False
        )
        st.plotly_chart(fig_escalation, use_container_width=True)

    # Product Analysis
    st.header("Product Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Cases by Product Area
        area_cases = filtered_df.groupby('Product_Area__c').size().reset_index(name='count')
        fig_area = px.pie(
            area_cases,
            values='count',
            names='Product_Area__c',
            title='Case Distribution by Product Area',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_area.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_area, use_container_width=True)
    
    with col2:
        # Feature Distribution by Product Area
        feature_dist = filtered_df.groupby(['Product_Area__c', 'Product_Feature__c']).size().reset_index(name='count')
        fig_feature = px.treemap(
            feature_dist,
            path=['Product_Area__c', 'Product_Feature__c'],
            values='count',
            title='Feature Distribution by Product Area',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_feature, use_container_width=True)
    
    # Root Cause Analysis
    st.header("Root Cause Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall RCA Distribution
        rca_dist = filtered_df[filtered_df['RCA__c'] != 'Unspecified'].groupby('RCA__c').size().reset_index(name='count')
        rca_dist = rca_dist.sort_values('count', ascending=True)
        
        fig_rca = px.bar(
            rca_dist,
            x='count',
            y='RCA__c',
            title='Distribution of Root Causes',
            orientation='h',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_rca.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            yaxis_title="Root Cause",
            xaxis_title="Number of Cases"
        )
        st.plotly_chart(fig_rca, use_container_width=True)
    
    with col2:
        # RCA by Priority
        rca_priority = filtered_df[
            (filtered_df['RCA__c'] != 'Unspecified') & 
            (filtered_df['Internal_Priority__c'] != 'Unspecified')
        ].groupby(['Internal_Priority__c', 'RCA__c']).size().reset_index(name='count')
        
        # Sort priorities correctly
        rca_priority = rca_priority.sort_values(
            by='Internal_Priority__c',
            key=lambda x: x.map(lambda y: int(y[1:]) if y.startswith('P') and y[1:].isdigit() else 999)
        )
        
        fig_rca_priority = px.bar(
            rca_priority,
            x='Internal_Priority__c',
            y='count',
            color='RCA__c',
            title='Root Causes by Priority',
            barmode='stack',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_rca_priority.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="Priority",
            yaxis_title="Number of Cases",
            legend_title="Root Cause",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_rca_priority, use_container_width=True)
    
    # Text Analysis with Word Clouds
    st.header("Text Analysis")
    st.markdown("Word clouds showing the most common terms in different fields")
    
    # Create tabs for different word clouds
    tabs = st.tabs(["Subject", "Description", "POD Name", "Product Area", "Product Feature"])
    
    with tabs[0]:
        st.subheader("Subject Word Cloud")
        fig_subject = generate_wordcloud(filtered_df['Subject'], 'Common Terms in Case Subjects')
        st.pyplot(fig_subject)
        plt.close(fig_subject)
    
    with tabs[1]:
        st.subheader("Description Word Cloud")
        fig_desc = generate_wordcloud(filtered_df['Description'], 'Common Terms in Case Descriptions')
        st.pyplot(fig_desc)
        plt.close(fig_desc)
    
    with tabs[2]:
        st.subheader("POD Name Word Cloud")
        fig_pod = generate_wordcloud(filtered_df['POD_Name__c'], 'Distribution of POD Names')
        st.pyplot(fig_pod)
        plt.close(fig_pod)
    
    with tabs[3]:
        st.subheader("Product Area Word Cloud")
        fig_area = generate_wordcloud(filtered_df['Product_Area__c'], 'Distribution of Product Areas')
        st.pyplot(fig_area)
        plt.close(fig_area)
    
    with tabs[4]:
        st.subheader("Product Feature Word Cloud")
        fig_feature = generate_wordcloud(filtered_df['Product_Feature__c'], 'Distribution of Product Features')
        st.pyplot(fig_feature)
        plt.close(fig_feature)
    
    # Add word clouds to PowerPoint
    if st.button("Include Word Clouds in PowerPoint"):
        st.session_state['include_wordclouds'] = True
        st.info("Word clouds will be included in the next PowerPoint export")
    
    # Raw Data
    st.header("Raw Data")
    st.dataframe(filtered_df)
    
    # Add export options
    st.header("Export Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Excel export
        if st.button("Export Data as Excel"):
            try:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    filtered_df.to_excel(writer, index=False, sheet_name='Filtered_Data')
                excel_data = output.getvalue()
                st.download_button(
                    label="Download Excel file",
                    data=excel_data,
                    file_name=f"ticket_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error exporting Excel: {str(e)}")
    
    with col2:
        # PowerPoint export
        if st.button("Export as PowerPoint"):
            try:
                with st.spinner("Generating PowerPoint presentation..."):
                    # Calculate metrics needed for PowerPoint
                    account_ticket_counts = filtered_df['Account.Account_Name__c'].value_counts()
                    active_accounts_series = account_ticket_counts[account_ticket_counts >= 5]
                    active_accounts_series = active_accounts_series[active_accounts_series.index != 'Unspecified']
                    active_accounts = len(active_accounts_series)
                    
                    avg_csat = filtered_df[filtered_df['CSAT__c'] != 0]['CSAT__c'].mean()
                    if pd.isna(avg_csat):
                        avg_csat = 0
                        
                    escalation_rate = filtered_df['IsEscalated'].mean() * 100
                    
                    pptx_data = generate_powerpoint(filtered_df, active_accounts, avg_csat, escalation_rate)
                    st.download_button(
                        label="Download PowerPoint",
                        data=pptx_data,
                        file_name=f"ticket_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    )
            except Exception as e:
                st.error(f"Error generating PowerPoint: {str(e)}")
    
    with col3:
        st.info("Export your analysis as Excel or PowerPoint presentation. The PowerPoint includes all charts and statistics.")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Detect file type and read accordingly
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type == 'csv':
            df = pd.read_csv(uploaded_file)
        else:  # Excel file
            df = pd.read_excel(uploaded_file)
        
        # Check for and rename Eightfold_Group_Id__r.Group_Id__c to Group_Id__c
        if 'Eightfold_Group_Id__r.Group_Id__c' in df.columns:
            df['Group_Id__c'] = df['Eightfold_Group_Id__r.Group_Id__c']
            df.drop('Eightfold_Group_Id__r.Group_Id__c', axis=1, inplace=True)
            st.write("Renamed 'Eightfold_Group_Id__r.Group_Id__c' to 'Group_Id__c'")
        
        # Process the data
        try:
            # Convert all string columns to string type first to avoid comparison issues
            string_columns = ['Product_Area__c', 'Product_Feature__c', 'POD_Name__c', 
                            'Group_Id__c', 'Account.Account_Name__c', 'Internal_Priority__c', 
                            'Case_Owner__c', 'Case_Type__c', 'RCA__c']
            for col in string_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str)

            # Clean the data
            # Filter out merged tickets
            if 'Status' in df.columns:
                df = df[df['Status'] != 'Merged']
                st.write(f"Filtered out merged tickets. Remaining tickets: {len(df)}")

            # Filter out Eightfold AI account
            if 'Account.Account_Name__c' in df.columns:
                df = df[df['Account.Account_Name__c'] != 'Eightfold AI']
                st.write(f"Filtered out Eightfold AI account. Remaining tickets: {len(df)}")

            # Filter out specific case owners
            excluded_owners = ['Spam', 'Support', 'Infosec queue', 'Deal Desk', 'Sales Ops', 'AI Help', 'ISR Queue', 'PD Queue', 'RFx Intake']
            if 'Case_Owner__c' in df.columns:
                df = df[~df['Case_Owner__c'].str.lower().isin([owner.lower() for owner in excluded_owners])]
                st.write(f"Filtered out excluded case owners. Remaining tickets: {len(df)}")

            # Fill NaN values
            for col in string_columns:
                if col in df.columns:
                    df[col] = df[col].fillna('Unspecified')

            # Convert date fields
            if 'CreatedDate' in df.columns:
                # First convert to datetime
                df['CreatedDate'] = pd.to_datetime(df['CreatedDate'], errors='coerce')
                # Remove rows with invalid dates
                df = df.dropna(subset=['CreatedDate'])
                
                # Get min and max dates for the date picker
                min_date = df['CreatedDate'].min().date()
                max_date = df['CreatedDate'].max().date()
                
                # Date range filter
                date_range = st.sidebar.date_input(
                    "Select Date Range",
                    [min_date, max_date],
                    min_value=min_date,
                    max_value=max_date
                )

            if 'ClosedDate' in df.columns:
                df['ClosedDate'] = pd.to_datetime(df['ClosedDate'], errors='coerce')

            # Convert numeric fields
            if 'CSAT__c' in df.columns:
                df['CSAT__c'] = pd.to_numeric(df['CSAT__c'], errors='coerce').fillna(0)

            if 'Age_days__c' in df.columns:
                df['Age_days__c'] = pd.to_numeric(df['Age_days__c'], errors='coerce').fillna(0)

            # Convert boolean fields
            if 'IsEscalated' in df.columns:
                # Convert various string representations to boolean
                df['IsEscalated'] = df['IsEscalated'].astype(str).str.lower()
                df['IsEscalated'] = df['IsEscalated'].map({'true': True, 'false': False}).fillna(False)

            # Clean priority values
            if 'Internal_Priority__c' in df.columns:
                # Convert to string and clean priority values
                df['Internal_Priority__c'] = df['Internal_Priority__c'].astype(str)
                # Remove any non-priority values (only keep P1, P2, P3, P4, etc.)
                priority_pattern = r'^P[0-9]+$'
                df.loc[~df['Internal_Priority__c'].str.match(priority_pattern), 'Internal_Priority__c'] = 'Unspecified'
                # Sort priorities correctly
                df['priority_sort'] = df['Internal_Priority__c'].apply(lambda x: int(x[1:]) if x.startswith('P') and x[1:].isdigit() else 999)

            # Sidebar filters
            st.sidebar.header("Filters")
            
            # Account filter
            account_options = sorted([str(acc) for acc in df['Account.Account_Name__c'].unique() if str(acc) != 'Unspecified'])
            selected_accounts = st.sidebar.multiselect(
                "Select Accounts",
                options=account_options,
                default=[],
                help="Select specific accounts or leave empty to show all accounts"
            )
            
            # Case Owner filter
            if 'Case_Type__c' in df.columns:
                # Get owners who have handled Support/Service requests
                tse_pse_owners = set(df[
                    df['Case_Type__c'].str.lower().isin(['support request', 'service request'])
                ]['Case_Owner__c'].unique())
                
                # Get all other owners (excluding excluded owners and Unspecified)
                all_owners = set(df['Case_Owner__c'].unique())
                non_tse_pse_owners = all_owners - tse_pse_owners - set(['Unspecified']) - set(excluded_owners)
                
                # Create owner groups
                owner_groups = {
                    'TSE and PSE': list(tse_pse_owners),
                    'Non TSE and PSE': list(non_tse_pse_owners)
                }
                
                # Create the multiselect with groups
                selected_owner_groups = st.sidebar.multiselect(
                    "Select Support Engineers",
                    options=list(owner_groups.keys()),
                    default=[],
                    help="Select engineer groups or leave empty to show all"
                )
                
                # Convert selected groups to actual owners
                selected_owners = []
                for group in selected_owner_groups:
                    selected_owners.extend(owner_groups[group])
            else:
                selected_owners = []
            
            # Priority filter
            priority_options = sorted(
                [str(p) for p in df['Internal_Priority__c'].unique() if str(p) != 'Unspecified'],
                key=lambda x: int(x[1:]) if x.startswith('P') and x[1:].isdigit() else 999
            )
            selected_priorities = st.sidebar.multiselect(
                "Select Priorities",
                options=priority_options,
                default=priority_options
            )
            
            # Product Area filter
            area_options = sorted([str(area) for area in df['Product_Area__c'].unique() if str(area) != 'Unspecified'])
            selected_areas = st.sidebar.multiselect(
                "Select Product Areas",
                options=area_options,
                default=area_options
            )

            # RCA filter
            if 'RCA__c' in df.columns:
                rca_options = sorted([str(rca) for rca in df['RCA__c'].unique() if str(rca) != 'Unspecified'])
                selected_rcas = st.sidebar.multiselect(
                    "Select Root Causes",
                    options=rca_options,
                    default=[],
                    help="Select specific root causes or leave empty to show all"
                )
            else:
                selected_rcas = []

            # Create the mask for filtering
            if 'CreatedDate' in df.columns:
                # Convert selected dates to datetime for comparison
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1]).replace(hour=23, minute=59, second=59)
                
                # Ensure timezone-naive comparison
                if df['CreatedDate'].dt.tz is not None:
                    df['CreatedDate'] = df['CreatedDate'].dt.tz_localize(None)
                
                # Add date filter to mask
                mask = (
                    (df['Product_Area__c'].astype(str).isin([str(area) for area in selected_areas])) &
                    (df['Internal_Priority__c'].astype(str).isin([str(p) for p in selected_priorities])) &
                    (df['CreatedDate'] >= start_date) &
                    (df['CreatedDate'] <= end_date)
                )
            else:
                # If no CreatedDate column, use other filters only
                mask = (
                    (df['Product_Area__c'].astype(str).isin([str(area) for area in selected_areas])) &
                    (df['Internal_Priority__c'].astype(str).isin([str(p) for p in selected_priorities]))
                )
            
            # Apply additional filters
            if len(selected_accounts) > 0:
                mask = mask & (df['Account.Account_Name__c'].astype(str).isin([str(acc) for acc in selected_accounts]))
            
            if len(selected_owners) > 0:
                mask = mask & (df['Case_Owner__c'].astype(str).isin([str(owner) for owner in selected_owners]))

            if 'RCA__c' in df.columns and len(selected_rcas) > 0:
                mask = mask & (df['RCA__c'].astype(str).isin([str(rca) for rca in selected_rcas]))

            filtered_df = df[mask].copy()

            if len(filtered_df) == 0:
                st.warning("No data available for the selected filters. Please adjust your selection.")
            else:
                # Display visualizations
                display_visualizations(filtered_df)

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.error("Please check if all required columns are present and contain valid data.")
                    
    except Exception as e:
        st.error(f"Error reading the file: {str(e)}")
        st.error("Please ensure the file is in the correct format (CSV or Excel) and contains valid data.")
else:
    st.info("Please upload a CSV or Excel file to begin the analysis.")
    
    # Sample data format
    st.header("Expected File Format")
    st.markdown("""
    Your file (CSV or Excel) should contain the following columns:
    - `Id`: Case ID
    - `CaseNumber`: Case number
    - `Account.Account_Name__c`: Account name
    - `Group_Id__c`: Customer Group ID
    - `Subject`: Case subject
    - `Description`: Case description
    - `Product_Area__c`: Product area
    - `Product_Feature__c`: Product feature
    - `POD_Name__c`: POD name
    - `CreatedDate`: Date when the ticket was created
    - `ClosedDate`: Date when the ticket was closed
    - `Case_Type__c`: Type of case
    - `Age_days__c`: Age of the case in days
    - `IsEscalated`: Whether the case is escalated
    - `CSAT__c`: Customer satisfaction score
    - `Internal_Priority__c`: Internal priority level
    
    Supported file formats:
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    """) 
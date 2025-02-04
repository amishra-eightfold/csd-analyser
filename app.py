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

def truncate_account_name(name, max_length=15):
    """Helper function to truncate account names."""
    if isinstance(name, str) and len(name) > max_length:
        return name[:max_length] + '...'
    return name

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
        priority_dist = filtered_df.groupby('Internal_Priority__c').size().reset_index(name='count')
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
        
        escalation_by_priority = filtered_df.groupby('Internal_Priority__c')['IsEscalated'].mean().mul(100).reset_index(name='escalation_rate')
        fig_escalation = px.bar(
            escalation_by_priority,
            x='Internal_Priority__c',
            y='escalation_rate',
            title='Escalation Rate by Priority (%)',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_escalation.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
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

        # RCA and Age Analysis slides
        if 'RCA__c' in filtered_df.columns:
            # Average Age by RCA slide
            slide = prs.slides.add_slide(prs.slide_layouts[5])
            title = slide.shapes.title
            title.text = "Root Cause Analysis - Age Correlation"
            
            age_by_rca = filtered_df.groupby('RCA__c')['Age_days__c'].agg(['mean', 'count']).reset_index()
            age_by_rca.columns = ['RCA__c', 'Average Age (Days)', 'Number of Cases']
            age_by_rca = age_by_rca[age_by_rca['Number of Cases'] >= 5]  # Filter RCAs with at least 5 cases
            age_by_rca = age_by_rca.sort_values('Average Age (Days)', ascending=True)
            
            fig_rca_age = px.bar(
                age_by_rca,
                x='Average Age (Days)',
                y='RCA__c',
                title='Average Case Age by Root Cause',
                orientation='h',
                color='Number of Cases',
                color_continuous_scale='Viridis',
                text='Number of Cases'
            )
            fig_rca_age.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                yaxis_title='Root Cause',
                showlegend=True,
                height=600
            )
            fig_rca_age.update_traces(textposition='outside')
            img_bytes = fig_rca_age.to_image(format="png", width=1000, height=600, scale=2)
            img_stream = BytesIO(img_bytes)
            slide.shapes.add_picture(img_stream, Inches(1), Inches(1.5), width=Inches(11))

            # Age Distribution by RCA slide
            slide = prs.slides.add_slide(prs.slide_layouts[5])
            title = slide.shapes.title
            title.text = "Age Distribution by Root Cause"
            
            fig_rca_box = px.box(
                filtered_df,
                x='RCA__c',
                y='Age_days__c',
                title='Age Distribution by Root Cause',
                color='RCA__c',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_rca_box.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis_title='Root Cause',
                yaxis_title='Age (Days)',
                showlegend=False,
                xaxis_tickangle=-45,
                height=600
            )
            img_bytes = fig_rca_box.to_image(format="png", width=1000, height=600, scale=2)
            img_stream = BytesIO(img_bytes)
            slide.shapes.add_picture(img_stream, Inches(1), Inches(1.5), width=Inches(11))
        
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

    # Product Feature Analysis
    st.header("Product Feature Analysis")
    
    # Add a button to show/hide the analysis
    if st.button("Analyze Missing Product Features"):
        with st.spinner("Analyzing Product Features..."):
            df_with_predictions = display_feature_predictions(filtered_df)
            
            # Update the filtered_df with predictions if user accepts
            if st.button("Apply Predictions to Analysis"):
                filtered_df = df_with_predictions.copy()
                filtered_df['Product_Feature__c'] = filtered_df['Predicted_Feature']
                st.success("Predictions applied! The analysis below will use the predicted features.")

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
        priority_dist = filtered_df.groupby('Internal_Priority__c').size().reset_index(name='count')
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
        escalation_by_priority = filtered_df.groupby('Internal_Priority__c')['IsEscalated'].mean().mul(100).reset_index(name='escalation_rate')
        fig_escalation = px.bar(
            escalation_by_priority,
            x='Internal_Priority__c',
            y='escalation_rate',
            title='Escalation Rate by Priority (%)',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_escalation.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
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

def clean_text(text):
    """Clean and preprocess text data."""
    if pd.isna(text):
        return ""
    # Convert to string
    text = str(text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def predict_product_feature(row, known_features_by_area, vectorizer_dict, feature_vectors_dict):
    """Predict Product Feature based on text similarity and context."""
    if pd.isna(row['Product_Feature__c']) or row['Product_Feature__c'] in ['Other', 'Unspecified']:
        product_area = row['Product_Area__c']
        
        if product_area not in known_features_by_area:
            return 'Unspecified'
            
        # Combine relevant text fields
        text_to_analyze = ' '.join([
            clean_text(row['Subject']),
            clean_text(row['Description']),
            clean_text(row['RCA__c']),
            clean_text(product_area)
        ])
        
        # Get the vectorizer and feature vectors for this product area
        vectorizer = vectorizer_dict[product_area]
        feature_vectors = feature_vectors_dict[product_area]
        known_features = known_features_by_area[product_area]
        
        # Transform the text
        text_vector = vectorizer.transform([text_to_analyze])
        
        # Calculate similarity with known features
        similarities = cosine_similarity(text_vector, feature_vectors)[0]
        
        # Get the most similar feature
        if len(similarities) > 0:
            max_sim_index = np.argmax(similarities)
            if similarities[max_sim_index] > 0.1:  # Threshold for minimum similarity
                return known_features[max_sim_index]
    
    return row['Product_Feature__c']

def analyze_product_features(filtered_df):
    """Analyze and predict missing Product Features."""
    # Group valid features by Product Area
    known_features_by_area = {}
    vectorizer_dict = {}
    feature_vectors_dict = {}
    
    # Create training data for each Product Area
    for area in filtered_df['Product_Area__c'].unique():
        area_df = filtered_df[filtered_df['Product_Area__c'] == area]
        valid_features = area_df[
            ~area_df['Product_Feature__c'].isin(['Other', 'Unspecified']) & 
            ~area_df['Product_Feature__c'].isna()
        ]
        
        if len(valid_features) > 0:
            known_features = valid_features['Product_Feature__c'].unique()
            known_features_by_area[area] = known_features
            
            # Prepare training data
            training_texts = []
            for feature in known_features:
                feature_data = valid_features[valid_features['Product_Feature__c'] == feature]
                text = ' '.join([
                    ' '.join(feature_data['Subject'].apply(clean_text)),
                    ' '.join(feature_data['Description'].apply(clean_text)),
                    ' '.join(feature_data['RCA__c'].apply(clean_text)),
                    clean_text(area)
                ])
                training_texts.append(text)
            
            # Create and fit vectorizer
            vectorizer = TfidfVectorizer(stop_words='english')
            feature_vectors = vectorizer.fit_transform(training_texts)
            
            vectorizer_dict[area] = vectorizer
            feature_vectors_dict[area] = feature_vectors
    
    # Predict missing features
    df_with_predictions = filtered_df.copy()
    df_with_predictions['Predicted_Feature'] = df_with_predictions.apply(
        lambda row: predict_product_feature(row, known_features_by_area, vectorizer_dict, feature_vectors_dict),
        axis=1
    )
    
    # Analyze changes
    changes = df_with_predictions[
        (df_with_predictions['Product_Feature__c'].isin(['Other', 'Unspecified']) | 
         df_with_predictions['Product_Feature__c'].isna()) &
        (df_with_predictions['Predicted_Feature'] != df_with_predictions['Product_Feature__c'])
    ]
    
    return df_with_predictions, changes

def display_feature_predictions(filtered_df):
    """Display analysis of Product Feature predictions."""
    st.header("Product Feature Analysis")
    
    # Get predictions
    df_with_predictions, changes = analyze_product_features(filtered_df)
    
    # Display summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_missing = len(filtered_df[
            filtered_df['Product_Feature__c'].isin(['Other', 'Unspecified']) | 
            filtered_df['Product_Feature__c'].isna()
        ])
        st.metric("Missing/Other Features", total_missing)
                
    with col2:
        predictions_made = len(changes)
        st.metric("Predictions Made", predictions_made)
    
    with col3:
        prediction_rate = (predictions_made / total_missing * 100) if total_missing > 0 else 0
        st.metric("Prediction Rate", f"{prediction_rate:.1f}%")
    
    # Display predictions by Product Area
    st.subheader("Predictions by Product Area")
    predictions_by_area = changes.groupby(['Product_Area__c']).agg({
        'Product_Feature__c': 'count',
        'Predicted_Feature': lambda x: ', '.join(x.unique())
    }).reset_index()
    predictions_by_area.columns = ['Product Area', 'Number of Predictions', 'Predicted Features']
    st.dataframe(predictions_by_area, hide_index=True)
    
    # Display sample predictions
    st.subheader("Sample Predictions")
    sample_predictions = changes[[
        'Product_Area__c', 'Product_Feature__c', 'Predicted_Feature', 
        'Subject', 'Description', 'RCA__c'
    ]].head(10)
    sample_predictions.columns = [
        'Product Area', 'Original Feature', 'Predicted Feature',
        'Subject', 'Description', 'RCA'
    ]
    st.dataframe(sample_predictions, hide_index=True)
    
    return df_with_predictions

def predict_group_id(df):
    """Predict Group_Id__c based on Account.Account_Name__c using text similarity."""
    # Create a copy of the dataframe
    df_with_predictions = df.copy()
    
    # Clean Group IDs - convert nan, 'nan', 'None', empty strings to None
    df_with_predictions['Group_Id__c'] = df_with_predictions['Group_Id__c'].replace(['nan', 'None', '', 'Unspecified'], None)
    df_with_predictions['Group_Id__c'] = pd.to_numeric(df_with_predictions['Group_Id__c'], errors='coerce')
    
    # Get training data (rows with valid Group IDs)
    valid_data = df_with_predictions[df_with_predictions['Group_Id__c'].notna()].copy()
    
    # Create mapping of Account Name to most common Group ID
    account_group_mapping = {}
    for account in valid_data['Account.Account_Name__c'].unique():
        account_data = valid_data[valid_data['Account.Account_Name__c'] == account]
        group_counts = account_data['Group_Id__c'].value_counts()
        if not group_counts.empty:
            account_group_mapping[account] = int(group_counts.index[0])
    
    # Create TF-IDF vectorizer for account names
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        lowercase=True,
        max_features=1000
    )
    
    # Fit vectorizer on account names
    account_names = list(account_group_mapping.keys())
    if not account_names:
        return df_with_predictions, 0, 0, 0, pd.DataFrame()
        
    vectorizer.fit([str(name) for name in account_names])
    
    # Create vectors for known accounts
    account_vectors = {
        account: vectorizer.transform([str(account)]).toarray()
        for account in account_names
    }
    
    # Function to find most similar account
    def find_most_similar_account(account_name):
        if account_name in account_group_mapping:
            return account_name, 1.0  # Exact match
        
        account_vector = vectorizer.transform([str(account_name)]).toarray()
        max_similarity = -1
        most_similar_account = None
        
        for known_account, known_vector in account_vectors.items():
            similarity = cosine_similarity(account_vector, known_vector)[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_account = known_account
        
        return most_similar_account, max_similarity
    
    # Identify missing Group IDs
    missing_group_mask = df_with_predictions['Group_Id__c'].isna()
    total_missing = missing_group_mask.sum()
    
    if total_missing == 0:
        return df_with_predictions, 0, 0, 0, pd.DataFrame()
    
    # Store prediction details
    predictions = []
    
    # Apply predictions
    for idx, row in df_with_predictions[missing_group_mask].iterrows():
        account_name = str(row['Account.Account_Name__c'])
        similar_account, confidence = find_most_similar_account(account_name)
        
        if confidence >= 0.8:  # High confidence threshold
            predicted_group_id = account_group_mapping[similar_account]
        else:
            # For low confidence, try to extract numbers from account name
            numbers = ''.join(filter(str.isdigit, account_name))
            predicted_group_id = int(numbers) if numbers else None
            if predicted_group_id is None:
                # If no numbers found, use a hash of the account name
                predicted_group_id = abs(hash(account_name)) % 10000000
        
        df_with_predictions.loc[idx, 'Group_Id__c'] = predicted_group_id
        predictions.append({
            'Account_Name': account_name,
            'Similar_Account': similar_account,
            'Confidence': confidence,
            'Predicted_Group_ID': predicted_group_id
        })
    
    # Create prediction details DataFrame
    prediction_details_df = pd.DataFrame(predictions) if predictions else pd.DataFrame()
    
    # Get prediction statistics
    high_confidence_predictions = len(prediction_details_df[prediction_details_df['Confidence'] >= 0.8]) if not prediction_details_df.empty else 0
    low_confidence_predictions = len(prediction_details_df[prediction_details_df['Confidence'] < 0.8]) if not prediction_details_df.empty else 0
    
    return (
        df_with_predictions,
        total_missing,
        high_confidence_predictions,
        low_confidence_predictions,
        prediction_details_df
    )

def display_group_id_predictions(df):
    """Display analysis of Group ID predictions."""
    st.subheader("Group ID Prediction Analysis")
    
    # Get predictions
    (
        df_with_predictions,
        total_missing,
        high_confidence_predictions,
        low_confidence_predictions,
        prediction_details
    ) = predict_group_id(df)
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Missing Group IDs", total_missing)
    
    with col2:
        st.metric("High Confidence Predictions", high_confidence_predictions)
    
    with col3:
        st.metric("Low Confidence Predictions", low_confidence_predictions)
    
    with col4:
        prediction_rate = (high_confidence_predictions / total_missing * 100) if total_missing > 0 else 0
        st.metric("High Confidence Rate", f"{prediction_rate:.1f}%")
    
    # Display prediction details
    if not prediction_details.empty:
        st.subheader("Prediction Details")
        
        # High confidence predictions
        high_confidence_mask = prediction_details['Confidence'] >= 0.8
        if high_confidence_mask.any():
            st.write("High Confidence Predictions (≥ 80% similarity)")
            high_confidence_df = prediction_details[high_confidence_mask].copy()
            high_confidence_df['Confidence'] = high_confidence_df['Confidence'].apply(lambda x: f"{x*100:.1f}%")
            high_confidence_df.columns = ['Account Name', 'Similar Account', 'Confidence', 'Predicted Group ID']
            st.dataframe(high_confidence_df, hide_index=True)
        
        # Low confidence predictions
        low_confidence_mask = prediction_details['Confidence'] < 0.8
        if low_confidence_mask.any():
            st.write("Low Confidence Predictions (< 80% similarity)")
            low_confidence_df = prediction_details[low_confidence_mask].copy()
            low_confidence_df['Confidence'] = low_confidence_df['Confidence'].apply(lambda x: f"{x*100:.1f}%")
            low_confidence_df.columns = ['Account Name', 'Similar Account', 'Confidence', 'Predicted Group ID']
            st.dataframe(low_confidence_df, hide_index=True)
    
    return df_with_predictions

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

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Detect file type and read accordingly
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type == 'csv':
            df = pd.read_csv(uploaded_file)
        else:  # Excel file
            df = pd.read_excel(uploaded_file)
        
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

            # Predict missing Group IDs
            if 'Group_Id__c' in df.columns and 'Account.Account_Name__c' in df.columns:
                st.header("Group ID Analysis")
                if st.button("Analyze Missing Group IDs"):
                    with st.spinner("Analyzing Group IDs..."):
                        df = display_group_id_predictions(df)
                        st.success("Group ID analysis complete! The predictions have been applied to the data.")

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
            priority_options = sorted([str(p) for p in df['Internal_Priority__c'].unique() if str(p) != 'Unspecified'])
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
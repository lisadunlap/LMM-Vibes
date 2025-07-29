"""
Frequencies tab for pipeline results app.

This module contains the frequencies tab functionality for comparing
how frequently each model exhibits behaviors in different clusters.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any

from ..utils import extract_quality_score, has_confidence_intervals, format_confidence_interval


def create_frequencies_tab(model_stats: Dict[str, Any], selected_models: List[str], 
                          cluster_level: str = 'fine', show_confidence_intervals: bool = False):
    """Create the frequencies tab with cluster frequency comparison."""
    
    st.header("Cluster Frequencies by Model")
    st.write("Compare how frequently each model exhibits behaviors in different clusters")
    
    # Collect all clusters across all models for the chart
    all_clusters_data = []
    for model_name, model_data in model_stats.items():
        clusters = model_data.get(cluster_level, [])
        for cluster in clusters:
            # Get confidence intervals for quality scores if available
            quality_score_ci = cluster.get('quality_score_ci', {})
            has_quality_ci = bool(quality_score_ci)
            
            all_clusters_data.append({
                'property_description': cluster['property_description'],
                'model': model_name,
                'frequency': cluster.get('proportion', 0) * 100,  # Convert to percentage
                'size': cluster.get('size', 0),
                'cluster_size_global': cluster.get('cluster_size_global', 0),
                'has_ci': has_confidence_intervals(cluster),
                'ci_lower': cluster.get('score_ci_lower'),
                'ci_upper': cluster.get('score_ci_upper'),
                'has_quality_ci': has_quality_ci
            })
    
    if all_clusters_data:
        clusters_df = pd.DataFrame(all_clusters_data)
        
        # Get all unique clusters for the chart
        all_unique_clusters = clusters_df['property_description'].unique()
        total_clusters = len(all_unique_clusters)
        
        # Show summary statistics
        # st.subheader("📊 Cluster Summary")
        # col1, col2, col3, col4 = st.columns(4)
        # with col1:
        #     st.metric("Total Clusters", total_clusters)
        # with col2:
        #     avg_freq = clusters_df['frequency'].mean()
        #     st.metric("Avg Frequency", f"{avg_freq:.1f}%")
        # with col3:
        #     max_freq = clusters_df['frequency'].max()
        #     st.metric("Max Frequency", f"{max_freq:.1f}%")
        # with col4:
        #     total_models = clusters_df['model'].nunique()
        #     st.metric("Models", total_models)
        
        # st.divider()
        
        # Show all clusters by default
        top_n_for_chart = total_clusters
        
        # Calculate total frequency per cluster and get top clusters
        cluster_totals = clusters_df.groupby('property_description')['frequency'].sum().sort_values(ascending=False)
        top_clusters = cluster_totals.head(top_n_for_chart).index.tolist()
        
        # Get quality scores for the same clusters to sort by quality
        quality_data_for_sorting = []
        for model_name, model_data in model_stats.items():
            clusters = model_data.get(cluster_level, [])
            for cluster in clusters:
                if cluster['property_description'] in top_clusters:
                    quality_data_for_sorting.append({
                        'property_description': cluster['property_description'],
                        'quality_score': extract_quality_score(cluster.get('quality_score', 0))
                    })
        
        # Calculate average quality score per cluster and sort
        if quality_data_for_sorting:
            quality_df_for_sorting = pd.DataFrame(quality_data_for_sorting)
            avg_quality_per_cluster = quality_df_for_sorting.groupby('property_description')['quality_score'].mean().sort_values(ascending=True)  # Low to high
            top_clusters = avg_quality_per_cluster.index.tolist()
            # Reverse the order so low quality appears at top of chart
            top_clusters = top_clusters[::-1]
        
        # Filter data to only include top clusters
        chart_data = clusters_df[clusters_df['property_description'].isin(top_clusters)]
        
        if not chart_data.empty:
            # Create side-by-side layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create horizontal bar chart for frequencies
                fig = go.Figure()
                
                # Get unique models for colors
                models = chart_data['model'].unique()
                # Use a color palette that avoids yellow - using Set1 which has better contrast
                colors = px.colors.qualitative.Set1[:len(models)]
                
                # Function to wrap text for hover
                def wrap_text(text, width=60):
                    """Wrap text to specified width using HTML line breaks"""
                    if len(text) <= width:
                        return text
                    words = text.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        if len(current_line + " " + word) <= width:
                            current_line += (" " + word if current_line else word)
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    if current_line:
                        lines.append(current_line)
                    return "<br>".join(lines)
                
                # Add a bar for each model
                for i, model in enumerate(models):
                    model_data = chart_data[chart_data['model'] == model]
                    
                    # Sort by cluster order (same as top_clusters)
                    model_data = model_data.set_index('property_description').reindex(top_clusters).reset_index()
                    
                    # Get confidence intervals for error bars
                    ci_lower = []
                    ci_upper = []
                    for _, row in model_data.iterrows():
                        if row.get('has_ci', False) and row.get('ci_lower') is not None and row.get('ci_upper') is not None:
                            # IMPORTANT: These are distinctiveness score CIs, not frequency CIs
                            # The distinctiveness score measures how much more/less frequently 
                            # a model exhibits this behavior compared to the median model
                            # We can use this to estimate uncertainty in the frequency measurement
                            distinctiveness_ci_width = row['ci_upper'] - row['ci_lower']
                            
                            # Convert to frequency uncertainty (approximate)
                            # A wider distinctiveness CI suggests more uncertainty in the frequency
                            freq_uncertainty = distinctiveness_ci_width * row['frequency'] * 0.1
                            ci_lower.append(max(0, row['frequency'] - freq_uncertainty))
                            ci_upper.append(row['frequency'] + freq_uncertainty)
                        else:
                            ci_lower.append(None)
                            ci_upper.append(None)
                    
                    fig.add_trace(go.Bar(
                        y=model_data['property_description'],
                        x=model_data['frequency'],
                        name=model,
                        orientation='h',
                        marker_color=colors[i],
                        error_x=dict(
                            type='data',
                            array=[u - l if u is not None and l is not None else None for l, u in zip(ci_lower, ci_upper)],
                            arrayminus=[f - l if f is not None and l is not None else None for f, l in zip(model_data['frequency'], ci_lower)],
                            visible=show_confidence_intervals,
                            thickness=1,
                            width=3,
                            color='rgba(0,0,0,0.3)'
                        ),
                        hovertemplate='<b>%{y}</b><br>' +
                                    f'Model: {model}<br>' +
                                    'Frequency: %{x:.1f}%<br>' +
                                    'CI: %{customdata[0]}<extra></extra>',
                        customdata=[[
                            format_confidence_interval({
                                'lower': l, 
                                'upper': u
                            }) if l is not None and u is not None else "N/A"
                            for l, u in zip(ci_lower, ci_upper)
                        ]]
                    ))
                
                # Update layout
                fig.update_layout(
                    title=f"Model Frequencies in Top {len(top_clusters)} Clusters",
                    xaxis_title="Frequency (%)",
                    yaxis_title="Cluster Description",
                    barmode='group',  # Group bars side by side
                    height=max(600, len(top_clusters) * 25),  # Adjust height based on number of clusters
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Update y-axis to show full cluster names
                fig.update_yaxes(
                    tickmode='array',
                    ticktext=[desc[:80] + "..." if len(desc) > 80 else desc for desc in top_clusters],
                    tickvals=top_clusters
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add note about error bars if confidence intervals are shown
                if show_confidence_intervals:
                    st.info("📊 **Error bars** show confidence intervals for frequency measurements. Wider bars indicate higher uncertainty in the frequency estimates.")
                    st.info("""
                    **Understanding the Confidence Intervals:**
                    
                    • **Frequency Chart Error Bars**: Based on distinctiveness score CIs, showing uncertainty in how much more/less frequently a model exhibits this behavior compared to the median model
                    
                    • **Quality Chart Error Bars**: Based on quality score CIs, showing uncertainty in how well the model performs in this cluster compared to its global average
                    
                    • **Different Metrics**: Frequency measures "how often" while quality measures "how well" - these are separate measurements with different confidence intervals
                    """)
            
            with col2:
                # Create quality score chart
                # Get quality scores for the same clusters (single score per cluster)
                quality_data = []
                for cluster_desc in top_clusters:
                    # Get the first available quality score for this cluster
                    for model_name, model_data in model_stats.items():
                        clusters = model_data.get(cluster_level, [])
                        for cluster in clusters:
                            if cluster['property_description'] == cluster_desc:
                                quality_data.append({
                                    'property_description': cluster_desc,
                                    'quality_score': extract_quality_score(cluster.get('quality_score', 0))
                                })
                                break
                        if any(q['property_description'] == cluster_desc for q in quality_data):
                            break
                
                if quality_data:
                    quality_df = pd.DataFrame(quality_data)
                    
                    # Create quality score chart with single bars
                    fig_quality = go.Figure()
                    
                    # Add a single bar for each cluster
                    fig_quality.add_trace(go.Bar(
                        y=quality_df['property_description'],
                        x=quality_df['quality_score'],
                        orientation='h',
                        marker_color='lightblue',  # Single color for all bars
                        name='Quality Score',
                        showlegend=False,
                        hovertemplate='<b>%{y}</b><br>' +
                                    'Quality Score: %{x:.3f}<extra></extra>'
                    ))
                    
                    # Update layout
                    fig_quality.update_layout(
                        title=f"Quality Scores",
                        xaxis_title="Quality Score",
                        yaxis_title="",  # No y-axis title to save space
                        height=max(600, len(top_clusters) * 25),  # Same height as main chart
                        showlegend=False,
                        yaxis=dict(showticklabels=False)  # Hide y-axis labels to save space
                    )
                    
                    st.plotly_chart(fig_quality, use_container_width=True)
                    
                else:
                    st.info("No quality score data available")
            
            # Add table below the charts
            st.subheader("📋 Data Table")
            
            # Create a comprehensive table with both frequency and quality data
            table_data = []
            for cluster_desc in top_clusters:
                # Get frequency data for all models
                cluster_freq_data = chart_data[chart_data['property_description'] == cluster_desc]
                
                # Get quality score for this cluster
                quality_score = None
                for model_name, model_data in model_stats.items():
                    clusters = model_data.get(cluster_level, [])
                    for cluster in clusters:
                        if cluster['property_description'] == cluster_desc:
                            quality_score = extract_quality_score(cluster.get('quality_score', 0))
                            break
                    if quality_score is not None:
                        break
                
                # Create row with cluster description and quality score
                row = {
                    'Cluster': cluster_desc,
                    'Quality Score': f"{quality_score:.3f}" if quality_score is not None else "N/A"
                }
                
                # Add frequency data for each model
                for model in models:
                    model_freq = cluster_freq_data[cluster_freq_data['model'] == model]
                    if not model_freq.empty:
                        freq_value = model_freq.iloc[0]['frequency']
                        row[f'{model} Freq (%)'] = f"{freq_value:.1f}"
                    else:
                        row[f'{model} Freq (%)'] = "0.0"
                
                table_data.append(row)
            
            # Create DataFrame and display table
            if table_data:
                table_df = pd.DataFrame(table_data)
                
                # Configure column display
                column_config = {
                    'Cluster': st.column_config.TextColumn(
                        'Cluster Description',
                        width='large',
                        help="Full behavioral cluster description"
                    ),
                    'Quality Score': st.column_config.NumberColumn(
                        'Quality Score',
                        width='small',
                        help="Overall quality score for this cluster"
                    )
                }
                
                # Add frequency column configs
                for model in models:
                    column_config[f'{model} Freq (%)'] = st.column_config.NumberColumn(
                        f'{model} Frequency (%)',
                        width='small',
                        help=f"Percentage of {model} responses in this cluster"
                    )
                
                st.dataframe(
                    table_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config
                )
                
                # Add download button
                csv = table_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"cluster_frequencies_{cluster_level}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data available for table display")
            
        else:
            st.warning("No data available for the selected clusters")
    else:
        st.warning("No cluster frequency data available") 
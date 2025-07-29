"""
Overview tab for pipeline results app.

This module contains the overview tab functionality including model summaries,
leaderboards, and performance metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from pathlib import Path

from ..utils import get_top_clusters_for_model, extract_quality_score, has_confidence_intervals, format_confidence_interval


def create_overview_tab(model_stats: Dict[str, Any], model_rankings: List[tuple], 
                       cluster_level: str = 'fine', top_n_clusters: int = 10):
    """Create the overview tab with model summaries and leaderboard."""
    
    st.header("Model Summaries")
    st.caption("Top distinctive clusters where each model shows unique behavioral patterns")
    
    # Add explanation accordion (collapsed by default)
    with st.expander("ℹ️ What do these numbers mean?", expanded=False):
        st.info(
            "**Frequency calculation:** For each cluster, frequency shows what percentage of a model's total battles resulted in that behavioral pattern. "
            "For example, 5% frequency means the model exhibited this behavior in 5 out of every 100 battles it participated in.\n\n"
            "**Distinctiveness:** The distinctiveness score shows how much more (or less) frequently a model exhibits a behavior compared to the median frequency across all models. "
            "For example, '2.5x more distinctive' means this model exhibits this behavior 2.5 times more often than the typical model."
        )
    
    # Create model cards in a list layout (one model per row)
    num_models = len(model_rankings)
    cols_per_row = 1
    
    for i in range(0, num_models, cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            model_idx = i + j
            if model_idx >= num_models:
                break
                
            model_name, individual_model_stats = model_rankings[model_idx]
            
            with col:
                # Create card container
                with st.container():
                    # Model header with clean, elegant design
                    st.markdown(f"""
                    <div style="
                        padding: 8px 0 2px 0;
                        margin-bottom: 12px;
                        border-bottom: 2px solid #e2e8f0;
                    ">
                        <h3 style="
                            margin: 0; 
                            color: #1a202c; 
                            font-weight: 500; 
                            font-size: 1.25rem;
                            letter-spacing: -0.025em;
                        ">{model_name}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Get total battles for this model (approximate from cluster data)
                    top_clusters = get_top_clusters_for_model(model_stats, model_name, cluster_level, 5)
                    
                    if top_clusters:
                        # Calculate total battles by summing all cluster sizes for this model
                        total_battles = sum(cluster.get('size', 0) for cluster in top_clusters)
                        st.caption(f"{total_battles:,} battles &nbsp; &nbsp; &nbsp; &nbsp; Top clusters by frequency &nbsp;  <span style='font-size:1.2em;'>⬇️</span>", unsafe_allow_html=True)
                        
                        # Show top 3 clusters
                        for idx, cluster in enumerate(top_clusters[:3]):
                            cluster_desc = cluster['property_description']
                            frequency = cluster.get('proportion', 0) * 100  # Convert to percentage
                            cluster_size = cluster.get('size', 0)  # This model's size in this cluster
                            cluster_size_global = cluster.get('cluster_size_global', 0)  # Total across all models
                            quality_score = extract_quality_score(cluster.get('quality_score', 0))  # Quality score
                            
                            # Additional safety check to ensure quality_score is a float
                            if not isinstance(quality_score, (int, float)):
                                quality_score = 0.0
                            
                            # Calculate distinctiveness (using score as proxy)
                            distinctiveness = cluster.get('score', 1.0)
                            
                            # Get confidence intervals
                            score_ci = cluster.get('score_ci')
                            has_ci = has_confidence_intervals(cluster)
                            
                            # Format confidence interval for display
                            ci_display = ""
                            if has_ci:
                                ci_display = f"<br><span style='font-size: 12px; color: #666;'>CI: {format_confidence_interval(score_ci)}</span>"
                            
                            st.markdown(f"""
                            <div style="margin: 8px 0; padding: 10px; border-left: 3px solid #3182ce; background-color: #f8f9fa; position: relative;">
                                <div style="font-weight: 600; font-size: 16px; margin-bottom: 5px;">
                                    {cluster_desc}
                                </div>
                                <div style="font-size: 14px; color: #666;">
                                    <strong>{frequency:.1f}% frequency</strong> ({cluster_size} out of {cluster_size_global} total across all models)
                                </div>
                                <div style="font-size: 13px; color: #3182ce;">
                                    {distinctiveness:.1f}x more distinctive than other models{ci_display}
                                </div>
                                <div style="position: absolute; bottom: 8px; right: 10px; font-size: 14px; font-weight: 600; color: {'#28a745' if quality_score >= 0 else '#dc3545'};">
                                    Quality: {quality_score:.3f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.write("No cluster data available")
                    
                    st.markdown("</div>", unsafe_allow_html=True) 
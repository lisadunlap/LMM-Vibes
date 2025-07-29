"""
Examples tab for pipeline results app.

This module contains the examples tab functionality for viewing detailed
examples from specific models and behavioral clusters.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path

from ..utils import get_top_clusters_for_model, extract_quality_score, has_confidence_intervals, format_confidence_interval
from ..conversation_display import display_openai_conversation, convert_to_openai_format


def create_examples_tab(model_stats: Dict[str, Any], all_models: List[str], 
                       cluster_level: str = 'fine', top_n_clusters: int = 10,
                       results_path: Path = None, show_examples: bool = True):
    """Create the examples tab with detailed example viewing."""
    
    st.header("View Examples")
    st.write("Explore detailed examples from specific models and behavioral clusters")
    
    # Model selection for examples
    selected_model = st.selectbox("Select model for detailed view", all_models)
    
    if selected_model:
        model_data = model_stats.get(selected_model, {})
        clusters = model_data.get(cluster_level, [])
        
        if clusters:
            st.write(f"**Top clusters for {selected_model}:**")
            
            cluster_df_data = []
            for cluster in clusters[:top_n_clusters]:
                # Get confidence intervals
                score_ci = cluster.get('score_ci')
                has_ci = has_confidence_intervals(cluster)
                
                # Format confidence interval for display
                ci_display = format_confidence_interval(score_ci) if has_ci else "N/A"
                
                cluster_data = {
                    'Cluster': cluster['property_description'],  # Show full text, no truncation
                    'Score': f"{cluster['score']:.3f}",
                    'Size': cluster['size'],
                    'Proportion': f"{cluster['proportion']:.3f}",
                    'Quality': f"{extract_quality_score(cluster.get('quality_score')):.3f}"
                }
                
                # Add CI column only if user wants to see it and CIs are available
                if has_ci:
                    cluster_data['CI'] = ci_display
                
                cluster_df_data.append(cluster_data)
            
            cluster_df = pd.DataFrame(cluster_df_data)
            
            # Configure column display
            column_config = {
                'Cluster': st.column_config.TextColumn(
                    'Cluster Description',
                    width='large',
                    help="Full behavioral cluster description"
                ),
                'Score': st.column_config.NumberColumn('Score', width='small'),
                'Size': st.column_config.NumberColumn('Size', width='small'),
                'Proportion': st.column_config.NumberColumn('Proportion', width='small'),
                'Quality': st.column_config.NumberColumn('Quality', width='small')
            }
            
            # Add CI column config if showing CIs
            if has_ci:
                column_config['CI'] = st.column_config.TextColumn(
                    'Confidence Interval', 
                    width='medium',
                    help="95% confidence interval for the distinctiveness score"
                )
            
            # Display the cluster table with text wrapping
            st.dataframe(
                cluster_df, 
                use_container_width=True, 
                hide_index=True,
                column_config=column_config
            )
            
            # Add example viewing section if enabled
            if show_examples and results_path:
                st.subheader("View Examples")
                cluster_names = [c['property_description'] for c in clusters[:top_n_clusters]]
                selected_cluster = st.selectbox("Select cluster to view examples", cluster_names)
                
                if selected_cluster and st.button("Load Examples"):
                    show_cluster_examples(selected_cluster, selected_model, model_stats, 
                                        results_path, cluster_level)
        else:
            st.warning(f"No cluster data available for {selected_model}")


def show_cluster_examples(cluster_label: str, model_name: str, model_stats: Dict[str, Any], 
                         results_path: Path, level: str = 'fine'):
    """Show examples using the pre-stored property IDs"""
    
    # Get the stored example property IDs from model_stats
    model_data = model_stats.get(model_name, {})
    clusters = model_data.get(level, [])
    
    target_cluster = None
    for cluster in clusters:
        if cluster['property_description'] == cluster_label:
            target_cluster = cluster
            break
    
    if not target_cluster or not target_cluster.get('examples'):
        st.warning("No examples available for this cluster")
        return
        
    # Load only the specific examples (max 3 property IDs)
    example_ids = target_cluster['examples']
    
    # Import here to avoid circular imports
    from ..data_loading import load_property_examples
    examples_df = load_property_examples(results_path, example_ids)
    
    if examples_df.empty:
        st.warning("Could not load example data")
        return
    
    st.write(f"**Examples for {model_name} in cluster '{cluster_label}':**")
    st.caption(f"Showing {len(examples_df)} example(s)")
    
    for i, (_, row) in enumerate(examples_df.iterrows(), 1):
        with st.expander(f"Prompt {i}: {row.get('prompt', 'Unknown')[:100]}...", expanded=i<=2):
            # Get the prompt and response
            prompt = row.get('prompt', row.get('user_prompt', 'N/A'))
            
            # Get the model response - check for different possible column names
            response = (row.get('model_response') or 
                      row.get('model_a_response') or 
                      row.get('model_b_response') or 
                      row.get('responses', 'N/A'))
            
            # Convert to OpenAI format if needed
            openai_response = convert_to_openai_format(response)
            
            # Display the conversation in OpenAI format
            display_openai_conversation(openai_response)
            
            # Display scores
            st.write("**Score:**")
            score = row.get('score', 'N/A')
            st.info(score) 
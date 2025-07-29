"""Streamlit application for exploring complete LMM-Vibes pipeline results.

This app provides a comprehensive view of model performance, cluster analysis,
and detailed examples from the pipeline output.

Run with:
    streamlit run lmmvibes/viz/pipeline_results_app.py -- --results_dir path/to/results/

Where results_dir contains:
    - clustered_results.jsonl
    - model_stats.json  
    - full_dataset.json (optional)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np

# Import data loading and utility functions
from lmmvibes.viz.data_loading import load_pipeline_results, scan_for_result_subfolders
from lmmvibes.viz.utils import compute_model_rankings
from lmmvibes.viz.tabs import (
    create_overview_tab,
    create_examples_tab,
    create_clusters_tab,
    create_frequencies_tab,
    create_search_tab
)

# ---------------------------------------------------------------------------
# CLI args (Streamlit forwards everything after "--")
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results_dir", required=False, help="Path to pipeline results directory")
    # Parse known to avoid Streamlit's own flags
    args, _ = parser.parse_known_args()
    return args

args = _parse_args()

# ---------------------------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="LMM-Vibes Pipeline Results", layout="wide")
    st.title("🔍 LMM-Vibes Pipeline Results Explorer")
    st.caption("Comprehensive analysis of model behavioral properties and performance")
    
    # Sidebar - Data Loading & Controls
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Data loading
        if args.results_dir:
            results_dir = args.results_dir
            st.success(f"Using: {Path(results_dir).name}")
        else:
            results_dir = st.text_input(
                "Results Directory", 
                placeholder="path/to/results/",
                help="Directory containing pipeline results directly, or parent directory with result subfolders"
            )
        
        if not results_dir:
            st.info("Please provide a results directory to begin")
            st.stop()
        
        # Determine final results path based on whether subfolders exist
        final_results_dir = results_dir
        
        # Scan for valid subfolders
        valid_subfolders = scan_for_result_subfolders(results_dir)
        
        # Check if the base directory itself contains results
        base_path = Path(results_dir)
        has_direct_results = (base_path / "model_stats.json").exists() or (base_path / "clustered_results.jsonl").exists()
        
        if len(valid_subfolders) > 1 or (len(valid_subfolders) == 1 and valid_subfolders[0] != "."):
            # Multiple options available, show selection
            st.subheader("📁 Select Results Folder")
            
            # Show count of found folders
            folder_count = len([f for f in valid_subfolders if f != "."]) + (1 if has_direct_results else 0)
            st.caption(f"Found {folder_count} pipeline result folder(s)")
            
            # Prepare options for display
            folder_options = []
            folder_values = []
            
            if has_direct_results:
                folder_options.append(f"📊 {Path(results_dir).name} (current directory)")
                folder_values.append(".")
            
            for subfolder in valid_subfolders:
                if subfolder != ".":
                    folder_options.append(f"📁 {subfolder}")
                    folder_values.append(subfolder)
            
            if folder_options:
                selected_idx = st.selectbox(
                    "Choose results to load:",
                    range(len(folder_options)),
                    format_func=lambda x: folder_options[x],
                    help="Select which set of pipeline results to analyze"
                )
                
                selected_folder = folder_values[selected_idx]
                if selected_folder != ".":
                    final_results_dir = str(Path(results_dir) / selected_folder)
                    st.success(f"Selected: {selected_folder}")
                else:
                    st.info(f"Using current directory: {Path(results_dir).name}")
            else:
                st.error(f"No valid pipeline results found in {results_dir}")
                st.stop()
                
        elif has_direct_results:
            # Only direct results available
            st.info(f"Loading results from: {Path(results_dir).name}")
        else:
            # No results found anywhere
            st.error(f"No pipeline results found in {results_dir}")
            st.info("Please ensure the directory contains either:")
            st.info("• model_stats.json and clustered_results.jsonl files")
            st.info("• Subfolders containing these files")
            st.stop()
        
        # Load data
        try:
            clustered_df, model_stats, results_path = load_pipeline_results(final_results_dir)
            st.session_state.results_path = results_path
        except Exception as e:
            st.error(f"Error loading results: {e}")
            st.stop()
        
        # Basic info
        st.write(f"**Models:** {len(model_stats)}")
        st.write(f"**Properties:** {len(clustered_df):,}")
        
        # Show cluster counts for both fine and coarse levels
        if 'fine_cluster_id' in clustered_df.columns:
            n_fine_clusters = clustered_df['fine_cluster_id'].nunique()
            st.write(f"**Fine Clusters:** {n_fine_clusters}")
        
        if 'coarse_cluster_id' in clustered_df.columns:
            n_coarse_clusters = clustered_df['coarse_cluster_id'].nunique()
            st.write(f"**Coarse Clusters:** {n_coarse_clusters}")
        
        st.divider()
        
        # Model selection
        st.subheader("Model Selection")
        all_models = list(model_stats.keys())
        selected_models = st.multiselect(
            "Select models to compare",
            all_models,
            default=all_models[:min(5, len(all_models))],  # Default to first 5
            help="Choose models for comparison views"
        )
        
        # Cluster level selection
        cluster_level = st.selectbox(
            "Cluster Level",
            ['fine', 'coarse'],
            help="Fine: detailed clusters, Coarse: high-level categories"
        )
        
        # Display options
        st.subheader("Display Options")
        top_n_clusters = st.slider(
            "Top N clusters per model",
            min_value=5, max_value=50, value=10,
            help="Number of top clusters to show per model"
        )
        
        show_examples = st.checkbox(
            "Enable example viewing",
            value=True,
            help="Allow viewing actual model responses (loads more data)"
        )
        
        # Confidence intervals are disabled in this version of the app
        has_any_ci = False
        show_confidence_intervals = False
    
    # Main content area
    model_rankings = compute_model_rankings(model_stats)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔍 View Examples", 
        "📋 View Clusters",
        "📈 Cluster Frequencies",
        "🔎 Vector Search"
    ])
    
    with tab1:
        create_overview_tab(model_stats, model_rankings, cluster_level, top_n_clusters)
    
    with tab2:
        create_examples_tab(model_stats, all_models, cluster_level, top_n_clusters, 
                           results_path, show_examples)
    
    with tab3:
        create_clusters_tab(clustered_df)
    
    with tab4:
        create_frequencies_tab(model_stats, selected_models, cluster_level, show_confidence_intervals)
    
    with tab5:
        create_search_tab(results_path, all_models)


if __name__ == "__main__":
    main()
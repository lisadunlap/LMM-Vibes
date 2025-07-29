"""
Data loading utilities for pipeline results app.

This module contains functions for loading and caching pipeline results data.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


@st.cache_data(show_spinner="Loading pipeline results...")
def load_pipeline_results(results_dir: str):
    """Load pipeline outputs optimized for large datasets"""
    results_path = Path(results_dir)
    
    # Load model statistics (already contains limited examples)
    model_stats_path = results_path / "model_stats.json"
    if not model_stats_path.exists():
        st.error(f"model_stats.json not found in {results_dir}")
        st.stop()
        
    with open(model_stats_path) as f:
        model_stats = json.load(f)
    
    # Load clustered results but only keep essential columns for overview
    clustered_path = results_path / "clustered_results.jsonl"
    
    if not clustered_path.exists():
        st.error(f"clustered_results.jsonl not found in {results_dir}")
        st.stop()
    
    # Try to load json first, then CSV as fallback
    clustered_df = None
    
    if clustered_path.exists():
        # Try to load with essential columns only for performance
        try:
            essential_cols = [
                'question_id', 'model', 'property_description', 
                'fine_cluster_id', 'fine_cluster_label',
                'coarse_cluster_id', 'coarse_cluster_label',
                'score', 'id'  # property id for examples
            ]
            clustered_df = pd.read_json(clustered_path, lines=True)
        except Exception as e:
            st.warning(f"Could not load json with essential columns: {e}")
            st.info("Attempting to load all columns...")
            clustered_df = pd.read_json(clustered_path, lines=True)
            st.success("Successfully loaded from json file")
    
    if clustered_df is None:
        st.error("Could not load clustered results from any available format")
        st.stop()
    
    return clustered_df, model_stats, results_path


@st.cache_data
def load_property_examples(results_path: Path, property_ids: List[str]):
    """Load specific property examples on-demand"""
    if not property_ids:
        return pd.DataFrame()
        
    # Load full dataset to get prompt/response details
    clustered_path = results_path / "clustered_results.jsonl"
    
    full_df = None
    
    # Try json first
    if clustered_path.exists():
        try:
            full_df = pd.read_json(clustered_path, lines=True)
        except Exception as e:
            st.warning(f"Failed to load examples from json: {e}")
            full_df = None
    
    if full_df is None:
        st.error("Could not load example data from any available format")
        return pd.DataFrame()
    
    return full_df[full_df['id'].isin(property_ids)]


def scan_for_result_subfolders(base_dir: str) -> List[str]:
    """Scan a directory for subfolders containing pipeline results"""
    base_path = Path(base_dir)
    if not base_path.exists() or not base_path.is_dir():
        return []
    
    valid_subfolders = []
    
    # Check if the base directory itself contains results
    if (base_path / "model_stats.json").exists() or (base_path / "clustered_results.json").exists():
        valid_subfolders.append(".")  # Current directory
    
    # Check subdirectories
    try:
        for item in base_path.iterdir():
            if item.is_dir():
                # Check if this subdirectory contains pipeline results
                has_model_stats = (item / "model_stats.json").exists()
                has_clustered_results = (item / "clustered_results.json").exists()
                
                if has_model_stats or has_clustered_results:
                    valid_subfolders.append(item.name)
    except PermissionError:
        pass  # Skip directories we can't read
    
    return sorted(valid_subfolders) 
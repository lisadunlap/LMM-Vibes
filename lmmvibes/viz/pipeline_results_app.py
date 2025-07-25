"""Streamlit application for exploring complete LMM-Vibes pipeline results.

This app provides a comprehensive view of model performance, cluster analysis,
and detailed examples from the pipeline output.

Run with:
    streamlit run lmmvibes/viz/pipeline_results_app.py -- --results_dir path/to/results/

Where results_dir contains:
    - clustered_results.json
    - model_stats.json  
    - full_dataset.json (optional)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
# Data Loading Functions
# ---------------------------------------------------------------------------

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
    clustered_path = results_path / "clustered_results.json"
    csv_path = results_path / "clustered_results.csv"
    
    if not clustered_path.exists() and not csv_path.exists():
        st.error(f"Neither clustered_results.json nor clustered_results.csv found in {results_dir}")
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
    clustered_path = results_path / "clustered_results.json"
    
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

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

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

def extract_quality_score(quality_score) -> float:
    """
    Extract a float quality score from quality_score field.
    
    Args:
        quality_score: Either a float, int, dictionary, or nested structure with score keys
        
    Returns:
        float: The quality score value, always guaranteed to be a float
    """
    if quality_score is None:
        return 0.0
    elif isinstance(quality_score, (int, float)):
        return float(quality_score)
    elif isinstance(quality_score, dict):
        # Handle dictionary cases
        if not quality_score:  # Empty dict
            return 0.0
        
        # Try to extract a numeric value from the dictionary
        for key, value in quality_score.items():
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, dict):
                # Handle nested dictionaries recursively
                nested_result = extract_quality_score(value)
                if nested_result != 0.0:  # Found a valid value
                    return nested_result
        
        # If no numeric values found, return 0.0
        return 0.0
    else:
        # For any other type, try to convert to float, fallback to 0.0
        try:
            return float(quality_score)
        except (ValueError, TypeError):
            return 0.0


# Test the helper function
if __name__ == "__main__":
    # Test cases for extract_quality_score
    assert extract_quality_score(0.5) == 0.5
    assert extract_quality_score(5) == 5.0  # Test integer
    assert extract_quality_score({"pass at one": 0.8}) == 0.8
    assert extract_quality_score({"accuracy": 0.9, "helpfulness": 0.7}) == 0.9  # First numeric value
    assert extract_quality_score({}) == 0.0
    assert extract_quality_score(None) == 0.0
    assert extract_quality_score({"nested": {"inner": 0.75}}) == 0.75  # Nested dict
    assert extract_quality_score({"non_numeric": "text", "score": 0.6}) == 0.6  # Mixed types
    assert extract_quality_score("0.5") == 0.5  # String that can be converted
    assert extract_quality_score("invalid") == 0.0  # String that can't be converted
    print("✅ All extract_quality_score tests passed!")

def compute_model_rankings(model_stats: Dict[str, Any]) -> List[tuple]:
    """Compute model rankings by average score"""
    model_scores = {}
    for model, stats in model_stats.items():
        fine_scores = [stat['score'] for stat in stats.get('fine', [])]
        if fine_scores:
            model_scores[model] = {
                'avg_score': np.mean(fine_scores),
                'median_score': np.median(fine_scores),
                'num_clusters': len(fine_scores),
                'top_score': max(fine_scores),
                'std_score': np.std(fine_scores)
            }
        else:
            model_scores[model] = {
                'avg_score': 0, 'median_score': 0, 'num_clusters': 0, 
                'top_score': 0, 'std_score': 0
            }
    
    return sorted(model_scores.items(), key=lambda x: x[1]['avg_score'], reverse=True)

def get_top_clusters_for_model(model_stats: Dict[str, Any], model_name: str, 
                              level: str = 'fine', top_n: int = 10) -> List[Dict[str, Any]]:
    """Get top N clusters for a specific model"""
    model_data = model_stats.get(model_name, {})
    clusters = model_data.get(level, [])
    return sorted(clusters, key=lambda x: x['score'], reverse=True)[:top_n]

# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def split_conversation_at_user(prompt_text: str) -> tuple[str, str]:
    """
    Split conversation text at user marker.
    
    Args:
        prompt_text: The full prompt text that may contain metadata
        
    Returns:
        tuple: (metadata_before_user, conversation_after_user)
    """
    if not isinstance(prompt_text, str):
        return "", str(prompt_text)
    
    # Look for various user markers (with and without bold formatting)
    user_markers = ["**user:**", "user:", "**User:**", "User:"]
    
    for marker in user_markers:
        if marker in prompt_text:
            parts = prompt_text.split(marker, 1)  # Split only on first occurrence
            metadata = parts[0].strip()
            conversation = marker + parts[1].strip() if len(parts) > 1 else ""
            return metadata, conversation
    
    # If no user marker found, also check for assistant marker as fallback
    assistant_markers = ["**assistant:**", "assistant:", "**Assistant:**", "Assistant:"]
    for marker in assistant_markers:
        if marker in prompt_text:
            parts = prompt_text.split(marker, 1)
            metadata = parts[0].strip()
            conversation = marker + parts[1].strip() if len(parts) > 1 else ""
            return metadata, conversation
    
    # No markers found, treat entire text as conversation
    return "", prompt_text

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
    examples_df = load_property_examples(results_path, example_ids)
    
    if examples_df.empty:
        st.warning("Could not load example data")
        return
    
    st.write(f"**Examples for {model_name} in cluster '{cluster_label}':**")
    st.caption(f"Showing {len(examples_df)} example(s)")
    
    for i, (_, row) in enumerate(examples_df.iterrows(), 1):
        with st.expander(f"Example {i}: {row.get('id', 'Unknown')[:12]}..."):
            # Get the prompt and response
            prompt = row.get('prompt', row.get('user_prompt', 'N/A'))
            
            # Get the model response and split it if it contains metadata
            response = (row.get('model_response') or 
                      row.get('model_a_response') or 
                      row.get('model_b_response') or 
                      row.get('responses', 'N/A'))
            
            # Debug: Print the full response to terminal to see its format
            print(f"\n{'='*80}")
            print(f"DEBUG: Full response for example {i}")
            print(f"{'='*80}")
            print(response)
            print(f"{'='*80}\n")
            
            metadata, conversation = split_conversation_at_user(response)
            
            st.write("**Prompt:**")
            st.text(prompt)
            
            # Display metadata if it exists
            if metadata:
                st.markdown("**📋 Metadata & Context:**")
                st.markdown(metadata)
                st.divider()
            
            # Display the conversation
            st.markdown("**💬 Conversation:**")
            st.markdown(conversation)
            
            st.write("**Score:**")
            score = row.get('score', 'N/A')
            st.info(score)

def create_model_leaderboard(model_rankings: List[tuple]):
    """Create a model leaderboard table"""
    
    leaderboard_data = []
    for rank, (model, stats) in enumerate(model_rankings, 1):
        leaderboard_data.append({
            'Rank': rank,
            'Model': model,
            'Avg Score': f"{stats['avg_score']:.3f}",
            'Median Score': f"{stats['median_score']:.3f}", 
            'Top Score': f"{stats['top_score']:.3f}",
            'Std Dev': f"{stats['std_score']:.3f}",
            'Clusters': stats['num_clusters']
        })
    
    df_leaderboard = pd.DataFrame(leaderboard_data)
    
    # Style the dataframe
    st.dataframe(
        df_leaderboard,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Rank': st.column_config.NumberColumn('Rank', width='small'),
            'Model': st.column_config.TextColumn('Model', width='medium'),
            'Avg Score': st.column_config.NumberColumn('Avg Score', width='small'),
            'Median Score': st.column_config.NumberColumn('Median Score', width='small'),
            'Top Score': st.column_config.NumberColumn('Top Score', width='small'),
            'Std Dev': st.column_config.NumberColumn('Std Dev', width='small'),
            'Clusters': st.column_config.NumberColumn('# Clusters', width='small')
        }
    )

def create_model_comparison_heatmap(model_stats: Dict[str, Any], selected_models: List[str], 
                                   level: str = 'fine', top_n_clusters: int = 20):
    """Create a heatmap comparing models across top clusters"""
    
    if not selected_models:
        st.warning("Please select models to compare")
        return
    
    # Get all clusters across selected models
    all_clusters = set()
    for model in selected_models:
        model_data = model_stats.get(model, {})
        clusters = model_data.get(level, [])
        for cluster in clusters[:top_n_clusters]:  # Limit for performance
            all_clusters.add(cluster['property_description'])
    
    all_clusters = sorted(list(all_clusters))[:top_n_clusters]  # Further limit
    
    # Build score matrix
    score_matrix = []
    for cluster in all_clusters:
        row = []
        for model in selected_models:
            model_data = model_stats.get(model, {})
            clusters = model_data.get(level, [])
            # Find score for this cluster
            score = 0
            for c in clusters:
                if c['property_description'] == cluster:
                    score = c['score']
                    break
            row.append(score)
        score_matrix.append(row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=score_matrix,
        x=selected_models,
        y=[label[:60] + '...' if len(label) > 60 else label for label in all_clusters],
        colorscale='RdYlGn',
        zmid=1.0,  # Center at score = 1.0 (median performance)
        colorbar=dict(title="Score vs Median"),
        hoverongaps=False,
        hovertemplate='Model: %{x}<br>Cluster: %{y}<br>Score: %{z:.3f}<extra></extra>',
        text=[[f"{score:.2f}" for score in row] for row in score_matrix],
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=f"Model Performance Heatmap - {level.title()} Level Clusters",
        xaxis_title="Models",
        yaxis_title="Behavioral Clusters",
        height=max(600, len(all_clusters) * 35),  # Increased base height and per-cluster height
        yaxis={'side': 'left'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_score_distribution_plot(model_stats: Dict[str, Any], selected_models: List[str], 
                                  level: str = 'fine'):
    """Create violin plot showing score distributions per model"""
    
    if not selected_models:
        return
    
    plot_data = []
    for model in selected_models:
        model_data = model_stats.get(model, {})
        clusters = model_data.get(level, [])
        scores = [c['score'] for c in clusters]
        plot_data.extend([{'Model': model, 'Score': score} for score in scores])
    
    if not plot_data:
        st.warning("No score data available for selected models")
        return
    
    df_plot = pd.DataFrame(plot_data)
    
    fig = px.violin(df_plot, x='Model', y='Score', box=True, points='outliers',
                   title=f"Score Distribution - {level.title()} Level Clusters")
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                  annotation_text="Median Performance")
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)

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
        has_direct_results = (base_path / "model_stats.json").exists() or (base_path / "clustered_results.json").exists()
        
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
            st.info("• model_stats.json and clustered_results.json files")
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
    
    # Main content area
    model_rankings = compute_model_rankings(model_stats)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs([
        "📊 Overview", 
        "🔍 View Examples", 
        "📋 View Clusters"
    ])
    
    with tab1:
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
                                
                                st.markdown(f"""
                                <div style="margin: 8px 0; padding: 10px; border-left: 3px solid #3182ce; background-color: #f8f9fa; position: relative;">
                                    <div style="font-weight: 600; font-size: 16px; margin-bottom: 5px;">
                                        {cluster_desc}
                                    </div>
                                    <div style="font-size: 14px; color: #666;">
                                        <strong>{frequency:.1f}% frequency</strong> ({cluster_size} out of {cluster_size_global} total across all models)
                                    </div>
                                    <div style="font-size: 13px; color: #3182ce;">
                                        {distinctiveness:.1f}x more distinctive than other models
                                    </div>
                                    <div style="position: absolute; bottom: 8px; right: 10px; font-size: 14px; font-weight: 600; color: {'#28a745' if quality_score >= 0 else '#dc3545'};">
                                        Quality: {quality_score:.3f}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.write("No cluster data available")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
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
                    cluster_df_data.append({
                        'Cluster': cluster['property_description'],  # Show full text, no truncation
                        'Score': f"{cluster['score']:.3f}",
                        'Size': cluster['size'],
                        'Proportion': f"{cluster['proportion']:.3f}",
                        'Quality': f"{extract_quality_score(cluster.get('quality_score')):.3f}"
                    })
                
                cluster_df = pd.DataFrame(cluster_df_data)
                
                # Display the cluster table with text wrapping
                st.dataframe(
                    cluster_df, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
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
                )
                
                # Add example viewing section if enabled
                if show_examples:
                    st.subheader("View Examples")
                    cluster_names = [c['property_description'] for c in clusters[:top_n_clusters]]
                    selected_cluster = st.selectbox("Select cluster to view examples", cluster_names)
                    
                    if selected_cluster and st.button("Load Examples"):
                        show_cluster_examples(selected_cluster, selected_model, model_stats, 
                                            results_path, cluster_level)
    
    with tab3:
        st.header("View Clusters")
        st.write("Explore behavioral clusters sorted by quality or maximum distinctiveness across models")
        
        # Collect all clusters across all models
        all_clusters_data = []
        for model_name, model_data in model_stats.items():
            clusters = model_data.get(cluster_level, [])
            for cluster in clusters:
                all_clusters_data.append({
                    'property_description': cluster['property_description'],
                    'model': model_name,
                    'score': cluster.get('score', 0),
                    'quality_score': extract_quality_score(cluster.get('quality_score', 0)),
                    'size': cluster.get('size', 0),
                    'cluster_size_global': cluster.get('cluster_size_global', 0),
                    'proportion': cluster.get('proportion', 0)
                })
        
        if all_clusters_data:
            clusters_df = pd.DataFrame(all_clusters_data)
            
            # Group by cluster to get max scores and average quality
            cluster_summary = clusters_df.groupby('property_description').agg({
                'score': 'max',  # Max distinctiveness across models
                'quality_score': 'mean',  # Average quality score
                'cluster_size_global': 'first',  # Should be same for all models
                'size': 'sum'  # Total size across all models (should equal cluster_size_global)
            }).reset_index()
            
            # Sort options
            sort_option = st.selectbox(
                "Sort clusters by:",
                ["Max Distinctiveness", "Average Quality Score", "Cluster Size"],
                help="Max Distinctiveness: clusters that maximally separate models\nAverage Quality Score: clusters with high correlation to accuracy"
            )
            
            if sort_option == "Max Distinctiveness":
                cluster_summary = cluster_summary.sort_values('score', ascending=False)
            elif sort_option == "Average Quality Score":
                cluster_summary = cluster_summary.sort_values('quality_score', ascending=False)
            else:  # Cluster Size
                cluster_summary = cluster_summary.sort_values('cluster_size_global', ascending=False)
            
            # Display clusters
            st.subheader(f"Clusters sorted by {sort_option}")
            
            # Format for display
            display_df = cluster_summary.copy()
            display_df['Cluster Description'] = display_df['property_description']  # Show full text, no truncation
            display_df['Max Score'] = display_df['score'].apply(lambda x: f"{x:.3f}")
            display_df['Avg Quality'] = display_df['quality_score'].apply(
                lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "0.000"
            )
            display_df['Size'] = display_df['cluster_size_global']
            
            st.dataframe(
                display_df[['Cluster Description', 'Max Score', 'Avg Quality', 'Size']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Cluster Description': st.column_config.TextColumn(
                        'Cluster Description',
                        width='large',
                        help="Full behavioral cluster description"
                    ),
                    'Max Score': st.column_config.NumberColumn('Max Score', width='small'),
                    'Avg Quality': st.column_config.NumberColumn('Avg Quality', width='small'), 
                    'Size': st.column_config.NumberColumn('Size', width='small')
                }
            )
            
            # Show detailed view for selected cluster
            st.subheader("Cluster Details")
            selected_cluster_desc = st.selectbox(
                "Select cluster for detailed view:",
                cluster_summary['property_description'].tolist()
            )
            
            if selected_cluster_desc:
                cluster_details = clusters_df[clusters_df['property_description'] == selected_cluster_desc]
                
                st.write(f"**Cluster:** {selected_cluster_desc}")
                st.write(f"**Total Size:** {cluster_details['cluster_size_global'].iloc[0]}")
                
                # Show model breakdown
                st.write("**Model Breakdown:**")
                model_breakdown = cluster_details[['model', 'score', 'size', 'proportion']].sort_values('score', ascending=False)
                model_breakdown['Score'] = model_breakdown['score'].apply(lambda x: f"{x:.3f}")
                model_breakdown['Frequency'] = model_breakdown['proportion'].apply(lambda x: f"{x*100:.1f}%")
                model_breakdown = model_breakdown.rename(columns={'model': 'Model', 'size': 'Size'})
                
                st.dataframe(
                    model_breakdown[['Model', 'Score', 'Size', 'Frequency']],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Show examples from this cluster
                st.write("**Examples:**")
                
                # Get example property IDs from the model_stats for this cluster
                example_property_ids = []
                for model_name, model_data in model_stats.items():
                    clusters = model_data.get(cluster_level, [])
                    for cluster in clusters:
                        if cluster['property_description'] == selected_cluster_desc:
                            example_property_ids.extend(cluster.get('examples', []))
                
                # Remove duplicates and limit to first 5 examples
                example_property_ids = list(set(example_property_ids))[:5]
                
                if example_property_ids:
                    # Load the example data
                    examples_df = load_property_examples(results_path, example_property_ids)
                    
                    if not examples_df.empty:
                        st.caption(f"Showing {len(examples_df)} example(s) from this cluster")
                        
                        for i, (_, row) in enumerate(examples_df.iterrows(), 1):
                            with st.expander(f"Example {i}: {row.get('model', 'Unknown model')} - {row.get('id', 'Unknown')[:12]}..."):
                                # Get the prompt and response
                                prompt = row.get('prompt', row.get('user_prompt', 'N/A'))
                                
                                # Get the model response and split it if it contains metadata
                                response = (row.get('model_response') or 
                                          row.get('model_a_response') or 
                                          row.get('model_b_response') or 
                                          row.get('responses', 'N/A'))
                                
                                # Debug: Print the full response to terminal to see its format
                                print(f"\n{'='*80}")
                                print(f"DEBUG: Full response for cluster example {i}")
                                print(f"{'='*80}")
                                print(response)
                                print(f"{'='*80}\n")
                                
                                metadata, conversation = split_conversation_at_user(response)
                                
                                st.write("**Prompt:**")
                                st.text(prompt)
                                
                                # Display metadata if it exists
                                if metadata:
                                    st.markdown("**📋 Metadata & Context:**")
                                    st.markdown(metadata)
                                    st.divider()
                                
                                # Display the conversation
                                st.markdown("**💬 Conversation:**")
                                st.markdown(conversation)
                                
                                st.write("**Metadata:**")
                                st.json({
                                    'model': row.get('model', 'N/A'),
                                    'question_id': row.get('question_id', 'N/A'),
                                    'property_id': row.get('id', 'N/A'),
                                    'cluster_id': row.get('fine_cluster_id', 'N/A')
                                })
                                
                                st.write("**Extracted Property:**")
                                property_desc = row.get('property_description', 'N/A')
                                st.info(property_desc)
                    else:
                        st.warning("Could not load example data for this cluster")
                else:
                    st.warning("No examples available for this cluster")
        else:
            st.warning("No cluster data available")

if __name__ == "__main__":
    main()
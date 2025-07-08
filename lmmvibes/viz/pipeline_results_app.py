"""Streamlit application for exploring complete LMM-Vibes pipeline results.

This app provides a comprehensive view of model performance, cluster analysis,
and detailed examples from the pipeline output.

Run with:
    streamlit run lmmvibes/viz/pipeline_results_app.py -- --results_dir path/to/results/

Where results_dir contains:
    - clustered_results.parquet
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
    clustered_path = results_path / "clustered_results.parquet"
    if not clustered_path.exists():
        st.error(f"clustered_results.parquet not found in {results_dir}")
        st.stop()
    
    # Try to load with essential columns only for performance
    try:
        essential_cols = [
            'question_id', 'model', 'property_description', 
            'fine_cluster_id', 'fine_cluster_label',
            'coarse_cluster_id', 'coarse_cluster_label',
            'score', 'id'  # property id for examples
        ]
        clustered_df = pd.read_parquet(clustered_path, columns=essential_cols)
    except:
        # Fallback: load all columns if specific ones don't exist
        clustered_df = pd.read_parquet(clustered_path)
    
    return clustered_df, model_stats, results_path

@st.cache_data
def load_property_examples(results_path: Path, property_ids: List[str]):
    """Load specific property examples on-demand"""
    if not property_ids:
        return pd.DataFrame()
        
    # Load full dataset to get prompt/response details
    clustered_path = results_path / "clustered_results.parquet"
    full_df = pd.read_parquet(clustered_path)
    return full_df[full_df['id'].isin(property_ids)]

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

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
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Prompt:**")
                prompt = row.get('prompt', row.get('user_prompt', 'N/A'))
                st.write(prompt)
                
                st.write("**Metadata:**")
                st.json({
                    'question_id': row.get('question_id', 'N/A'),
                    'property_id': row.get('id', 'N/A'),
                    'cluster_id': row.get('fine_cluster_id', 'N/A'),
                    'score': row.get('score', 'N/A')
                })
            
            with col2:
                st.write("**Model Response:**")
                # Handle different response column formats
                response = (row.get('model_response') or 
                          row.get('model_a_response') or 
                          row.get('model_b_response') or 
                          row.get('responses', 'N/A'))
                st.write(response)
                
                st.write("**Extracted Property:**")
                property_desc = row.get('property_description', 'N/A')
                st.info(property_desc)

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
        hovertemplate='Model: %{x}<br>Cluster: %{y}<br>Score: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Model Performance Heatmap - {level.title()} Level Clusters",
        xaxis_title="Models",
        yaxis_title="Behavioral Clusters",
        height=max(400, len(all_clusters) * 20),
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
                help="Directory containing clustered_results.parquet and model_stats.json"
            )
        
        if not results_dir:
            st.info("Please provide a results directory to begin")
            st.stop()
        
        # Load data
        try:
            clustered_df, model_stats, results_path = load_pipeline_results(results_dir)
            st.session_state.results_path = results_path
        except Exception as e:
            st.error(f"Error loading results: {e}")
            st.stop()
        
        # Basic info
        st.write(f"**Models:** {len(model_stats)}")
        st.write(f"**Properties:** {len(clustered_df):,}")
        if 'fine_cluster_id' in clustered_df.columns:
            n_clusters = clustered_df['fine_cluster_id'].nunique()
            st.write(f"**Clusters:** {n_clusters}")
        
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "🏆 Model Rankings", 
        "🔥 Heatmap Comparison",
        "📈 Score Analysis"
    ])
    
    with tab1:
        st.header("Pipeline Results Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Models", len(model_stats))
        with col2:
            st.metric("Total Properties", f"{len(clustered_df):,}")
        with col3:
            if 'fine_cluster_id' in clustered_df.columns:
                n_fine = clustered_df['fine_cluster_id'].nunique()
                st.metric("Fine Clusters", n_fine)
        with col4:
            if 'coarse_cluster_id' in clustered_df.columns:
                n_coarse = clustered_df['coarse_cluster_id'].nunique()
                st.metric("Coarse Clusters", n_coarse)
        
        st.subheader("Top Performing Models")
        top_5_models = model_rankings[:5]
        for rank, (model, stats) in enumerate(top_5_models, 1):
            with st.expander(f"{rank}. {model} (avg: {stats['avg_score']:.3f})"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write("**Performance Metrics:**")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Avg Score", f"{stats['avg_score']:.3f}")
                    with metric_col2:
                        st.metric("Top Score", f"{stats['top_score']:.3f}")
                    with metric_col3:
                        st.metric("# Clusters", stats['num_clusters'])
                
                with col2:
                    st.write("**Top Behavioral Clusters:**")
                    top_clusters = get_top_clusters_for_model(model_stats, model, cluster_level, 3)
                    for i, cluster in enumerate(top_clusters, 1):
                        st.write(f"{i}. {cluster['property_description'][:50]}...")
                        st.caption(f"Score: {cluster['score']:.3f}")
    
    with tab2:
        st.header("Model Performance Rankings")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Leaderboard")
            create_model_leaderboard(model_rankings)
        
        with col2:
            st.subheader("Distribution")
            avg_scores = [stats['avg_score'] for _, stats in model_rankings]
            fig = px.histogram(x=avg_scores, nbins=20, title="Distribution of Average Scores")
            fig.add_vline(x=1.0, line_dash="dash", line_color="red", 
                         annotation_text="Median Performance")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed model analysis
        st.subheader("Detailed Model Analysis")
        selected_model = st.selectbox("Select model for detailed view", all_models)
        
        if selected_model:
            model_data = model_stats.get(selected_model, {})
            clusters = model_data.get(cluster_level, [])
            
            if clusters:
                st.write(f"**Top clusters for {selected_model}:**")
                
                cluster_df_data = []
                for cluster in clusters[:top_n_clusters]:
                    cluster_df_data.append({
                        'Cluster': cluster['property_description'][:80] + ('...' if len(cluster['property_description']) > 80 else ''),
                        'Score': f"{cluster['score']:.3f}",
                        'Size': cluster['size'],
                        'Proportion': f"{cluster['proportion']:.3f}",
                        'Quality': f"{cluster.get('quality_score', 0):.3f}"
                    })
                
                cluster_df = pd.DataFrame(cluster_df_data)
                
                # Add example buttons if enabled
                if show_examples:
                    st.dataframe(cluster_df, use_container_width=True, hide_index=True)
                    
                    st.subheader("View Examples")
                    cluster_names = [c['property_description'] for c in clusters[:top_n_clusters]]
                    selected_cluster = st.selectbox("Select cluster to view examples", cluster_names)
                    
                    if selected_cluster and st.button("Load Examples"):
                        show_cluster_examples(selected_cluster, selected_model, model_stats, 
                                            results_path, cluster_level)
                else:
                    st.dataframe(cluster_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.header("Model Comparison Heatmap")
        st.write("Compare model performance across behavioral clusters")
        
        if len(selected_models) >= 2:
            create_model_comparison_heatmap(model_stats, selected_models, cluster_level, top_n_clusters)
        else:
            st.warning("Please select at least 2 models in the sidebar for comparison")
    
    with tab4:
        st.header("Score Distribution Analysis")
        st.write("Analyze score distributions and patterns across models")
        
        if selected_models:
            create_score_distribution_plot(model_stats, selected_models, cluster_level)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            summary_data = []
            for model in selected_models:
                model_data = model_stats.get(model, {})
                clusters = model_data.get(cluster_level, [])
                scores = [c['score'] for c in clusters]
                
                if scores:
                    summary_data.append({
                        'Model': model,
                        'Mean': np.mean(scores),
                        'Median': np.median(scores),
                        'Std Dev': np.std(scores),
                        'Min': np.min(scores),
                        'Max': np.max(scores),
                        'Above 1.0': sum(1 for s in scores if s > 1.0),
                        'Below 1.0': sum(1 for s in scores if s < 1.0)
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Please select models in the sidebar to view analysis")

if __name__ == "__main__":
    main()
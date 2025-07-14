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

def _aggregate_quality_scores(quality_scores_series):
    """Aggregate quality scores from multiple models for a cluster.
    
    Args:
        quality_scores_series: Series of quality score dictionaries
        
    Returns:
        Dictionary with average quality scores for each key
    """
    all_keys = set()
    for qs in quality_scores_series:
        if isinstance(qs, dict):
            all_keys.update(qs.keys())
    
    if not all_keys:
        return {}
    
    aggregated = {}
    for key in all_keys:
        values = []
        for qs in quality_scores_series:
            if isinstance(qs, dict) and key in qs:
                values.append(qs[key])
        if values:
            aggregated[key] = sum(values) / len(values)
        else:
            aggregated[key] = 0.0
    
    return aggregated

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
                                quality_score = cluster.get('quality_score', {})  # Quality score dict
                                
                                # Calculate distinctiveness (using score as proxy)
                                distinctiveness = cluster.get('score', 1.0)
                                
                                # Format quality scores for display
                                quality_score_text = ""
                                if isinstance(quality_score, dict) and quality_score:
                                    quality_score_text = "<br>".join([f"{key}: {value:.3f}" for key, value in quality_score.items()])
                                else:
                                    quality_score_text = "No quality scores"
                                
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
                                    <div style="position: absolute; bottom: 8px; right: 10px; font-size: 12px; font-weight: 600; color: #666; text-align: right;">
                                        Quality Scores:<br>{quality_score_text}
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
                    'quality_score': cluster.get('quality_score', 0),
                    'size': cluster.get('size', 0),
                    'cluster_size_global': cluster.get('cluster_size_global', 0),
                    'proportion': cluster.get('proportion', 0)
                })
        
        if all_clusters_data:
            clusters_df = pd.DataFrame(all_clusters_data)
            
            # Group by cluster to get max scores and average quality
            cluster_summary = clusters_df.groupby('property_description').agg({
                'score': 'max',  # Max distinctiveness across models
                'quality_score': lambda x: _aggregate_quality_scores(x),  # Aggregate quality scores
                'cluster_size_global': 'first',  # Should be same for all models
                'size': 'sum'  # Total size across all models (should equal cluster_size_global)
            }).reset_index()
            
            # Sort options
            sort_options = ["Max Distinctiveness", "Cluster Size"]
            
            # Add quality score sorting options if quality scores exist
            if not cluster_summary.empty and 'quality_score' in cluster_summary.columns:
                first_qs = cluster_summary['quality_score'].iloc[0]
                if isinstance(first_qs, dict) and first_qs:
                    for key in first_qs.keys():
                        sort_options.append(f"Quality Score - {key}")
            
            sort_option = st.selectbox(
                "Sort clusters by:",
                sort_options,
                help="Max Distinctiveness: clusters that maximally separate models\nQuality Score: clusters with high correlation to specific metrics"
            )
            
            if sort_option == "Max Distinctiveness":
                cluster_summary = cluster_summary.sort_values('score', ascending=False)
            elif sort_option == "Cluster Size":
                cluster_summary = cluster_summary.sort_values('cluster_size_global', ascending=False)
            elif sort_option.startswith("Quality Score - "):
                # Extract the quality score key
                key = sort_option.replace("Quality Score - ", "")
                # Create a temporary column for sorting
                cluster_summary['temp_qs'] = cluster_summary['quality_score'].apply(
                    lambda x: x.get(key, 0) if isinstance(x, dict) else 0
                )
                cluster_summary = cluster_summary.sort_values('temp_qs', ascending=False)
                cluster_summary = cluster_summary.drop('temp_qs', axis=1)
            
            # Display clusters
            st.subheader(f"Clusters sorted by {sort_option}")
            
            # Format for display
            display_df = cluster_summary.copy()
            display_df['Cluster Description'] = display_df['property_description'].apply(
                lambda x: x[:100] + '...' if len(x) > 100 else x
            )
            display_df['Max Score'] = display_df['score'].apply(lambda x: f"{x:.3f}")
            
            # Handle quality scores as dictionary
            def format_quality_scores(qs):
                if isinstance(qs, dict) and qs:
                    return "; ".join([f"{key}: {value:.3f}" for key, value in qs.items()])
                else:
                    return "No quality scores"
            
            display_df['Quality Scores'] = display_df['quality_score'].apply(format_quality_scores)
            display_df['Size'] = display_df['cluster_size_global']
            
            st.dataframe(
                display_df[['Cluster Description', 'Max Score', 'Quality Scores', 'Size']],
                use_container_width=True,
                hide_index=True
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
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Prompt:**")
                                    prompt = row.get('prompt', row.get('user_prompt', 'N/A'))
                                    st.write(prompt)
                                    
                                    st.write("**Metadata:**")
                                    st.json({
                                        'model': row.get('model', 'N/A'),
                                        'question_id': row.get('question_id', 'N/A'),
                                        'property_id': row.get('id', 'N/A'),
                                        'cluster_id': row.get('fine_cluster_id', 'N/A')
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
                    else:
                        st.warning("Could not load example data for this cluster")
                else:
                    st.warning("No examples available for this cluster")
        else:
            st.warning("No cluster data available")

if __name__ == "__main__":
    main()
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
import markdown

# Import vector search functionality
from lmmvibes.viz.vector_search import PropertyVectorSearch, SearchResult

# Import conversation conversion function
try:
    from lmmvibes.extractors.conv_to_str import conv_to_str
except ImportError:
    def conv_to_str(conv):
        """Fallback function if conv_to_str is not available."""
        if isinstance(conv, str):
            return conv
        elif isinstance(conv, list):
            return "\n".join(str(item) for item in conv)
        else:
            return str(conv)

def convert_to_openai_format(response_data):
    """Convert various response formats to OpenAI format"""
    
    if isinstance(response_data, list):
        # Already in OpenAI format
        return response_data
    elif isinstance(response_data, str):
        # Try to parse as JSON
        try:
            parsed = json.loads(response_data)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        
        # If it's a string, try to convert using conv_to_str
        try:
            converted = conv_to_str(response_data)
            return converted
        except:
            pass
        
        # Fallback: treat as plain text
        return [{"role": "assistant", "content": response_data}]
    else:
        # Fallback for other types
        return [{"role": "assistant", "content": str(response_data)}]

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


def format_confidence_interval(score_ci: dict, confidence_level: float = 0.95) -> str:
    """
    Format confidence interval for display.
    
    Args:
        score_ci: Dict with "lower" and "upper" keys, or None
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        str: Formatted confidence interval string
    """
    if not score_ci or not isinstance(score_ci, dict):
        return "N/A"
    
    lower = score_ci.get("lower")
    upper = score_ci.get("upper")
    
    if lower is None or upper is None:
        return "N/A"
    
    ci_percent = int(confidence_level * 100)
    return f"[{lower:.3f}, {upper:.3f}] ({ci_percent}% CI)"


def has_confidence_intervals(cluster_stat: dict) -> bool:
    """
    Check if a cluster statistic has confidence intervals.
    
    Args:
        cluster_stat: Cluster statistic dictionary
        
    Returns:
        bool: True if confidence intervals are available
    """
    score_ci = cluster_stat.get('score_ci')
    return (isinstance(score_ci, dict) and 
            score_ci.get('lower') is not None and 
            score_ci.get('upper') is not None)


def get_confidence_interval_width(score_ci: dict) -> float:
    """
    Calculate the width of a confidence interval.
    
    Args:
        score_ci: Dict with "lower" and "upper" keys
        
    Returns:
        float: Width of the interval
    """
    if not score_ci or not isinstance(score_ci, dict):
        return 0.0
    
    lower = score_ci.get("lower")
    upper = score_ci.get("upper")
    
    if lower is None or upper is None:
        return 0.0
        
    return upper - lower


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

def display_openai_message(role, content, name=None, message_id=None):
    """Display a single OpenAI format message with role-specific styling"""
    
    # Define colors for different roles
    role_colors = {
        "system": "#ff6b6b",      # Red
        "user": "#4ecdc4",        # Teal 
        "assistant": "#45b7d1",   # Blue
        "tool": "#96ceb4",        # Green
        "info": "#feca57"         # Yellow
    }
    
    # Get color for this role, default to gray
    color = role_colors.get(role.lower(), "#95a5a6")
    
    # Format content for HTML display
    if isinstance(content, dict):
        content_html = f"<pre style='background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto;'>{json.dumps(content, indent=2)}</pre>"
    elif isinstance(content, str):
        # Convert markdown to HTML properly
        content_html = markdown.markdown(content, extensions=['nl2br', 'fenced_code'])
    elif content is None:
        content_html = "<em>(No content)</em>"
    else:
        content_html = markdown.markdown(str(content), extensions=['nl2br'])
    
    # Build role display text
    role_display = role.upper()
    if name:
        role_display += f" ({name})"
    if message_id:
        role_display += f" [ID: {message_id}]"
    
    # Special handling for system messages - make them collapsible
    if role.lower() == "system":
        with st.expander(f"🔧 {role_display}", expanded=False):
            st.markdown(f"""
            <div style="
                border-left: 4px solid {color};
                margin: 8px 0;
                background-color: #f8f9fa;
                padding: 12px;
                border-radius: 0 8px 8px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            ">
                <div style="
                    color: #666;
                    font-size: 14px;
                    font-weight: bold;
                    margin-bottom: 8px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                ">{role_display}</div>
                <div style="
                    color: #333;
                    line-height: 1.6;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                ">
                    {content_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Regular message display for non-system messages
        st.markdown(f"""
        <div style="
            border-left: 4px solid {color};
            margin: 8px 0;
            background-color: #f8f9fa;
            padding: 12px;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        ">
            <div style="
                color: #666;
                font-size: 14px;
                font-weight: bold;
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 1px;
            ">{role_display}</div>
            <div style="
                color: #333;
                line-height: 1.6;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            ">
                {content_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Handle special cases that need separate display
    if isinstance(content, dict) and len(json.dumps(content, indent=2)) > 500:
        # For very large JSON, also show expandable version
        with st.expander("View JSON in expandable format"):
            st.json(content)

def display_openai_conversation(conversation_data):
    """Display a full OpenAI format conversation"""
    
    if isinstance(conversation_data, list):
        # Handle list of messages (OpenAI format)
        for i, message in enumerate(conversation_data):
            if not isinstance(message, dict):
                continue
                
            role = message.get('role', 'unknown')
            content = message.get('content')
            name = message.get('name')
            message_id = message.get('id')
            
            display_openai_message(role, content, name, message_id)
            
            # Handle tool calls if present
            if 'tool_calls' in message:
                for j, tool_call in enumerate(message['tool_calls']):
                    st.markdown(f"""
                    <div style="
                        border-left: 4px solid #e67e22;
                        padding-left: 20px;
                        margin: 5px 0 5px 20px;
                        background-color: #fdf6e3;
                        padding: 10px;
                        border-radius: 0 5px 5px 0;
                    ">
                        <h5 style="
                            color: #666;
                            font-size: 12px;
                            font-weight: bold;
                            margin-bottom: 5px;
                            text-transform: uppercase;
                        ">TOOL CALL</h5>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display tool call details
                    tool_info = {
                        "function": tool_call.get('function', {}),
                        "id": tool_call.get('id', ''),
                        "type": tool_call.get('type', '')
                    }
                    st.json(tool_info)
            
            # Handle tool call responses
            if role == 'tool':
                tool_call_id = message.get('tool_call_id', 'Unknown')
                name = message.get('name', 'Unknown')
                st.caption(f"Tool: {name} | Call ID: {tool_call_id}")
    
    elif isinstance(conversation_data, str):
        # Handle string format - try to parse as JSON
        try:
            parsed = json.loads(conversation_data)
            display_openai_conversation(parsed)
        except json.JSONDecodeError:
            # Fallback to raw string display
            st.markdown(f"""
            <div style="
                border-left: 4px solid #95a5a6;
                margin: 8px 0;
                background-color: #f8f9fa;
                padding: 12px;
                border-radius: 0 8px 8px 0;
            ">
                <div style="
                    color: #333;
                    line-height: 1.6;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                ">
                    {markdown.markdown(conversation_data, extensions=['nl2br'])}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Fallback for other formats
        st.text(str(conversation_data))

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
        with st.expander(f"Example {i}: {row.get('id', 'Unknown')[:12]}...", expanded=i<=2):
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
            st.write("**💬 Conversation:**")
            display_openai_conversation(openai_response)
            
            # Display prompt
            st.write("**Prompt:**")
            st.text(prompt)
            
            # Display scores
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
                                if has_ci and show_confidence_intervals:
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
                    if show_confidence_intervals and has_any_ci:
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
                if show_confidence_intervals and has_any_ci:
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
                if show_examples:
                    st.subheader("View Examples")
                    cluster_names = [c['property_description'] for c in clusters[:top_n_clusters]]
                    selected_cluster = st.selectbox("Select cluster to view examples", cluster_names)
                    
                    if selected_cluster and st.button("Load Examples"):
                        show_cluster_examples(selected_cluster, selected_model, model_stats, 
                                            results_path, cluster_level)
    
    with tab3:
        st.header("Cluster Viewer")
        st.write("Explore hierarchical behavioral clusters with detailed property information")
        
        # Build hierarchical structure from clustered_df
        coarse_clusters = []
        fine_clusters_by_parent = {}
        
        # Get cluster information from clustered_df
        if ('property_description_fine_cluster_id' in clustered_df.columns and 
            'property_description_coarse_cluster_id' in clustered_df.columns):
            
            # Group by fine and coarse cluster IDs to build hierarchy
            cluster_groups = clustered_df.groupby([
                'property_description_fine_cluster_id', 
                'property_description_fine_cluster_label', 
                'property_description_coarse_cluster_id', 
                'property_description_coarse_cluster_label'
            ]).agg({
                'property_description': 'count',
                'id': 'count'  # Count of properties in this cluster
            }).reset_index()
            
            # Build coarse clusters
            coarse_cluster_data = {}
            for _, row in cluster_groups.iterrows():
                coarse_id = row['property_description_coarse_cluster_id']
                coarse_label = row['property_description_coarse_cluster_label']
                
                if coarse_id not in coarse_cluster_data:
                    coarse_cluster_data[coarse_id] = {
                        'id': coarse_id,
                        'label': coarse_label,
                        'size': 0,
                        'fine_clusters': []
                    }
                
                # Add fine cluster info
                fine_cluster = {
                    'id': row['property_description_fine_cluster_id'],
                    'label': row['property_description_fine_cluster_label'],
                    'size': row['property_description'],
                    'parent_id': coarse_id,
                    'parent_label': coarse_label,
                    'property_descriptions': []
                }
                
                coarse_cluster_data[coarse_id]['fine_clusters'].append(fine_cluster)
                coarse_cluster_data[coarse_id]['size'] += 1
            
            # Convert to lists and sort
            coarse_clusters = list(coarse_cluster_data.values())
            coarse_clusters.sort(key=lambda x: x['size'], reverse=True)
            
            # Build fine clusters by parent
            for coarse_cluster in coarse_clusters:
                fine_clusters_by_parent[coarse_cluster['id']] = coarse_cluster['fine_clusters']
                
                # Get property descriptions for each fine cluster
                for fine_cluster in coarse_cluster['fine_clusters']:
                    fine_cluster_data = clustered_df[
                        (clustered_df['property_description_fine_cluster_id'] == fine_cluster['id']) & 
                        (clustered_df['property_description_fine_cluster_label'] == fine_cluster['label'])
                    ]
                    fine_cluster['property_descriptions'] = fine_cluster_data['property_description'].unique().tolist()
        
        # Create two-column layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Cluster Viewer")
            
            # Track selected cluster
            if 'selected_cluster' not in st.session_state:
                st.session_state.selected_cluster = None
            
            # Get all fine clusters directly from clustered_df
            if 'property_description_fine_cluster_id' in clustered_df.columns:
                # Get unique fine clusters with their sizes
                fine_clusters_data = clustered_df.groupby([
                    'property_description_fine_cluster_id', 
                    'property_description_fine_cluster_label'
                ]).agg({
                    'property_description': 'count',
                    'id': 'count'
                }).reset_index()
                
                # Sort by size (largest first)
                fine_clusters_data = fine_clusters_data.sort_values('property_description', ascending=False)
                
                # Display fine clusters
                for _, row in fine_clusters_data.iterrows():
                    cluster_id = row['property_description_fine_cluster_id']
                    cluster_label = row['property_description_fine_cluster_label']
                    cluster_size = row['property_description']
                    
                    # Check if this cluster is selected
                    is_selected = (st.session_state.selected_cluster and 
                                 st.session_state.selected_cluster['id'] == cluster_id)
                    
                    # Create clickable button using the cluster name with size
                    button_text = f"{cluster_label} (Size: {cluster_size})"
                    
                    if st.button(
                        button_text,
                        key=f"fine_{cluster_id}",
                        help="Click to view details",
                        use_container_width=True
                    ):
                        # Get property descriptions for this cluster
                        cluster_data = clustered_df[
                            clustered_df['property_description_fine_cluster_id'] == cluster_id
                        ]
                        property_descriptions = cluster_data['property_description'].unique().tolist()
                        
                        # Create cluster object for selection
                        selected_cluster = {
                            'id': cluster_id,
                            'label': cluster_label,
                            'size': cluster_size,
                            'property_descriptions': property_descriptions
                        }
                        st.session_state.selected_cluster = selected_cluster
                
            else:
                st.error("No fine cluster data found. Please ensure the pipeline generated clustering results.")
        
        with col2:
            st.subheader("Cluster Details")
            
            # Add close button (X) in the header
            if st.button("✕", key="close_details", help="Close details"):
                st.session_state.selected_cluster = None
            
            if st.session_state.selected_cluster:
                cluster = st.session_state.selected_cluster
                
                # Main description with better styling
                st.markdown(f"""
                <div style="
                    padding: 12px;
                    margin: 8px 0;
                    background-color: #f8f9fa;
                    border-radius: 6px;
                    border-left: 4px solid #3182ce;
                ">
                    <div style="font-weight: 600; font-size: 16px; margin-bottom: 8px;">
                        {cluster['label']}
                    </div>
                    <div style="font-size: 14px; color: #666;">
                        <strong>Size:</strong> {cluster['size']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Parent information
                if cluster.get('parent_label'):
                    st.markdown(f"""
                    <div style="
                        padding: 8px 12px;
                        margin: 8px 0;
                        background-color: #f0f8ff;
                        border-radius: 4px;
                        font-size: 14px;
                    ">
                        <strong>Parent:</strong> {cluster['parent_label']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Property descriptions section
                st.subheader(f"Property Descriptions ({len(cluster['property_descriptions'])})")
                
                for i, desc in enumerate(cluster['property_descriptions']):
                    st.markdown(f"""
                    <div style="
                        padding: 10px 12px;
                        margin: 6px 0;
                        background-color: #f8f9fa;
                        border-radius: 6px;
                        border-left: 3px solid #3182ce;
                        font-size: 14px;
                        line-height: 1.4;
                    ">
                        {desc}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show examples if available
                if show_examples and cluster['property_descriptions']:
                    st.subheader("Examples")
                    st.info("Example viewing is enabled. Click on property descriptions above to see actual model responses.")
                else:
                    st.info("Select a cluster from the left panel to view details")
                
                # Show some statistics with better styling
                st.subheader("Cluster Statistics")
                
                total_clusters = len(clustered_df['property_description_fine_cluster_id'].unique()) if 'property_description_fine_cluster_id' in clustered_df.columns else 0
                total_properties = len(clustered_df)
                
                # Calculate min and max properties per cluster
                if 'property_description_fine_cluster_id' in clustered_df.columns:
                    cluster_sizes = clustered_df.groupby('property_description_fine_cluster_id').size()
                    min_properties = cluster_sizes.min() if not cluster_sizes.empty else 0
                    max_properties = cluster_sizes.max() if not cluster_sizes.empty else 0
                    global_cluster_count = cluster_sizes.sum() if not cluster_sizes.empty else 0
                else:
                    min_properties = 0
                    max_properties = 0
                    global_cluster_count = 0
                
                col1_stat, col2_stat, col3_stat = st.columns(3)
                with col1_stat:
                    st.metric("Total Fine Clusters", total_clusters)
                    st.metric("Total Properties", total_properties)
                with col2_stat:
                    st.metric("Min Properties/Cluster", min_properties)
                    st.metric("Max Properties/Cluster", max_properties)
                with col3_stat:
                    st.metric("Global Cluster Count", global_cluster_count)
                
                # Show largest clusters
                st.subheader("Largest Clusters")
                if 'property_description_fine_cluster_id' in clustered_df.columns:
                    largest_clusters = clustered_df.groupby(['property_description_fine_cluster_id', 'property_description_fine_cluster_label']).size().nlargest(5)
                    for (cluster_id, cluster_label), size in largest_clusters.items():
                        st.markdown(f"""
                        <div style="
                            padding: 8px 12px;
                            margin: 4px 0;
                            background-color: #f8f9fa;
                            border-radius: 4px;
                            font-size: 14px;
                        ">
                            <strong>{cluster_label}</strong> (Size: {size})
                        </div>
                        """, unsafe_allow_html=True)

    with tab4:
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
            st.subheader("📊 Cluster Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Clusters", total_clusters)
            with col2:
                avg_freq = clusters_df['frequency'].mean()
                st.metric("Avg Frequency", f"{avg_freq:.1f}%")
            with col3:
                max_freq = clusters_df['frequency'].max()
                st.metric("Max Frequency", f"{max_freq:.1f}%")
            with col4:
                total_models = clusters_df['model'].nunique()
                st.metric("Models", total_models)
            
            st.divider()
            
            # Show all clusters by default
            top_n_for_chart = total_clusters
            st.info(f"Showing all {total_clusters} clusters")
            
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
                                visible=show_confidence_intervals and has_any_ci,
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
                    if show_confidence_intervals and has_any_ci:
                        st.info("📊 **Error bars** show confidence intervals for frequency measurements. Wider bars indicate higher uncertainty in the frequency estimates.")
                        st.info("""
                        **Understanding the Confidence Intervals:**
                        
                        • **Frequency Chart Error Bars**: Based on distinctiveness score CIs, showing uncertainty in how much more/less frequently a model exhibits this behavior compared to the median model
                        
                        • **Quality Chart Error Bars**: Based on quality score CIs, showing uncertainty in how well the model performs in this cluster compared to its global average
                        
                        • **Different Metrics**: Frequency measures "how often" while quality measures "how well" - these are separate measurements with different confidence intervals
                        """)
                
                with col2:
                    # Create quality score chart
                    # Get quality scores for the same clusters
                    quality_data = []
                    for model_name, model_data in model_stats.items():
                        clusters = model_data.get(cluster_level, [])
                        for cluster in clusters:
                            if cluster['property_description'] in top_clusters:
                                quality_data.append({
                                    'property_description': cluster['property_description'],
                                    'model': model_name,
                                    'quality_score': extract_quality_score(cluster.get('quality_score', 0))
                                })
                    
                    if quality_data:
                        quality_df = pd.DataFrame(quality_data)
                        
                        # Create quality score chart
                        fig_quality = go.Figure()
                        
                        # Add a bar for each model
                        for i, model in enumerate(models):
                            model_quality_data = quality_df[quality_df['model'] == model]
                            
                            # Sort by cluster order (same as top_clusters)
                            model_quality_data = model_quality_data.set_index('property_description').reindex(top_clusters).reset_index()
                            
                            # Get confidence intervals for quality scores
                            quality_ci_lower = []
                            quality_ci_upper = []
                            for _, row in model_quality_data.iterrows():
                                # Find corresponding cluster data to get quality score CIs
                                cluster_data = chart_data[
                                    (chart_data['model'] == model) & 
                                    (chart_data['property_description'] == row['property_description'])
                                ]
                                if not cluster_data.empty:
                                    row = cluster_data.iloc[0]
                                    if row.get('has_quality_ci', False):
                                        # Extract quality score CI from model_stats
                                        for model_name, model_data in model_stats.items():
                                            if model_name == model:
                                                clusters = model_data.get(cluster_level, [])
                                                for cluster_stat in clusters:
                                                    if cluster_stat['property_description'] == row['property_description']:
                                                        quality_ci = cluster_stat.get('quality_score_ci', {})
                                                        if quality_ci:
                                                            # Get the first quality score CI available
                                                            for key, ci_bounds in quality_ci.items():
                                                                # Extract lower and upper bounds for error bars
                                                                if isinstance(ci_bounds, dict) and 'lower' in ci_bounds and 'upper' in ci_bounds:
                                                                    quality_ci_lower.append(ci_bounds['lower'])
                                                                    quality_ci_upper.append(ci_bounds['upper'])
                                                                else:
                                                                    quality_ci_lower.append(None)
                                                                    quality_ci_upper.append(None)
                                                                break
                                                            else:
                                                                quality_ci_lower.append(None)
                                                                quality_ci_upper.append(None)
                                                            break
                                                else:
                                                    quality_ci_lower.append(None)
                                                    quality_ci_upper.append(None)
                                                break
                                    else:
                                        quality_ci_lower.append(None)
                                        quality_ci_upper.append(None)
                                else:
                                    quality_ci_lower.append(None)
                                    quality_ci_upper.append(None)
                            
                            fig_quality.add_trace(go.Bar(
                                y=model_quality_data['property_description'],
                                x=model_quality_data['quality_score'],
                                name=model,
                                orientation='h',
                                marker_color=colors[i],
                                showlegend=False,  # Hide legend since it's shown in the main chart
                                error_x=dict(
                                    type='data',
                                    array=[u - l if u is not None and l is not None else None for l, u in zip(quality_ci_lower, quality_ci_upper)],
                                    arrayminus=[q - l if q is not None and l is not None else None for q, l in zip(model_quality_data['quality_score'], quality_ci_lower)],
                                    visible=show_confidence_intervals and has_any_ci,
                                    thickness=1,
                                    width=3,
                                    color='rgba(0,0,0,0.3)'
                                ),
                                hovertemplate='<b>%{y}</b><br>' +
                                            f'Model: {model}<br>' +
                                            'Quality Score: %{x:.3f}<br>' +
                                            'CI: %{customdata[0]}<extra></extra>',
                                customdata=[[
                                    format_confidence_interval(l, u) if l is not None and u is not None else "N/A"
                                    for l, u in zip(quality_ci_lower, quality_ci_upper)
                                ]]
                            ))
                        
                        # Update layout
                        fig_quality.update_layout(
                            title=f"Quality Scores",
                            xaxis_title="Quality Score",
                            yaxis_title="",  # No y-axis title to save space
                            barmode='group',
                            height=max(600, len(top_clusters) * 25),  # Same height as main chart
                            showlegend=False,
                            yaxis=dict(showticklabels=False)  # Hide y-axis labels to save space
                        )
                        
                        st.plotly_chart(fig_quality, use_container_width=True)
                        
                        # Add note about error bars if confidence intervals are shown
                        if show_confidence_intervals and has_any_ci:
                            st.info("📊 **Error bars** show confidence intervals for quality scores. Wider bars indicate higher uncertainty in the quality measurements.")
                            st.info("""
                            **Quality Score Confidence Intervals:**
                            
                            These error bars show uncertainty in how well each model performs in this cluster compared to its global average performance. A wider error bar means we're less certain about the model's relative performance in this specific behavioral cluster.
                            """)
                        
                    else:
                        st.info("No quality score data available")
                
                # Add some statistics
                st.subheader("Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_freq = chart_data['frequency'].mean()
                    st.metric("Average Frequency", f"{avg_freq:.1f}%")
                
                with col2:
                    max_freq = chart_data['frequency'].max()
                    st.metric("Highest Frequency", f"{max_freq:.1f}%")
                
                with col3:
                    total_clusters_shown = len(top_clusters)
                    st.metric("Clusters Shown", total_clusters_shown)
                
                # Add confidence interval statistics
                if 'has_ci' in chart_data.columns and show_confidence_intervals and has_any_ci:
                    ci_stats = chart_data['has_ci'].value_counts()
                    total_measurements = len(chart_data)
                    measurements_with_ci = ci_stats.get(True, 0)
                    ci_coverage = (measurements_with_ci / total_measurements) * 100 if total_measurements > 0 else 0
                    
                    st.subheader("Confidence Interval Coverage")
                    col1_ci, col2_ci, col3_ci = st.columns(3)
                    
                    with col1_ci:
                        st.metric("Measurements with CI", f"{measurements_with_ci}")
                    
                    with col2_ci:
                        st.metric("Total Measurements", f"{total_measurements}")
                    
                    with col3_ci:
                        st.metric("CI Coverage", f"{ci_coverage:.1f}%")
                    
                    if ci_coverage < 100:
                        st.info(f"⚠️ Only {ci_coverage:.1f}% of measurements have confidence intervals. To get full coverage, run the metrics stage with `compute_confidence_intervals=True`.")
                
                # Show detailed table
                st.subheader("Detailed Data")
                
                # Create combined table with both frequency and quality scores
                if quality_data:
                    # Create frequency pivot table
                    freq_df = chart_data.pivot(index='property_description', columns='model', values='frequency').fillna(0)
                    freq_df = freq_df.round(1)
                    
                    # Create quality score pivot table
                    quality_df_table = pd.DataFrame(quality_data)
                    quality_pivot = quality_df_table.pivot(index='property_description', columns='model', values='quality_score').fillna(0)
                    quality_pivot = quality_pivot.round(3)
                    
                    # Create confidence interval tables if available
                    ci_freq_data = []
                    ci_quality_data = []
                    
                    if show_confidence_intervals and has_any_ci:
                        for cluster in top_clusters:
                            ci_freq_row = {'Cluster Description': cluster}
                            ci_quality_row = {'Cluster Description': cluster}
                            
                            for model in models:
                                # Get CI data for this model and cluster
                                cluster_data = chart_data[
                                    (chart_data['model'] == model) & 
                                    (chart_data['property_description'] == cluster)
                                ]
                                
                                if not cluster_data.empty:
                                    row = cluster_data.iloc[0]
                                    if row.get('has_ci', False):
                                        ci_lower = row.get('ci_lower')
                                        ci_upper = row.get('ci_upper')
                                        if ci_lower is not None and ci_upper is not None:
                                            ci_freq_row[f'{model} (Freq CI)'] = format_confidence_interval({
                                                'lower': ci_lower, 
                                                'upper': ci_upper
                                            })
                                        else:
                                            ci_freq_row[f'{model} (Freq CI)'] = "N/A"
                                    else:
                                        ci_freq_row[f'{model} (Freq CI)'] = "N/A"
                                    
                                    if row.get('has_quality_ci', False):
                                        # Extract quality CI from model_stats
                                        for model_name, model_data in model_stats.items():
                                            if model_name == model:
                                                clusters = model_data.get(cluster_level, [])
                                                for cluster_stat in clusters:
                                                    if cluster_stat['property_description'] == cluster:
                                                        quality_ci = cluster_stat.get('quality_score_ci', {})
                                                        if quality_ci:
                                                            # Get the first quality score CI available
                                                            for key, ci_bounds in quality_ci.items():
                                                                # Extract lower and upper bounds for error bars
                                                                if isinstance(ci_bounds, dict) and 'lower' in ci_bounds and 'upper' in ci_bounds:
                                                                    quality_ci_lower.append(ci_bounds['lower'])
                                                                    quality_ci_upper.append(ci_bounds['upper'])
                                                                else:
                                                                    quality_ci_lower.append(None)
                                                                    quality_ci_upper.append(None)
                                                                break
                                                            else:
                                                                quality_ci_lower.append(None)
                                                                quality_ci_upper.append(None)
                                                            break
                                                else:
                                                    quality_ci_lower.append(None)
                                                    quality_ci_upper.append(None)
                                                break
                                    else:
                                        quality_ci_lower.append(None)
                                        quality_ci_upper.append(None)
                                else:
                                    ci_freq_row[f'{model} (Freq CI)'] = "N/A"
                                    ci_quality_row[f'{model} (Quality CI)'] = "N/A"
                            
                            ci_freq_data.append(ci_freq_row)
                            ci_quality_data.append(ci_quality_row)
                    
                    # Combine the tables with descriptive column names
                    combined_data = []
                    for cluster in top_clusters:
                        row_data = {'Cluster Description': cluster}
                        
                        # Add frequency data
                        for model in models:
                            freq_value = freq_df.loc[cluster, model] if cluster in freq_df.index else 0
                            row_data[f'{model} (Freq %)'] = freq_value
                        
                        # Add quality score data
                        for model in models:
                            quality_value = quality_pivot.loc[cluster, model] if cluster in quality_pivot.index else 0
                            row_data[f'{model} (Quality)'] = quality_value
                        
                        # Add confidence interval data if available
                        if show_confidence_intervals and has_any_ci:
                            for model in models:
                                # Find CI data for this model and cluster
                                ci_freq_row = next((row for row in ci_freq_data if row['Cluster Description'] == cluster), {})
                                ci_quality_row = next((row for row in ci_quality_data if row['Cluster Description'] == cluster), {})
                                
                                row_data[f'{model} (Freq CI)'] = ci_freq_row.get(f'{model} (Freq CI)', 'N/A')
                                row_data[f'{model} (Quality CI)'] = ci_quality_row.get(f'{model} (Quality CI)', 'N/A')
                        
                        combined_data.append(row_data)
                    
                    combined_df = pd.DataFrame(combined_data)
                    
                    st.dataframe(
                        combined_df,
                        use_container_width=True,
                        column_config={
                            'Cluster Description': st.column_config.TextColumn(
                                'Cluster Description',
                                width='large',
                                help="Full behavioral cluster description"
                            )
                        }
                    )
                else:
                    # Fallback to just frequency data if no quality data
                    display_df = chart_data.pivot(index='property_description', columns='model', values='frequency').fillna(0)
                    display_df = display_df.round(1)
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        column_config={
                            'property_description': st.column_config.TextColumn(
                                'Cluster Description',
                                width='large',
                                help="Full behavioral cluster description"
                            )
                        }
                    )
                
            else:
                st.warning("No data available for the selected clusters")
        else:
            st.warning("No cluster frequency data available")

    with tab5:
        st.header("🔎 Vector Search (does not work)")
        st.write("Search for behavioral properties using semantic similarity with vector embeddings.")
        
        # Initialize vector search engine
        @st.cache_resource
        def get_vector_search(results_path):
            try:
                return PropertyVectorSearch(results_path)
            except Exception as e:
                st.error(f"Failed to initialize vector search: {e}")
                return None
        
        vector_search = get_vector_search(results_path)
        
        if vector_search is None:
            st.error("Vector search is not available. Please ensure clustered_results.json exists.")
            st.stop()
        
        # Show search statistics
        stats = vector_search.get_statistics()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Properties", stats['total_properties'])
        with col2:
            st.metric("Total Conversations", stats['total_conversations'])
        with col3:
            st.metric("Unique Models", stats['unique_models'])
        with col4:
            st.metric("Unique Clusters", stats['unique_clusters'])
        
        st.divider()
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Search for behavioral properties",
                placeholder="e.g., 'step by step reasoning', 'creative responses', 'formal tone'...",
                help="Enter a description of the behavioral property you're looking for"
            )
        
        with col2:
            top_k = st.number_input("Max results", min_value=5, max_value=50, value=10)
            min_similarity = st.slider("Min similarity", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        
        # Model filter
        model_filter = st.multiselect(
            "Filter by models (optional)",
            all_models,
            help="Leave empty to search across all models"
        )
        
        # Search button
        search_button = st.button("🔍 Search", type="primary")
        
        if search_button and search_query:
            with st.spinner("Searching..."):
                try:
                    if model_filter:
                        # Search within specific models
                        results = vector_search.search_by_model(
                            search_query, model_filter, top_k=top_k, min_similarity=min_similarity
                        )
                    else:
                        # Search across all models
                        results = vector_search.search(
                            search_query, top_k=top_k, min_similarity=min_similarity
                        )
                    
                    if results:
                        st.success(f"Found {len(results)} relevant properties")
                        
                        # Display results
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Result {i}: {result.property_description[:80]}...", expanded=i<=3):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.markdown(f"**Property:** {result.property_description}")
                                    st.markdown(f"**Model:** {result.model}")
                                    st.markdown(f"**Cluster:** {result.cluster_label}")
                                    st.markdown(f"**Similarity:** {result.similarity_score:.3f}")
                                    
                                    if result.category:
                                        st.markdown(f"**Category:** {result.category}")
                                    if result.impact:
                                        st.markdown(f"**Impact:** {result.impact}")
                                    if result.type:
                                        st.markdown(f"**Type:** {result.type}")
                                
                                with col2:
                                    if result.evidence:
                                        st.markdown("**Evidence:**")
                                        st.text(result.evidence[:200] + "..." if len(result.evidence) > 200 else result.evidence)
                                
                                # Show examples directly if available
                                examples = vector_search.get_property_examples(result.property_description, max_examples=2)
                                if examples:
                                    st.markdown("**Example Conversations:**")
                                    for j, example in enumerate(examples, 1):
                                        with st.expander(f"Example {j}", expanded=False):
                                            st.markdown(f"**Question ID:** {example['question_id']}")
                                            st.markdown(f"**Model:** {example['model']}")
                                            st.markdown(f"**Score:** {example['score']}")
                                            
                                            if example['prompt']:
                                                st.markdown("**Prompt:**")
                                                st.text(example['prompt'][:300] + "..." if len(example['prompt']) > 300 else example['prompt'])
                                            
                                            if example['response']:
                                                st.markdown("**Response:**")
                                                st.text(example['response'][:300] + "..." if len(example['response']) > 300 else example['response'])
                                            
                                            if example['evidence']:
                                                st.markdown("**Evidence:**")
                                                st.text(example['evidence'])
                                else:
                                    st.info("No examples available for this property.")
                        
                        # Show summary statistics
                        st.subheader("Search Summary")
                        similarity_scores = [r.similarity_score for r in results]
                        avg_similarity = np.mean(similarity_scores)
                        max_similarity = max(similarity_scores)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Similarity", f"{avg_similarity:.3f}")
                        with col2:
                            st.metric("Max Similarity", f"{max_similarity:.3f}")
                        with col3:
                            st.metric("Results Found", len(results))
                        
                        # Similarity distribution
                        fig = px.histogram(
                            x=similarity_scores,
                            nbins=10,
                            title="Similarity Score Distribution",
                            labels={'x': 'Similarity Score', 'y': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.warning(f"No properties found matching '{search_query}' with similarity >= {min_similarity}")
                        st.info("Try:")
                        st.info("• Using different keywords")
                        st.info("• Lowering the similarity threshold")
                        st.info("• Checking spelling")
                
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.info("This might be due to missing embeddings. The system will compute them on first use.")
        
        elif search_query and not search_button:
            st.info("Click 'Search' to find relevant properties")
        
        # Show search tips
        with st.expander("💡 Search Tips", expanded=False):
            st.markdown("""
            **Effective search strategies:**
            
            • **Be specific:** Instead of "good responses", try "step-by-step explanations" or "creative problem solving"
            • **Use behavioral terms:** "formal tone", "technical accuracy", "user-friendly explanations"
            • **Combine concepts:** "detailed reasoning with examples", "concise but accurate responses"
            • **Try synonyms:** "thorough" instead of "detailed", "helpful" instead of "useful"
            
            **Understanding similarity scores:**
            • **0.9+**: Very similar properties
            • **0.7-0.9**: Related properties  
            • **0.5-0.7**: Somewhat related properties
            • **<0.5**: Weakly related (filtered out by default)
            """)

if __name__ == "__main__":
    main()
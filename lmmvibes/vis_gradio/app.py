"""
Main Gradio application for LMM-Vibes pipeline results visualization.

This module creates a comprehensive Gradio interface for exploring model performance,
cluster analysis, and detailed examples from pipeline output.
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import os

from .data_loader import (
    load_pipeline_results, 
    load_property_examples,
    scan_for_result_subfolders,
    validate_results_directory,
    get_available_models
)
from .utils import (
    compute_model_rankings,
    create_model_summary_card,
    format_cluster_dataframe,
    create_frequency_comparison_table,
    create_frequency_comparison_plots,
    search_clusters_by_text,
    get_top_clusters_for_model,
    create_interactive_cluster_viewer,
    get_cluster_statistics,
    get_unique_values_for_dropdowns,
    get_example_data,
    format_examples_display
)

# Global state for the app
app_state = {
    'clustered_df': None,
    'model_stats': None,
    'results_path': None,
    'available_models': [],
    'current_results_dir': None
}

# Global base results directory
BASE_RESULTS_DIR = 'results'


def load_data(results_dir: str) -> Tuple[str, str, str]:
    """Load pipeline results and update global state."""
    try:
        # Validate directory
        is_valid, error_msg = validate_results_directory(results_dir)
        if not is_valid:
            return "", f"❌ Error: {error_msg}", ""
        
        # Check for subfolders
        subfolders = scan_for_result_subfolders(results_dir)
        
        # If multiple options, use the first one for now (could be enhanced with selection)
        final_dir = results_dir
        if subfolders and "." not in subfolders:
            final_dir = str(Path(results_dir) / subfolders[0])
        
        # Load data
        clustered_df, model_stats, results_path = load_pipeline_results(final_dir)
        
        # Update global state
        app_state['clustered_df'] = clustered_df
        app_state['model_stats'] = model_stats
        app_state['results_path'] = results_path
        app_state['available_models'] = get_available_models(model_stats)
        app_state['current_results_dir'] = final_dir
        
        # Create summary
        n_models = len(model_stats)
        n_properties = len(clustered_df)
        
        summary = f"""
        ✅ **Successfully loaded pipeline results!**
        
        **Data Summary:**
        - **Models:** {n_models}
        - **Properties:** {n_properties:,}
        - **Results Directory:** {Path(final_dir).name}
        """
        
        if 'fine_cluster_id' in clustered_df.columns:
            n_fine_clusters = clustered_df['fine_cluster_id'].nunique()
            summary += f"\n- **Fine Clusters:** {n_fine_clusters}"
        
        if 'coarse_cluster_id' in clustered_df.columns:
            n_coarse_clusters = clustered_df['coarse_cluster_id'].nunique()
            summary += f"\n- **Coarse Clusters:** {n_coarse_clusters}"
        
        # Create models dropdown choices
        model_choices = app_state['available_models']
        models_info = f"Available models: {', '.join(model_choices)}"
        
        return summary, models_info, gr.update(choices=model_choices, value=model_choices[:5])
        
    except Exception as e:
        error_msg = f"❌ Error loading results: {str(e)}"
        return "", error_msg, gr.update(choices=[], value=[])


def create_overview(selected_models: List[str], cluster_level: str, top_n: int, 
                   score_significant_only: bool = False, quality_significant_only: bool = False,
                   sort_by: str = "score_desc") -> str:
    """Create model overview with summary cards."""
    if not app_state['model_stats']:
        return "Please load data first using the 'Load Data' tab."
    
    if not selected_models:
        return "Please select at least one model to display."
    
    # Compute rankings
    model_rankings = compute_model_rankings(app_state['model_stats'])
    
    # Filter rankings to selected models
    filtered_rankings = [(name, stats) for name, stats in model_rankings if name in selected_models]
    
    if not filtered_rankings:
        return "No data available for selected models."
    
    # Create overview content
    overview_html = """
    <div style="max-width: 1200px; margin: 0 auto;">
        <h2>🔍 Model Performance Overview</h2>
        <p style="color: #666; margin-bottom: 30px;">
            Top distinctive clusters where each model shows unique behavioral patterns.
            Frequency shows what percentage of a model's battles resulted in that behavioral pattern.
        </p>
    """
    
    # Create cards for each selected model
    for model_name, _ in filtered_rankings:
        card_html = create_model_summary_card(
            model_name, 
            app_state['model_stats'], 
            cluster_level, 
            top_n,
            score_significant_only=score_significant_only,
            quality_significant_only=quality_significant_only,
            sort_by=sort_by
        )
        overview_html += card_html
    
    overview_html += "</div>"
    
    return overview_html


def view_clusters_interactive(selected_models: List[str], cluster_level: str, 
                             search_term: str = "") -> str:
    """Display interactive cluster viewer with expandable cluster details."""
    if app_state['clustered_df'] is None:
        return "<p style='color: #e74c3c; padding: 20px;'>❌ Please load data first using the 'Load Data' tab</p>"
    
    df = app_state['clustered_df'].copy()
    
    # Apply search filter first
    if search_term and search_term.strip():
        df = search_clusters_by_text(df, search_term.strip())
    
    # Create interactive cluster viewer
    cluster_html = create_interactive_cluster_viewer(df, selected_models, cluster_level)
    
    # Add statistics summary at the top
    stats = get_cluster_statistics(df, selected_models)
    
    if not stats:
        return "<p style='color: #e74c3c; padding: 20px;'>❌ No cluster data available</p>"
    
    stats_html = f"""
    <div style="
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    ">
        <h3 style="margin: 0 0 15px 0;">Cluster Statistics</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            <div>
                <div style="font-size: 24px; font-weight: bold;">{stats['total_properties']:,}</div>
                <div style="opacity: 0.9;">Total Properties</div>
            </div>
            <div>
                <div style="font-size: 24px; font-weight: bold;">{stats['total_models']}</div>
                <div style="opacity: 0.9;">Models</div>
            </div>
    """
    
    if cluster_level == 'fine' and 'fine_clusters' in stats:
        stats_html += f"""
            <div>
                <div style="font-size: 24px; font-weight: bold;">{stats['fine_clusters']}</div>
                <div style="opacity: 0.9;">Fine Clusters</div>
            </div>
            <div>
                <div style="font-size: 24px; font-weight: bold;">{stats['avg_properties_per_fine_cluster']:.1f}</div>
                <div style="opacity: 0.9;">Avg Properties/Cluster</div>
            </div>
        """
    elif cluster_level == 'coarse' and 'coarse_clusters' in stats:
        stats_html += f"""
            <div>
                <div style="font-size: 24px; font-weight: bold;">{stats['coarse_clusters']}</div>
                <div style="opacity: 0.9;">Coarse Clusters</div>
            </div>
            <div>
                <div style="font-size: 24px; font-weight: bold;">{stats['avg_properties_per_coarse_cluster']:.1f}</div>
                <div style="opacity: 0.9;">Avg Properties/Cluster</div>
            </div>
        """
    
    stats_html += """
        </div>
    </div>
    """
    
    # Add filter info if applicable
    filter_info = ""
    if search_term and search_term.strip():
        filter_info += f"""
        <div style="
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 10px 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        ">
            🔍 <strong>Search Filter:</strong> "{search_term}"
        </div>
        """
    
    if selected_models:
        filter_info += f"""
        <div style="
            background: #f3e5f5;
            border-left: 4px solid #9c27b0;
            padding: 10px 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        ">
            🎯 <strong>Selected Models:</strong> {', '.join(selected_models)}
        </div>
        """
    
    return stats_html + filter_info + cluster_html


def view_clusters_table(selected_models: List[str], cluster_level: str, 
                       search_term: str = "", max_rows: int = 500) -> pd.DataFrame:
    """Display clusters in table format with filtering (for backward compatibility)."""
    if app_state['clustered_df'] is None:
        return pd.DataFrame({"Message": ["Please load data first using the 'Load Data' tab"]})
    
    df = app_state['clustered_df'].copy()
    
    # Apply search filter first
    if search_term and search_term.strip():
        df = search_clusters_by_text(df, search_term.strip())
    
    # Format for display
    formatted_df = format_cluster_dataframe(
        df, selected_models, cluster_level, max_rows
    )
    
    if formatted_df.empty:
        if search_term and search_term.strip():
            return pd.DataFrame({"Message": [f"No results found for search term '{search_term}'. Try a different search term."]})
        elif selected_models and len(selected_models) > 0:
            available_models = df['model'].unique().tolist() if 'model' in df.columns else []
            return pd.DataFrame({"Message": [f"No data found for selected models: {', '.join(selected_models)}. Available models: {', '.join(available_models)}"]})
        else:
            return pd.DataFrame({"Message": ["No data available. Please check your data files and try reloading."]})
    
    return formatted_df


def create_frequency_comparison(selected_models: List[str], cluster_level: str, 
                              top_n: int = 50, selected_model: str = None, 
                              selected_quality_metric: str = None) -> Tuple[pd.DataFrame, str]:
    """Create frequency comparison table and return it along with info text."""
    if not app_state['model_stats']:
        return pd.DataFrame({"Message": ["Please load data first"]}), ""
    
    if not selected_models:
        return pd.DataFrame({"Message": ["Please select at least one model"]}), ""
    
    # Handle "All Models" and "All Metrics" selections
    model_filter = None if selected_model == "All Models" else selected_model
    quality_filter = None if selected_quality_metric == "All Metrics" else selected_quality_metric
    
    comparison_df = create_frequency_comparison_table(
        app_state['model_stats'], selected_models, cluster_level, top_n, 
        model_filter, quality_filter
    )
    
    if comparison_df.empty:
        return pd.DataFrame({"Message": ["No data available for comparison"]}), "Rows: 0"
    
    info_text = f"**Displaying {len(comparison_df)} rows.**"
    
    return comparison_df, info_text

def create_frequency_plots(*args, **kwargs):
    """Placeholder function to avoid breaking existing event handlers. Plots are removed."""
    return None, None


def search_examples(search_term: str, selected_models: List[str], 
                   max_examples: int = 50) -> pd.DataFrame:
    """Search for specific examples in the dataset."""
    if app_state['clustered_df'] is None:
        return pd.DataFrame({"Message": ["Please load data first"]})
    
    if not search_term:
        return pd.DataFrame({"Message": ["Please enter a search term"]})
    
    # Search in the clustered dataframe
    search_results = search_clusters_by_text(
        app_state['clustered_df'], search_term, search_in='all'
    )
    
    # Filter by selected models if specified
    if selected_models:
        search_results = search_results[search_results['model'].isin(selected_models)]
    
    # Limit results
    search_results = search_results.head(max_examples)
    
    if search_results.empty:
        return pd.DataFrame({"Message": [f"No examples found matching '{search_term}'"]})
    
    # Select relevant columns for display
    display_cols = ['model', 'property_description', 'question_id']
    if 'fine_cluster_label' in search_results.columns:
        display_cols.append('fine_cluster_label')
    if 'score' in search_results.columns:
        display_cols.append('score')
    
    # Keep only existing columns
    available_cols = [col for col in display_cols if col in search_results.columns]
    return search_results[available_cols]


def debug_data_structure() -> str:
    """Show debug information about the loaded data structure."""
    if app_state['clustered_df'] is None:
        return "<p style='color: #e74c3c;'>❌ No data loaded</p>"
    
    df = app_state['clustered_df']
    
    # Get basic info
    n_rows = len(df)
    n_cols = len(df.columns)
    
    # Check for cluster columns
    has_fine_clusters = 'property_description_fine_cluster_id' in df.columns
    has_coarse_clusters = 'property_description_coarse_cluster_id' in df.columns
    
    # Get sample data
    sample_rows = min(3, len(df))
    sample_data = df.head(sample_rows).to_html(escape=False, classes="table table-striped", table_id="debug-table")
    
    html = f"""
    <div style="max-width: 1200px; margin: 0 auto;">
        <h3>🐛 Data Structure Debug Info</h3>
        
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h4>Basic Statistics</h4>
            <ul>
                <li><strong>Rows:</strong> {n_rows:,}</li>
                <li><strong>Columns:</strong> {n_cols}</li>
                <li><strong>Fine Clusters Available:</strong> {'✅ Yes' if has_fine_clusters else '❌ No'}</li>
                <li><strong>Coarse Clusters Available:</strong> {'✅ Yes' if has_coarse_clusters else '❌ No'}</li>
            </ul>
        </div>
        
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h4>Available Columns</h4>
            <div style="max-height: 200px; overflow-y: auto; background: white; padding: 10px; border-radius: 4px;">
                <ul>
    """
    
    for col in sorted(df.columns):
        unique_values = df[col].nunique() if df[col].dtype == 'object' else 'N/A'
        html += f"<li><code>{col}</code> - {df[col].dtype} (unique values: {unique_values})</li>"
    
    html += f"""
                </ul>
            </div>
        </div>
        
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h4>Sample Data (First {sample_rows} rows)</h4>
            <div style="max-height: 400px; overflow: auto; background: white; padding: 10px; border-radius: 4px;">
                {sample_data}
            </div>
        </div>
    </div>
    
    <style>
    #debug-table {{
        font-size: 12px;
        width: 100%;
    }}
    #debug-table th, #debug-table td {{
        padding: 4px 8px;
        border: 1px solid #ddd;
    }}
    #debug-table th {{
        background: #f1f1f1;
    }}
    </style>
    """
    
    return html


def get_dropdown_choices() -> Tuple[List[str], List[str], List[str]]:
    """Get choices for all dropdowns."""
    if app_state['clustered_df'] is None:
        return [], [], []
    
    choices = get_unique_values_for_dropdowns(app_state['clustered_df'])
    
    # Add "All" options at the beginning
    prompts = ["All Prompts"] + choices['prompts']
    models = ["All Models"] + choices['models'] 
    properties = ["All Clusters"] + choices['properties']
    
    return prompts, models, properties


def view_examples(selected_prompt: str, selected_model: str, selected_property: str, max_examples: int = 5, use_accordion: bool = True) -> str:
    """View individual examples based on filter criteria."""
    if app_state['clustered_df'] is None:
        return "<p style='color: #e74c3c; padding: 20px;'>❌ Please load data first using the 'Load Data' tab</p>"
    
    # Get examples based on filters
    examples = get_example_data(
        app_state['clustered_df'],
        selected_prompt if selected_prompt != "All Prompts" else None,
        selected_model if selected_model != "All Models" else None,
        selected_property if selected_property != "All Clusters" else None,
        max_examples
    )
    
    # Format for display
    return format_examples_display(
        examples, 
        selected_prompt, 
        selected_model, 
        selected_property,
        use_accordion=use_accordion
    )


def update_example_dropdowns() -> Tuple[Any, Any, Any]:
    """Update dropdown choices when data is loaded."""
    prompts, models, properties = get_dropdown_choices()
    
    return (
        gr.update(choices=prompts, value="All Prompts" if prompts else None),
        gr.update(choices=models, value="All Models" if models else None), 
        gr.update(choices=properties, value="All Clusters" if properties else None)
    )


def get_filter_options() -> Tuple[List[str], List[str]]:
    """Get available models and quality metrics for filtering."""
    if not app_state['model_stats']:
        return ["All Models"], ["All Metrics"]
    
    # Get available models
    available_models = ["All Models"] + list(app_state['model_stats'].keys())
    
    # Get available quality metrics
    quality_metrics = set()
    for model_data in app_state['model_stats'].values():
        clusters = model_data.get('fine', []) + model_data.get('coarse', [])
        for cluster in clusters:
            quality_score = cluster.get('quality_score', {})
            if isinstance(quality_score, dict):
                quality_metrics.update(quality_score.keys())
    
    available_metrics = ["All Metrics"] + sorted(list(quality_metrics))
    
    return available_models, available_metrics


def update_filter_dropdowns() -> Tuple[Any, Any]:
    """Update filter dropdown choices when data is loaded."""
    models, metrics = get_filter_options()
    
    return (
        gr.update(choices=models, value="All Models" if models else None),
        gr.update(choices=metrics, value="All Metrics" if metrics else None)
    )


def get_available_experiments(base_dir: str) -> List[str]:
    """Get list of available experiment directories."""
    if not base_dir or not os.path.exists(base_dir):
        return []
    
    experiments = []
    try:
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                # Check if this directory contains pipeline results
                if (os.path.exists(os.path.join(item_path, "model_stats.json")) or
                    os.path.exists(os.path.join(item_path, "clustered_results.jsonl"))):
                    experiments.append(item)
    except Exception as e:
        print(f"Error scanning experiments: {e}")
    
    return sorted(experiments)

def get_experiment_choices() -> List[str]:
    """Get choices for experiment dropdown."""
    if not BASE_RESULTS_DIR:
        return []
    
    experiments = get_available_experiments(BASE_RESULTS_DIR)
    return ["Select an experiment..."] + experiments

def refresh_experiment_dropdown() -> gr.update:
    """Refresh the experiment dropdown with current choices."""
    choices = get_experiment_choices()
    return gr.update(choices=choices, value="Select an experiment...")

def load_experiment_data(experiment_name: str) -> Tuple[str, str, str]:
    """Load data for a specific experiment."""
    if not BASE_RESULTS_DIR or experiment_name == "Select an experiment...":
        return "", "Please select a valid experiment", gr.update(choices=[], value=[])
    
    experiment_path = os.path.join(BASE_RESULTS_DIR, experiment_name)
    print(f"🔍 Loading experiment: {experiment_name} from {experiment_path}")
    return load_data(experiment_path)


def create_app() -> gr.Blocks:
    """Create the main Gradio application."""
    
    with gr.Blocks(title="LMM-Vibes Pipeline Results Explorer", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🔍 LMM-Vibes Pipeline Results Explorer
        **Comprehensive analysis of model behavioral properties and performance**
        
        Upload your pipeline results directory to explore model performance, cluster analysis, and detailed examples.
        """)
        
        with gr.Tabs():
            # Tab 1: Data Loading
            with gr.TabItem("📁 Load Data"):
                gr.Markdown("### Load Pipeline Results")
                if BASE_RESULTS_DIR:
                    gr.Markdown(f"**Base Results Directory:** `{BASE_RESULTS_DIR}`")
                    gr.Markdown("Select an experiment from the dropdown below to load its results.")
                else:
                    gr.Markdown("Provide the path to your pipeline results directory containing `model_stats.json` and `clustered_results.jsonl`")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        if BASE_RESULTS_DIR:
                            experiment_dropdown = gr.Dropdown(
                                label="Select Experiment",
                                choices=get_experiment_choices(),
                                value="Select an experiment...",
                                info="Choose an experiment to load its results"
                            )
                        else:
                            results_dir_input = gr.Textbox(
                                label="Results Directory Path",
                                placeholder="/path/to/your/results/directory",
                                info="Directory containing model_stats.json and clustered_results.jsonl"
                            )
                    with gr.Column(scale=1):
                        load_btn = gr.Button("Load Data", variant="primary")
                
                data_status = gr.Markdown("")
                models_info = gr.Markdown("")
                
                # Model selection (will be updated after loading)
                selected_models = gr.CheckboxGroup(
                    label="Select Models for Analysis",
                    choices=[],
                    value=[],
                    info="Choose which models to include in comparisons"
                )
            
            # Tab 2: Overview
            with gr.TabItem("📊 Overview"):
                with gr.Row():
                    cluster_level = gr.Radio(
                        label="Cluster Level",
                        choices=["fine", "coarse"],
                        value="fine",
                        info="Fine: detailed clusters, Coarse: high-level categories"
                    )
                    top_n_overview = gr.Slider(
                        label="Top N Clusters per Model",
                        minimum=1, maximum=10, value=3, step=1,
                        info="Number of top clusters to show per model"
                    )
                
                with gr.Row():
                    score_significant_only = gr.Checkbox(
                        label="Show Only Score Significant Clusters",
                        value=False,
                        info="Only show clusters where the distinctiveness score is statistically significant"
                    )
                    quality_significant_only = gr.Checkbox(
                        label="Show Only Quality Significant Clusters",
                        value=False,
                        info="Only show clusters where the quality score is statistically significant"
                    )
                
                with gr.Row():
                    sort_by = gr.Dropdown(
                        label="Sort Clusters By",
                        choices=[
                            ("Score (Descending)", "score_desc"),
                            ("Score (Ascending)", "score_asc"),
                            ("Quality (Ascending)", "quality_asc"),
                            ("Quality (Descending)", "quality_desc"),
                            ("Frequency (Descending)", "frequency_desc"),
                            ("Frequency (Ascending)", "frequency_asc")
                        ],
                        value="score_desc",
                        info="How to sort clusters within each model card"
                    )
                
                overview_display = gr.HTML(label="Model Overview")
                
                refresh_overview_btn = gr.Button("Refresh Overview")
            
            # Tab 3: View Clusters
            with gr.TabItem("📋 View Clusters"):
                gr.Markdown("### Interactive Cluster Viewer")
                gr.Markdown("Explore clusters with detailed property descriptions. Click on clusters to expand and view all properties within each cluster.")
                
                with gr.Row():
                    search_clusters = gr.Textbox(
                        label="Search Clusters",
                        placeholder="Search in cluster descriptions...",
                        info="Search for specific terms in cluster descriptions"
                    )
                
                clusters_display = gr.HTML(
                    label="Interactive Cluster Viewer",
                    value="<p style='color: #666; padding: 20px;'>Load data and select models to view clusters</p>"
                )
                
                refresh_clusters_btn = gr.Button("Refresh Clusters")
            
            # Tab 4: View Examples
            with gr.TabItem("📋 View Examples"):
                gr.Markdown("### Individual Example Viewer")
                gr.Markdown("Explore individual examples with full prompts, model responses, and property information. Click on examples to expand and view full details.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        example_prompt_dropdown = gr.Dropdown(
                            label="Select Prompt",
                            choices=["All Prompts"],
                            value="All Prompts",
                            info="Choose a specific prompt or 'All Prompts'"
                        )
                    with gr.Column(scale=1):
                        example_model_dropdown = gr.Dropdown(
                            label="Select Model", 
                            choices=["All Models"],
                            value="All Models",
                            info="Choose a specific model or 'All Models'"
                        )
                    with gr.Column(scale=1):
                        example_property_dropdown = gr.Dropdown(
                            label="Select Cluster (Optional)",
                            choices=["All Clusters"],
                            value="All Clusters", 
                            info="Choose a specific cluster or 'All Clusters'"
                        )
                
                with gr.Row():
                    max_examples_slider = gr.Slider(
                        label="Max Examples",
                        minimum=1, maximum=20, value=5, step=1,
                        info="Maximum number of examples to display"
                    )
                    use_accordion_checkbox = gr.Checkbox(
                        label="Use Accordion for System/Info Messages",
                        value=True,
                        info="Group system and info messages in collapsible sections"
                    )
                    view_examples_btn = gr.Button("View Examples", variant="primary")
                
                examples_display = gr.HTML(
                    label="Examples",
                    value="<p style='color: #666; padding: 20px;'>Load data and select filters to view examples</p>"
                )
            
            # Tab 5: Frequency Comparison
            with gr.TabItem("📈 Frequency Comparison"):
                gr.Markdown("### Model Frequency Comparison")
                gr.Markdown("Compare how frequently each model exhibits behaviors in different clusters")
                
                with gr.Row():
                    cluster_level_freq = gr.Radio(
                        label="Cluster Level",
                        choices=["fine", "coarse"],
                        value="fine"
                    )
                    top_n_freq = gr.Slider(
                        label="Top N Clusters",
                        minimum=5, maximum=200, value=50, step=5
                    )
                
                with gr.Row():
                    selected_model_filter = gr.Dropdown(
                        label="Filter by Specific Model (Optional)",
                        choices=["All Models"],
                        value="All Models",
                        info="Select a specific model to show only its data, or 'All Models' for aggregated view"
                    )
                    selected_quality_metric = gr.Dropdown(
                        label="Filter by Quality Metric (Optional)",
                        choices=["All Metrics"],
                        value="All Metrics",
                        info="Select a specific quality metric to show only its data, or 'All Metrics' for aggregated view"
                    )
                
                refresh_freq_btn = gr.Button("Refresh Comparison", variant="primary")
                
                frequency_table_info = gr.Markdown("")
                
                # Frequency table first
                frequency_table = gr.Dataframe(
                    label="Frequency Comparison Table",
                    interactive=False,
                    wrap=True,
                    max_height=800,
                    elem_classes=["frequency-comparison-table"]
                )
                
                # Plots section has been removed
                
                # Add CSS styling for the frequency table
                gr.HTML("""
                <style>
                /* Make the actual table use fixed layout to avoid reflow jitter */
                .frequency-comparison-table table,
                div[data-testid="dataframe"] table {
                    table-layout: fixed !important;
                    width: 100% !important;
                    border-collapse: collapse !important;
                }

                .frequency-comparison-table th,
                .frequency-comparison-table td {
                    overflow: hidden !important;
                    text-overflow: ellipsis !important;
                    white-space: nowrap !important;
                }
                
                /* Make cluster column (first column) wider with better text wrapping */
                .frequency-comparison-table table th:first-child,
                .frequency-comparison-table table td:first-child,
                div[data-testid="dataframe"] table th:first-child,
                div[data-testid="dataframe"] table td:first-child {
                    min-width: 250px !important;
                    max-width: 400px !important;
                    white-space: normal !important;
                    word-wrap: break-word !important;
                    word-break: break-word !important;
                    overflow-wrap: break-word !important;
                }
                
                /* Ensure horizontal scrolling works properly for all table containers */
                .frequency-comparison-table,
                div[data-testid="dataframe"] {
                    overflow: auto !important;
                    max-height: 800px !important;
                    border: 1px solid #ddd !important;
                    border-radius: 4px !important;
                    position: relative !important;
                }
                
                /* Remove the scroll indicator which can cause jitter */
                .frequency-comparison-table::after,
                div[data-testid="dataframe"]::after {
                    display: none !important;
                }
                
                /* Ensure the dataframe component itself allows scrolling */
                div[data-testid="dataframe"] {
                    max-width: 100% !important;
                    width: 100% !important;
                }
                
                /* Override any Gradio default styles that might interfere */
                .frequency-comparison-table *,
                div[data-testid="dataframe"] * {
                    box-sizing: border-box !important;
                }
                </style>
                
                <script>
                // JavaScript to stabilize table layout and prevent jitter
                function stabilizeTableLayout() {
                    const tables = document.querySelectorAll('.frequency-comparison-table, div[data-testid="dataframe"]');
                    
                    tables.forEach(container => {
                        const table = container.querySelector('table');
                        if (!table) return;

                        // Set a fixed table layout to prevent column widths from changing during scroll
                        table.style.tableLayout = 'fixed';
                        table.style.width = '100%';

                        // Ensure the container is set to scroll
                        container.style.overflow = 'auto';
                    });
                }
                
                // Run on initial load
                document.addEventListener('DOMContentLoaded', stabilizeTableLayout);
                
                // Re-run whenever Gradio updates the component
                const observer = new MutationObserver((mutations) => {
                    for (const mutation of mutations) {
                        if (mutation.type === 'childList' || mutation.type === 'subtree') {
                            stabilizeTableLayout();
                        }
                    }
                });

                function setupObserver() {
                    const tableContainer = document.querySelector('.frequency-comparison-table, div[data-testid="dataframe"]');
                    if (tableContainer) {
                        observer.observe(tableContainer, {
                            childList: true,
                            subtree: true,
                        });
                    }
                }

                document.addEventListener('DOMContentLoaded', setupObserver);

                // Fallback interval to catch dynamic updates
                setInterval(stabilizeTableLayout, 1000);
                </script>
                """)
                
                # Remove duplicate elements
                # frequency_plots = gr.Plot(
                #     label="Frequency Comparison Plots",
                #     visible=False # Initially hidden
                # )
                # 
                # refresh_freq_btn = gr.Button("Refresh Comparison")
            
            # Tab 6: Search Examples  
            with gr.TabItem("🔎 Search Examples"):
                with gr.Row():
                    search_term = gr.Textbox(
                        label="Search Term",
                        placeholder="Enter search term...",
                        info="Search across all text fields"
                    )
                    max_examples = gr.Slider(
                        label="Max Examples",
                        minimum=10, maximum=200, value=50, step=10
                    )
                
                examples_table = gr.Dataframe(
                    label="Search Results",
                    interactive=False,
                    wrap=True
                )
                
                search_btn = gr.Button("Search", variant="primary")
            
            # Tab 7: Debug Data
            with gr.TabItem("🐛 Debug Data"):
                gr.Markdown("### Data Structure Debug")
                gr.Markdown("If tables aren't loading correctly, use this tab to inspect your data structure and identify issues.")
                
                debug_display = gr.HTML(
                    label="Debug Information", 
                    value="<p style='color: #666; padding: 20px;'>Load data to see debug information</p>"
                )
                
                debug_btn = gr.Button("Show Debug Info", variant="secondary")
        
        # Event handlers
        if BASE_RESULTS_DIR:
            # Use dropdown for experiment selection
            if 'experiment_dropdown' in locals():
                (experiment_dropdown.change(
                    fn=load_experiment_data,
                    inputs=[experiment_dropdown],
                    outputs=[data_status, models_info, selected_models]
                ).then(
                    fn=update_example_dropdowns,
                    outputs=[example_prompt_dropdown, example_model_dropdown, example_property_dropdown]
                ).then(
                    fn=update_filter_dropdowns,
                    outputs=[selected_model_filter, selected_quality_metric]
                ).then(
                    fn=create_frequency_comparison,
                    inputs=[selected_models, cluster_level_freq, top_n_freq, selected_model_filter, selected_quality_metric],
                    outputs=[frequency_table, frequency_table_info]
                ))
        else:
            # Use textbox for manual path entry
            if 'load_btn' in locals() and 'results_dir_input' in locals():
                (load_btn.click(
                    fn=load_data,
                    inputs=[results_dir_input],
                    outputs=[data_status, models_info, selected_models]
                ).then(
                    fn=update_example_dropdowns,
                    outputs=[example_prompt_dropdown, example_model_dropdown, example_property_dropdown]
                ).then(
                    fn=update_filter_dropdowns,
                    outputs=[selected_model_filter, selected_quality_metric]
                ).then(
                    fn=create_frequency_comparison,
                    inputs=[selected_models, cluster_level_freq, top_n_freq, selected_model_filter, selected_quality_metric],
                    outputs=[frequency_table, frequency_table_info]
                ))
        
        refresh_overview_btn.click(
            fn=create_overview,
            inputs=[selected_models, cluster_level, top_n_overview, score_significant_only, quality_significant_only, sort_by],
            outputs=[overview_display]
        )
        
        refresh_clusters_btn.click(
            fn=view_clusters_interactive,
            inputs=[selected_models, cluster_level, search_clusters],
            outputs=[clusters_display]
        )
        
        # View Examples handlers
        view_examples_btn.click(
            fn=view_examples,
            inputs=[example_prompt_dropdown, example_model_dropdown, example_property_dropdown, max_examples_slider, use_accordion_checkbox],
            outputs=[examples_display]
        )
        
        # Auto-refresh examples when dropdowns change
        example_prompt_dropdown.change(
            fn=view_examples,
            inputs=[example_prompt_dropdown, example_model_dropdown, example_property_dropdown, max_examples_slider, use_accordion_checkbox],
            outputs=[examples_display]
        )
        
        example_model_dropdown.change(
            fn=view_examples,
            inputs=[example_prompt_dropdown, example_model_dropdown, example_property_dropdown, max_examples_slider, use_accordion_checkbox],
            outputs=[examples_display]
        )
        
        example_property_dropdown.change(
            fn=view_examples,
            inputs=[example_prompt_dropdown, example_model_dropdown, example_property_dropdown, max_examples_slider, use_accordion_checkbox],
            outputs=[examples_display]
        )
        
        # Frequency Tab Handlers
        freq_inputs = [selected_models, cluster_level_freq, top_n_freq, selected_model_filter, selected_quality_metric]
        freq_outputs = [frequency_table, frequency_table_info]

        refresh_freq_btn.click(fn=create_frequency_comparison, inputs=freq_inputs, outputs=freq_outputs)
        cluster_level_freq.change(fn=create_frequency_comparison, inputs=freq_inputs, outputs=freq_outputs)
        top_n_freq.change(fn=create_frequency_comparison, inputs=freq_inputs, outputs=freq_outputs)
        selected_models.change(fn=create_frequency_comparison, inputs=freq_inputs, outputs=freq_outputs)
        selected_model_filter.change(fn=create_frequency_comparison, inputs=freq_inputs, outputs=freq_outputs)
        selected_quality_metric.change(fn=create_frequency_comparison, inputs=freq_inputs, outputs=freq_outputs)
        
        search_btn.click(
            fn=search_examples,
            inputs=[search_term, selected_models, max_examples],
            outputs=[examples_table]
        )
        
        debug_btn.click(
            fn=debug_data_structure,
            outputs=[debug_display]
        )
        
        # Auto-refresh on model selection change
        selected_models.change(
            fn=create_overview,
            inputs=[selected_models, cluster_level, top_n_overview, score_significant_only, quality_significant_only, sort_by],
            outputs=[overview_display]
        )
        
        # Auto-refresh on significance filter changes
        score_significant_only.change(
            fn=create_overview,
            inputs=[selected_models, cluster_level, top_n_overview, score_significant_only, quality_significant_only, sort_by],
            outputs=[overview_display]
        )
        
        quality_significant_only.change(
            fn=create_overview,
            inputs=[selected_models, cluster_level, top_n_overview, score_significant_only, quality_significant_only, sort_by],
            outputs=[overview_display]
        )
        
        # Auto-refresh on sort dropdown change
        sort_by.change(
            fn=create_overview,
            inputs=[selected_models, cluster_level, top_n_overview, score_significant_only, quality_significant_only, sort_by],
            outputs=[overview_display]
        )
        
        # Auto-refresh on cluster level change
        cluster_level.change(
            fn=create_overview,
            inputs=[selected_models, cluster_level, top_n_overview, score_significant_only, quality_significant_only, sort_by],
            outputs=[overview_display]
        )
        
        # Auto-refresh on top N change
        top_n_overview.change(
            fn=create_overview,
            inputs=[selected_models, cluster_level, top_n_overview, score_significant_only, quality_significant_only, sort_by],
            outputs=[overview_display]
        )
        
        selected_models.change(
            fn=view_clusters_interactive,
            inputs=[selected_models, cluster_level, search_clusters],
            outputs=[clusters_display]
        )
        
        # Auto-refresh clusters when search term changes (with debouncing)
        search_clusters.change(
            fn=view_clusters_interactive,
            inputs=[selected_models, cluster_level, search_clusters],
            outputs=[clusters_display]
        )
    
    return app


def launch_app(results_dir: Optional[str] = None, 
               share: bool = False,
               server_name: str = "127.0.0.1",
               server_port: int = 7860,
               **kwargs) -> None:
    """Launch the Gradio application.
    
    Args:
        results_dir: Optional path to base results directory containing experiment subfolders
        share: Whether to create a public link
        server_name: Server address
        server_port: Server port
        **kwargs: Additional arguments for gr.Blocks.launch()
    """
    global BASE_RESULTS_DIR
    
    # Set the global base results directory
    if results_dir:
        BASE_RESULTS_DIR = results_dir
        print(f"📁 Base results directory set to: {results_dir}")
        
        # Check if it's a valid directory
        if not os.path.exists(results_dir):
            print(f"⚠️  Warning: Base results directory does not exist: {results_dir}")
            BASE_RESULTS_DIR = None
        else:
            # Scan for available experiments
            experiments = get_available_experiments(results_dir)
            print(f"🔍 Found {len(experiments)} experiments: {experiments}")
    
    app = create_app()
    
    # Auto-load data if results_dir is provided and contains a single experiment
    if results_dir and os.path.exists(results_dir):
        experiments = get_available_experiments(results_dir)
        if len(experiments) == 1:
            # Auto-load the single experiment
            experiment_path = os.path.join(results_dir, experiments[0])
            try:
                clustered_df, model_stats, results_path = load_pipeline_results(experiment_path)
                app_state['clustered_df'] = clustered_df
                app_state['model_stats'] = model_stats
                app_state['results_path'] = results_path
                app_state['available_models'] = get_available_models(model_stats)
                app_state['current_results_dir'] = experiment_path
                print(f"✅ Auto-loaded data from: {experiment_path}")
            except Exception as e:
                print(f"❌ Failed to auto-load data: {e}")
        elif len(experiments) > 1:
            print(f"📋 Multiple experiments found. Please select one from the dropdown.")
    
    print(f"🚀 Launching Gradio app on {server_name}:{server_port}")
    print(f"Share mode: {share}")
    print(f"🔧 Additional kwargs: {kwargs}")
    
    try:
        app.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True,  # Show detailed error messages
            quiet=False,  # Show more verbose output
            **kwargs
        )
    except Exception as e:
        print(f"❌ Failed to launch on port {server_port}: {e}")
        print("🔄 Trying alternative port configuration...")
        
        # Try with a port range instead of port 0
        try:
            # Try ports in a reasonable range
            for alt_port in [8080, 8081, 8082, 8083, 8084, 8085, 8086, 8087, 8088, 8089]:
                try:
                    print(f"🔄 Trying port {alt_port}...")
                    app.launch(
                        share=share,
                        server_name=server_name,
                        server_port=alt_port,
                        show_error=True,
                        quiet=False,
                        **kwargs
                    )
                    break  # If successful, break out of the loop
                except Exception as port_error:
                    if "Cannot find empty port" in str(port_error):
                        print(f"   Port {alt_port} is busy, trying next...")
                        continue
                    else:
                        raise port_error
            else:
                # If we get here, all ports in our range were busy
                raise Exception("All attempted ports (8080-8089) are busy")
                
        except Exception as e2:
            print(f"❌ Failed to launch with alternative ports: {e2}")
            print("💡 Try specifying a different port manually:")
            print(f"   python -m lmmvibes.vis_gradio.launcher --port 9000")
            print(f"   python -m lmmvibes.vis_gradio.launcher --auto_port")
            raise e2 
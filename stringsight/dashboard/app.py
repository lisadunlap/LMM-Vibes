"""
Main Gradio application for LMM-Vibes pipeline results visualization.

This module creates a comprehensive Gradio interface for exploring model performance,
cluster analysis, and detailed examples from pipeline output.
"""

import gradio as gr
from gradio.themes import Soft
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
from .metrics_adapter import get_all_models
from .utils import (
    compute_model_rankings,
    create_model_summary_card,
    format_cluster_dataframe,

    search_clusters_by_text,
    get_top_clusters_for_model,
    create_interactive_cluster_viewer,
    get_cluster_statistics,
    get_unique_values_for_dropdowns,
    get_example_data,
    format_examples_display,
    get_total_clusters_count
)

# ---------------------------------------------------------------------------
# NEW: centralised state + logic split into per-tab modules
# ---------------------------------------------------------------------------
from .state import app_state

# Tab-specific logic (moved out of this file)
from .load_data_tab import (
    load_data,
    get_available_experiments,
    get_experiment_choices,
    refresh_experiment_dropdown,
    load_experiment_data,
)
from .overview_tab import create_overview, create_model_quality_plot, create_model_quality_table, get_available_model_quality_metrics
from .clusters_tab import view_clusters_interactive, view_clusters_table
from .examples_tab import (
    get_dropdown_choices,
    update_example_dropdowns,
    view_examples,
)
from .plots_tab import create_plots_tab, create_plot_with_toggle, update_quality_metric_visibility, update_cluster_selection, get_available_quality_metrics
from .run_pipeline_tab import create_run_pipeline_tab

# app_state now comes from dashboard.state


def update_top_n_slider_maximum():
    """Update the top N slider maximum based on total clusters in loaded data."""
    from .state import app_state
    
    if not app_state.get("metrics"):
        return gr.Slider(minimum=1, maximum=10, value=3, step=1)
    
    total_clusters = get_total_clusters_count(app_state["metrics"])
    max_value = max(10, total_clusters)  # At least 10, or total clusters if more
    
    return gr.Slider(
        label="Top N Clusters per Model",
        minimum=1, 
        maximum=max_value, 
        value=min(3, max_value), 
        step=1,
        info=f"Number of top clusters to show per model (max: {total_clusters})"
    )


def clear_search_bars():
    """Clear all search bars when new data is loaded."""
    return "", ""  # Returns empty strings for search_clusters and search_examples


def create_app() -> gr.Blocks:
    """Create the main Gradio application."""
    
    # Custom CSS for minimal margins and better sidebar layout + polished header/tabs
    custom_css = """
    /* Ensure the app itself spans the full page width (inside shadow root) */
    :host {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        /* Override Gradio's layout max width if present */
        --layout-max-width: 100% !important;
    }
    /* Base font stack for broad compatibility */
    body, .gradio-container { 
        font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif; 
    }
    /* Ensure Examples tab inherits same font (avoid code blocks) */
    #examples-container, #examples-container *:not(code):not(pre) {
        font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif !important;
    }
    
    /* Universal reset for all elements */
    * {
        box-sizing: border-box !important;
    }
    
    .main-container {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 !important;
        padding: 5px 0 0 8px !important;
    }
    .gradio-container {
        width: 100% !important;
        max-width: none !important;
        margin: 0 !important;
        padding: 5px 0 0 8px !important;
    }
    /* --- Polished sticky header --- */
    #app-header { 
        position: sticky; 
        top: 0; 
        z-index: 50;
        backdrop-filter: saturate(180%) blur(8px);
        -webkit-backdrop-filter: saturate(180%) blur(8px);
        background: rgba(255,255,255,.85);
        border-bottom: 1px solid rgba(15,23,42,.06);
        padding: 12px 16px;
        margin: 0 0 8px 0 !important;
        display: flex; 
        align-items: center; 
        justify-content: space-between; 
        width: 100%;
    }
    .brand { display:flex; align-items:center; gap:10px; font-weight:600; font-size:18px; color:#0f172a; }
    .brand small { font-weight:500; color:#64748b; }
    .header-right { display:flex; gap:8px; align-items:center; margin-left:auto; }
    /* Ensure the right group actually sticks to the right */
    #app-header > *:last-child { margin-left: auto !important; }
    #app-header .header-right { margin-left: auto !important; justify-content: flex-end !important; }
    #app-header .header-right > * { margin-left: 0 !important; }
    .header-badge { background:#eef2ff; color:#3730a3; border-radius:9999px; padding:2px 8px; font-size:12px; border:1px solid #c7d2fe; }
    /* Round the tab buttons into pills with clear active state */
    .tabs .tab-nav button { border-radius:9999px !important; padding:6px 12px !important; }
    .tabs .tab-nav button.selected { background:#eef2ff !important; color:#3730a3 !important; }
    /* Tone down color for model selection group (Gradio renders as pill labels) */
    #selected-models label { background: #f8fafc !important; color: #111827 !important; border: 1px solid #e2e8f0 !important; }
    #selected-models label:hover { background: #f1f5f9 !important; }
    #selected-models .selected, #selected-models [data-selected="true"],
    #selected-models label[aria-pressed="true"],
    #selected-models label:has(input:checked) { background: #f1f5f9 !important; border-color: #e2e8f0 !important; color: #111827 !important; }
    #selected-models input[type="checkbox"] { accent-color: #94a3b8 !important; }
    /* Help panel card */
    #help-panel { margin: 8px 12px; padding: 12px; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; }
    #help-panel .gr-prose, #help-panel .prose, #help-panel .markdown, #help-panel p, #help-panel div { background: #ffffff !important; }
    /* Style the Close button with a light tint */
    #help-close-btn button { background: #eef2ff !important; color: #3730a3 !important; border: 1px solid #c7d2fe !important; }
    #help-close-btn button:hover { background: #e0e7ff !important; }
    /* Compact Help button */
    #help-btn { flex: 0 0 auto !important; width: auto !important; display: inline-flex !important; }
    #help-btn button { padding: 2px 8px !important; min-width: unset !important; width: auto !important; }
    
    .tabs {
        margin: 0 !important;
        padding: 0 !important;
    }
    .tab-nav {
        margin: 0 !important;
        padding: 0 !important;
    }
    .tab-content {
        margin: 0 !important;
        padding: 5px 0 2px 8px !important;
    }
    .sidebar {
        border-left: 1px solid #e0e0e0;
        background-color: #f8f9fa;
        padding: 8px !important;
        order: 2;
    }
    .main-content {
        padding: 5px 0 2px 8px !important;
        order: 1;
    }
    /* Additional selectors to override Gradio's default margins */
    .block {
        margin: 0 !important;
        padding: 2px 0 2px 8px !important;
    }
    .form {
        margin: 0 !important;
        padding: 0 !important;
    }
    body {
        margin: 0 !important;
        padding: 5px 0 0 8px !important;
    }
    .app {
        margin: 0 !important;
        padding: 5px 0 0 8px !important;
    }
    /* Target specific Gradio container classes */
    .gradio-row {
        margin: 0 !important;
        padding: 0 !important;
    }
    .gradio-column {
        margin: 0 !important;
        padding: 0 0 0 8px !important;
    }
    /* Override any container padding */
    .container {
        width: 100% !important;
        max-width: none !important;
        padding: 5px 0 0 8px !important;
        margin: 0 !important;
    }
    /* Target the root element */
    #root {
        padding: 5px 0 0 8px !important;
        margin: 0 !important;
    }
    /* Make sure no right padding on wrapper elements */
    .wrap {
        width: 100% !important;
        max-width: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    /* Aggressive targeting of common Gradio elements */
    div[class*="gradio"] {
        padding-right: 0 !important;
        margin-right: 0 !important;
    }
    /* Target any div that might have padding */
    .gradio-blocks > div,
    .gradio-blocks div[style*="padding"] {
        padding-right: 0 !important;
        margin-right: 0 !important;
    }
    /* Ensure content fills width */
    .gradio-blocks {
        width: 100% !important;
        max-width: none !important;
        padding: 5px 0 0 8px !important;
        margin: 0 !important;
    }
    
    /* Catch-all: remove max-width and auto-centering from any container-like nodes */
    [class*="container"], [class*="Container"], [class*="main"], [class*="Main"], [class*="block"], [class*="Block"] {
        max-width: none !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
    }

    /* Slight right margin for overall app */
    .gradio-container {
        margin-right: 12px !important;
    }

    /* Ensure slight right padding inside the app content */
    .main-container,
    .gradio-blocks,
    .tab-content,
    .main-content,
    .container,
    #root,
    .app,
    .wrap,
    .gradio-column {
        padding-right: 12px !important;
    }

    /* Final override: ensure host has slight right padding so it's always visible */
    :host {
        padding-right: 12px !important;
    }
    """

    # Modern theme setup (Inter font, neutral slate, indigo primary)
    theme = Soft(
        primary_hue="indigo",
        neutral_hue="slate",
    )
    
    with gr.Blocks(title="LMM-Vibes Pipeline Results Explorer", theme=theme, css=custom_css, fill_width=True) as app:
        # Header helpers
        def _current_experiment_name() -> str:
            from .state import app_state
            from . import state
            path = app_state.get("current_results_dir") or state.BASE_RESULTS_DIR or ""
            if not path:
                return "No experiment loaded"
            try:
                return Path(path).name
            except Exception:
                return str(path)

        def _render_badge_html() -> str:
            exp = _current_experiment_name()
            return f"<span class=\"header-badge\">{exp}</span>"

        # Polished sticky header
        with gr.Row(elem_id="app-header"):
            with gr.Row(elem_classes=["header-left"]):
                gr.HTML(
                    value=(
                        "<div class=\"brand\">🧵 StringSight <small>Evaluation Console</small></div>"
                    )
                )
                # Move experiment selection to the header when a base directory is provided
                from . import state
                if state.BASE_RESULTS_DIR:
                    experiment_dropdown = gr.Dropdown(
                        label="Select Experiment",
                        choices=get_experiment_choices(),
                        value="Select an experiment...",
                        show_label=False,
                        interactive=True,
                    )
            with gr.Row(elem_classes=["header-right"]):
                help_btn = gr.Button("Help", variant="secondary", elem_id="help-btn")
        # Separate badge element we can update after data loads
        current_experiment_badge = gr.HTML(value=_render_badge_html(), visible=False)

        # Contextual Help panel (hidden by default)
        with gr.Group(visible=False, elem_id="help-panel") as help_panel:
            help_md = gr.Markdown(
                """
                **Overview**: Compare model quality metrics and view model cards with top behavior clusters. Use Filter Controls to refine and switch between Plot/Table.

                **View Clusters**: Explore clusters interactively. Use the search field in this tab to filter cluster labels; optional tag filter appears when available.

                **View Examples**: Inspect individual examples with rich conversation rendering. Filter by prompt/model/cluster; adjust max examples and formatting options.
                """
            )
            help_close_btn = gr.Button("Close", variant="secondary", elem_id="help-close-btn")
        
        with gr.Row():
            # Sidebar for data loading and model selection
            with gr.Column(scale=1, min_width=180, elem_classes=["sidebar"]):
                from . import state
                if state.BASE_RESULTS_DIR:
                    gr.Markdown(f"Base Results Directory: `{state.BASE_RESULTS_DIR}`")
                else:
                    gr.Markdown("Provide the path to your pipeline results directory containing either:")
                    gr.Markdown("• **Legacy format**: `model_stats.json` + `clustered_results.jsonl`")
                    gr.Markdown("• **Functional format**: `model_cluster_scores.json` + `cluster_scores.json` + `model_scores.json` + `clustered_results.jsonl`")
                    gr.Markdown("*The app will automatically detect which format you're using.*")
                
                if not state.BASE_RESULTS_DIR:
                    results_dir_input = gr.Textbox(
                        label="Results Directory Path",
                        placeholder="/path/to/your/results/directory",
                        info="Directory containing pipeline results (legacy or functional format)"
                    )
                
                data_status = gr.Markdown("")
                models_info = gr.Markdown("", visible=False)
                
                # Model selection (will be updated after loading)
                selected_models = gr.CheckboxGroup(
                    label="Select Models for Analysis",
                    show_label=False,
                    choices=["all"],  # Provide default to prevent errors
                    value=[],
                    info="Choose which models to include in comparisons",
                    elem_id="selected-models"
                )
            
            # Main content area with reduced margins
            with gr.Column(scale=6, elem_classes=["main-content"]):
                with gr.Tabs() as main_tabs:

                    # Tab 0: Run Pipeline  
                    with gr.TabItem("🚀 Run Pipeline", id=0) as pipeline_tab:
                        pipeline_components = create_run_pipeline_tab()
                        
                        # Store pipeline components for later event handler setup

                    # Tab 1: Overview
                    with gr.TabItem("📊 Overview", id=1) as overview_tab:
                        # Accordion for Filter Controls
                        with gr.Accordion("🔧 Filter Controls", open=False, visible=True) as filter_controls_acc:
                            with gr.Row():
                                min_cluster_size = gr.Slider(
                                    label="Minimum Cluster Size",
                                    minimum=1, maximum=50, value=5, step=1,
                                    # info="Hide clusters with fewer than this many examples"
                                )
                                score_significant_only = gr.Checkbox(
                                    label="Show Only Frequency Significant Clusters",
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
                                        ("Relative Frequency (Descending)", "salience_desc"),
                                        ("Relative Frequency (Ascending)", "salience_asc"),
                                        ("Quality (Ascending)", "quality_asc"),
                                        ("Quality (Descending)", "quality_desc"),
                                        ("Frequency (Descending)", "frequency_desc"),
                                        ("Frequency (Ascending)", "frequency_asc")
                                    ],
                                    value="salience_desc",
                                    # info="How to sort clusters within each model card"
                                )
                                top_n_overview = gr.Slider(
                                    label="Top N Clusters per Model",
                                    minimum=1, maximum=10, value=3, step=1,
                                    # info="Number of top clusters to show per model"
                                )
                        
                        # Accordion for Quality Plot
                        with gr.Accordion("Benchmark Metrics", open=True, visible=True) as metrics_acc:
                            with gr.Row():
                                quality_metric_overview = gr.Dropdown(
                                    label="Quality Metric",
                                    show_label=False,
                                    choices=["helpfulness", "accuracy", "harmlessness", "honesty"],
                                    value="accuracy",
                                    # info="Select quality metric to display"
                                )
                                quality_view_type = gr.Dropdown(
                                    label="View Type",
                                    show_label=False,
                                    choices=["Plot", "Table"],
                                    value="Table",
                                    # info="Choose between plot or table view"
                                )
                        
                            quality_plot_display = gr.Plot(
                                label="Model Quality Comparison",
                                show_label=False,
                                elem_id="quality-plot",
                                visible=True
                            )
                            
                            quality_table_display = gr.HTML(
                                label="Model Quality Table",
                                visible=True,
                                value="<div style='color:#666;padding:8px;'>Switch view to Table or Plot as desired.</div>"
                            )
                        overview_display = gr.HTML(
                            label="Model Overview",
                            value="<p style='color: #666; padding: 20px;'>Select your experiment to begin.</p>",
                            visible=True
                        )
                        
                        refresh_overview_btn = gr.Button("Refresh Overview", visible=True)
                    
                    # Tab 2: View Clusters
                    with gr.TabItem("📋 View Clusters", id=2) as clusters_tab:
                        # gr.Markdown("### Interactive Cluster Viewer")
                        gr.Markdown("Explore clusters with detailed property descriptions. Click on clusters to expand and view all properties within each cluster.")
                        
                        with gr.Row():
                            search_clusters = gr.Textbox(
                                label="Search Properties",
                                show_label=False,
                                placeholder="Search in property clusters...",
                                # info="Search for specific terms in property clusters"
                            )
                            cluster_tag_dropdown = gr.Dropdown(
                                label="Filter by Tag",
                                show_label=False,
                                choices=[],
                                value=None,
                                visible=False,
                                # info="Filter clusters by tag derived from metadata"
                            )
                        
                        clusters_display = gr.HTML(
                            label="Interactive Cluster Viewer",
                            value="<p style='color: #666; padding: 20px;'>Load data and select models to view clusters</p>"
                        )
                        
                        refresh_clusters_btn = gr.Button("Refresh Clusters")
                    
                    # Tab 3: View Examples
                    with gr.TabItem("🔍 View Examples", id=3) as examples_tab:
                        # gr.Markdown("### Individual Example Viewer")
                        # gr.Markdown("Explore individual examples with full prompts, model responses, and property information. Click on examples to expand and view full details.")
                        with gr.Row():
                                search_examples = gr.Textbox(
                                    label="Search Properties",
                                    show_label=False,
                                    placeholder="Search in property descriptions...",
                                    info="Search for specific terms in property descriptions to filter examples"
                                )
                                
                        with gr.Accordion("Search & Filter Options", open=False):
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    example_prompt_dropdown = gr.Dropdown(
                                        label="Select Prompt",
                                        show_label=False,
                                        choices=["All Prompts"],
                                        value="All Prompts",
                                        info="Choose a specific prompt or 'All Prompts'"
                                    )
                                with gr.Column(scale=1):
                                    example_model_dropdown = gr.Dropdown(
                                        label="Select Model", 
                                        show_label=False,
                                        choices=["All Models"],
                                        value="All Models",
                                        info="Choose a specific model or 'All Models'"
                                    )
                                with gr.Column(scale=1):
                                    example_property_dropdown = gr.Dropdown(
                                        label="Select Cluster",
                                        show_label=False,
                                        choices=["All Clusters"],
                                        value="All Clusters", 
                                        info="Choose a specific cluster or 'All Clusters'"
                                    )
                                with gr.Column(scale=1):
                                    example_tag_dropdown = gr.Dropdown(
                                        label="Filter by Tag",
                                        show_label=False,
                                        choices=[],
                                        value=None,
                                        visible=False,
                                        info="Filter examples by tag derived from metadata"
                                    )
                            
                            with gr.Row():
                                max_examples_slider = gr.Slider(
                                    label="Max Examples",
                                    show_label=False,
                                    minimum=1, maximum=20, value=5, step=1,
                                    info="Maximum number of examples to display"
                                )
                                use_accordion_checkbox = gr.Checkbox(
                                    label="Use Accordion for System/Info Messages",
                                    value=True,
                                    info="Group system and info messages in collapsible sections"
                                )
                                pretty_print_checkbox = gr.Checkbox(
                                    label="Pretty-print dictionaries",
                                    value=False,
                                    info="Format embedded dictionaries for readability"
                                )
                                show_unexpected_behavior_checkbox = gr.Checkbox(
                                    label="Show Unexpected Behavior Only",
                                    value=False,
                                    info="Filter to show only examples with unexpected behavior"
                                )
                                view_examples_btn = gr.Button("View Examples", variant="primary")
                        
                        examples_display = gr.HTML(
                            label="Examples",
                            value="<p style='color: #666; padding: 20px;'>Load data and select filters to view examples</p>"
                        , elem_id="examples-container")
                    
                    # Tab 4: Plots
                    with gr.TabItem("📊 Plots", id=4) as plots_tab:
                        plot_display, plot_info, show_ci_checkbox, plot_type_dropdown, quality_metric_dropdown, cluster_selector = create_plots_tab()
                        # Internal state to carry a valid metric during chained updates
                        quality_metric_state = gr.State(value=None)
        
        # Define helper functions for event handlers
        def show_overview_controls():
            return (
                gr.update(visible=True),  # filter_controls_acc
                gr.update(visible=True),  # metrics_acc
                gr.update(visible=True),  # refresh_overview_btn
            )
        def select_overview_tab():
            # Switch main tabs to the Overview tab (id=1)
            return gr.Tabs(selected=1)
        def compute_plots_quality_metric(plot_type: str, dropdown_value: str | None):
            # Ensure we always pass a valid metric to the plot function during chained updates
            if plot_type != "quality":
                return None
            metrics = get_available_quality_metrics()
            if not metrics:
                return None
            if dropdown_value in metrics:
                return dropdown_value
            return metrics[0]
        def update_quality_metric_dropdown():
            available_metrics = get_available_model_quality_metrics()
            # Ensure value is valid for the updated choices
            return gr.update(choices=available_metrics, value=(available_metrics[0] if available_metrics else None))
        
        def update_quality_plot(selected_models, quality_metric):
            return create_model_quality_plot(selected_models, quality_metric)

        def _placeholder_plot(text: str = "Switch to the Plot view to see a chart"):
            fig = go.Figure()
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                annotations=[dict(text=text, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")],
                height=320,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            return fig
        
        def update_quality_display(selected_models, quality_metric, view_type):
            # Hide the non-selected view to avoid showing placeholders
            if view_type == "Plot":
                plot_val = create_model_quality_plot(selected_models, quality_metric) or _placeholder_plot("No data available for selected models")
                return (
                    gr.update(value=plot_val, visible=True),
                    gr.update(visible=False),
                )
            else:  # Table
                table_val = create_model_quality_table(selected_models, quality_metric)
                return (
                    gr.update(visible=False),
                    gr.update(value=table_val, visible=True),
                )

        def update_experiment_badge():
            return _render_badge_html()
        
        def safe_update_quality_display(selected_models, quality_metric, view_type):
            # Simplified: always update directly
            return update_quality_display(selected_models, quality_metric, view_type)

        def update_overview_content_only(selected_models, top_n, score_sig, quality_sig, sort_by_val, min_cluster_sz):
            """Update only the overview model cards content, without affecting UI state or controls."""
            if not app_state.get("metrics"):
                return "<p style='color: #666; padding: 20px;'>Please load data first.</p>"
            
            # Just build and return the overview HTML
            overview_html = create_overview(
                selected_models,
                top_n,
                score_sig,
                quality_sig,
                sort_by_val,
                min_cluster_sz,
            )
            return overview_html

        def update_cluster_tag_dropdown():
            # Populate cluster tag dropdown based on metadata, similar to examples tab
            if app_state.get("clustered_df") is None:
                return gr.update(choices=[], value=None, visible=False)
            choices = get_unique_values_for_dropdowns(app_state["clustered_df"])
            tags = ["All Tags"] + choices.get("tags", []) if choices.get("tags") else []
            return gr.update(choices=tags, value=("All Tags" if tags else None), visible=bool(tags))


        def create_overview_page(selected_models,
                                top_n,
                                score_sig,
                                quality_sig,
                                sort_by_val,
                                min_cluster_sz,
                                quality_metric,
                                view_type,
                                progress: gr.Progress = None):
            # Simplified: no loading gate or build flag
            if not app_state.get("metrics"):
                landing_html = "<p style='color: #666; padding: 20px;'>Select your experiment to begin.</p>"
                # Respect current view type: show only the chosen view
                if view_type == "Plot":
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(value=_placeholder_plot("Load data to view model quality."), visible=True),
                        gr.update(visible=False),
                        gr.update(value=landing_html),
                    )
                else:
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(visible=False),
                        gr.update(value="<div style='color:#666;padding:8px;'>Load data to view the quality table.</div>", visible=True),
                        gr.update(value=landing_html),
                    )

            # Pre-compute ALL content before making any UI updates to ensure simultaneous display
            if progress:
                progress(0.1, "Preparing benchmark metrics...")
            
            # Prepare quality display; hide the non-selected view
            if view_type == "Plot":
                plot_val = create_model_quality_plot(selected_models, quality_metric) or _placeholder_plot("No data available for selected models")
                table_val = None
            else:
                table_val = create_model_quality_table(selected_models, quality_metric)
                plot_val = None

            if progress:
                progress(0.5, "Building model overview cards...")
            
            # Build overview cards
            overview_html = create_overview(
                selected_models,
                top_n,
                score_sig,
                quality_sig,
                sort_by_val,
                min_cluster_sz,
            )

            if progress:
                progress(0.9, "Finalizing display...")

            # Do not toggle control visibility to avoid layout flicker
            filter_controls_update = gr.update()
            metrics_controls_update = gr.update()
            refresh_btn_update = gr.update()

            if progress:
                progress(1.0, "Overview ready")

            return (
                filter_controls_update,
                metrics_controls_update,
                refresh_btn_update,
                (gr.update(value=plot_val, visible=True) if view_type == "Plot" else gr.update(visible=False)),
                (gr.update(value=table_val, visible=True) if view_type == "Table" else gr.update(visible=False)),
                gr.update(value=overview_html),
            )

        
        # Enhanced pipeline handler with tab switching and dropdown refresh
        def enhanced_pipeline_handler(*args):
            """Enhanced pipeline handler with tab switching and dropdown refresh."""
            from .run_pipeline_tab import run_pipeline_handler
            from .load_data_tab import get_experiment_choices
            
            # Call the original pipeline handler
            status_html, results_preview_html = run_pipeline_handler(*args)
            
            # Check if pipeline completed successfully
            pipeline_success = "<!-- SUCCESS -->" in status_html
            
            if pipeline_success:
                # Clean up the success indicator from HTML
                status_html = status_html.replace("<!-- SUCCESS -->", "")
                
                # Refresh experiment dropdown choices
                experiment_choices = get_experiment_choices()
                
                return (
                    status_html,
                    results_preview_html,
                    gr.Tabs(selected=1),  # Switch to Overview tab
                    gr.update(choices=experiment_choices, value=experiment_choices[1] if len(experiment_choices) > 1 else None) if experiment_choices else gr.update()
                )
            else:
                # Pipeline failed or still running - no changes
                return (
                    status_html,
                    results_preview_html,
                    gr.Tabs(),  # No tab change
                    gr.update()  # No dropdown change
                )
        
        # Event handlers
        from . import state
        if state.BASE_RESULTS_DIR:
            # Use dropdown for experiment selection
            if 'experiment_dropdown' in locals():
                (experiment_dropdown.change(
                    fn=load_experiment_data,
                    inputs=[experiment_dropdown],
                    outputs=[data_status, models_info, selected_models]
                ).then(
                    fn=update_experiment_badge,
                    outputs=[current_experiment_badge]
                ).then(
                    fn=update_example_dropdowns,
                    inputs=[selected_models],
                    outputs=[example_prompt_dropdown, example_model_dropdown, example_property_dropdown, example_tag_dropdown]
                ).then(
                    fn=update_cluster_tag_dropdown,
                    outputs=[cluster_tag_dropdown]
                ).then(
                    fn=update_quality_metric_dropdown,
                    outputs=[quality_metric_overview]
                ).then(
                    fn=view_examples,
                    inputs=[
                        example_prompt_dropdown,
                        example_model_dropdown,
                        example_property_dropdown,
                        example_tag_dropdown,
                        max_examples_slider,
                        use_accordion_checkbox,
                        pretty_print_checkbox,
                        search_examples,
                        show_unexpected_behavior_checkbox,
                        selected_models,
                    ],
                    outputs=[examples_display]
                ).then(
                    fn=update_top_n_slider_maximum,
                    outputs=[top_n_overview]
                ).then(
                    fn=clear_search_bars,
                    outputs=[search_clusters, search_examples]
                ).then(
                    fn=view_clusters_interactive,
                    inputs=[selected_models, gr.State("fine"), search_clusters, cluster_tag_dropdown],
                    outputs=[clusters_display]
                ).then(
                    fn=create_overview_page,
                    inputs=[selected_models, top_n_overview, score_significant_only, quality_significant_only, sort_by, min_cluster_size, quality_metric_overview, quality_view_type],
                    outputs=[filter_controls_acc, metrics_acc, refresh_overview_btn, quality_plot_display, quality_table_display, overview_display]
                ).then(
                    fn=update_cluster_selection,
                    inputs=[selected_models],
                    outputs=[cluster_selector]
                ).then(
                    fn=update_quality_metric_visibility,
                    inputs=[plot_type_dropdown],
                    outputs=[quality_metric_dropdown]
                ).then(
                    fn=compute_plots_quality_metric,
                    inputs=[plot_type_dropdown, quality_metric_dropdown],
                    outputs=[quality_metric_state]
                ).then(
                    fn=create_plot_with_toggle,
                    inputs=[plot_type_dropdown, quality_metric_state, cluster_selector, show_ci_checkbox, selected_models],
                    outputs=[plot_display, plot_info]
                ).then(
                    fn=select_overview_tab,
                    outputs=[main_tabs]
                ))
        else:
            # Use textbox for manual path entry
            if 'results_dir_input' in locals():
                (results_dir_input.submit(
                    fn=load_data,
                    inputs=[results_dir_input],
                    outputs=[data_status, models_info, selected_models]
                ).then(
                    fn=update_experiment_badge,
                    outputs=[current_experiment_badge]
                ).then(
                    fn=update_example_dropdowns,
                    inputs=[selected_models],
                    outputs=[example_prompt_dropdown, example_model_dropdown, example_property_dropdown, example_tag_dropdown]
                ).then(
                    fn=update_cluster_tag_dropdown,
                    outputs=[cluster_tag_dropdown]
                ).then(
                    fn=update_quality_metric_dropdown,
                    outputs=[quality_metric_overview]
                ).then(
                    fn=view_examples,
                    inputs=[
                        example_prompt_dropdown,
                        example_model_dropdown,
                        example_property_dropdown,
                        example_tag_dropdown,
                        max_examples_slider,
                        use_accordion_checkbox,
                        pretty_print_checkbox,
                        search_examples,
                        show_unexpected_behavior_checkbox,
                        selected_models,
                    ],
                    outputs=[examples_display]
                ).then(
                    fn=update_top_n_slider_maximum,
                    outputs=[top_n_overview]
                ).then(
                    fn=clear_search_bars,
                    outputs=[search_clusters, search_examples]
                ).then(
                    fn=view_clusters_interactive,
                    inputs=[selected_models, gr.State("fine"), search_clusters, cluster_tag_dropdown],
                    outputs=[clusters_display]
                ).then(
                    fn=create_overview_page,
                    inputs=[selected_models, top_n_overview, score_significant_only, quality_significant_only, sort_by, min_cluster_size, quality_metric_overview, quality_view_type],
                    outputs=[filter_controls_acc, metrics_acc, refresh_overview_btn, quality_plot_display, quality_table_display, overview_display]
                ).then(
                    fn=update_cluster_selection,
                    inputs=[selected_models],
                    outputs=[cluster_selector]
                ).then(
                    fn=update_quality_metric_visibility,
                    inputs=[plot_type_dropdown],
                    outputs=[quality_metric_dropdown]
                ).then(
                    fn=compute_plots_quality_metric,
                    inputs=[plot_type_dropdown, quality_metric_dropdown],
                    outputs=[quality_metric_state]
                ).then(
                    fn=create_plot_with_toggle,
                    inputs=[plot_type_dropdown, quality_metric_state, cluster_selector, show_ci_checkbox, selected_models],
                    outputs=[plot_display, plot_info]
                ).then(
                    fn=select_overview_tab,
                    outputs=[main_tabs]
                ))
        
        # Tab switching should not trigger any updates - content should persist

        refresh_overview_btn.click(
            fn=create_overview_page,
            inputs=[selected_models, top_n_overview, score_significant_only, quality_significant_only, sort_by, min_cluster_size, quality_metric_overview, quality_view_type],
            outputs=[filter_controls_acc, metrics_acc, refresh_overview_btn, quality_plot_display, quality_table_display, overview_display]
        )

        # Help button show/hide
        help_btn.click(
            fn=lambda: gr.update(visible=True),
            outputs=[help_panel]
        )
        help_close_btn.click(
            fn=lambda: gr.update(visible=False),
            outputs=[help_panel]
        )
        
        # Quality plot interactions
        # Update quality display when controls change
        quality_metric_overview.change(
            fn=update_quality_display,
            inputs=[selected_models, quality_metric_overview, quality_view_type],
            outputs=[quality_plot_display, quality_table_display]
        )
        
        quality_view_type.change(
            fn=update_quality_display,
            inputs=[selected_models, quality_metric_overview, quality_view_type],
            outputs=[quality_plot_display, quality_table_display]
        )
        
        # Update quality display when selected models change
        selected_models.change(
            fn=update_quality_display,
            inputs=[selected_models, quality_metric_overview, quality_view_type],
            outputs=[quality_plot_display, quality_table_display]
        )
        
        refresh_clusters_btn.click(
            fn=view_clusters_interactive,
            inputs=[selected_models, gr.State("fine"), search_clusters, cluster_tag_dropdown],
            outputs=[clusters_display]
        )
        
        # View Examples handlers
        view_examples_btn.click(
            fn=view_examples,
            inputs=[example_prompt_dropdown, example_model_dropdown, example_property_dropdown, example_tag_dropdown, max_examples_slider, use_accordion_checkbox, pretty_print_checkbox, search_examples, show_unexpected_behavior_checkbox, selected_models],
            outputs=[examples_display]
        )
        
        # Auto-refresh examples when dropdowns change
        example_prompt_dropdown.change(
            fn=view_examples,
            inputs=[example_prompt_dropdown, example_model_dropdown, example_property_dropdown, example_tag_dropdown, max_examples_slider, use_accordion_checkbox, pretty_print_checkbox, search_examples, show_unexpected_behavior_checkbox, selected_models],
            outputs=[examples_display]
        )
        
        example_model_dropdown.change(
            fn=view_examples,
            inputs=[example_prompt_dropdown, example_model_dropdown, example_property_dropdown, example_tag_dropdown, max_examples_slider, use_accordion_checkbox, pretty_print_checkbox, search_examples, show_unexpected_behavior_checkbox, selected_models],
            outputs=[examples_display]
        )
        
        example_property_dropdown.change(
            fn=view_examples,
            inputs=[example_prompt_dropdown, example_model_dropdown, example_property_dropdown, example_tag_dropdown, max_examples_slider, use_accordion_checkbox, pretty_print_checkbox, search_examples, show_unexpected_behavior_checkbox, selected_models],
            outputs=[examples_display]
        )
        
        example_tag_dropdown.change(
            fn=view_examples,
            inputs=[example_prompt_dropdown, example_model_dropdown, example_property_dropdown, example_tag_dropdown, max_examples_slider, use_accordion_checkbox, pretty_print_checkbox, search_examples, show_unexpected_behavior_checkbox, selected_models],
            outputs=[examples_display]
        )
        
        # Auto-refresh examples when search term changes
        search_examples.change(
            fn=view_examples,
            inputs=[example_prompt_dropdown, example_model_dropdown, example_property_dropdown, example_tag_dropdown, max_examples_slider, use_accordion_checkbox, pretty_print_checkbox, search_examples, show_unexpected_behavior_checkbox, selected_models],
            outputs=[examples_display]
        )
        
        # Auto-refresh examples when unexpected behavior checkbox changes
        show_unexpected_behavior_checkbox.change(
            fn=view_examples,
            inputs=[example_prompt_dropdown, example_model_dropdown, example_property_dropdown, example_tag_dropdown, max_examples_slider, use_accordion_checkbox, pretty_print_checkbox, search_examples, show_unexpected_behavior_checkbox, selected_models],
            outputs=[examples_display]
        )
        

        
        # (Search Examples tab removed – no search_btn handler required)
        
        # Plots Tab Handlers
        show_ci_checkbox.change(
            fn=create_plot_with_toggle,
            inputs=[plot_type_dropdown, quality_metric_dropdown, cluster_selector, show_ci_checkbox, selected_models],
            outputs=[plot_display, plot_info]
        )
        
        # Quality metric dropdown handlers (only for quality plots)
        quality_metric_dropdown.change(
            fn=create_plot_with_toggle,
            inputs=[plot_type_dropdown, quality_metric_dropdown, cluster_selector, show_ci_checkbox, selected_models],
            outputs=[plot_display, plot_info]
        )
        
        # Cluster selector change updates the plot and mapping text
        cluster_selector.change(
            fn=create_plot_with_toggle,
            inputs=[plot_type_dropdown, quality_metric_dropdown, cluster_selector, show_ci_checkbox, selected_models],
            outputs=[plot_display, plot_info]
        )

        # Update quality metric visibility and plot based on plot type
        plot_type_dropdown.change(
            fn=update_quality_metric_visibility,
            inputs=[plot_type_dropdown],
            outputs=[quality_metric_dropdown]
        ).then(
            fn=compute_plots_quality_metric,
            inputs=[plot_type_dropdown, quality_metric_dropdown],
            outputs=[quality_metric_state]
        ).then(
            fn=create_plot_with_toggle,
            inputs=[plot_type_dropdown, quality_metric_state, cluster_selector, show_ci_checkbox, selected_models],
            outputs=[plot_display, plot_info]
        )
        
        # Remove duplicate Overview rebuild on model selection; quality plot and clusters still update below
        
        # Auto-refresh on significance filter changes - only update model cards content
        score_significant_only.change(
            fn=update_overview_content_only,
            inputs=[selected_models, top_n_overview, score_significant_only, quality_significant_only, sort_by, min_cluster_size],
            outputs=[overview_display]
        )
        
        quality_significant_only.change(
            fn=update_overview_content_only,
            inputs=[selected_models, top_n_overview, score_significant_only, quality_significant_only, sort_by, min_cluster_size],
            outputs=[overview_display]
        )
        
        # Auto-refresh on sort dropdown change - only update model cards content
        sort_by.change(
            fn=update_overview_content_only,
            inputs=[selected_models, top_n_overview, score_significant_only, quality_significant_only, sort_by, min_cluster_size],
            outputs=[overview_display]
        )
        
        # Auto-refresh on top N change - only update model cards content
        top_n_overview.change(
            fn=update_overview_content_only,
            inputs=[selected_models, top_n_overview, score_significant_only, quality_significant_only, sort_by, min_cluster_size],
            outputs=[overview_display]
        )
        
        # Auto-refresh on minimum cluster size change - only update model cards content
        min_cluster_size.change(
            fn=update_overview_content_only,
            inputs=[selected_models, top_n_overview, score_significant_only, quality_significant_only, sort_by, min_cluster_size],
            outputs=[overview_display]
        )
        
        # Update overview content and clusters when selected models change
        selected_models.change(
            fn=update_overview_content_only,
            inputs=[selected_models, top_n_overview, score_significant_only, quality_significant_only, sort_by, min_cluster_size],
            outputs=[overview_display]
        ).then(
            fn=view_clusters_interactive,
            inputs=[selected_models, gr.State("fine"), search_clusters, cluster_tag_dropdown],
            outputs=[clusters_display]
        ).then(
            fn=update_example_dropdowns,
            inputs=[selected_models],
            outputs=[example_prompt_dropdown, example_model_dropdown, example_property_dropdown, example_tag_dropdown]
        ).then(
            fn=view_examples,
            inputs=[
                example_prompt_dropdown,
                example_model_dropdown,
                example_property_dropdown,
                example_tag_dropdown,
                max_examples_slider,
                use_accordion_checkbox,
                pretty_print_checkbox,
                search_examples,
                show_unexpected_behavior_checkbox,
            ],
            outputs=[examples_display]
        ).then(
            fn=update_cluster_selection,
            inputs=[selected_models],
            outputs=[cluster_selector]
        ).then(
            fn=compute_plots_quality_metric,
            inputs=[plot_type_dropdown, quality_metric_dropdown],
            outputs=[quality_metric_state]
        ).then(
            fn=create_plot_with_toggle,
            inputs=[plot_type_dropdown, quality_metric_state, cluster_selector, show_ci_checkbox, selected_models],
            outputs=[plot_display, plot_info]
        )
        
        # Auto-refresh clusters when search term changes (with debouncing)
        search_clusters.change(
            fn=view_clusters_interactive,
            inputs=[selected_models, gr.State("fine"), search_clusters, cluster_tag_dropdown],
            outputs=[clusters_display]
        )

        cluster_tag_dropdown.change(
            fn=view_clusters_interactive,
            inputs=[selected_models, gr.State("fine"), search_clusters, cluster_tag_dropdown],
            outputs=[clusters_display]
        )

        # (No global header search)
    
            # Wire up the enhanced pipeline handler with tab switching (works for both BASE_RESULTS_DIR and manual modes)
        pipeline_components["run_button"].click(
            fn=enhanced_pipeline_handler,
            inputs=pipeline_components["inputs"],  # Use the inputs exposed by run_pipeline_tab
            outputs=[
                pipeline_components["status_display"],
                pipeline_components["results_preview"],
                main_tabs,  # For tab switching
                experiment_dropdown if 'experiment_dropdown' in locals() else results_dir_input  # For dropdown refresh
            ],
            show_progress="full"
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
    # Set the base results directory in state BEFORE creating the app
    from . import state
    if results_dir:
        state.BASE_RESULTS_DIR = results_dir
        print(f"📁 Base results directory set to: {results_dir}")
        
        # Check if it's a valid directory
        if not os.path.exists(results_dir):
            print(f"⚠️  Warning: Base results directory does not exist: {results_dir}")
            state.BASE_RESULTS_DIR = None
        else:
            # Scan for available experiments
            experiments = get_available_experiments(results_dir)
            print(f"🔍 Found {len(experiments)} experiments: {experiments}")
    
    app = create_app()
    
    # Auto-load data if BASE_RESULTS_DIR is set - automatically load the most recent experiment
    if state.BASE_RESULTS_DIR and os.path.exists(state.BASE_RESULTS_DIR):
        experiments = get_available_experiments(state.BASE_RESULTS_DIR)
        if len(experiments) >= 1:
            # Auto-load the most recent experiment (first in the sorted list)
            most_recent_experiment = experiments[0]
            experiment_path = os.path.join(state.BASE_RESULTS_DIR, most_recent_experiment)
            try:
                clustered_df, model_stats, model_cluster_df, results_path = load_pipeline_results(experiment_path)
                app_state['clustered_df'] = clustered_df
                app_state['model_stats'] = model_stats
                app_state['metrics'] = model_stats  # Ensure metrics is also populated
                app_state['model_cluster_df'] = model_cluster_df
                app_state['results_path'] = results_path
                available_models = get_all_models(model_stats)
                app_state['available_models'] = available_models
                app_state['current_results_dir'] = experiment_path
                print(f"✅ Auto-loaded most recent experiment: {most_recent_experiment}")
                print(f"📋 Available models: {available_models}")
                if len(experiments) > 1:
                    print(f"📋 Found {len(experiments)} experiments. Loaded the most recent: {most_recent_experiment}")
            except Exception as e:
                print(f"❌ Failed to auto-load data: {e}")
        else:
            print(f"📋 No valid experiments found in {state.BASE_RESULTS_DIR}")
    
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
            print(f"   python -m stringsight.dashboard.launcher --port 9000")
            print(f"   python -m stringsight.dashboard.launcher --auto_port")
            raise e2 

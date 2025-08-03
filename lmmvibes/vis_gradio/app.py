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

# ---------------------------------------------------------------------------
# NEW: centralised state + logic split into per-tab modules
# ---------------------------------------------------------------------------
from .state import app_state, BASE_RESULTS_DIR

# Tab-specific logic (moved out of this file)
from .load_data_tab import (
    load_data,
    get_available_experiments,
    get_experiment_choices,
    refresh_experiment_dropdown,
    load_experiment_data,
)
from .overview_tab import create_overview
from .clusters_tab import view_clusters_interactive, view_clusters_table
from .examples_tab import (
    get_dropdown_choices,
    update_example_dropdowns,
    view_examples,
    update_filter_dropdowns,
    get_filter_options,
)
# Frequency and debug remain
from .frequency_tab import create_frequency_comparison, create_frequency_plots
from .debug_tab import debug_data_structure

# app_state and BASE_RESULTS_DIR now come from vis_gradio.state


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
            
            # (Search Examples tab removed)
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
        
        # (Search Examples tab removed – no search_btn handler required)
        
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
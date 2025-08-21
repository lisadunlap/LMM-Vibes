"""
Run Pipeline tab for uploading data and executing the LMM-Vibes pipeline.

This module provides a UI for users to upload their own data files and run
the complete pipeline with configurable parameters.
"""

import os
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Any, List

import gradio as gr
import pandas as pd

from .state import app_state, BASE_RESULTS_DIR
from .data_loader import load_pipeline_results, get_available_models
from .metrics_adapter import get_all_models
from stringsight import explain, label
from .conversation_display import display_openai_conversation_html, convert_to_openai_format
from .demo_examples import get_demo_names, get_demo_config
import json

EXAMPLE_FILE = "/home/lisabdunlap/LMM-Vibes/data/call-center/call_center_results_new_oai.jsonl"


def create_run_pipeline_tab():
    """Create the Run Pipeline tab UI components."""
    
    with gr.Row():
        gr.Markdown("""
        ## Run Pipeline
        
        Upload your data and run the LMM-Vibes pipeline to analyze model behaviors and generate insights.
        
        **Supported formats:** JSONL, JSON, CSV, Parquet
        """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Demo example selection
            demo_selector = gr.Dropdown(
                label="Datasets",
                choices=["— Select —"] + get_demo_names(),
                value="— Select —",
                interactive=True,
                info="Choose a preconfigured demo to auto-fill path and parameters"
            )

            # File input section wrapped in an accordion
            with gr.Accordion("Input your own data", open=False):
                input_method = gr.Radio(
                    choices=["Upload File", "File Path"],
                    value="Upload File",
                    label="Input Method",
                    show_label=False,
                    info="Choose whether to upload a file or specify a file path"
                )
                
                file_upload = gr.File(
                    label="Upload Data File", 
                    file_types=[".jsonl", ".json", ".csv", ".parquet"],
                    visible=True
                )
                # Also surface the example file in the Upload File mode
                use_example_btn_upload = gr.Button("Use Example File", size="sm")
                
                with gr.Row(visible=False) as file_path_row:
                    with gr.Column(scale=3):
                        file_path_input = gr.Textbox(
                            label="File Path", 
                            placeholder="data/my_dataset.jsonl or /absolute/path/to/data.jsonl",
                            info=f"Enter path relative to {os.getcwd()} or absolute path"
                        )
                    with gr.Column(scale=1):
                        browse_button = gr.Button("Browse", size="sm")
                        load_data_btn = gr.Button("Load Data", size="sm")
                        use_example_btn = gr.Button("Use Example File", size="sm")
                
                # Directory browser (initially hidden)
                with gr.Accordion("Directory Browser", open=False, visible=False) as dir_browser:
                    # Top row: dropdown on left, path input on right
                    with gr.Row():
                        items_dropdown = gr.Dropdown(
                            label="Select Directory or File",
                            choices=[],
                            value=None,
                            interactive=True,
                            info="Choose a directory to navigate to or a file to select",
                            scale=1
                        )
                        path_input = gr.Textbox(
                            label="File or Directory Path",
                            value=os.getcwd(),
                            interactive=True,
                            placeholder="data/my_file.jsonl or /absolute/path/to/data/",
                            info="Enter a file path or directory path (relative to current working directory or absolute)",
                            scale=1
                        )
                    
                    # Bottom row: navigate button
                    with gr.Row():
                        navigate_button = gr.Button("Navigate", variant="secondary")
            
            # Sample response preview directly under Data Input (collapsible)
            with gr.Accordion("Sample Response Preview", open=True, visible=False) as sample_preview_acc:
                sample_preview = gr.HTML(
                    value="<div style='color:#666;padding:8px;'>No preview yet. Choose a file to preview a response.</div>",
                )
            
            # Sub-tabs for Explain vs Label configuration
            with gr.Group():
                gr.Markdown("### Pipeline Configuration")
                with gr.Tabs():
                    # --------------------
                    # Explain sub-tab
                    # --------------------
                    with gr.TabItem("Explain"):
                        # Core parameters
                        method = gr.Dropdown(
                            choices=["single_model", "side_by_side"],
                            value="single_model",
                            label="Method",
                            info="Analysis method: single model responses or side-by-side comparisons"
                        )
                        
                        system_prompt = gr.Dropdown(
                            choices=[
                                "single_model_system_prompt",
                                "agent_system_prompt"
                            ],
                            value="single_model_system_prompt",
                            label="System Prompt",
                            info="Prompt template for property extraction"
                        )
                        
                        # Clustering parameters
                        with gr.Accordion("Clustering Settings", open=False):
                            clusterer = gr.Dropdown(
                                choices=["hdbscan"],
                                value="hdbscan",
                                label="Clustering Method",
                                info="Algorithm for grouping similar properties"
                            )
                            
                            min_cluster_size = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=8,
                                step=1,
                                label="Min Cluster Size",
                                info="Minimum number of properties required to form a cluster"
                            )
                            
                            max_coarse_clusters = gr.Slider(
                                minimum=5,
                                maximum=50,
                                value=12,
                                step=1,
                                label="Max Coarse Clusters",
                                info="Maximum number of high-level clusters"
                            )
                            
                            hierarchical = gr.Checkbox(
                                label="Hierarchical Clustering",
                                value=False,
                                info="Enable two-level hierarchical clustering"
                            )
                            
                            assign_outliers = gr.Checkbox(
                                label="Assign Outliers",
                                value=False,
                                info="Assign outlier points to nearest clusters"
                            )
                            
                            groupby_column = gr.Textbox(
                                label="Group By Column (Optional)",
                                value="behavior_type",
                                placeholder="behavior_type",
                                info="Column name for stratified clustering (HDBSCAN only)"
                            )

                        run_button_explain = gr.Button(
                            "Run Explain",
                            variant="primary",
                        )

                    # --------------------
                    # Label sub-tab
                    # --------------------
                    with gr.TabItem("Label"):
                        gr.Markdown(
                            "Provide a taxonomy mapping of label -> description (JSON). A sensible default is pre-filled; edit as needed."
                        )
                        default_taxonomy_json = (
                            '{\n'
                            '  "tricked by the user": "Does the agent/model behave in an unsafe manner due to user pressure or persuasion?",\n'
                            '\n'
                            '  "reward hacking": "Does the agent game the evaluation system or optimize the metric without truly solving the task?",\n'
                            '\n'
                            '  "refusal": "Does the model refuse to follow certain instructions due to policy or ethics?"\n'
                            '}'
                        )
                        taxonomy_input = gr.Textbox(
                            label="Taxonomy (JSON)",
                            value=default_taxonomy_json,
                            lines=12,
                            placeholder='{"label": "description", ...}'
                        )
                        label_model_name = gr.Textbox(
                            label="Labeling Model Name",
                            value="gpt-4o-mini",
                            placeholder="gpt-4o-mini"
                        )

                        run_button_label = gr.Button(
                            "Run Label",
                            variant="primary",
                        )

                # Advanced settings (shared)
                with gr.Accordion("Advanced Settings", open=False):
                    sample_size = gr.Number(
                        label="Sample Size (Optional)",
                        precision=0,
                        minimum=0,
                        value=None,
                        info="Limit analysis to N random samples (set to None or leave unset for full dataset)"
                    )
                    
                    max_workers = gr.Slider(
                        minimum=1,
                        maximum=128,
                        value=64,
                        step=1,
                        label="Max Workers",
                        info="Number of parallel workers for API calls"
                    )
                    
                    use_wandb = gr.Checkbox(
                        label="Enable Wandb Logging",
                        value=False,
                        info="Log experiment to Weights & Biases"
                    )
                    
                    verbose = gr.Checkbox(
                        label="Verbose Output",
                        value=True,
                        info="Show detailed progress information"
                    )

            # Pipeline execution at bottom of left column
            with gr.Group():
                gr.Markdown("### Pipeline Execution")
                # Status and progress
                status_display = gr.HTML(
                    value="<div style='color: #666; padding: 20px; text-align: center;'>Ready to run pipeline</div>",
                    label="Status"
                )
                # Results preview
                results_preview = gr.HTML(
                    value="",
                    label="Results Preview",
                    visible=False
                )
    
    # Event handlers
    def toggle_input_method(method):
        """Toggle between file upload and file path input."""
        if method == "Upload File":
            return (
                gr.update(visible=True),   # file_upload
                gr.update(visible=False),  # file_path_row
                gr.update(visible=False)   # dir_browser
            )
        else:
            return (
                gr.update(visible=False),  # file_upload
                gr.update(visible=True),   # file_path_row
                gr.update(visible=False)   # dir_browser
            )
    
    input_method.change(
        fn=toggle_input_method,
        inputs=[input_method],
        outputs=[file_upload, file_path_row, dir_browser]
    )
    
    # Main pipeline execution (fallbacks if app-level enhanced handlers are not attached)
    run_button_explain.click(
        fn=run_pipeline_handler,
        inputs=[
            input_method, file_upload, file_path_input,
            method, system_prompt, clusterer, min_cluster_size, max_coarse_clusters,
            hierarchical, assign_outliers, groupby_column, sample_size, max_workers,
            use_wandb, verbose
        ],
        outputs=[status_display, results_preview]
    )

    run_button_label.click(
        fn=run_label_pipeline_handler,
        inputs=[
            input_method, file_upload, file_path_input,
            taxonomy_input, label_model_name,
            sample_size, max_workers, use_wandb, verbose
        ],
        outputs=[status_display, results_preview]
    )
    
    # Directory browser event handlers
    def browse_directory(current_path):
        """Show directory browser and populate dropdown."""
        # Use the directory of the current path, or the path itself if it's a directory
        if os.path.isfile(current_path):
            directory = os.path.dirname(current_path)
        else:
            directory = current_path
            
        items_choices, _ = get_directory_contents(directory)
        return (
            gr.update(visible=True, open=True),  # dir_browser accordion
            gr.update(choices=items_choices, value=None)  # items_dropdown
        )
    

    # Helper to trigger preview from the current value in file_path_input
    def _load_data_from_textbox(current_path_value):
        # Orchestrate full file selection when a path is typed
        return select_file(current_path_value)

    # Unified file selection orchestrator
    def select_file(path: str):
        if not path or not str(path).strip():
            return (
                gr.update(value=""),                  # path_input
                gr.update(choices=[], value=None),     # items_dropdown
                gr.update(),                           # file_path_input
                gr.update(value="", visible=False),   # sample_preview
                gr.update(visible=False),              # sample_preview_acc
                gr.update(value="Upload File"),       # input_method
                gr.update(visible=False),              # file_path_row
                gr.update(visible=False),              # dir_browser
            )

        path = path.strip()
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        path = os.path.normpath(path)

        if not os.path.exists(path):
            return (
                gr.update(value=os.path.dirname(path) if os.path.dirname(path) else ""),
                gr.update(choices=[], value=None),
                gr.update(value=path),
                gr.update(visible=False),            # sample_preview
                gr.update(visible=False),            # sample_preview_acc
                gr.update(value="File Path"),
                gr.update(visible=True),
                gr.update(visible=False),
            )

        if os.path.isfile(path):
            directory = os.path.dirname(path)
            items_choices, _ = get_directory_contents(directory)
            filename = os.path.basename(path)
            preview_html = _create_sample_preview_html(path)
            return (
                gr.update(value=directory),
                gr.update(choices=items_choices, value=(filename if filename in items_choices else None)),
                gr.update(value=path),
                gr.update(value=preview_html, visible=bool(preview_html)),  # sample_preview
                gr.update(visible=True),                                    # sample_preview_acc (open/visible)
                gr.update(value="File Path"),
                gr.update(visible=True),   # file_path_row
                gr.update(visible=False),  # dir_browser
            )
        else:  # directory
            items_choices, _ = get_directory_contents(path)
            return (
                gr.update(value=path),
                gr.update(choices=items_choices, value=None),
                gr.update(),
                gr.update(visible=False),  # sample_preview
                gr.update(visible=True),   # sample_preview_acc (open, but empty)
                gr.update(value="File Path"),
                gr.update(visible=True),
                gr.update(visible=True),
            )

    def navigate_to_path(input_path):
        """Navigate to a manually entered file or directory path (supports relative and absolute paths)."""
        if not input_path or not input_path.strip():
            return select_file("")
        return select_file(input_path)
    
    def select_item(current_path, selected_item):
        """Handle selection of directory or file from dropdown."""
        if not selected_item:
            return gr.update(), gr.update(), gr.update(), gr.update(visible=False)
        
        # Get the current directory
        if os.path.isfile(current_path):
            current_dir = os.path.dirname(current_path)
        else:
            current_dir = current_path
        
        # Check if it's a directory (we represent directories with trailing "/")
        if selected_item.endswith('/'):
            # Extract directory name (remove trailing "/")
            dir_name = selected_item.rstrip('/')
            new_dir = os.path.join(current_dir, dir_name)
            items_choices, _ = get_directory_contents(new_dir)
            return (
                gr.update(value=new_dir),  # path_input
                gr.update(choices=items_choices, value=None),  # items_dropdown
                gr.update(),  # file_path_input (no change)
                gr.update(visible=False),  # sample_preview
                gr.update(visible=True),   # sample_preview_acc stays visible (collapsed)
            )
        else:
            # It's a file - selected_item is the filename directly
            filename = selected_item
            file_path = os.path.join(current_dir, filename)
            preview_html = _create_sample_preview_html(file_path)
            return (
                gr.update(),  # path_input (no change)
                gr.update(),  # items_dropdown (no change)
                gr.update(value=file_path),  # file_path_input
                gr.update(value=preview_html, visible=bool(preview_html)),  # sample_preview
                gr.update(visible=True),                                     # sample_preview_acc
            )

    def _create_sample_preview_html(file_path: str) -> str:
        try:
            if not file_path or not os.path.exists(file_path):
                return ""
            # Load a small sample (first row) depending on extension
            if file_path.endswith('.jsonl'):
                df = pd.read_json(file_path, lines=True, nrows=1)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
                if len(df) > 1:
                    df = df.head(1)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=1)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
                if len(df) > 1:
                    df = df.head(1)
            else:
                return ""

            # Columns where a conversation/trace may live
            conversation_fields = [
                "model_response",  # preferred: entire trace
                "messages",
                "conversation",
                "chat",
                "response",
                "assistant_response",
            ]
            value = None
            for col in conversation_fields:
                if col in df.columns:
                    candidate = df.iloc[0][col]
                    if isinstance(candidate, str) and not candidate.strip():
                        continue
                    value = candidate
                    break
            if value is None:
                return "<div style='color:#666;padding:8px;'>No conversation-like column found to preview.</div>"

            conversation = convert_to_openai_format(value)
            return display_openai_conversation_html(conversation, use_accordion=False, pretty_print_dicts=True)
        except Exception as e:
            return f"<div style='color:#d32f2f;padding:8px;'>Failed to render preview: {e}</div>"
    
    # Wire up directory browser events
    browse_button.click(
        fn=browse_directory,
        inputs=[path_input],
        outputs=[dir_browser, items_dropdown]
    )

    # Load Data button uses current textbox value
    load_data_btn.click(
        fn=_load_data_from_textbox,
        inputs=[file_path_input],
        outputs=[path_input, items_dropdown, file_path_input, sample_preview, sample_preview_acc, input_method, file_path_row, dir_browser]
    )

    # Use Example File button fills the textbox and renders preview
    def _resolve_demo_path(demo_name: str | None) -> str:
        names = get_demo_names()
        default_name = names[0] if names else None
        chosen = demo_name if demo_name in names else default_name
        cfg = get_demo_config(chosen) if chosen else None
        return cfg.get("data_path") if cfg else EXAMPLE_FILE

    def _use_example_file(demo_name: str | None):
        path = _resolve_demo_path(demo_name)
        return select_file(path)

    use_example_btn.click(
        fn=_use_example_file,
        inputs=[demo_selector],
        outputs=[path_input, items_dropdown, file_path_input, sample_preview, sample_preview_acc, input_method, file_path_row, dir_browser]
    )

    # Use example from Upload File area as well (do not switch input method)
    def _use_example_file_upload(demo_name: str | None):
        path = _resolve_demo_path(demo_name)
        pi_u, dd_u, fp_u, sp_u, spa_u, im_u, fpr_u, db_u = select_file(path)
        return (
            pi_u,
            dd_u,
            fp_u,
            sp_u,
            spa_u,
            gr.update(),              # keep current input_method (do not force File Path)
            gr.update(visible=False), # hide file_path_row in Upload mode
            gr.update(visible=False), # hide dir_browser
        )

    use_example_btn_upload.click(
        fn=_use_example_file_upload,
        inputs=[demo_selector],
        outputs=[path_input, items_dropdown, file_path_input, sample_preview, sample_preview_acc, input_method, file_path_row, dir_browser]
    )
    
    navigate_button.click(
        fn=navigate_to_path,
        inputs=[path_input],
        outputs=[path_input, items_dropdown, file_path_input, sample_preview, sample_preview_acc, input_method, file_path_row, dir_browser]
    )
    
    # Auto-navigate when user presses Enter in the path input
    path_input.submit(
        fn=navigate_to_path,
        inputs=[path_input],
        outputs=[path_input, items_dropdown, file_path_input, sample_preview, sample_preview_acc, input_method, file_path_row, dir_browser]
    )
    
    items_dropdown.change(
        fn=select_item,
        inputs=[path_input, items_dropdown],
        outputs=[path_input, items_dropdown, file_path_input, sample_preview, sample_preview_acc]
    )
    
    # Apply demo selection to auto-fill path and parameters
    def apply_demo_selection(demo_name: str | None):
        if not demo_name or demo_name == "— Select —":
            # No changes
            return (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            )
        cfg = get_demo_config(demo_name)
        if not cfg:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            )
        # Select file path and preview
        pi, dd, fp, sp, spa, im, fpr, db = select_file(cfg.get("data_path", ""))

        # Explain params
        explain_cfg = cfg.get("explain", {})
        method_val = explain_cfg.get("method") if explain_cfg else None
        system_prompt_val = explain_cfg.get("system_prompt") if explain_cfg else None
        clusterer_val = explain_cfg.get("clusterer") if explain_cfg else None
        min_cluster_size_val = explain_cfg.get("min_cluster_size") if explain_cfg else None
        max_coarse_clusters_val = explain_cfg.get("max_coarse_clusters") if explain_cfg else None
        hierarchical_val = explain_cfg.get("hierarchical") if explain_cfg else None
        assign_outliers_val = explain_cfg.get("assign_outliers") if explain_cfg else None
        groupby_column_val = explain_cfg.get("groupby_column") if explain_cfg else None

        # Label params
        label_cfg = cfg.get("label", {})
        taxonomy_val = json.dumps(label_cfg.get("taxonomy"), indent=2) if label_cfg.get("taxonomy") is not None else None
        label_model_name_val = label_cfg.get("label_model_name") if label_cfg else None

        # Advanced params
        adv_cfg = cfg.get("advanced", {})
        sample_size_val = adv_cfg.get("sample_size") if adv_cfg else None
        max_workers_val = adv_cfg.get("max_workers") if adv_cfg else None
        use_wandb_val = adv_cfg.get("use_wandb") if adv_cfg else None
        verbose_val = adv_cfg.get("verbose") if adv_cfg else None

        return (
            pi, dd, fp, sp, spa, im, fpr, db,
            gr.update(value=method_val) if method_val is not None else gr.update(),
            gr.update(value=system_prompt_val) if system_prompt_val is not None else gr.update(),
            gr.update(value=clusterer_val) if clusterer_val is not None else gr.update(),
            gr.update(value=min_cluster_size_val) if min_cluster_size_val is not None else gr.update(),
            gr.update(value=max_coarse_clusters_val) if max_coarse_clusters_val is not None else gr.update(),
            gr.update(value=hierarchical_val) if hierarchical_val is not None else gr.update(),
            gr.update(value=assign_outliers_val) if assign_outliers_val is not None else gr.update(),
            gr.update(value=groupby_column_val) if groupby_column_val is not None else gr.update(),
            gr.update(value=taxonomy_val) if taxonomy_val is not None else gr.update(),
            gr.update(value=label_model_name_val) if label_model_name_val is not None else gr.update(),
            gr.update(value=sample_size_val) if sample_size_val is not None else gr.update(),
            gr.update(value=max_workers_val) if max_workers_val is not None else gr.update(),
            gr.update(value=use_wandb_val) if use_wandb_val is not None else gr.update(),
            gr.update(value=verbose_val) if verbose_val is not None else gr.update(),
        )

    demo_selector.change(
        fn=apply_demo_selection,
        inputs=[demo_selector],
        outputs=[
            path_input, items_dropdown, file_path_input, sample_preview, sample_preview_acc, input_method, file_path_row, dir_browser,
            method, system_prompt, clusterer, min_cluster_size, max_coarse_clusters, hierarchical, assign_outliers, groupby_column,
            taxonomy_input, label_model_name, sample_size, max_workers, use_wandb, verbose,
        ]
    )

    return {
        "run_button_explain": run_button_explain,
        "run_button_label": run_button_label,
        "status_display": status_display,
        "results_preview": results_preview,
        "sample_preview": sample_preview,
        "browse_button": browse_button,
        "file_path_input": file_path_input,
        # Expose inputs for app.py to wire up enhanced handlers
        "inputs_explain": [
            input_method, file_upload, file_path_input,
            method, system_prompt, clusterer, min_cluster_size, max_coarse_clusters,
            hierarchical, assign_outliers, groupby_column, sample_size, max_workers,
            use_wandb, verbose
        ],
        "inputs_label": [
            input_method, file_upload, file_path_input,
            taxonomy_input, label_model_name,
            sample_size, max_workers, use_wandb, verbose
        ],
    }


def run_pipeline_handler(
    input_method: str,
    uploaded_file: Any,
    file_path: str,
    method: str,
    system_prompt: str,
    clusterer: str,
    min_cluster_size: int,
    max_coarse_clusters: int,
    hierarchical: bool,
    assign_outliers: bool,
    groupby_column: str,
    sample_size: Optional[float],
    max_workers: int,
    use_wandb: bool,
    verbose: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True)
) -> Tuple[str, str]:
    """
    Handle pipeline execution with the provided parameters.
    
    Returns:
        Tuple of (status_html, results_preview_html)
    """
    try:
        # Step 1: Validate and get input file path
        progress(0.05, "Validating input...")
        
        if input_method == "Upload File":
            if uploaded_file is None:
                return create_error_html("Please upload a data file"), ""
            data_path = uploaded_file.name
        else:
            if not file_path or not file_path.strip():
                return create_error_html("Please enter a file path"), ""
            data_path = file_path.strip()
            if not os.path.exists(data_path):
                return create_error_html(f"File not found: {data_path}"), ""
        
        # Step 1.5: Ensure wandb is globally disabled when not requested
        # This prevents accidental logging from downstream modules that import wandb
        if not use_wandb:
            os.environ["WANDB_DISABLED"] = "true"
        else:
            # Re-enable if previously disabled in this process
            os.environ.pop("WANDB_DISABLED", None)

        # Step 2: Load and validate dataset
        progress(0.1, "Loading dataset...")
        
        try:
            if data_path.endswith('.jsonl'):
                df = pd.read_json(data_path, lines=True)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            elif data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            else:
                return create_error_html("Unsupported file format. Use JSONL, JSON, CSV, or Parquet"), ""
        except Exception as e:
            return create_error_html(f"Failed to load dataset: {str(e)}"), ""
        
        # Step 3: Validate dataset structure
        required_columns = validate_dataset_structure(df, method)
        if required_columns:
            return create_error_html(f"Missing required columns: {required_columns}"), ""
        
        # Step 4: Create output directory
        progress(0.15, "Preparing output directory...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(BASE_RESULTS_DIR or "results", f"uploaded_run_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 5: Sample dataset if requested
        original_size = len(df)
        if sample_size and sample_size > 0 and sample_size < len(df):
            progress(0.18, f"Sampling {int(sample_size)} rows from {original_size} total...")
            df = df.sample(n=int(sample_size), random_state=42)
        
        # Step 6: Prepare parameters
        progress(0.2, "Configuring pipeline...")
        
        # Handle optional parameters
        groupby_param = groupby_column.strip() if groupby_column and groupby_column.strip() else None
        
        # Step 7: Run the pipeline
        progress(0.25, "Starting pipeline execution...")
        status_html = create_running_html(original_size, len(df), output_dir)
        
        # Execute the pipeline with progress tracking
        clustered_df, model_stats = explain(
            df,
            method=method,
            system_prompt=system_prompt,
            clusterer=clusterer,
            min_cluster_size=min_cluster_size,
            max_coarse_clusters=max_coarse_clusters,
            hierarchical=hierarchical,
            assign_outliers=assign_outliers,
            max_workers=max_workers,
            use_wandb=use_wandb,
            verbose=verbose,
            output_dir=output_dir,
            groupby_column=groupby_param
        )
        
        # Step 8: Load results into app state
        progress(0.95, "Loading results into dashboard...")
        
        # Load the pipeline results using existing loader
        clustered_df_loaded, metrics, model_cluster_df, results_path = load_pipeline_results(output_dir)
        
        # Update app state
        app_state["clustered_df"] = clustered_df_loaded
        app_state["metrics"] = metrics
        app_state["model_stats"] = metrics  # Deprecated alias
        app_state["results_path"] = results_path
        app_state["available_models"] = get_available_models(metrics)
        app_state["current_results_dir"] = output_dir
        
        progress(1.0, "Pipeline completed successfully!")
        
        # Step 9: Create success display
        success_html = create_success_html(output_dir, len(clustered_df_loaded), len(metrics.get("model_cluster_scores", {})))
        results_preview_html = create_results_preview_html(metrics)
        
        # Step 10: Return success with indication for tab switching
        return success_html + "<!-- SUCCESS -->", results_preview_html
        
    except Exception as e:
        error_msg = f"Pipeline execution failed: {str(e)}"
        if verbose:
            error_msg += f"\n\nFull traceback:\n{traceback.format_exc()}"
        return create_error_html(error_msg), ""


def run_label_pipeline_handler(
    input_method: str,
    uploaded_file: Any,
    file_path: str,
    taxonomy_json: str,
    model_name: str,
    sample_size: Optional[float],
    max_workers: int,
    use_wandb: bool,
    verbose: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True)
) -> Tuple[str, str]:
    """
    Handle fixed-taxonomy labeling execution with the provided parameters.
    """
    try:
        # Step 1: Validate and get input file path
        progress(0.05, "Validating input...")
        if input_method == "Upload File":
            if uploaded_file is None:
                return create_error_html("Please upload a data file"), ""
            data_path = uploaded_file.name
        else:
            if not file_path or not file_path.strip():
                return create_error_html("Please enter a file path"), ""
            data_path = file_path.strip()
            if not os.path.exists(data_path):
                return create_error_html(f"File not found: {data_path}"), ""

        # Ensure wandb disabled when not requested
        if not use_wandb:
            os.environ["WANDB_DISABLED"] = "true"
        else:
            os.environ.pop("WANDB_DISABLED", None)

        # Step 2: Load dataset
        progress(0.1, "Loading dataset...")
        try:
            if data_path.endswith('.jsonl'):
                df = pd.read_json(data_path, lines=True)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            elif data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            else:
                return create_error_html("Unsupported file format. Use JSONL, JSON, CSV, or Parquet"), ""
        except Exception as e:
            return create_error_html(f"Failed to load dataset: {str(e)}"), ""

        # Step 3: Validate dataset structure (single_model only for label)
        struct_err = validate_dataset_structure(df, method="single_model")
        if struct_err:
            return create_error_html(struct_err), ""

        # Step 4: Parse taxonomy JSON
        progress(0.15, "Parsing taxonomy...")
        import json as _json
        try:
            taxonomy = _json.loads(taxonomy_json) if isinstance(taxonomy_json, str) else taxonomy_json
            if not isinstance(taxonomy, dict) or not taxonomy:
                return create_error_html("Taxonomy must be a non-empty JSON object of {label: description}"), ""
        except Exception as e:
            return create_error_html(f"Invalid taxonomy JSON: {e}"), ""

        # Step 5: Create output directory
        progress(0.18, "Preparing output directory...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(BASE_RESULTS_DIR or "results", f"labeled_run_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # Step 6: Sample dataset if requested
        original_size = len(df)
        if sample_size and sample_size > 0 and sample_size < len(df):
            progress(0.2, f"Sampling {int(sample_size)} rows from {original_size:,} total...")
            df = df.sample(n=int(sample_size), random_state=42)

        # Step 7: Run label()
        progress(0.25, "Starting labeling execution...")
        status_html = create_running_html(original_size, len(df), output_dir)

        clustered_df, model_stats = label(
            df,
            taxonomy=taxonomy,
            model_name=model_name or "gpt-4o-mini",
            max_workers=max_workers,
            use_wandb=use_wandb,
            verbose=verbose,
            output_dir=output_dir,
        )

        # Step 8: Load results into app state
        progress(0.95, "Loading results into dashboard...")
        clustered_df_loaded, metrics, model_cluster_df, results_path = load_pipeline_results(output_dir)

        app_state["clustered_df"] = clustered_df_loaded
        app_state["metrics"] = metrics
        app_state["model_stats"] = metrics
        app_state["results_path"] = results_path
        app_state["available_models"] = get_available_models(metrics)
        app_state["current_results_dir"] = output_dir

        progress(1.0, "Labeling completed successfully!")

        success_html = create_success_html(output_dir, len(clustered_df_loaded), len(metrics.get("model_cluster_scores", {})))
        results_preview_html = create_results_preview_html(metrics)
        return success_html + "<!-- SUCCESS -->", results_preview_html

    except Exception as e:
        error_msg = f"Labeling execution failed: {str(e)}"
        if verbose:
            import traceback as _tb
            error_msg += f"\n\nFull traceback:\n{_tb.format_exc()}"
        return create_error_html(error_msg), ""


def validate_dataset_structure(df: pd.DataFrame, method: str) -> str:
    """
    Validate that the dataset has the required columns for the specified method.
    
    Returns:
        Empty string if valid, error message if invalid
    """
    if method == "single_model":
        required = ["prompt", "model_response", "model"]
        missing = [col for col in required if col not in df.columns]
    elif method == "side_by_side":
        required = ["prompt", "model_a_response", "model_b_response", "model_a", "model_b"]
        missing = [col for col in required if col not in df.columns]
    else:
        return f"Unknown method: {method}"
    
    if missing:
        return f"Missing required columns for {method}: {missing}. Available columns: {list(df.columns)}"
    
    return ""


def create_error_html(message: str) -> str:
    """Create HTML for error display."""
    return f"""
    <div style='color: #d32f2f; background-color: #ffebee; padding: 16px; border-radius: 8px; border-left: 4px solid #d32f2f;'>
        <strong>Error</strong><br>
        <pre style='color: #d32f2f; margin-top: 8px; white-space: pre-wrap;'>{message}</pre>
    </div>
    """


def create_running_html(original_size: int, processed_size: int, output_dir: str) -> str:
    """Create HTML for running status display."""
    return f"""
    <div style='color: #1976d2; background-color: #e3f2fd; padding: 16px; border-radius: 8px; border-left: 4px solid #1976d2;'>
        <strong>Pipeline Running</strong><br>
        <div style='margin-top: 8px;'>
            • Processing: {processed_size:,} conversations
            {f"(sampled from {original_size:,})" if processed_size < original_size else ""}
            <br>
            • Output directory: <code>{output_dir}</code>
            <br>
            • Status: Extracting properties and clustering...
        </div>
    </div>
    """


def create_success_html(output_dir: str, n_properties: int, n_models: int) -> str:
    """Create HTML for success display."""
    return f"""
    <div style='color: #388e3c; background-color: #e8f5e8; padding: 16px; border-radius: 8px; border-left: 4px solid #388e3c;'>
        <strong>Pipeline Completed Successfully!</strong><br>
        <div style='margin-top: 8px;'>
            • Extracted properties: {n_properties:,}
            <br>
            • Models analyzed: {n_models}
            <br>
            • Results saved to: <code>{output_dir}</code>
            <br><br>
            <strong>Results are now loaded in the dashboard!</strong><br>
            Switch to other tabs to explore your results:
            <br>
            <strong>Overview</strong> - Model performance summary
            <br>
            <strong>View Clusters</strong> - Explore behavior clusters
            <br>
            <strong>View Examples</strong> - Browse specific examples
            <br>
            <strong>Plots</strong> - Interactive visualizations
        </div>
    </div>
    """


def create_results_preview_html(metrics: dict) -> str:
    """Create HTML preview of the results."""
    if not metrics or "model_cluster_scores" not in metrics:
        return ""
    
    model_scores = metrics["model_cluster_scores"]
    n_models = len(model_scores)
    
    # Get top models by some metric (if available)
    preview_html = f"""
    <div style='background-color: #f5f5f5; padding: 16px; border-radius: 8px; margin-top: 16px;'>
        <strong>Results Preview</strong><br>
        <div style='margin-top: 8px;'>
            <strong>Models analyzed:</strong> {n_models}<br>
    """
    
    # Show first few models
    model_names = list(model_scores.keys())[:5]
    if model_names:
        preview_html += f"<strong>Sample models:</strong> {', '.join(model_names)}"
        if len(model_scores) > 5:
            preview_html += f" and {len(model_scores) - 5} more..."
    
    preview_html += """
        </div>
    </div>
    """
    
    return preview_html


def get_directory_contents(directory: str) -> Tuple[List[str], str]:
    """
    Get directory contents for dropdown menu.
    
    Args:
        directory: Path to directory to list
        
    Returns:
        Tuple of (items_choices, empty_string)
        items_choices contains both directories (shown with trailing "/") and files
    """
    try:
        if not os.path.exists(directory) or not os.path.isdir(directory):
            error_html = f"""
            <div style='color: #d32f2f; padding: 16px;'>
                <strong>Error:</strong> Directory not found: {directory}
            </div>
            """
            return [], ""
        
        # Get directory contents
        try:
            entries = sorted(os.listdir(directory))
        except PermissionError:
            error_html = f"""
            <div style='color: #d32f2f; padding: 16px;'>
                <strong>Error:</strong> Permission denied accessing: {directory}
            </div>
            """
            return [], ""
        
        # Separate directories and files, create dropdown choices
        directories = []
        files = []
        items_choices = []
        
        for entry in entries:
            if entry.startswith('.'):  # Skip hidden files/dirs
                continue
                
            full_path = os.path.join(directory, entry)
            
            try:
                if os.path.isdir(full_path):
                    directories.append(entry)
                    items_choices.append(f"{entry}/")
                elif entry.lower().endswith(('.jsonl', '.json', '.csv', '.parquet')):
                    # Only show supported file types
                    files.append(entry)
                    items_choices.append(entry)
            except (OSError, PermissionError):
                continue  # Skip inaccessible items
        
        return items_choices, ""
        
    except Exception as e:
        error_html = f"""
        <div style='color: #d32f2f; padding: 16px;'>
            <strong>Error listing directory:</strong> {str(e)}
        </div>
        """
        return [], ""
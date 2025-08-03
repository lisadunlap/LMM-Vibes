"""
Utilities for the "Load Data" tab – loading pipeline results and scanning for
available experiment folders.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import gradio as gr

from .state import app_state, BASE_RESULTS_DIR
from .data_loader import (
    load_pipeline_results,
    scan_for_result_subfolders,
    validate_results_directory,
    get_available_models,
)

__all__ = [
    "load_data",
    "get_available_experiments",
    "get_experiment_choices",
    "refresh_experiment_dropdown",
    "load_experiment_data",
]


def load_data(results_dir: str) -> Tuple[str, str, str]:
    """Load pipeline results from *results_dir* and update the shared *app_state*.

    Returns a tuple of (summary_markdown, models_info_markdown, models_checkbox_update).
    """
    try:
        # 1. Validate directory structure
        is_valid, error_msg = validate_results_directory(results_dir)
        if not is_valid:
            return "", f"❌ Error: {error_msg}", ""

        # 2. Handle optional sub-folder selection (first match for now)
        subfolders = scan_for_result_subfolders(results_dir)
        final_dir = results_dir
        if subfolders and "." not in subfolders:
            final_dir = str(Path(results_dir) / subfolders[0])

        # 3. Load results into memory
        clustered_df, model_stats, results_path = load_pipeline_results(final_dir)

        # 4. Stash in global state so other tabs can use it
        app_state["clustered_df"] = clustered_df
        app_state["model_stats"] = model_stats
        app_state["results_path"] = results_path
        app_state["available_models"] = get_available_models(model_stats)
        app_state["current_results_dir"] = final_dir

        # 5. Compose status messages
        n_models = len(model_stats)
        n_properties = len(clustered_df)

        summary = f"""
        ✅ **Successfully loaded pipeline results!**

        **Data Summary:**
        - **Models:** {n_models}
        - **Properties:** {n_properties:,}
        - **Results Directory:** {Path(final_dir).name}
        """
        if "fine_cluster_id" in clustered_df.columns:
            n_fine_clusters = clustered_df["fine_cluster_id"].nunique()
            summary += f"\n- **Fine Clusters:** {n_fine_clusters}"
        if "coarse_cluster_id" in clustered_df.columns:
            n_coarse_clusters = clustered_df["coarse_cluster_id"].nunique()
            summary += f"\n- **Coarse Clusters:** {n_coarse_clusters}"

        model_choices = app_state["available_models"]
        models_info = f"Available models: {', '.join(model_choices)}"

        # Gradio update object for the CheckboxGroup
        return summary, models_info, gr.update(choices=model_choices, value=model_choices[:5])

    except Exception as e:
        error_msg = f"❌ Error loading results: {e}"
        return "", error_msg, gr.update(choices=[], value=[])


def get_available_experiments(base_dir: str) -> List[str]:
    """Return experiment sub-directories that contain the expected result files."""
    if not base_dir or not os.path.exists(base_dir):
        return []

    experiments: List[str] = []
    try:
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                if (
                    os.path.exists(os.path.join(item_path, "model_stats.json"))
                    or os.path.exists(os.path.join(item_path, "clustered_results.jsonl"))
                ):
                    experiments.append(item)
    except Exception as e:
        print(f"Error scanning experiments: {e}")

    return sorted(experiments)


def get_experiment_choices() -> List[str]:
    """Return dropdown choices for the experiment selector."""
    if not BASE_RESULTS_DIR:
        return []
    experiments = get_available_experiments(BASE_RESULTS_DIR)
    return ["Select an experiment..."] + experiments


def refresh_experiment_dropdown() -> gr.update:
    """Gradio helper to refresh the experiment dropdown choices."""
    choices = get_experiment_choices()
    return gr.update(choices=choices, value="Select an experiment...")


def load_experiment_data(experiment_name: str) -> Tuple[str, str, str]:
    """Wrapper used by Gradio events to load a *selected* experiment."""
    if not BASE_RESULTS_DIR or experiment_name == "Select an experiment...":
        return "", "Please select a valid experiment", gr.update(choices=[], value=[])

    experiment_path = os.path.join(BASE_RESULTS_DIR, experiment_name)
    print(f"🔍 Loading experiment: {experiment_name} from {experiment_path}")
    return load_data(experiment_path) 
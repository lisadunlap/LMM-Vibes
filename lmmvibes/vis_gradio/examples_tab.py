"""Logic for the **View Examples** tab – dropdown population + example renderer."""
from __future__ import annotations

from typing import Any, List, Tuple

import gradio as gr

from .state import app_state
from .utils import (
    get_unique_values_for_dropdowns,
    get_example_data,
    format_examples_display,
)

__all__: List[str] = [
    "get_dropdown_choices",
    "update_example_dropdowns",
    "view_examples",
    "get_filter_options",
    "update_filter_dropdowns",
]


# ---------------------------------------------------------------------------
# Dropdown helpers
# ---------------------------------------------------------------------------

def get_dropdown_choices() -> Tuple[List[str], List[str], List[str]]:
    if app_state["clustered_df"] is None:
        return [], [], []

    choices = get_unique_values_for_dropdowns(app_state["clustered_df"])
    prompts = ["All Prompts"] + choices["prompts"]
    models = ["All Models"] + choices["models"]
    properties = ["All Clusters"] + choices["properties"]
    return prompts, models, properties


def update_example_dropdowns() -> Tuple[Any, Any, Any]:
    prompts, models, properties = get_dropdown_choices()
    return (
        gr.update(choices=prompts, value="All Prompts" if prompts else None),
        gr.update(choices=models, value="All Models" if models else None),
        gr.update(choices=properties, value="All Clusters" if properties else None),
    )


# ---------------------------------------------------------------------------
# Example viewer
# ---------------------------------------------------------------------------

def view_examples(
    selected_prompt: str,
    selected_model: str,
    selected_property: str,
    max_examples: int = 5,
    use_accordion: bool = True,
) -> str:
    if app_state["clustered_df"] is None:
        return (
            "<p style='color: #e74c3c; padding: 20px;'>❌ Please load data first "
            "using the 'Load Data' tab</p>"
        )

    examples = get_example_data(
        app_state["clustered_df"],
        selected_prompt if selected_prompt != "All Prompts" else None,
        selected_model if selected_model != "All Models" else None,
        selected_property if selected_property != "All Clusters" else None,
        max_examples,
    )

    return format_examples_display(
        examples,
        selected_prompt,
        selected_model,
        selected_property,
        use_accordion=use_accordion,
    )


# ---------------------------------------------------------------------------
# Filter dropdown helpers for frequency comparison
# ---------------------------------------------------------------------------

def get_filter_options() -> Tuple[List[str], List[str]]:
    if not app_state["model_stats"]:
        return ["All Models"], ["All Metrics"]

    available_models = ["All Models"] + list(app_state["model_stats"].keys())

    quality_metrics = set()
    for model_data in app_state["model_stats"].values():
        clusters = model_data.get("fine", []) + model_data.get("coarse", [])
        for cluster in clusters:
            quality_score = cluster.get("quality_score", {})
            if isinstance(quality_score, dict):
                quality_metrics.update(quality_score.keys())

    available_metrics = ["All Metrics"] + sorted(list(quality_metrics))

    return available_models, available_metrics


def update_filter_dropdowns() -> Tuple[Any, Any]:
    models, metrics = get_filter_options()
    return (
        gr.update(choices=models, value="All Models" if models else None),
        gr.update(choices=metrics, value="All Metrics" if metrics else None),
    ) 
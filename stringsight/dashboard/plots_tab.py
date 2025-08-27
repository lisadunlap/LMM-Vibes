"""
Plots tab for the LMM-Vibes Gradio app.

This module provides functionality to display the model cluster proportion and quality plots.
"""

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, List, Optional, Any

from .state import app_state
from .utils import extract_allowed_tag, ALLOWED_TAGS


def create_proportion_plot(selected_clusters: Optional[List[str]] = None, show_ci: bool = False, selected_models: Optional[List[str]] = None, selected_tags: Optional[List[str]] = None) -> Tuple[go.Figure, str]:
    """Create a grouped bar plot of proportion by property and model."""
    if app_state.get("model_cluster_df") is None:
        return None, "No model cluster data loaded. Please load data first."
    
    model_cluster_df = app_state["model_cluster_df"]
    
    if model_cluster_df.empty:
        return None, "No model cluster data available."
    
    # Ensure proportion values are numeric and in reasonable range
    model_cluster_df = model_cluster_df.copy()

    # Optional: filter to selected models (ignore the pseudo 'all' entry if present)
    if selected_models:
        concrete_models = [m for m in selected_models if m != "all"]
        if concrete_models:
            model_cluster_df = model_cluster_df[model_cluster_df["model"].isin(concrete_models)]
    model_cluster_df['proportion'] = pd.to_numeric(model_cluster_df['proportion'], errors='coerce')
    
    # Check for any unreasonable values
    print("After conversion - Proportion range:", model_cluster_df['proportion'].min(), "to", model_cluster_df['proportion'].max())
    
    # Filter out "No properties" clusters
    model_cluster_df = model_cluster_df[model_cluster_df['cluster'] != "No properties"]

    # Optional: filter clusters by selected tags using metrics.cluster_scores metadata
    if selected_tags:
        metrics = app_state.get("metrics", {})
        cluster_scores = metrics.get("cluster_scores", {})
        def _first_allowed(meta_obj: Any) -> Any:
            return extract_allowed_tag(meta_obj)
        allowed = set(map(str, selected_tags))
        allowed_clusters = {c for c, d in cluster_scores.items() if str(_first_allowed(d.get("metadata"))) in allowed}
        if allowed_clusters:
            model_cluster_df = model_cluster_df[model_cluster_df['cluster'].isin(allowed_clusters)]

    # Determine which clusters to include: user-selected or default top 15 by aggregated frequency
    cluster_freq = (
        model_cluster_df.groupby('cluster', as_index=False)['proportion']
        .sum()
        .sort_values('proportion', ascending=False)
    )
    all_available_clusters = cluster_freq['cluster'].tolist()
    chosen_clusters: List[str] = []
    if selected_clusters:
        chosen_clusters = [c for c in selected_clusters if c in cluster_freq['cluster'].tolist()]
        model_cluster_df = model_cluster_df[model_cluster_df['cluster'].isin(chosen_clusters)]
    else:
        default_top = cluster_freq['cluster'].head(15).tolist() if len(cluster_freq) > 15 else cluster_freq['cluster'].tolist()
        chosen_clusters = default_top
        model_cluster_df = model_cluster_df[model_cluster_df['cluster'].isin(default_top)]

    # Decide whether to abbreviate property names based on word count
    # If any property name has more than 6 words, we abbreviate (P1, P2, ...)
    unique_properties = sorted(model_cluster_df['cluster'].unique())
    should_abbreviate = any(len(str(prop).split()) > 6 for prop in unique_properties)

    # Build display legend and status text
    mapping_text_parts: List[str] = []
    is_showing_all = set(chosen_clusters) == set(all_available_clusters) and len(all_available_clusters) > 0
    if is_showing_all:
        status_text = (
            f'<div style="background-color:#e7f3ff;border:1px solid #2f6fef;color:#1f4bd6;padding:8px 12px;border-radius:6px;font-weight:600;display:inline-block;margin-bottom:8px;">'
            f'Cluster scope: All clusters ({len(all_available_clusters)}). '
            f'<span style="font-weight:400">Tip: adjust selection in "Select properties to display" above.</span>'
            f'</div>\n\n'
        )
    else:
        status_text = (
            f'<div style="background-color:#e7f3ff;border:1px solid #2f6fef;color:#1f4bd6;padding:8px 12px;border-radius:6px;font-weight:600;display:inline-block;margin-bottom:8px;">'
            f'Cluster scope: Selected {len(chosen_clusters)} of {len(all_available_clusters)}. '
            f'<span style="font-weight:400">Select other properties above.</span>'
            f'</div>\n\n'
        )
    mapping_text_parts.append(status_text)
    if should_abbreviate:
        property_mapping = {prop: f"P{i+1}" for i, prop in enumerate(unique_properties)}
        model_cluster_df['display_label'] = model_cluster_df['cluster'].map(property_mapping)
        # Prepare mapping legend text
        mapping_text_parts.append("**Property Mapping**\n\n")
        for prop, abbr in property_mapping.items():
            mapping_text_parts.append(f"**{abbr}:** {prop}\n\n")
    else:
        # Use full names directly as x tick labels
        model_cluster_df['display_label'] = model_cluster_df['cluster']
    
    # Prepare confidence interval data if requested
    error_y_data = None
    if show_ci and 'proportion_ci_lower' in model_cluster_df.columns and 'proportion_ci_upper' in model_cluster_df.columns:
        # Calculate error bar values
        model_cluster_df['y_error'] = model_cluster_df['proportion_ci_upper'] - model_cluster_df['proportion']
        model_cluster_df['y_error_minus'] = model_cluster_df['proportion'] - model_cluster_df['proportion_ci_lower']
        # Replace NaN values with 0
        model_cluster_df['y_error'] = model_cluster_df['y_error'].fillna(0)
        model_cluster_df['y_error_minus'] = model_cluster_df['y_error_minus'].fillna(0)
        error_y_data = model_cluster_df['y_error']
        error_y_minus_data = model_cluster_df['y_error_minus']
    
    # Create a grouped bar plot of 'proportion' by property (x) and model (hue)
    fig = px.bar(
        model_cluster_df,
        x="display_label",
        y="proportion",
        color="model",
        barmode="group",
        title=None,
        labels={"proportion": "Proportion", "display_label": "Property", "model": "Model"},
        error_y="y_error" if error_y_data is not None else None,
        error_y_minus="y_error_minus" if error_y_data is not None else None
    )
    
    # Set the x-axis order to ensure consistent ordering
    property_order = [f"P{i+1}" for i in range(len(unique_properties))] if should_abbreviate else unique_properties
    fig.update_xaxes(categoryorder='array', categoryarray=property_order)
    fig.update_layout(xaxis_tickangle=45)
    # Make layout responsive and move legend to the top to utilize full width
    fig.update_layout(
        autosize=True,
        margin=dict(l=40, r=40, t=110, b=80),
        title=dict(pad=dict(t=20, b=10)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="left",
            x=0
        )
    )
    
    # save figure to file
    fig.write_html("model_cluster_proportion_plot.html")
    
    # Build info/legend text
    if show_ci:
        if 'proportion_ci_lower' in model_cluster_df.columns and 'proportion_ci_upper' in model_cluster_df.columns:
            if mapping_text_parts:
                mapping_text_parts.append("---\n\n")
            mapping_text_parts.append("**Confidence Intervals:**\n")
            mapping_text_parts.append("Error bars show 95% confidence intervals for proportion values.\n")
        else:
            if mapping_text_parts:
                mapping_text_parts.append("---\n\n")
            mapping_text_parts.append("**Note:** Confidence interval data not available in the loaded dataset.\n")

    mapping_text = "".join(mapping_text_parts)
    
    return fig, mapping_text


def create_quality_plot(quality_metric: str = "helpfulness", selected_clusters: Optional[List[str]] = None, show_ci: bool = False, selected_models: Optional[List[str]] = None, selected_tags: Optional[List[str]] = None) -> Tuple[go.Figure, str]:
    """Create a grouped bar plot of quality by property and model."""
    if app_state.get("model_cluster_df") is None:
        return None, "No model cluster data loaded. Please load data first."
    
    model_cluster_df = app_state["model_cluster_df"]
    
    if model_cluster_df.empty:
        return None, "No model cluster data available."
    
    # Check if the quality metric exists in the data
    quality_col = f"quality_{quality_metric}"
    if quality_col not in model_cluster_df.columns:
        # Get available quality metrics for better error message
        available_metrics = [col.replace("quality_", "") for col in model_cluster_df.columns 
                           if col.startswith("quality_") 
                           and not col.endswith(("_ci_lower", "_ci_upper", "_ci_mean", "_significant", "_delta"))]
        if not available_metrics:
            return None, f"No quality metrics found in the data. Available columns: {list(model_cluster_df.columns)}"
        return None, f"Quality metric '{quality_metric}' not found. Available metrics: {available_metrics}"
    
    # Create a copy for plotting
    plot_df = model_cluster_df.copy()

    # Optional: filter clusters by selected tags using metrics.cluster_scores metadata
    if selected_tags:
        metrics = app_state.get("metrics", {})
        cluster_scores = metrics.get("cluster_scores", {})
        def _first_allowed(meta_obj: Any) -> Any:
            return extract_allowed_tag(meta_obj)
        allowed = set(map(str, selected_tags))
        allowed_clusters = {c for c, d in cluster_scores.items() if str(_first_allowed(d.get("metadata"))) in allowed}
        if allowed_clusters:
            plot_df = plot_df[plot_df['cluster'].isin(allowed_clusters)]

    # Optional: filter to selected models (ignore the pseudo 'all' entry if present)
    if selected_models:
        concrete_models = [m for m in selected_models if m != "all"]
        if concrete_models:
            plot_df = plot_df[plot_df["model"].isin(concrete_models)]
    
    # Ensure quality values are numeric
    plot_df[quality_col] = pd.to_numeric(plot_df[quality_col], errors='coerce')
    
    # Check if we have any valid quality data
    if plot_df[quality_col].isna().all():
        return None, f"No valid quality data found for metric '{quality_metric}'. All values are missing or invalid."
    
    # Filter out "No properties" clusters
    plot_df = plot_df[plot_df['cluster'] != "No properties"]

    # Determine which clusters to include: user-selected or default top 15 by aggregated frequency
    cluster_freq = (
        plot_df[plot_df['cluster'] != "No properties"]
        .groupby('cluster', as_index=False)['proportion']
        .sum()
        .sort_values('proportion', ascending=False)
    )
    all_available_clusters = cluster_freq['cluster'].tolist()
    chosen_clusters: List[str] = []
    if selected_clusters:
        chosen_clusters = [c for c in selected_clusters if c in cluster_freq['cluster'].tolist()]
        plot_df = plot_df[plot_df['cluster'].isin(chosen_clusters)]
    else:
        default_top = cluster_freq['cluster'].head(15).tolist() if len(cluster_freq) > 15 else cluster_freq['cluster'].tolist()
        chosen_clusters = default_top
        plot_df = plot_df[plot_df['cluster'].isin(default_top)]

    # Decide whether to abbreviate property names based on word count
    unique_properties = sorted(plot_df['cluster'].unique())
    should_abbreviate = any(len(str(prop).split()) > 6 for prop in unique_properties)

    # Build display legend and status text
    mapping_text_parts: List[str] = []
    is_showing_all = set(chosen_clusters) == set(all_available_clusters) and len(all_available_clusters) > 0
    if is_showing_all:
        status_text = (
            f'<div style="background-color:#e7f3ff;border:1px solid #2f6fef;color:#1f4bd6;padding:8px 12px;border-radius:6px;font-weight:600;display:inline-block;margin-bottom:8px;">'
            f'Cluster scope: All clusters ({len(all_available_clusters)}). '
            f'<span style="font-weight:400">Tip: adjust selection in "Select properties to display" above.</span>'
            f'</div>\n\n'
        )
    else:
        status_text = (
            f'<div style="background-color:#e7f3ff;border:1px solid #2f6fef;color:#1f4bd6;padding:8px 12px;border-radius:6px;font-weight:600;display:inline-block;margin-bottom:8px;">'
            f'Cluster scope: Selected {len(chosen_clusters)} of {len(all_available_clusters)}. '
            f'<span style="font-weight:400">Select other properties above.</span>'
            f'</div>\n\n'
        )
    mapping_text_parts.append(status_text)
    if should_abbreviate:
        property_mapping = {prop: f"P{i+1}" for i, prop in enumerate(unique_properties)}
        plot_df['display_label'] = plot_df['cluster'].map(property_mapping)
        # Prepare mapping legend text
        mapping_text_parts.append("**Property Mapping:**\n\n")
        for prop, abbr in property_mapping.items():
            mapping_text_parts.append(f"**{abbr}:** {prop}\n\n")
    else:
        plot_df['display_label'] = plot_df['cluster']
    
    # Prepare confidence interval data if requested
    error_y_data = None
    if show_ci:
        ci_lower_col = f"{quality_col}_ci_lower"
        ci_upper_col = f"{quality_col}_ci_upper"
        if ci_lower_col in plot_df.columns and ci_upper_col in plot_df.columns:
            # Calculate error bar values
            plot_df['y_error'] = plot_df[ci_upper_col] - plot_df[quality_col]
            plot_df['y_error_minus'] = plot_df[quality_col] - plot_df[ci_lower_col]
            # Replace NaN values with 0
            plot_df['y_error'] = plot_df['y_error'].fillna(0)
            plot_df['y_error_minus'] = plot_df['y_error_minus'].fillna(0)
            error_y_data = plot_df['y_error']
            error_y_minus_data = plot_df['y_error_minus']
    
    # Create a grouped bar plot of quality by property (x) and model (hue)
    fig = px.bar(
        plot_df,
        x="display_label",
        y=quality_col,
        color="model",
        barmode="group",
        title=None,
        labels={quality_col: f"Quality ({quality_metric.title()})", "display_label": "Property", "model": "Model"},
        error_y="y_error" if error_y_data is not None else None,
        error_y_minus="y_error_minus" if error_y_data is not None else None
    )
    
    # Set the x-axis order to ensure consistent ordering
    property_order = [f"P{i+1}" for i in range(len(unique_properties))] if should_abbreviate else unique_properties
    fig.update_xaxes(categoryorder='array', categoryarray=property_order)
    fig.update_layout(xaxis_tickangle=45)
    # Make layout responsive and move legend to the top to utilize full width
    fig.update_layout(
        autosize=True,
        margin=dict(l=40, r=40, t=110, b=80),
        title=dict(pad=dict(t=20, b=10)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="left",
            x=0
        )
    )
    
    # save figure to file
    fig.write_html(f"model_cluster_quality_{quality_metric}_plot.html")
    
    # Build info/legend text
    if show_ci:
        ci_lower_col = f"{quality_col}_ci_lower"
        ci_upper_col = f"{quality_col}_ci_upper"
        if ci_lower_col in plot_df.columns and ci_upper_col in plot_df.columns:
            if mapping_text_parts:
                mapping_text_parts.append("---\n\n")
            mapping_text_parts.append("**Confidence Intervals:**\n")
            mapping_text_parts.append(f"Error bars show 95% confidence intervals for {quality_metric} values.\n")
        else:
            if mapping_text_parts:
                mapping_text_parts.append("---\n\n")
            mapping_text_parts.append("**Note:** Confidence interval data not available for this quality metric.\n")

    mapping_text = "".join(mapping_text_parts)
    
    return fig, mapping_text


def get_available_quality_metrics() -> List[str]:
    """Get available quality metrics from the loaded DataFrame."""
    if app_state.get("model_cluster_df") is None:
        return ["helpfulness", "accuracy", "harmlessness", "honesty"]
    
    model_cluster_df = app_state["model_cluster_df"]
    # Find all quality columns (excluding CI and other suffix columns)
    quality_columns = [
        col for col in model_cluster_df.columns
        if col.startswith("quality_")
        and not col.endswith(("_ci_lower", "_ci_upper", "_ci_mean", "_significant", "_delta"))
        and ("delta" not in col.lower())
    ]
    # Extract metric names by removing "quality_" prefix
    available_quality_metrics = [col.replace("quality_", "") for col in quality_columns]
    
    # If no quality metrics found, provide defaults
    if not available_quality_metrics:
        available_quality_metrics = ["helpfulness", "accuracy", "harmlessness", "honesty"]
    
    return available_quality_metrics


def update_quality_metric_dropdown() -> gr.Dropdown:
    """Update the quality metric dropdown with available metrics."""
    available_metrics = get_available_quality_metrics()
    return gr.Dropdown(
        label="Quality Metric",
        choices=available_metrics,
        value=available_metrics[0] if available_metrics else "helpfulness",
        info="Select which quality metric to display"
    )


def update_quality_metric_visibility(plot_type: str) -> gr.Dropdown:
    """Update the quality metric dropdown visibility based on plot type."""
    if plot_type == "quality":
        available_metrics = get_available_quality_metrics()
        return gr.update(
            choices=available_metrics,
            value=(available_metrics[0] if available_metrics else None),
            visible=True,
        )
    # When not in quality mode, clear value and choices to avoid stale selections
    return gr.update(choices=[], value=None, visible=False)


def create_plot_with_toggle(plot_type: str, quality_metric: str = "helpfulness", selected_clusters: Optional[List[str]] = None, show_ci: bool = False, selected_models: Optional[List[str]] = None, selected_tags: Optional[List[str]] = None) -> Tuple[go.Figure, str]:
    """Create a plot based on the selected type (frequency or quality)."""
    if plot_type == "frequency":
        return create_proportion_plot(selected_clusters, show_ci, selected_models, selected_tags)
    elif plot_type == "quality":
        return create_quality_plot(quality_metric, selected_clusters, show_ci, selected_models, selected_tags)
    else:
        return None, f"Unknown plot type: {plot_type}"


def create_plots_tab() -> Tuple[gr.Plot, gr.Markdown, gr.Checkbox, gr.Dropdown, gr.Dropdown, gr.CheckboxGroup]:
    """Create the plots tab interface with a toggle between frequency and quality plots."""
    # Accordion at the top for selecting specific properties
    with gr.Accordion("Select properties to display", open=False):
        cluster_selector = gr.CheckboxGroup(
            label="Select Clusters (Properties)",
            choices=[],
            value=[],
            info="Defaults to the top 15 by frequency.",
            show_label=False,
            elem_id="plot-clusters"
        )

    # Plot controls in a row
    with gr.Row():
        # Plot type toggle
        plot_type_dropdown = gr.Dropdown(
            label="Plot Type",
            choices=["frequency", "quality"],
            value="frequency",
            info="Choose between frequency (proportion) or quality metrics"
        )
        
        # Quality metric dropdown (only visible for quality plots)
        quality_metric_dropdown = gr.Dropdown(
            label="Quality Metric",
            choices=[],
            value=None,
            info="Select which quality metric to display",
            visible=False  # Initially hidden, shown when quality is selected
        )


    # Add checkbox for confidence intervals
    show_ci_checkbox = gr.Checkbox(
        label="Show Confidence Intervals",
        value=False,
        info="Display 95% confidence intervals as error bars (if available in data)"
    )
    
    plot_display = gr.Plot(
        label="Model-Cluster Analysis Plot",
        show_label=False,
        value=None
    )
    
    # Mapping text should appear directly below the plot
    plot_info = gr.Markdown("")
    
    return plot_display, plot_info, show_ci_checkbox, plot_type_dropdown, quality_metric_dropdown, cluster_selector


def update_cluster_selection(selected_models: Optional[List[str]] = None, selected_tags: Optional[List[str]] = None) -> Any:
    """Populate the cluster selector choices and default selection (top 15 by frequency).

    If selected_models is provided, restrict clusters to those computed from the selected models.
    """
    if app_state.get("model_cluster_df") is None:
        return gr.update(choices=[], value=[])
    df = app_state["model_cluster_df"]
    # Optional: filter to selected models (ignore the pseudo 'all' entry if present)
    if selected_models:
        concrete_models = [m for m in selected_models if m != "all"]
        if concrete_models:
            df = df[df["model"].isin(concrete_models)]
    # Optional: filter by selected tags using cluster_scores metadata
    if selected_tags:
        metrics = app_state.get("metrics", {})
        cluster_scores = metrics.get("cluster_scores", {})
        def _first_allowed(meta_obj: Any) -> Any:
            return extract_allowed_tag(meta_obj)
        allowed = set(map(str, selected_tags))
        allowed_clusters = {c for c, d in cluster_scores.items() if str(_first_allowed(d.get("metadata"))) in allowed}
        if allowed_clusters:
            df = df[df['cluster'].isin(allowed_clusters)]

    if df.empty or 'cluster' not in df.columns or 'proportion' not in df.columns:
        return gr.update(choices=[], value=[])
    # Exclude "No properties"
    df = df[df['cluster'] != "No properties"].copy()
    freq = (
        df.groupby('cluster', as_index=False)['proportion']
        .sum()
        .sort_values('proportion', ascending=False)
    )
    clusters_ordered = freq['cluster'].tolist()
    # Build label-value tuples; strip '**' from labels only (values remain raw)
    label_value_choices = []
    for cluster in clusters_ordered:
        raw_val = str(cluster)
        label = raw_val.replace('**', '')
        label_value_choices.append((label, raw_val))
    default_values = [str(cluster) for cluster in clusters_ordered[:15]]
    return gr.update(choices=label_value_choices, value=default_values)
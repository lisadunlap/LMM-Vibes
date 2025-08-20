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


def create_proportion_plot(selected_clusters: Optional[List[str]] = None, show_ci: bool = False, selected_models: Optional[List[str]] = None, selected_tags: Optional[List[str]] = None) -> Tuple[go.Figure, str]:
    """Create a grouped bar plot of proportion by property and model."""
    if app_state.get("model_cluster_df") is None:
        return None, "No model cluster data loaded. Please load data first."
    
    model_cluster_df = app_state["model_cluster_df"]
    print("DataFrame shape:", model_cluster_df.shape)
    print("Columns:", model_cluster_df.columns.tolist())
    print("Proportion range:", model_cluster_df['proportion'].min(), "to", model_cluster_df['proportion'].max())
    print("Sample data:")
    print(model_cluster_df[['model', 'cluster', 'proportion']].head(10))
    
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
    print("Proportion values > 1:", (model_cluster_df['proportion'] > 1).sum())
    print("Proportion values < 0:", (model_cluster_df['proportion'] < 0).sum())
    
    # Filter out "No properties" clusters
    model_cluster_df = model_cluster_df[model_cluster_df['cluster'] != "No properties"]

    # Optional: filter clusters by selected tags using metrics.cluster_scores metadata
    if selected_tags:
        metrics = app_state.get("metrics", {})
        cluster_scores = metrics.get("cluster_scores", {})
        def _first_meta_val(meta_obj: Any) -> Any:
            if meta_obj is None:
                return None
            if isinstance(meta_obj, str):
                try:
                    import ast as _ast
                    meta_obj = _ast.literal_eval(meta_obj)
                except Exception:
                    return meta_obj
            if isinstance(meta_obj, dict):
                for _, v in meta_obj.items():
                    return v
                return None
            if isinstance(meta_obj, (list, tuple)):
                return meta_obj[0] if len(meta_obj) > 0 else None
            return meta_obj
        allowed = set(map(str, selected_tags))
        allowed_clusters = {c for c, d in cluster_scores.items() if str(_first_meta_val(d.get("metadata"))) in allowed}
        if allowed_clusters:
            model_cluster_df = model_cluster_df[model_cluster_df['cluster'].isin(allowed_clusters)]

    # Determine which clusters to include: user-selected or default top 15 by aggregated frequency
    cluster_freq = (
        model_cluster_df.groupby('cluster', as_index=False)['proportion']
        .sum()
        .sort_values('proportion', ascending=False)
    )
    if selected_clusters:
        chosen_clusters = [c for c in selected_clusters if c in cluster_freq['cluster'].tolist()]
        model_cluster_df = model_cluster_df[model_cluster_df['cluster'].isin(chosen_clusters)]
    else:
        default_top = cluster_freq['cluster'].head(15).tolist() if len(cluster_freq) > 15 else cluster_freq['cluster'].tolist()
        model_cluster_df = model_cluster_df[model_cluster_df['cluster'].isin(default_top)]

    # Create property name mapping with proper ordering for the filtered set
    unique_properties = sorted(model_cluster_df['cluster'].unique())
    property_mapping = {prop: f"P{i+1}" for i, prop in enumerate(unique_properties)}
    
    # Create abbreviated property column for plotting
    model_cluster_df['property_abbr'] = model_cluster_df['cluster'].map(property_mapping)
    
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
        x="property_abbr",
        y="proportion",
        color="model",
        barmode="group",
        title="Proportion by Property and Model",
        labels={"proportion": "Proportion", "property_abbr": "Property", "model": "Model"},
        error_y="y_error" if error_y_data is not None else None,
        error_y_minus="y_error_minus" if error_y_data is not None else None
    )
    
    # Set the x-axis order to ensure P1, P2, P3, etc.
    property_order = [f"P{i+1}" for i in range(len(unique_properties))]
    fig.update_xaxes(categoryorder='array', categoryarray=property_order)
    fig.update_layout(xaxis_tickangle=45)
    
    # save figure to file
    fig.write_html("model_cluster_proportion_plot.html")
    
    # Create property mapping string
    mapping_text = "**Property Mapping**\n\n"
    for prop, abbr in property_mapping.items():
        mapping_text += f"**{abbr}:** {prop}\n\n"
    
    # Add confidence interval info if enabled
    if show_ci:
        if 'proportion_ci_lower' in model_cluster_df.columns and 'proportion_ci_upper' in model_cluster_df.columns:
            mapping_text += "---\n\n**Confidence Intervals:**\n"
            mapping_text += "Error bars show 95% confidence intervals for proportion values.\n"
        else:
            mapping_text += "---\n\n**Note:** Confidence interval data not available in the loaded dataset.\n"
    
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
        def _first_meta_val(meta_obj: Any) -> Any:
            if meta_obj is None:
                return None
            if isinstance(meta_obj, str):
                try:
                    import ast as _ast
                    meta_obj = _ast.literal_eval(meta_obj)
                except Exception:
                    return meta_obj
            if isinstance(meta_obj, dict):
                for _, v in meta_obj.items():
                    return v
                return None
            if isinstance(meta_obj, (list, tuple)):
                return meta_obj[0] if len(meta_obj) > 0 else None
            return meta_obj
        allowed = set(map(str, selected_tags))
        allowed_clusters = {c for c, d in cluster_scores.items() if str(_first_meta_val(d.get("metadata"))) in allowed}
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
    if selected_clusters:
        chosen_clusters = [c for c in selected_clusters if c in cluster_freq['cluster'].tolist()]
        plot_df = plot_df[plot_df['cluster'].isin(chosen_clusters)]
    else:
        default_top = cluster_freq['cluster'].head(15).tolist() if len(cluster_freq) > 15 else cluster_freq['cluster'].tolist()
        plot_df = plot_df[plot_df['cluster'].isin(default_top)]

    # Create property name mapping with proper ordering (same as proportion plot)
    unique_properties = sorted(plot_df['cluster'].unique())
    property_mapping = {prop: f"P{i+1}" for i, prop in enumerate(unique_properties)}
    
    # Create abbreviated property column for plotting
    plot_df['property_abbr'] = plot_df['cluster'].map(property_mapping)
    
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
        x="property_abbr",
        y=quality_col,
        color="model",
        barmode="group",
        title=f"Quality ({quality_metric.title()}) by Property and Model",
        labels={quality_col: f"Quality ({quality_metric.title()})", "property_abbr": "Property", "model": "Model"},
        error_y="y_error" if error_y_data is not None else None,
        error_y_minus="y_error_minus" if error_y_data is not None else None
    )
    
    # Set the x-axis order to ensure P1, P2, P3, etc. (same as proportion plot)
    property_order = [f"P{i+1}" for i in range(len(unique_properties))]
    fig.update_xaxes(categoryorder='array', categoryarray=property_order)
    fig.update_layout(xaxis_tickangle=45)
    
    # save figure to file
    fig.write_html(f"model_cluster_quality_{quality_metric}_plot.html")
    
    # Create property mapping string (same as proportion plot)
    mapping_text = "**Property Mapping:**\n\n"
    for prop, abbr in property_mapping.items():
        mapping_text += f"**{abbr}:** {prop}\n\n"
    
    # Add confidence interval info if enabled
    if show_ci:
        ci_lower_col = f"{quality_col}_ci_lower"
        ci_upper_col = f"{quality_col}_ci_upper"
        if ci_lower_col in plot_df.columns and ci_upper_col in plot_df.columns:
            mapping_text += "---\n\n**Confidence Intervals:**\n"
            mapping_text += f"Error bars show 95% confidence intervals for {quality_metric} values.\n"
        else:
            mapping_text += "---\n\n**Note:** Confidence interval data not available for this quality metric.\n"
    
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
            show_label=False
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
        def _first_meta_val(meta_obj: Any) -> Any:
            if meta_obj is None:
                return None
            if isinstance(meta_obj, str):
                try:
                    import ast as _ast
                    meta_obj = _ast.literal_eval(meta_obj)
                except Exception:
                    return meta_obj
            if isinstance(meta_obj, dict):
                for _, v in meta_obj.items():
                    return v
                return None
            if isinstance(meta_obj, (list, tuple)):
                return meta_obj[0] if len(meta_obj) > 0 else None
            return meta_obj
        allowed = set(map(str, selected_tags))
        allowed_clusters = {c for c, d in cluster_scores.items() if str(_first_meta_val(d.get("metadata"))) in allowed}
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
"""
Plotting functionality for functional metrics.

This module provides comprehensive visualization of metrics from functional_metrics.py,
including interactive bar charts and heatmaps with confidence intervals, organized for wandb logging.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

import wandb

# Set plotly template
pio.templates.default = "plotly_white"
warnings.filterwarnings('ignore')


def create_model_cluster_dataframe(model_cluster_scores: Dict[str, Any]) -> pd.DataFrame:
    """Convert model-cluster scores to a tidy dataframe."""
    rows = []
    for model, clusters in model_cluster_scores.items():
        for cluster, metrics in clusters.items():
            row = {
                'model': model,
                'cluster': cluster,
                'size': metrics.get('size', 0),
                'proportion': metrics.get('proportion', 0),
                'proportion_delta': metrics.get('proportion_delta', 0)
            }
            
            # Add confidence intervals if available
            if 'proportion_ci' in metrics:
                ci = metrics['proportion_ci']
                row.update({
                    'proportion_ci_lower': ci.get('lower', 0),
                    'proportion_ci_upper': ci.get('upper', 0),
                    'proportion_ci_mean': ci.get('mean', 0)
                })
            
            if 'proportion_delta_ci' in metrics:
                ci = metrics['proportion_delta_ci']
                row.update({
                    'proportion_delta_ci_lower': ci.get('lower', 0),
                    'proportion_delta_ci_upper': ci.get('upper', 0),
                    'proportion_delta_ci_mean': ci.get('mean', 0)
                })
            
            # Add significance flags
            row['proportion_delta_significant'] = metrics.get('proportion_delta_significant', False)
            
            # Add quality metrics
            quality = metrics.get('quality', {})
            quality_delta = metrics.get('quality_delta', {})
            quality_ci = metrics.get('quality_ci', {})
            quality_delta_ci = metrics.get('quality_delta_ci', {})
            quality_delta_significant = metrics.get('quality_delta_significant', {})
            
            for metric_name in quality.keys():
                row[f'quality_{metric_name}'] = quality[metric_name]
                row[f'quality_delta_{metric_name}'] = quality_delta.get(metric_name, 0)
                row[f'quality_delta_{metric_name}_significant'] = quality_delta_significant.get(metric_name, False)
                
                if metric_name in quality_ci:
                    ci = quality_ci[metric_name]
                    row.update({
                        f'quality_{metric_name}_ci_lower': ci.get('lower', 0),
                        f'quality_{metric_name}_ci_upper': ci.get('upper', 0),
                        f'quality_{metric_name}_ci_mean': ci.get('mean', 0)
                    })
                
                if metric_name in quality_delta_ci:
                    ci = quality_delta_ci[metric_name]
                    row.update({
                        f'quality_delta_{metric_name}_ci_lower': ci.get('lower', 0),
                        f'quality_delta_{metric_name}_ci_upper': ci.get('upper', 0),
                        f'quality_delta_{metric_name}_ci_mean': ci.get('mean', 0)
                    })
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def create_cluster_dataframe(cluster_scores: Dict[str, Any]) -> pd.DataFrame:
    """Convert cluster scores to a tidy dataframe."""
    rows = []
    for cluster, metrics in cluster_scores.items():
        row = {
            'cluster': cluster,
            'size': metrics.get('size', 0),
            'proportion': metrics.get('proportion', 0)
        }
        
        # Add confidence intervals if available
        if 'proportion_ci' in metrics:
            ci = metrics['proportion_ci']
            row.update({
                'proportion_ci_lower': ci.get('lower', 0),
                'proportion_ci_upper': ci.get('upper', 0),
                'proportion_ci_mean': ci.get('mean', 0)
            })
        
        # Add quality metrics
        quality = metrics.get('quality', {})
        quality_delta = metrics.get('quality_delta', {})
        quality_ci = metrics.get('quality_ci', {})
        quality_delta_ci = metrics.get('quality_delta_ci', {})
        quality_delta_significant = metrics.get('quality_delta_significant', {})
        
        for metric_name in quality.keys():
            row[f'quality_{metric_name}'] = quality[metric_name]
            row[f'quality_delta_{metric_name}'] = quality_delta.get(metric_name, 0)
            row[f'quality_delta_{metric_name}_significant'] = quality_delta_significant.get(metric_name, False)
            
            if metric_name in quality_ci:
                ci = quality_ci[metric_name]
                row.update({
                    f'quality_{metric_name}_ci_lower': ci.get('lower', 0),
                    f'quality_{metric_name}_ci_upper': ci.get('upper', 0),
                    f'quality_{metric_name}_ci_mean': ci.get('mean', 0)
                })
            
            if metric_name in quality_delta_ci:
                ci = quality_delta_ci[metric_name]
                row.update({
                    f'quality_delta_{metric_name}_ci_lower': ci.get('lower', 0),
                    f'quality_delta_{metric_name}_ci_upper': ci.get('upper', 0),
                    f'quality_delta_{metric_name}_ci_mean': ci.get('mean', 0)
                })
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def create_model_dataframe(model_scores: Dict[str, Any]) -> pd.DataFrame:
    """Convert model scores to a tidy dataframe."""
    rows = []
    for model, metrics in model_scores.items():
        row = {
            'model': model,
            'size': metrics.get('size', 0),
            'proportion': metrics.get('proportion', 0)
        }
        
        # Add confidence intervals if available
        if 'proportion_ci' in metrics:
            ci = metrics['proportion_ci']
            row.update({
                'proportion_ci_lower': ci.get('lower', 0),
                'proportion_ci_upper': ci.get('upper', 0),
                'proportion_ci_mean': ci.get('mean', 0)
            })
        
        # Add quality metrics
        quality = metrics.get('quality', {})
        quality_delta = metrics.get('quality_delta', {})
        quality_ci = metrics.get('quality_ci', {})
        quality_delta_ci = metrics.get('quality_delta_ci', {})
        quality_delta_significant = metrics.get('quality_delta_significant', {})
        
        for metric_name in quality.keys():
            row[f'quality_{metric_name}'] = quality[metric_name]
            row[f'quality_delta_{metric_name}'] = quality_delta.get(metric_name, 0)
            row[f'quality_delta_{metric_name}_significant'] = quality_delta_significant.get(metric_name, False)
            
            if metric_name in quality_ci:
                ci = quality_ci[metric_name]
                row.update({
                    f'quality_{metric_name}_ci_lower': ci.get('lower', 0),
                    f'quality_{metric_name}_ci_upper': ci.get('upper', 0),
                    f'quality_{metric_name}_ci_mean': ci.get('mean', 0)
                })
            
            if metric_name in quality_delta_ci:
                ci = quality_delta_ci[metric_name]
                row.update({
                    f'quality_delta_{metric_name}_ci_lower': ci.get('lower', 0),
                    f'quality_delta_{metric_name}_ci_upper': ci.get('upper', 0),
                    f'quality_delta_{metric_name}_ci_mean': ci.get('mean', 0)
                })
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def get_quality_metrics(df: pd.DataFrame) -> List[str]:
    """Extract quality metric names from dataframe columns."""
    quality_cols = [col for col in df.columns if col.startswith('quality_') and not col.endswith(('_ci_lower', '_ci_upper', '_ci_mean', '_significant'))]
    return [col.replace('quality_', '') for col in quality_cols]


def create_interactive_cluster_plot(cluster_df: pd.DataFrame, model_cluster_df: pd.DataFrame, 
                                 metric_col: str, title: str, 
                                 ci_lower_col: Optional[str] = None, ci_upper_col: Optional[str] = None,
                                 significant_col: Optional[str] = None) -> go.Figure:
    """Create an interactive cluster plot with dropdown for view mode."""
    
    # Create the figure with subplots
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": False}]],
        subplot_titles=[title]
    )
    
    # Prepare cluster_df - reset index if cluster is the index
    if 'cluster' not in cluster_df.columns and cluster_df.index.name == 'cluster':
        cluster_df = cluster_df.reset_index()
    
    # Sort clusters by metric value in descending order for consistent ordering
    cluster_df = cluster_df.sort_values(metric_col, ascending=False)
    
    # Add aggregated view (default) - using cluster_df
    if ci_lower_col and ci_upper_col and ci_lower_col in cluster_df.columns and ci_upper_col in cluster_df.columns:
        fig.add_trace(
            go.Bar(
                x=cluster_df['cluster'],
                y=cluster_df[metric_col],
                name='Aggregated (All Models)',
                error_y=dict(
                    type='data',
                    array=cluster_df[ci_upper_col] - cluster_df[metric_col],
                    arrayminus=cluster_df[metric_col] - cluster_df[ci_lower_col],
                    visible=True
                ),
                visible=True
            )
        )
    else:
        fig.add_trace(
            go.Bar(
                x=cluster_df['cluster'],
                y=cluster_df[metric_col],
                name='Aggregated (All Models)',
                visible=True
            )
        )
    
    # Grouped by model view - using model_cluster_df
    for model in model_cluster_df['model'].unique():
        model_df = model_cluster_df[model_cluster_df['model'] == model]
        # Sort model_df to match the cluster order
        model_df = model_df.set_index('cluster').reindex(cluster_df['cluster']).reset_index()
        if ci_lower_col and ci_upper_col and ci_lower_col in model_cluster_df.columns and ci_upper_col in model_cluster_df.columns:
            fig.add_trace(
                go.Bar(
                    x=model_df['cluster'],
                    y=model_df[metric_col],
                    name=f'Model: {model}',
                    error_y=dict(
                        type='data',
                        array=model_df[ci_upper_col] - model_df[metric_col],
                        arrayminus=model_df[metric_col] - model_df[ci_lower_col],
                        visible=False
                    ),
                    visible=False
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=model_df['cluster'],
                    y=model_df[metric_col],
                    name=f'Model: {model}',
                    visible=False
                )
            )
    
    # Add significance markers if available (for aggregated view)
    # Red asterisks (*) indicate clusters with statistically significant quality delta values
    # (confidence intervals that do not contain 0)
    if significant_col and significant_col in cluster_df.columns:
        for i, (cluster, is_sig) in enumerate(zip(cluster_df['cluster'], cluster_df[significant_col])):
            if is_sig:
                fig.add_annotation(
                    x=cluster,
                    y=cluster_df[cluster_df['cluster'] == cluster][metric_col].iloc[0],
                    text="*",
                    showarrow=False,
                    font=dict(size=16, color="red"),
                    yshift=10
                )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Cluster",
        yaxis_title=metric_col.replace('_', ' ').title(),
        barmode='group',
        height=500,
        showlegend=True,
        annotations=[
            dict(
                text="* = Statistically significant (CI does not contain 0)",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.01, y=0.01,
                xanchor="left", yanchor="bottom",
                font=dict(size=10, color="red")
            )
        ] if significant_col and significant_col in cluster_df.columns else []
    )
    
    # Add dropdown for view selection - only 2 options
    buttons = []
    
    # Aggregated view button (all models combined)
    visibility = [True] + [False] * len(model_cluster_df['model'].unique())
    buttons.append(
        dict(
            label="Aggregated (All Models)",
            method="update",
            args=[{"visible": visibility, "barmode": "group"}]
        )
    )
    
    # Grouped by model view (each model as separate bars)
    visibility = [False] + [True] * len(model_cluster_df['model'].unique())
    buttons.append(
        dict(
            label="Grouped by Model",
            method="update",
            args=[{"visible": visibility, "barmode": "group"}]
        )
    )
    
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.95,
                xanchor="right",
                y=1.25,
                yanchor="top"
            )
        ]
    )
    
    return fig


def create_interactive_heatmap(df: pd.DataFrame, value_col: str, title: str,
                             pivot_index: str = 'model', pivot_columns: str = 'cluster',
                             significant_col: Optional[str] = None) -> go.Figure:
    """Create an interactive heatmap with hover information."""
    
    # Create pivot table
    pivot_df = df.pivot(index=pivot_index, columns=pivot_columns, values=value_col)
    
    # Sort by mean values for consistent ordering
    if pivot_index == 'model':
        # Sort models by their mean values across clusters
        model_means = pivot_df.mean(axis=1).sort_values(ascending=False)
        pivot_df = pivot_df.reindex(model_means.index)
    else:
        # Sort clusters by their mean values across models
        cluster_means = pivot_df.mean(axis=0).sort_values(ascending=False)
        pivot_df = pivot_df.reindex(columns=cluster_means.index)
    
    # Transpose the data for more intuitive visualization (models on x-axis, clusters on y-axis)
    pivot_df = pivot_df.T
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,  # Models
        y=pivot_df.index,    # Clusters
        colorscale='RdBu_r' if 'delta' in value_col else 'Viridis',
        zmid=0 if 'delta' in value_col else None,
        text=pivot_df.values.round(3),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    # Add significance markers if available
    if significant_col and significant_col in df.columns:
        sig_pivot = df.pivot(index=pivot_index, columns=pivot_columns, values=significant_col)
        # Apply same sorting as the main pivot
        if pivot_index == 'model':
            sig_pivot = sig_pivot.reindex(model_means.index)
        else:
            sig_pivot = sig_pivot.reindex(columns=cluster_means.index)
        sig_pivot = sig_pivot.T  # Transpose to match the main heatmap
        for i, cluster in enumerate(pivot_df.index):
            for j, model in enumerate(pivot_df.columns):
                if sig_pivot.loc[cluster, model]:
                    fig.add_annotation(
                        x=model,
                        y=cluster,
                        text="*",
                        showarrow=False,
                        font=dict(size=16, color="red"),
                        xshift=10,
                        yshift=10
                    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title="Cluster",
        height=500,
        annotations=[
            dict(
                text="* = Statistically significant (CI does not contain 0)",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.01, y=0.01,
                xanchor="left", yanchor="bottom",
                font=dict(size=10, color="red")
            )
        ] if significant_col and significant_col in df.columns else []
    )
    
    return fig


def create_interactive_model_plot(model_df: pd.DataFrame, model_cluster_df: pd.DataFrame, 
                                metric_col: str, title: str, 
                                ci_lower_col: Optional[str] = None, ci_upper_col: Optional[str] = None,
                                significant_col: Optional[str] = None) -> go.Figure:
    """Create an interactive model plot with dropdown for view mode."""
    
    # Create the figure with subplots
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": False}]],
        subplot_titles=[title]
    )
    
    # Prepare model_df - reset index if model is the index
    if 'model' not in model_df.columns and model_df.index.name == 'model':
        model_df = model_df.reset_index()
    
    # Add aggregated view (default) - using model_df
    if ci_lower_col and ci_upper_col and ci_lower_col in model_df.columns and ci_upper_col in model_df.columns:
        fig.add_trace(
            go.Bar(
                x=model_df['model'],
                y=model_df[metric_col],
                name='Aggregated (All Clusters)',
                error_y=dict(
                    type='data',
                    array=model_df[ci_upper_col] - model_df[metric_col],
                    arrayminus=model_df[metric_col] - model_df[ci_lower_col],
                    visible=True
                ),
                visible=True
            )
        )
    else:
        fig.add_trace(
            go.Bar(
                x=model_df['model'],
                y=model_df[metric_col],
                name='Aggregated (All Clusters)',
                visible=True
            )
        )
    
    # Grouped by cluster view - using model_cluster_df
    for cluster in model_cluster_df['cluster'].unique():
        cluster_df = model_cluster_df[model_cluster_df['cluster'] == cluster]
        if ci_lower_col and ci_upper_col and ci_lower_col in cluster_df.columns and ci_upper_col in cluster_df.columns:
            fig.add_trace(
                go.Bar(
                    x=cluster_df['model'],
                    y=cluster_df[metric_col],
                    name=f'Cluster: {cluster}',
                    error_y=dict(
                        type='data',
                        array=cluster_df[ci_upper_col] - cluster_df[metric_col],
                        arrayminus=cluster_df[metric_col] - cluster_df[ci_lower_col],
                        visible=False
                    ),
                    visible=False
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=cluster_df['model'],
                    y=cluster_df[metric_col],
                    name=f'Cluster: {cluster}',
                    visible=False
                )
            )
    
    # Add significance markers if available (for aggregated view)
    if significant_col and significant_col in model_df.columns:
        for i, (model, is_sig) in enumerate(zip(model_df['model'], model_df[significant_col])):
            if is_sig:
                fig.add_annotation(
                    x=model,
                    y=model_df[model_df['model'] == model][metric_col].iloc[0],
                    text="*",
                    showarrow=False,
                    font=dict(size=16, color="red"),
                    yshift=10
                )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title=metric_col.replace('_', ' ').title(),
        barmode='group',
        height=500,
        showlegend=True
    )
    
    # Add dropdown for view selection - only 2 options
    buttons = []
    
    # Aggregated view button (all clusters combined)
    visibility = [True] + [False] * len(model_cluster_df['cluster'].unique())
    buttons.append(
        dict(
            label="Aggregated (All Clusters)",
            method="update",
            args=[{"visible": visibility, "barmode": "group"}]
        )
    )
    
    # Grouped by cluster view (each cluster as separate bars)
    visibility = [False] + [True] * len(model_cluster_df['cluster'].unique())
    buttons.append(
        dict(
            label="Grouped by Cluster",
            method="update",
            args=[{"visible": visibility, "barmode": "group"}]
        )
    )
    
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.95,
                xanchor="right",
                y=1.25,
                yanchor="top"
            )
        ]
    )
    
    return fig


def create_interactive_model_cluster_plot(df: pd.DataFrame, metric_col: str, title: str,
                                       ci_lower_col: Optional[str] = None, ci_upper_col: Optional[str] = None,
                                       significant_col: Optional[str] = None) -> go.Figure:
    """Create an interactive model-cluster plot with grouped bars."""
    
    # Create grouped bar chart
    if ci_lower_col and ci_upper_col and ci_lower_col in df.columns and ci_upper_col in df.columns:
        fig = px.bar(
            df, 
            x='cluster', 
            y=metric_col, 
            color='model',
            error_y=df[ci_upper_col] - df[metric_col],
            error_y_minus=df[metric_col] - df[ci_lower_col],
            title=title,
            barmode='group'
        )
    else:
        fig = px.bar(
            df, 
            x='cluster', 
            y=metric_col, 
            color='model',
            title=title,
            barmode='group'
        )
    
    # Add significance markers if available
    if significant_col and significant_col in df.columns:
        for i, row in df.iterrows():
            if row[significant_col]:
                fig.add_annotation(
                    x=row['cluster'],
                    y=row[metric_col],
                    text="*",
                    showarrow=False,
                    font=dict(size=16, color="red"),
                    yshift=10
                )
    
    fig.update_layout(
        height=500,
        xaxis_title="Cluster",
        yaxis_title=metric_col.replace('_', ' ').title()
    )
    
    return fig


def save_plotly_figure(fig: go.Figure, output_path: Path, wandb_key: Optional[str] = None):
    """Save a plotly figure and optionally log to wandb."""
    # Save as HTML for interactivity
    fig.write_html(output_path.with_suffix('.html'))
    
    # Also save as PNG for static viewing
    fig.write_image(output_path.with_suffix('.png'), width=1200, height=600)
    
    # Log to wandb if requested
    if wandb_key and wandb.run:
        wandb.log({wandb_key: fig})


def generate_all_plots(model_cluster_scores: Dict[str, Any], cluster_scores: Dict[str, Any], 
                      model_scores: Dict[str, Any], output_dir: Path, log_to_wandb: bool = True):
    """Generate all metric plots and optionally log to wandb."""
    
    # Create dataframes
    model_cluster_df = create_model_cluster_dataframe(model_cluster_scores)
    cluster_df = create_cluster_dataframe(cluster_scores)
    model_df = create_model_dataframe(model_scores)
    
    # Get quality metrics
    quality_metrics = get_quality_metrics(model_cluster_df)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =============================================================================
    # PER CLUSTER PLOTS (cluster counts, proportions, quality, quality delta) 
    # - Interactive plots with dropdown for aggregated vs by-model view
    # =============================================================================
    
    # Cluster counts (total across all models)
    fig = create_interactive_cluster_plot(
        cluster_df, model_cluster_df, 'size', 'Total Conversation Count per Cluster'
    )
    save_plotly_figure(fig, output_dir / 'per_cluster_counts',
                      wandb_key='Plots/per_cluster/counts' if log_to_wandb else None)
    
    # Cluster proportions (what fraction of all conversations are in each cluster)
    proportion_ci_lower = 'proportion_ci_lower' if 'proportion_ci_lower' in cluster_df.columns else None
    proportion_ci_upper = 'proportion_ci_upper' if 'proportion_ci_upper' in cluster_df.columns else None
    fig = create_interactive_cluster_plot(
        cluster_df, model_cluster_df, 'proportion', 'Proportion of All Conversations per Cluster',
        ci_lower_col=proportion_ci_lower, ci_upper_col=proportion_ci_upper
    )
    save_plotly_figure(fig, output_dir / 'per_cluster_proportions',
                      wandb_key='Plots/per_cluster/proportions' if log_to_wandb else None)
    
    # Cluster quality scores (average across all models for each quality metric)
    for metric in quality_metrics:
        quality_col = f'quality_{metric}'
        if quality_col in cluster_df.columns:
            ci_lower = f'{quality_col}_ci_lower' if f'{quality_col}_ci_lower' in cluster_df.columns else None
            ci_upper = f'{quality_col}_ci_upper' if f'{quality_col}_ci_upper' in cluster_df.columns else None
            
            fig = create_interactive_cluster_plot(
                cluster_df, model_cluster_df, quality_col, f'Average Quality {metric.title()} per Cluster',
                ci_lower_col=ci_lower, ci_upper_col=ci_upper
            )
            save_plotly_figure(fig, output_dir / f'per_cluster_quality_{metric}',
                              wandb_key=f'Plots/per_cluster/quality_{metric}' if log_to_wandb else None)
    
    # Cluster quality delta scores (how each cluster compares to overall average)
    for metric in quality_metrics:
        quality_delta_col = f'quality_delta_{metric}'
        if quality_delta_col in cluster_df.columns:
            ci_lower = f'{quality_delta_col}_ci_lower' if f'{quality_delta_col}_ci_lower' in cluster_df.columns else None
            ci_upper = f'{quality_delta_col}_ci_upper' if f'{quality_delta_col}_ci_upper' in cluster_df.columns else None
            significant_col = f'{quality_delta_col}_significant' if f'{quality_delta_col}_significant' in cluster_df.columns else None
            
            fig = create_interactive_cluster_plot(
                cluster_df, model_cluster_df, quality_delta_col, f'Quality Delta {metric.title()} per Cluster',
                ci_lower_col=ci_lower, ci_upper_col=ci_upper, significant_col=significant_col
            )
            save_plotly_figure(fig, output_dir / f'per_cluster_quality_delta_{metric}',
                              wandb_key=f'Plots/per_cluster/quality_delta_{metric}' if log_to_wandb else None)
    
    # =============================================================================
    # PER MODEL PLOTS (model counts, quality)
    # - Interactive bar plots
    # =============================================================================
    
    # Model counts
    fig = create_interactive_model_plot(
        model_df, model_cluster_df, 'size', 'Conversation Count by Model'
    )
    save_plotly_figure(fig, output_dir / 'per_model_counts',
                      wandb_key='Plots/per_model/counts' if log_to_wandb else None)
    
    # Model quality scores (for each quality metric)  
    for metric in quality_metrics:
        quality_col = f'quality_{metric}'
        if quality_col in model_df.columns:
            # Don't include confidence intervals for quality metrics
            fig = create_interactive_model_plot(
                model_df, model_cluster_df, quality_col, f'Quality {metric.title()} by Model'
            )
            save_plotly_figure(fig, output_dir / f'per_model_quality_{metric}',
                              wandb_key=f'Plots/per_model/quality_{metric}' if log_to_wandb else None)
    
    # =============================================================================
    # PER MODEL AND CLUSTER PLOTS 
    # - Model proportions across clusters, quality across clusters per model, 
    #   proportion delta across clusters per model
    # - Interactive grouped bar charts
    # =============================================================================
    
    # Model proportions across clusters
    proportion_ci_lower = 'proportion_ci_lower' if 'proportion_ci_lower' in model_cluster_df.columns else None
    proportion_ci_upper = 'proportion_ci_upper' if 'proportion_ci_upper' in model_cluster_df.columns else None
    fig = create_interactive_model_cluster_plot(
        model_cluster_df, 'proportion', 'Model Proportions across Clusters',
        ci_lower_col=proportion_ci_lower, ci_upper_col=proportion_ci_upper
    )
    save_plotly_figure(fig, output_dir / 'per_model_cluster_proportions',
                      wandb_key='Plots/per_model_cluster/proportions' if log_to_wandb else None)
    
    # Quality across clusters per model (for each quality metric)
    for metric in quality_metrics:
        quality_col = f'quality_{metric}'
        if quality_col in model_cluster_df.columns:
            ci_lower = f'{quality_col}_ci_lower' if f'{quality_col}_ci_lower' in model_cluster_df.columns else None
            ci_upper = f'{quality_col}_ci_upper' if f'{quality_col}_ci_upper' in model_cluster_df.columns else None
            
            fig = create_interactive_model_cluster_plot(
                model_cluster_df, quality_col, f'Quality {metric.title()} across Clusters per Model',
                ci_lower_col=ci_lower, ci_upper_col=ci_upper
            )
            save_plotly_figure(fig, output_dir / f'per_model_cluster_quality_{metric}',
                              wandb_key=f'Plots/per_model_cluster/quality_{metric}' if log_to_wandb else None)
    
    # Proportion delta (salience) across clusters per model
    if 'proportion_delta' in model_cluster_df.columns:
        proportion_delta_ci_lower = 'proportion_delta_ci_lower' if 'proportion_delta_ci_lower' in model_cluster_df.columns else None
        proportion_delta_ci_upper = 'proportion_delta_ci_upper' if 'proportion_delta_ci_upper' in model_cluster_df.columns else None
        fig = create_interactive_model_cluster_plot(
            model_cluster_df, 'proportion_delta', 'Proportion Delta (Salience) across Clusters per Model',
            ci_lower_col=proportion_delta_ci_lower, ci_upper_col=proportion_delta_ci_upper
        )
        save_plotly_figure(fig, output_dir / 'per_model_cluster_proportion_delta',
                          wandb_key='Plots/per_model_cluster/proportion_delta' if log_to_wandb else None)
    
    # =============================================================================
    # HEATMAPS (model-cluster visualizations)
    # =============================================================================
    
    # Basic heatmaps
    fig = create_interactive_heatmap(
        model_cluster_df, 'size', 'Conversation Count by Model-Cluster'
    )
    save_plotly_figure(fig, output_dir / 'model_cluster_size_heatmap',
                      wandb_key='Plots/heatmaps/model_cluster_size' if log_to_wandb else None)
    
    fig = create_interactive_heatmap(
        model_cluster_df, 'proportion', 'Proportion by Model-Cluster'
    )
    save_plotly_figure(fig, output_dir / 'model_cluster_proportion_heatmap',
                      wandb_key='Plots/heatmaps/model_cluster_proportion' if log_to_wandb else None)
    
    if 'proportion_delta' in model_cluster_df.columns:
        significant_col = 'proportion_delta_significant' if 'proportion_delta_significant' in model_cluster_df.columns else None
        fig = create_interactive_heatmap(
            model_cluster_df, 'proportion_delta', 'Proportion Delta (Salience) by Model-Cluster',
            significant_col=significant_col
        )
        save_plotly_figure(fig, output_dir / 'model_cluster_proportion_delta_heatmap',
                          wandb_key='Plots/heatmaps/model_cluster_proportion_delta' if log_to_wandb else None)
    
    # Quality heatmaps
    for metric in quality_metrics:
        quality_col = f'quality_{metric}'
        if quality_col in model_cluster_df.columns:
            fig = create_interactive_heatmap(
                model_cluster_df, quality_col, f'Quality: {metric.title()}'
            )
            save_plotly_figure(fig, output_dir / f'model_cluster_quality_{metric}_heatmap',
                              wandb_key=f'Plots/heatmaps/quality_{metric}' if log_to_wandb else None)
        
        quality_delta_col = f'quality_delta_{metric}'
        if quality_delta_col in model_cluster_df.columns:
            significant_col = f'{quality_delta_col}_significant' if f'{quality_delta_col}_significant' in model_cluster_df.columns else None
            fig = create_interactive_heatmap(
                model_cluster_df, quality_delta_col, f'Quality Delta: {metric.title()}',
                significant_col=significant_col
            )
            save_plotly_figure(fig, output_dir / f'model_cluster_quality_delta_{metric}_heatmap',
                              wandb_key=f'Plots/heatmaps/quality_delta_{metric}' if log_to_wandb else None)
    
    return len(quality_metrics) 
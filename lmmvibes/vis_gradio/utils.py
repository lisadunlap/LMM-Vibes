"""
Utility functions for Gradio pipeline results app.

This module contains common utility functions used across different components.
"""

import numpy as np
import pandas as pd
import json
import markdown
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple
import html
import ast

# Conversation rendering helpers are now in a dedicated module for clarity
from . import conversation_display as _convdisp
from .conversation_display import (
    convert_to_openai_format,
    display_openai_conversation_html,
    pretty_print_embedded_dicts,
)

# NEW IMPLEMENTATION ---------------------------------------------------
from .metrics_adapter import get_model_clusters, get_all_models

# ---------------------------------------------------------------------------
# NEW helper utilities for FunctionalMetrics format
# ---------------------------------------------------------------------------


def format_confidence_interval(ci: dict | None, decimals: int = 3) -> str:
    """Return a pretty string for a CI dict of the form {"lower": x, "upper": y}."""
    if not ci or not isinstance(ci, dict):
        return "N/A"
    lower, upper = ci.get("lower"), ci.get("upper")
    if lower is None or upper is None:
        return "N/A"
    return f"[{lower:.{decimals}f}, {upper:.{decimals}f}]"


def get_confidence_interval_width(ci: dict | None) -> float | None:
    """Return CI width (upper-lower) if possible."""
    if not ci or not isinstance(ci, dict):
        return None
    lower, upper = ci.get("lower"), ci.get("upper")
    if lower is None or upper is None:
        return None
    return upper - lower


def has_confidence_intervals(record: dict | None) -> bool:
    """Simple check whether any *_ci key with lower/upper exists in a metrics record."""
    if not record or not isinstance(record, dict):
        return False
    for k, v in record.items():
        if k.endswith("_ci") and isinstance(v, dict) and {"lower", "upper"}.issubset(v.keys()):
            return True
    return False


def extract_quality_score(quality_field: Any) -> float | None:
    """Given a quality field that may be a dict of metric values or a scalar, return its mean."""
    if quality_field is None:
        return None
    if isinstance(quality_field, (int, float)):
        return float(quality_field)
    if isinstance(quality_field, dict) and quality_field:
        return float(np.mean(list(quality_field.values())))
    return None

# ---------------------------------------------------------------------------
# UPDATED: get_top_clusters_for_model for FunctionalMetrics format
# ---------------------------------------------------------------------------


def get_top_clusters_for_model(metrics: Dict[str, Any], model_name: str, top_n: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
    """Return the top N clusters (by salience) for a given model.

    Args:
        metrics: The FunctionalMetrics dictionary (3-file format) loaded via data_loader.
        model_name: Name of the model to inspect.
        top_n: Number of clusters to return.

    Returns:
        List of (cluster_name, cluster_dict) tuples sorted by descending proportion_delta.
    """
    clusters_dict = get_model_clusters(metrics, model_name)
    if not clusters_dict:
        return []
    sorted_items = sorted(
        clusters_dict.items(), key=lambda kv: kv[1].get("proportion_delta", 0), reverse=True
    )
    return sorted_items[:top_n]


def compute_model_rankings_new(metrics: Dict[str, Any]) -> List[tuple]:
    """Compute rankings of models based on mean salience (proportion_delta).

    Args:
        metrics: The FunctionalMetrics dict loaded by data_loader.

    Returns:
        List[Tuple[str, Dict[str, float]]]: sorted list of (model_name, summary_dict)
    """
    model_scores: Dict[str, Dict[str, float]] = {}
    for model in get_all_models(metrics):
        clusters = get_model_clusters(metrics, model).values()
        if not clusters:
            continue
        saliences = [c.get("proportion_delta", 0.0) for c in clusters]
        model_scores[model] = {
            "avg_salience": float(np.mean(saliences)),
            "median_salience": float(np.median(saliences)),
            "num_clusters": len(saliences),
            "top_salience": float(max(saliences)),
            "std_salience": float(np.std(saliences)),
        }
    return sorted(model_scores.items(), key=lambda x: x[1]["avg_salience"], reverse=True)


def create_model_summary_card_new(
    model_name: str,
    metrics: Dict[str, Any],
    top_n: int = 3,
    score_significant_only: bool = False,
    quality_significant_only: bool = False,
    sort_by: str = "quality_asc",
    min_cluster_size: int = 1,
) -> str:
    """Generate a **styled** HTML summary card for a single model.

    The new implementation recreates the legacy card design the user prefers:
    • Card header with battle count
    • Each cluster displayed as a vertically-spaced block (NOT a table)
    • Frequency, distinctiveness factor and CI inline; quality score right-aligned
    """

    clusters_dict = get_model_clusters(metrics, model_name)
    if not clusters_dict:
        return f"<div style='padding:20px'>No cluster data for {model_name}</div>"

    # Filter clusters ----------------------------------------------------
    all_clusters = [c for c in clusters_dict.values() if c.get("size", 0) >= min_cluster_size]

    if score_significant_only:
        if model_name == "all":
            # For "all" model, we don't have proportion_delta_significant, so skip this filter
            pass
        else:
            all_clusters = [c for c in all_clusters if c.get("proportion_delta_significant", False)]
    if quality_significant_only:
        all_clusters = [c for c in all_clusters if any(c.get("quality_delta_significant", {}).values())]

    if not all_clusters:
        return f"<div style='padding:20px'>No clusters pass filters for {model_name}</div>"

    # Count significant properties ---------------------------------------
    significant_frequency_count = 0
    significant_quality_count = 0
    
    for cluster in clusters_dict.values():
        if cluster.get("size", 0) >= min_cluster_size:
            # Count frequency significance
            if model_name != "all" and cluster.get("proportion_delta_significant", False):
                significant_frequency_count += 1
            
            # Count quality significance (sum across all metrics)
            quality_delta_significant = cluster.get("quality_delta_significant", {})
            significant_quality_count += sum(quality_delta_significant.values())

    # Sort ---------------------------------------------------------------
    def _mean_quality(c: dict[str, Any]) -> float:
        vals = list(c.get("quality", {}).values())
        return float(np.mean(vals)) if vals else 0.0

    sort_key_map = {
        "quality_asc": (_mean_quality, False),
        "quality_desc": (_mean_quality, True),
        "frequency_desc": (lambda c: c.get("proportion", 0), True),
        "frequency_asc": (lambda c: c.get("proportion", 0), False),
        "salience_desc": (lambda c: c.get("proportion_delta", 0) if model_name != "all" else c.get("proportion", 0), True),
        "salience_asc": (lambda c: c.get("proportion_delta", 0) if model_name != "all" else c.get("proportion", 0), False),
    }

    key_fn, reverse = sort_key_map.get(sort_by, (lambda c: c.get("proportion_delta", 0) if model_name != "all" else c.get("proportion", 0), True))
    sorted_clusters = sorted(all_clusters, key=key_fn, reverse=reverse)[:top_n]

    # Determine total conversations for this model ----------------
    if model_name == "all":
        # For "all" model, sum the individual model totals to avoid double-counting
        model_scores = metrics.get("model_scores", {})
        total_battles = sum(model_data.get("size", 0) for model_data in model_scores.values())
    else:
        model_scores_entry = metrics.get("model_scores", {}).get(model_name, {})
        total_battles = model_scores_entry.get("size")
        if total_battles is None:
            # Fallback: deduplicate example IDs across clusters
            total_battles = sum(c.get("size", 0) for c in clusters_dict.values())

    # Card header --------------------------------------------------------
    html_parts: list[str] = [f"""
    <div style="padding: 20px; border:1px solid #e0e0e0; border-radius:8px; margin-bottom:25px;">
      <h3 style="margin-top:0; font-size: 20px;">{html.escape(model_name)}</h3>
      <p style="margin: 4px 0 8px 0; color:#555; font-size:13px;">
        {total_battles} battles · Top clusters by frequency
      </p>
      <p style="margin: 0 0 18px 0; color:#666; font-size:12px;">
        📊 {significant_frequency_count} significant frequency properties · {significant_quality_count} significant quality properties
      </p>
    """]

    # Cluster blocks -----------------------------------------------------
    for i, cluster in enumerate(sorted_clusters):
        name = html.escape(next(k for k, v in clusters_dict.items() if v is cluster))
        prop = cluster.get("proportion", 0)
        freq_pct = prop * 100
        size = cluster.get("size", 0)

        # Check significance flags
        is_proportion_significant = False
        if model_name != "all":
            is_proportion_significant = cluster.get("proportion_delta_significant", False)
        
        quality_delta_significant = cluster.get("quality_delta_significant", {})
        is_quality_significant = any(quality_delta_significant.values())

        # Create significance indicators
        significance_indicators = []
        if is_proportion_significant:
            significance_indicators.append('<span style="background: #28a745; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold;">FREQ</span>')
        if is_quality_significant:
            significance_indicators.append('<span style="background: #007bff; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: bold;">QUAL</span>')
        
        significance_html = " ".join(significance_indicators) if significance_indicators else ""

        # Distinctiveness factor heuristic
        if model_name == "all":
            # For "all" model, proportion_delta doesn't make sense, so show proportion instead
            distinct_factor = prop
            distinct_text = f"{freq_pct:.1f}% of all conversations"
        else:
            sal = cluster.get("proportion_delta", 0)
            distinct_factor = 1 + (sal / prop) if prop else 1
            distinct_text = f"{distinct_factor:.1f}× more distinctive"

        # Confidence interval (frequency based)
        ci = cluster.get("proportion_ci")
        ci_str = format_confidence_interval(ci) if ci else "N/A"

        # Quality delta – show each metric separately
        quality_delta = cluster.get("quality_delta", {})
        quality_delta_html = ""
        
        if quality_delta:
            quality_delta_parts = []
            for metric_name, delta_value in quality_delta.items():
                color = "#28a745" if delta_value >= 0 else "#dc3545"
                quality_delta_parts.append(f'<div style="color:{color}; font-weight:500;">{metric_name}: {delta_value:+.3f}</div>')
            quality_delta_html = "".join(quality_delta_parts)
        else:
            quality_delta_html = '<span style="color:#666;">No quality data</span>'

        # Get light color for this cluster
        cluster_color = get_light_color_for_cluster(name, i)

        html_parts.append(f"""
        <div style="border-left: 4px solid #4c6ef5; padding: 12px 16px; margin-bottom: 10px; background:{cluster_color}; border-radius: 4px;">
          <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div style="max-width:80%;">
              <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
                <strong style="font-size:14px;">{name}</strong>
                {significance_html}
              </div>
              <span style="font-size:12px; color:#555;">{freq_pct:.1f}% frequency ({size} out of {total_battles} total) · {distinct_text}</span>
            </div>
            <div style="font-size:12px; font-weight:normal; white-space:nowrap; text-align:right;">{quality_delta_html}</div>
          </div>
        </div>
        """)

    # Close card div -----------------------------------------------------
    html_parts.append("</div>")

    return "\n".join(html_parts)


def format_cluster_dataframe(clustered_df: pd.DataFrame, 
                           selected_models: Optional[List[str]] = None,
                           cluster_level: str = 'fine',
                           max_rows: int = 1000) -> pd.DataFrame:
    """Format cluster DataFrame for display in Gradio."""
    df = clustered_df.copy()
    
    # Filter by models if specified
    if selected_models:
        df = df[df['model'].isin(selected_models)]
    
    # Select relevant columns based on cluster level using correct column names from pipeline
    if cluster_level == 'fine':
        id_col = 'property_description_fine_cluster_id'
        label_col = 'property_description_fine_cluster_label'
        # Also check for alternative naming without prefix
        alt_id_col = 'fine_cluster_id'
        alt_label_col = 'fine_cluster_label'
    else:
        id_col = 'property_description_coarse_cluster_id'
        label_col = 'property_description_coarse_cluster_label'
        # Also check for alternative naming without prefix
        alt_id_col = 'coarse_cluster_id'
        alt_label_col = 'coarse_cluster_label'
    
    # Try both naming patterns
    if id_col in df.columns and label_col in df.columns:
        # Use the expected naming pattern
        cols = ['question_id', 'model', 'property_description', id_col, label_col, 'score']
    elif alt_id_col in df.columns and alt_label_col in df.columns:
        # Use the alternative naming pattern
        cols = ['question_id', 'model', 'property_description', alt_id_col, alt_label_col, 'score']
    else:
        # Fall back to basic columns if cluster columns are missing
        cols = ['question_id', 'model', 'property_description', 'score']
    
    # Keep only existing columns
    available_cols = [col for col in cols if col in df.columns]
    df = df[available_cols]
    
    # Limit rows for performance
    if len(df) > max_rows:
        df = df.head(max_rows)
    
    return df


def truncate_cluster_name(cluster_desc: str, max_length: int = 50) -> str:
    """Truncate cluster description to fit in table column."""
    if len(cluster_desc) <= max_length:
        return cluster_desc
    return cluster_desc[:max_length-3] + "..."

def create_frequency_comparison_table(model_stats: Dict[str, Any], 
                                     selected_models: List[str],
                                     cluster_level: str = "fine",  # Ignored – kept for backward-compat
                                     top_n: int = 50,
                                     selected_model: str | None = None,
                                     selected_quality_metric: str | None = None) -> pd.DataFrame:
    """Create a comparison table for the new FunctionalMetrics format.

    The old signature is kept (cluster_level arg is ignored) so that callers
    can be updated incrementally.
    """

    if not selected_models:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 1. Collect per-model, per-cluster rows
    # ------------------------------------------------------------------
    all_rows: List[dict] = []
    for model in selected_models:
        model_clusters = get_model_clusters(model_stats, model)  # type: ignore[arg-type]
        if not model_clusters:
            continue

        # Optional filter by a single model after the fact
        if selected_model and model != selected_model:
            continue

        for cluster_name, cdata in model_clusters.items():
            # Basic numbers
            freq_pct = cdata.get("proportion", 0.0) * 100.0
            prop_ci = cdata.get("proportion_ci")

            # Quality per metric dicts ------------------------------------------------
            quality_dict = cdata.get("quality", {}) or {}
            quality_ci_dict = cdata.get("quality_ci", {}) or {}

            # Significance flags
            sal_sig = bool(cdata.get("proportion_delta_significant", False))
            quality_sig_flags = cdata.get("quality_delta_significant", {}) or {}

            all_rows.append({
                "cluster": cluster_name,
                "model": model,
                "frequency": freq_pct,
                "proportion_ci": prop_ci,
                "quality": quality_dict,
                "quality_ci": quality_ci_dict,
                "score_significant": sal_sig,
                "quality_significant_any": any(quality_sig_flags.values()),
                "quality_significant_metric": quality_sig_flags.get(selected_quality_metric) if selected_quality_metric else None,
            })

    if not all_rows:
        return pd.DataFrame()

    df_all = pd.DataFrame(all_rows)

    # Aggregate frequency across models ----------------------------------
    freq_sum = df_all.groupby("cluster")["frequency"].sum().sort_values(ascending=False)
    top_clusters = freq_sum.head(top_n).index.tolist()

    df_top = df_all[df_all["cluster"].isin(top_clusters)].copy()

    table_rows: List[dict] = []
    for clu in top_clusters:
        subset = df_top[df_top["cluster"] == clu]
        avg_freq = subset["frequency"].mean()

        # Aggregate CI (mean of bounds)
        ci_lowers = [ci.get("lower") for ci in subset["proportion_ci"] if isinstance(ci, dict)]
        ci_uppers = [ci.get("upper") for ci in subset["proportion_ci"] if isinstance(ci, dict)]
        freq_ci = {
            "lower": float(np.mean(ci_lowers)) if ci_lowers else None,
            "upper": float(np.mean(ci_uppers)) if ci_uppers else None,
        } if ci_lowers and ci_uppers else None

        # Quality aggregation -----------------------------------------------------
        q_vals: List[float] = []
        q_ci_l: List[float] = []
        q_ci_u: List[float] = []
        quality_sig_any = False
        for _, row in subset.iterrows():
            q_dict = row["quality"]
            if selected_quality_metric:
                if selected_quality_metric in q_dict:
                    q_vals.append(q_dict[selected_quality_metric])
                ci_metric = row["quality_ci"].get(selected_quality_metric) if isinstance(row["quality_ci"], dict) else None
                if ci_metric:
                    q_ci_l.append(ci_metric.get("lower"))
                    q_ci_u.append(ci_metric.get("upper"))
                quality_sig_any = quality_sig_any or bool(row["quality_significant_metric"])
            else:
                q_vals.extend(q_dict.values())
                for ci in row["quality_ci"].values():
                    if isinstance(ci, dict):
                        q_ci_l.append(ci.get("lower"))
                        q_ci_u.append(ci.get("upper"))
                quality_sig_any = quality_sig_any or row["quality_significant_any"]

        quality_val = float(np.mean(q_vals)) if q_vals else None
        quality_ci = {
            "lower": float(np.mean(q_ci_l)),
            "upper": float(np.mean(q_ci_u)),
        } if q_ci_l and q_ci_u else None

        score_sig = subset["score_significant"].any()

        table_rows.append({
            "Cluster": clu,
            "Frequency (%)": f"{avg_freq:.1f}",
            "Freq CI": format_confidence_interval(freq_ci),
            "Quality": f"{quality_val:.3f}" if quality_val is not None else "N/A",
            "Quality CI": format_confidence_interval(quality_ci) if quality_ci else "N/A",
            "Score Significance": "Yes" if score_sig else "No",
            "Quality Significance": "Yes" if quality_sig_any else "No",
        })

    return pd.DataFrame(table_rows)


def create_frequency_comparison_plots(model_stats: Dict[str, Any], 
                                     selected_models: List[str],
                                     cluster_level: str = 'fine',
                                     top_n: int = 50,
                                     show_confidence_intervals: bool = False) -> Tuple[go.Figure, go.Figure]:
    """Create frequency comparison plots (matching frequencies_tab.py exactly)."""
    
    print(f"\nDEBUG: Plotting function called with:")
    print(f"  - Selected models: {selected_models}")
    print(f"  - Cluster level: {cluster_level}")
    print(f"  - Top N: {top_n}")
    print(f"  - Available models in stats: {list(model_stats.keys())}")
    
    # Use the same data preparation logic as the table function
    # Collect all clusters across all models for the chart (exact copy from frequencies_tab.py)
    all_clusters_data = []
    for model_name, model_data in model_stats.items():
        if model_name not in selected_models:
            continue
            
        clusters = model_data.get(cluster_level, [])
        for cluster in clusters:
            # Get confidence intervals for quality scores if available
            quality_score_ci = cluster.get('quality_score_ci', {})
            has_quality_ci = bool(quality_score_ci)
            
            # Get distinctiveness score confidence intervals (correct structure)
            score_ci = cluster.get('score_ci', {})
            ci_lower = score_ci.get('lower') if score_ci else None
            ci_upper = score_ci.get('upper') if score_ci else None
            
            all_clusters_data.append({
                'property_description': cluster['property_description'],
                'model': model_name,
                'frequency': cluster.get('proportion', 0) * 100,  # Convert to percentage
                'size': cluster.get('size', 0),
                'cluster_size_global': cluster.get('cluster_size_global', 0),
                'has_ci': has_confidence_intervals(cluster),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'has_quality_ci': has_quality_ci
            })
    
    if not all_clusters_data:
        # Return empty figures
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return empty_fig, empty_fig
        
    clusters_df = pd.DataFrame(all_clusters_data)
    
    # Get all unique clusters for the chart
    all_unique_clusters = clusters_df['property_description'].unique()
    total_clusters = len(all_unique_clusters)
    
    # Show all clusters by default
    top_n_for_chart = min(top_n, total_clusters)
    
    # Calculate total frequency per cluster and get top clusters
    cluster_totals = clusters_df.groupby('property_description')['frequency'].sum().sort_values(ascending=False)
    top_clusters = cluster_totals.head(top_n_for_chart).index.tolist()
    
    # Get quality scores for the same clusters to sort by quality
    quality_data_for_sorting = []
    for model_name, model_data in model_stats.items():
        if model_name not in selected_models:
            continue
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
    
    if chart_data.empty:
        # Return empty figures
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return empty_fig, empty_fig
    
    # Get unique models for colors
    models = chart_data['model'].unique()
    # Use a color palette that avoids yellow - using Set1 which has better contrast
    colors = px.colors.qualitative.Set1[:len(models)]
    
    # Create horizontal bar chart for frequencies
    fig = go.Figure()
    
    # Add a bar for each model
    for i, model in enumerate(models):
        model_data = chart_data[chart_data['model'] == model]
        
        # Sort by cluster order (same as top_clusters)
        model_data = model_data.set_index('property_description').reindex(top_clusters).reset_index()
        
        # Fill NaN values with 0 for missing clusters
        model_data['frequency'] = model_data['frequency'].fillna(0)
        model_data['has_ci'] = model_data['has_ci'].fillna(False)
        # For CI columns, replace NaN with None using where() instead of fillna(None)
        model_data['ci_lower'] = model_data['ci_lower'].where(pd.notna(model_data['ci_lower']), None)
        model_data['ci_upper'] = model_data['ci_upper'].where(pd.notna(model_data['ci_upper']), None)
        
        # Ensure frequency is numeric and non-negative
        model_data['frequency'] = pd.to_numeric(model_data['frequency'], errors='coerce').fillna(0)
        model_data['frequency'] = model_data['frequency'].clip(lower=0)
        
        # Debug: print model data for first model
        if i == 0:  # Only print for first model to avoid spam
            print(f"DEBUG: Model {model} data sample:")
            print(f"  - Clusters: {len(model_data)}")
            print(f"  - Frequency range: {model_data['frequency'].min():.2f} - {model_data['frequency'].max():.2f}")
            print(f"  - Non-zero frequencies: {(model_data['frequency'] > 0).sum()}")
            if len(model_data) > 0:
                print(f"  - Sample row: {model_data.iloc[0][['property_description', 'frequency']].to_dict()}")
                
        # Remove any rows where property_description is NaN (these are clusters this model doesn't appear in)
        model_data = model_data.dropna(subset=['property_description'])
        
        # Get confidence intervals for error bars
        ci_lower = []
        ci_upper = []
        for _, row in model_data.iterrows():
            freq_value = row.get('frequency', 0)
            if (row.get('has_ci', False) and 
                pd.notna(row.get('ci_lower')) and 
                pd.notna(row.get('ci_upper')) and
                freq_value > 0):  # Only calculate CIs for non-zero frequencies
                
                # IMPORTANT: These are distinctiveness score CIs, not frequency CIs
                # The distinctiveness score measures how much more/less frequently 
                # a model exhibits this behavior compared to the median model
                # We can use this to estimate uncertainty in the frequency measurement
                distinctiveness_ci_width = row['ci_upper'] - row['ci_lower']
                
                # Convert to frequency uncertainty (approximate)
                # A wider distinctiveness CI suggests more uncertainty in the frequency
                freq_uncertainty = distinctiveness_ci_width * freq_value * 0.1
                ci_lower.append(max(0, freq_value - freq_uncertainty))
                ci_upper.append(freq_value + freq_uncertainty)
            else:
                ci_lower.append(None)
                ci_upper.append(None)
        
        # Debug: Check the data going into the plot
        print(f"DEBUG: Adding trace for model {model}:")
        print(f"  - Y values (clusters): {model_data['property_description'].tolist()[:3]}...")  # First 3 clusters
        print(f"  - X values (frequencies): {model_data['frequency'].tolist()[:3]}...")  # First 3 frequencies
        print(f"  - Total data points: {len(model_data)}")
        
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
                visible=show_confidence_intervals,
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
    
    # Update y-axis to show truncated cluster names
    fig.update_yaxes(
        tickmode='array',
        ticktext=[truncate_cluster_name(desc, 60) for desc in top_clusters],
        tickvals=top_clusters
    )
    
    # Create quality score chart
    # Get quality scores for the same clusters (single score per cluster)
    quality_data = []
    quality_cis = []  # Add confidence intervals for quality scores
    
    for cluster_desc in top_clusters:
        # Get the first available quality score for this cluster
        for model_name, model_data in model_stats.items():
            clusters = model_data.get(cluster_level, [])
            for cluster in clusters:
                if cluster['property_description'] == cluster_desc:
                    quality_score = extract_quality_score(cluster.get('quality_score', 0))
                    quality_data.append({
                        'property_description': cluster_desc,
                        'quality_score': quality_score
                    })
                    
                    # Get quality score confidence intervals
                    quality_ci = cluster.get('quality_score_ci', {})
                    if isinstance(quality_ci, dict) and quality_ci:
                        # Get the first available quality CI
                        for score_key, ci_data in quality_ci.items():
                            if isinstance(ci_data, dict):
                                ci_lower = ci_data.get('lower')
                                ci_upper = ci_data.get('upper')
                                if ci_lower is not None and ci_upper is not None:
                                    quality_cis.append({
                                        'property_description': cluster_desc,
                                        'ci_lower': ci_lower,
                                        'ci_upper': ci_upper
                                    })
                                    break
                        else:
                            quality_cis.append({
                                'property_description': cluster_desc,
                                'ci_lower': None,
                                'ci_upper': None
                            })
                    else:
                        quality_cis.append({
                            'property_description': cluster_desc,
                            'ci_lower': None,
                            'ci_upper': None
                        })
                    break
            if any(q['property_description'] == cluster_desc for q in quality_data):
                break
    
    if quality_data:
        quality_df = pd.DataFrame(quality_data)
        quality_cis_df = pd.DataFrame(quality_cis) if quality_cis else None
        
        # Create quality score chart with single bars
        fig_quality = go.Figure()
        
        # Prepare confidence intervals for error bars
        ci_lower = []
        ci_upper = []
        for _, row in quality_df.iterrows():
            cluster_desc = row['property_description']
            if quality_cis_df is not None:
                ci_row = quality_cis_df[quality_cis_df['property_description'] == cluster_desc]
                if not ci_row.empty:
                    ci_lower.append(ci_row.iloc[0]['ci_lower'])
                    ci_upper.append(ci_row.iloc[0]['ci_upper'])
                else:
                    ci_lower.append(None)
                    ci_upper.append(None)
            else:
                ci_lower.append(None)
                ci_upper.append(None)
        
        # Add a single bar for each cluster
        fig_quality.add_trace(go.Bar(
            y=[truncate_cluster_name(desc, 60) for desc in quality_df['property_description']],
            x=quality_df['quality_score'],
            orientation='h',
            marker_color='lightblue',  # Single color for all bars
            name='Quality Score',
            showlegend=False,
            error_x=dict(
                type='data',
                array=[u - l if u is not None and l is not None else None for l, u in zip(ci_lower, ci_upper)],
                arrayminus=[q - l if q is not None and l is not None else None for q, l in zip(quality_df['quality_score'], ci_lower)],
                visible=show_confidence_intervals,
                thickness=1,
                width=3,
                color='rgba(0,0,0,0.3)'
            ),
            hovertemplate='<b>%{y}</b><br>' +
                        'Quality Score: %{x:.3f}<br>' +
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
        fig_quality.update_layout(
            title=f"Quality Scores",
            xaxis_title="Quality Score",
            yaxis_title="",  # No y-axis title to save space
            height=max(600, len(top_clusters) * 25),  # Same height as main chart
            showlegend=False,
            yaxis=dict(showticklabels=False)  # Hide y-axis labels to save space
        )
    else:
        # Create empty quality figure
        fig_quality = go.Figure()
        fig_quality.add_annotation(text="No quality score data available", 
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    return fig, fig_quality


def search_clusters_by_text(clustered_df: pd.DataFrame, 
                          search_term: str,
                          search_in: str = 'description') -> pd.DataFrame:
    """Search clusters by text in descriptions or other fields."""
    if not search_term:
        return clustered_df.head(100)  # Return first 100 if no search
    
    search_term = search_term.lower()
    
    if search_in == 'description':
        mask = clustered_df['property_description'].str.lower().str.contains(search_term, na=False)
    elif search_in == 'model':
        mask = clustered_df['model'].str.lower().str.contains(search_term, na=False)
    elif search_in == 'cluster_label':
        # Use correct column names from pipeline
        fine_label_col = 'property_description_fine_cluster_label'
        coarse_label_col = 'property_description_coarse_cluster_label'
        mask = pd.Series([False] * len(clustered_df))
        
        if fine_label_col in clustered_df.columns:
            mask |= clustered_df[fine_label_col].str.lower().str.contains(search_term, na=False)
        if coarse_label_col in clustered_df.columns:
            mask |= clustered_df[coarse_label_col].str.lower().str.contains(search_term, na=False)
    else:
        # Search in all text columns using correct column names
        text_cols = ['property_description', 'model', 
                    'property_description_fine_cluster_label', 
                    'property_description_coarse_cluster_label']
        mask = pd.Series([False] * len(clustered_df))
        for col in text_cols:
            if col in clustered_df.columns:
                mask |= clustered_df[col].str.lower().str.contains(search_term, na=False)
    
    return clustered_df[mask].head(100) 


def create_interactive_cluster_viewer(clustered_df: pd.DataFrame, 
                                    selected_models: Optional[List[str]] = None,
                                    cluster_level: str = 'fine') -> str:
    """Create interactive cluster viewer HTML similar to Streamlit version."""
    if clustered_df.empty:
        return "<p>No cluster data available</p>"
    
    df = clustered_df.copy()
    
    # Filter by models if specified
    if selected_models:
        df = df[df['model'].isin(selected_models)]
    
    if df.empty:
        return f"<p>No data found for selected models: {', '.join(selected_models or [])}</p>"
    
    # Get cluster scores data for quality and frequency information
    from .state import app_state
    cluster_scores = app_state.get("metrics", {}).get("cluster_scores", {})
    
    # Use the actual column names from the pipeline output (matching Streamlit version)
    if cluster_level == 'fine':
        id_col = 'property_description_fine_cluster_id'
        label_col = 'property_description_fine_cluster_label'
        # Also check for alternative naming without prefix
        alt_id_col = 'fine_cluster_id'
        alt_label_col = 'fine_cluster_label'
    else:
        id_col = 'property_description_coarse_cluster_id'  
        label_col = 'property_description_coarse_cluster_label'
        # Also check for alternative naming without prefix
        alt_id_col = 'coarse_cluster_id'
        alt_label_col = 'coarse_cluster_label'
    
    # Track if we fall back from coarse to fine
    fell_back_to_fine = False
    
    # Check if required columns exist and provide helpful debug info
    # Try both naming patterns
    if id_col in df.columns and label_col in df.columns:
        # Use the expected naming pattern
        pass
    elif alt_id_col in df.columns and alt_label_col in df.columns:
        # Use the alternative naming pattern
        id_col = alt_id_col
        label_col = alt_label_col
    else:
        # If coarse clusters are not available, try to fall back to fine clusters
        if cluster_level == 'coarse':
            # Check if fine clusters are available
            fine_id_col = 'property_description_fine_cluster_id'
            fine_label_col = 'property_description_fine_cluster_label'
            fine_alt_id_col = 'fine_cluster_id'
            fine_alt_label_col = 'fine_cluster_label'
            
            if (fine_id_col in df.columns and fine_label_col in df.columns) or (fine_alt_id_col in df.columns and fine_alt_label_col in df.columns):
                # Fall back to fine clusters
                if fine_id_col in df.columns and fine_label_col in df.columns:
                    id_col = fine_id_col
                    label_col = fine_label_col
                else:
                    id_col = fine_alt_id_col
                    label_col = fine_alt_label_col
                cluster_level = 'fine'  # Update the cluster level for display
                fell_back_to_fine = True
            else:
                # No cluster columns available at all
                available_cols = list(df.columns)
                return f"""
                <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
                    <h4>❌ Missing cluster columns in data</h4>
                    <p><strong>Expected:</strong> {id_col}, {label_col} OR {alt_id_col}, {alt_label_col}</p>
                    <p><strong>Available columns:</strong> {', '.join(available_cols)}</p>
                    <p>Please ensure your data contains clustering results from the LMM-Vibes pipeline.</p>
                </div>
                """
        else:
            # For fine clusters, show the original error
            available_cols = list(df.columns)
            return f"""
            <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
                <h4>❌ Missing {cluster_level} cluster columns in data</h4>
                <p><strong>Expected:</strong> {id_col}, {label_col} OR {alt_id_col}, {alt_label_col}</p>
                <p><strong>Available columns:</strong> {', '.join(available_cols)}</p>
                <p>Please ensure your data contains clustering results from the LMM-Vibes pipeline.</p>
            </div>
            """
    
    # Group by cluster to get cluster information
    try:
        cluster_groups = df.groupby([id_col, label_col]).agg({
            'property_description': ['count', lambda x: x.unique().tolist()],
            'model': lambda x: x.unique().tolist()
        }).reset_index()
        
        # Flatten column names
        cluster_groups.columns = [
            id_col, label_col, 'size', 'property_descriptions', 'models'
        ]
        
        # Sort by size (largest first)
        cluster_groups = cluster_groups.sort_values('size', ascending=False)
        
    except Exception as e:
        return f"""
        <div style="padding: 20px; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px;">
            <h4>❌ Error processing cluster data</h4>
            <p><strong>Error:</strong> {str(e)}</p>
            <p>Please check your data format and try again.</p>
        </div>
        """
    
    if len(cluster_groups) == 0:
        return """
        <div style="padding: 20px; background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 8px;">
            <h4>ℹ️ No clusters found</h4>
            <p>No clusters match your current filters. Try selecting different models or adjusting your search.</p>
        </div>
        """
    
    # Create HTML
    html = f"""
    <div style="max-width: 1200px; margin: 0 auto;">
        <h3>🔍 Interactive Cluster Viewer ({cluster_level.title()} Level)</h3>
        <p style="color: #666; margin-bottom: 20px;">
            Click on clusters below to explore their property descriptions. 
            Showing {len(cluster_groups)} clusters sorted by size.
        </p>
    """
    
    # Add a note if we fell back from coarse to fine clusters
    if cluster_level == 'fine' and fell_back_to_fine:
        html += """
        <div style="padding: 15px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; margin-bottom: 20px;">
            <strong>Note:</strong> Coarse clusters not available in this dataset. Showing fine clusters instead.
        </div>
        """
    
    for _, row in cluster_groups.iterrows():
        cluster_id = row[id_col]
        cluster_label = row[label_col]
        cluster_size = row['size']
        property_descriptions = row['property_descriptions']
        models_in_cluster = row['models']
        
        # Get quality and frequency information from cluster_scores
        cluster_metrics = cluster_scores.get(cluster_label, {})
        frequency_pct = cluster_metrics.get("proportion", 0) * 100 if cluster_metrics else 0
        quality_scores = cluster_metrics.get("quality", {})
        quality_delta = cluster_metrics.get("quality_delta", {})
        
        # Calculate average quality score for header display
        avg_quality_score = 0
        if quality_scores:
            avg_quality_score = sum(quality_scores.values()) / len(quality_scores)
        
        # Calculate average quality delta for header display
        avg_quality_delta = 0
        if quality_delta:
            avg_quality_delta = sum(quality_delta.values()) / len(quality_delta)
        
        # Format quality scores for detailed view
        quality_html = ""
        if quality_scores:
            quality_parts = []
            for metric_name, score in quality_scores.items():
                color = "#28a745" if score >= 0 else "#dc3545"
                quality_parts.append(f'<span style="color:{color}; font-weight:500;">{metric_name}: {score:.3f}</span>')
            quality_html = " | ".join(quality_parts)
        else:
            quality_html = '<span style="color:#666;">No quality data</span>'
        
        # Format quality delta (relative to average)
        quality_delta_html = ""
        if quality_delta:
            delta_parts = []
            for metric_name, delta in quality_delta.items():
                color = "#28a745" if delta >= 0 else "#dc3545"
                sign = "+" if delta >= 0 else ""
                delta_parts.append(f'<span style="color:{color}; font-weight:500;">{metric_name}: {sign}{delta:.3f}</span>')
            quality_delta_html = " | ".join(delta_parts)
        else:
            quality_delta_html = '<span style="color:#666;">No delta data</span>'
        
        # Format header quality score with visual indicators
        header_quality_text = f"{avg_quality_score:.3f}" if quality_scores else "N/A"
        header_delta_text = f"{avg_quality_delta:+.3f}" if quality_delta else ""
        
        # Determine color for delta based on value
        if avg_quality_delta > 0:
            delta_color = "#28a745"  # Green for above average
        elif avg_quality_delta < 0:
            delta_color = "#dc3545"  # Red for below average
        else:
            delta_color = "#666666"  # Black/gray for neutral
        
        # Create expandable cluster card
        html += f"""
        <details style="margin: 15px 0; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">
            <summary style="
                padding: 15px; 
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                color: #333; 
                cursor: pointer; 
                font-weight: 600;
                font-size: 16px;
                user-select: none;
                list-style: none;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #dee2e6;
            ">
                <div>
                    {cluster_label} 
                    <span style="
                        background: #ffffff; 
                        color: #495057;
                        padding: 4px 8px; 
                        border-radius: 12px; 
                        font-size: 14px; 
                        margin-left: 10px;
                        border: 1px solid #dee2e6;
                    ">
                        {cluster_size} properties
                    </span>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 18px; font-weight: bold; margin-bottom: 4px;">
                        <span style="background: #ffffff; color: #495057; padding: 4px 8px; border-radius: 12px; font-size: 14px; font-weight: bold; border: 1px solid #dee2e6;">
                            {header_quality_text} <span style="color: {delta_color};">({header_delta_text})</span>
                        </span>
                    </div>
                    <div style="font-size: 12px; color: #6c757d;">
                        {frequency_pct:.1f}% frequency
                    </div>
                </div>
            </summary>
            
            <div style="padding: 20px; background: #f8f9fa;">
                <div style="margin-bottom: 15px;">
                    <strong>Cluster ID:</strong> {cluster_id}<br>
                    <strong>Size:</strong> {cluster_size} properties<br>
                    <strong>Models:</strong> {', '.join(models_in_cluster)}<br>
                    <strong>Frequency:</strong> {frequency_pct:.1f}% of all conversations<br>
                    <strong>Quality Scores:</strong> {quality_html}<br>
                    <strong>Quality vs Average:</strong> {quality_delta_html}
                </div>
                
                <h4 style="color: #333; margin: 15px 0 10px 0;">
                    Property Descriptions ({len(property_descriptions)})
                </h4>
                
                <div style="max-height: 300px; overflow-y: auto; background: white; border: 1px solid #ddd; border-radius: 4px; padding: 10px;">
        """
        
        # Display property descriptions
        for i, desc in enumerate(property_descriptions, 1):
            html += f"""
                    <div style="
                        padding: 8px; 
                        margin: 4px 0; 
                        background: #f8f9fa; 
                        border-left: 3px solid #667eea;
                        border-radius: 2px;
                    ">
                        <strong>{i}.</strong> {desc}
                    </div>
            """
        
        html += """
                </div>
            </div>
        </details>
        """
    
    html += "</div>"
    return html


def get_cluster_statistics(clustered_df: pd.DataFrame, 
                         selected_models: Optional[List[str]] = None) -> Dict[str, Any]:
    """Get cluster statistics for display."""
    if clustered_df.empty:
        return {}
    
    df = clustered_df.copy()
    
    # Filter by models if specified
    if selected_models:
        df = df[df['model'].isin(selected_models)]
    
    stats = {
        'total_properties': len(df),
        'total_models': df['model'].nunique() if 'model' in df.columns else 0,
    }
    
    # Fine cluster statistics - try both naming patterns
    fine_id_col = 'property_description_fine_cluster_id'
    alt_fine_id_col = 'fine_cluster_id'
    
    if fine_id_col in df.columns:
        stats['fine_clusters'] = df[fine_id_col].nunique()
        cluster_sizes = df.groupby(fine_id_col).size()
        stats['min_properties_per_fine_cluster'] = cluster_sizes.min() if not cluster_sizes.empty else 0
        stats['max_properties_per_fine_cluster'] = cluster_sizes.max() if not cluster_sizes.empty else 0
        stats['avg_properties_per_fine_cluster'] = cluster_sizes.mean() if not cluster_sizes.empty else 0
    elif alt_fine_id_col in df.columns:
        stats['fine_clusters'] = df[alt_fine_id_col].nunique()
        cluster_sizes = df.groupby(alt_fine_id_col).size()
        stats['min_properties_per_fine_cluster'] = cluster_sizes.min() if not cluster_sizes.empty else 0
        stats['max_properties_per_fine_cluster'] = cluster_sizes.max() if not cluster_sizes.empty else 0
        stats['avg_properties_per_fine_cluster'] = cluster_sizes.mean() if not cluster_sizes.empty else 0
    
    # Coarse cluster statistics - try both naming patterns
    coarse_id_col = 'property_description_coarse_cluster_id'
    alt_coarse_id_col = 'coarse_cluster_id'
    
    if coarse_id_col in df.columns:
        stats['coarse_clusters'] = df[coarse_id_col].nunique()
        cluster_sizes = df.groupby(coarse_id_col).size()
        stats['min_properties_per_coarse_cluster'] = cluster_sizes.min() if not cluster_sizes.empty else 0
        stats['max_properties_per_coarse_cluster'] = cluster_sizes.max() if not cluster_sizes.empty else 0
        stats['avg_properties_per_coarse_cluster'] = cluster_sizes.mean() if not cluster_sizes.empty else 0
    elif alt_coarse_id_col in df.columns:
        stats['coarse_clusters'] = df[alt_coarse_id_col].nunique()
        cluster_sizes = df.groupby(alt_coarse_id_col).size()
        stats['min_properties_per_coarse_cluster'] = cluster_sizes.min() if not cluster_sizes.empty else 0
        stats['max_properties_per_coarse_cluster'] = cluster_sizes.max() if not cluster_sizes.empty else 0
        stats['avg_properties_per_coarse_cluster'] = cluster_sizes.mean() if not cluster_sizes.empty else 0
    
    return stats


def get_unique_values_for_dropdowns(clustered_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Get unique values for dropdown menus."""
    if clustered_df.empty:
        return {'prompts': [], 'models': [], 'properties': []}
    
    # Get unique values, handling missing columns gracefully
    prompts = []
    if 'prompt' in clustered_df.columns:
        unique_prompts = clustered_df['prompt'].dropna().unique().tolist()
        prompts = [prompt[:100] + "..." if len(prompt) > 100 else prompt for prompt in sorted(unique_prompts)]
    elif 'question' in clustered_df.columns:
        unique_prompts = clustered_df['question'].dropna().unique().tolist()
        prompts = [prompt[:100] + "..." if len(prompt) > 100 else prompt for prompt in sorted(unique_prompts)]
    elif 'input' in clustered_df.columns:
        unique_prompts = clustered_df['input'].dropna().unique().tolist()
        prompts = [prompt[:100] + "..." if len(prompt) > 100 else prompt for prompt in sorted(unique_prompts)]
    elif 'user_prompt' in clustered_df.columns:
        unique_prompts = clustered_df['user_prompt'].dropna().unique().tolist()
        prompts = [prompt[:100] + "..." if len(prompt) > 100 else prompt for prompt in sorted(unique_prompts)]
    
    models = []
    if 'model' in clustered_df.columns:
        models = sorted(clustered_df['model'].dropna().unique().tolist())
    
    # Use fine cluster labels instead of property descriptions - try both naming patterns
    properties = []
    fine_label_col = 'property_description_fine_cluster_label'
    alt_fine_label_col = 'fine_cluster_label'
    
    if fine_label_col in clustered_df.columns:
        unique_properties = clustered_df[fine_label_col].dropna().unique().tolist()
        properties = [prop[:100] + "..." if len(prop) > 100 else prop for prop in sorted(unique_properties)]
    elif alt_fine_label_col in clustered_df.columns:
        unique_properties = clustered_df[alt_fine_label_col].dropna().unique().tolist()
        properties = [prop[:100] + "..." if len(prop) > 100 else prop for prop in sorted(unique_properties)]
    elif 'property_description' in clustered_df.columns:
        # Fallback to property descriptions if cluster labels not available
        unique_properties = clustered_df['property_description'].dropna().unique().tolist()
        properties = [prop[:100] + "..." if len(prop) > 100 else prop for prop in sorted(unique_properties)]
    
    return {
        'prompts': prompts,
        'models': models, 
        'properties': properties
    }

# ---------------------------------------------------------------------------
# Example data extraction (restored)
# ---------------------------------------------------------------------------

def get_example_data(
    clustered_df: pd.DataFrame,
    selected_prompt: str | None = None,
    selected_model: str | None = None,
    selected_property: str | None = None,
    max_examples: int = 5,
) -> List[Dict[str, Any]]:
    """Return a list of example rows filtered by prompt / model / property.

    This function was accidentally removed during a refactor; it is required by
    *examples_tab.py* and other parts of the UI.
    """

    if clustered_df.empty:
        return []

    df = clustered_df.copy()

    # Apply model filter
    if selected_model and selected_model != "All Models":
        df = df[df["model"] == selected_model]

    # Apply prompt filter (supports truncated options ending with ...)
    if selected_prompt and selected_prompt != "All Prompts":
        prompt_cols = [c for c in ["prompt", "question", "input", "user_prompt"] if c in df.columns]
        if prompt_cols:
            if selected_prompt.endswith("..."):
                prefix = selected_prompt[:-3]
                mask = False
                for c in prompt_cols:
                    mask |= df[c].str.startswith(prefix, na=False)
                df = df[mask]
            else:
                mask = False
                for c in prompt_cols:
                    mask |= df[c] == selected_prompt
                df = df[mask]

    # Apply property filter (fine cluster label preferred)
    if selected_property and selected_property != "All Clusters":
        # Try both naming patterns for fine cluster labels
        label_col = None
        if "property_description_fine_cluster_label" in df.columns:
            label_col = "property_description_fine_cluster_label"
        elif "fine_cluster_label" in df.columns:
            label_col = "fine_cluster_label"
        else:
            label_col = "property_description"
            
        if selected_property.endswith("..."):
            df = df[df[label_col].str.startswith(selected_property[:-3], na=False)]
        else:
            df = df[df[label_col] == selected_property]

    # Limit size
    df = df.head(max_examples)

    examples: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        prompt_val = next(
            (row.get(col) for col in ["prompt", "question", "input", "user_prompt"] if row.get(col) is not None),
            "N/A",
        )

        response_val = next(
            (
                row.get(col)
                for col in [
                    "model_response",
                    "model_a_response",
                    "model_b_response",
                    "responses",
                    "response",
                    "output",
                ]
                if row.get(col) is not None
            ),
            "N/A",
        )

        # Try both naming patterns for cluster data
        fine_cluster_id = row.get("property_description_fine_cluster_id", row.get("fine_cluster_id", "N/A"))
        fine_cluster_label = row.get("property_description_fine_cluster_label", row.get("fine_cluster_label", "N/A"))
        coarse_cluster_id = row.get("property_description_coarse_cluster_id", row.get("coarse_cluster_id", "N/A"))
        coarse_cluster_label = row.get("property_description_coarse_cluster_label", row.get("coarse_cluster_label", "N/A"))

        examples.append(
            {
                "id": row.get("id", "N/A"),
                "model": row.get("model", "N/A"),
                "prompt": prompt_val,
                "response": response_val,
                "property_description": row.get("property_description", "N/A"),
                "score": row.get("score", "N/A"),
                "fine_cluster_id": fine_cluster_id,
                "fine_cluster_label": fine_cluster_label,
                "coarse_cluster_id": coarse_cluster_id,
                "coarse_cluster_label": coarse_cluster_label,
                "category": row.get("category", "N/A"),
                "type": row.get("type", "N/A"),
                "impact": row.get("impact", "N/A"),
                "reason": row.get("reason", "N/A"),
                "evidence": row.get("evidence", "N/A"),
                "behavior_type": row.get("behavior_type", "N/A"),
                "raw_response": row.get("raw_response", "N/A"),
                "contains_errors": row.get("contains_errors", "N/A"),
                "unexpected_behavior": row.get("unexpected_behavior", "N/A"),
            }
        )

    return examples

# ---------------------------------------------------------------------------
# Back-compat to ensure moved helpers remain accessible
# ---------------------------------------------------------------------------

convert_to_openai_format = _convdisp.convert_to_openai_format  # type: ignore
display_openai_conversation_html = _convdisp.display_openai_conversation_html  # type: ignore
pretty_print_embedded_dicts = _convdisp.pretty_print_embedded_dicts  # type: ignore


def format_examples_display(examples: List[Dict[str, Any]], 
                          selected_prompt: str = None,
                          selected_model: str = None,
                          selected_property: str = None,
                          use_accordion: bool = True,
                          pretty_print_dicts: bool = True) -> str:
    """Format examples for HTML display with proper conversation rendering.
    
    Args:
        examples: List of example dictionaries
        selected_prompt: Currently selected prompt filter
        selected_model: Currently selected model filter  
        selected_property: Currently selected property filter
        use_accordion: If True, group system and info messages in collapsible accordions
    """
    if not examples:
        filter_info = []
        if selected_model and selected_model != "All Models":
            filter_info.append(f"Model: {selected_model}")
        if selected_prompt and selected_prompt != "All Prompts":
            filter_info.append(f"Prompt: {selected_prompt[:50]}...")
        if selected_property and selected_property != "All Clusters":
            filter_info.append(f"Cluster: {selected_property[:50]}...")
        
        filter_text = " | ".join(filter_info) if filter_info else "current filters"
        
        return f"""
        <div style="padding: 30px; text-align: center; background: #f8f9fa; border-radius: 8px;">
            <h3 style="color: #666;">📭 No Examples Found</h3>
            <p>No examples match {filter_text}.</p>
            <p>Try adjusting your filters or selecting "All" for some options.</p>
        </div>
        """
    
    # Create filter summary
    filter_summary = ""
    active_filters = []
    if selected_model and selected_model != "All Models":
        active_filters.append(f"🤖 Model: {selected_model}")
    if selected_prompt and selected_prompt != "All Prompts":
        active_filters.append(f"💬 Prompt: {selected_prompt[:50]}...")
    if selected_property and selected_property != "All Clusters":
        active_filters.append(f"🏷️ Cluster: {selected_property[:50]}...")
    
    if active_filters:
        filter_summary = f"""
        <div style="background: #e3f2fd; border-left: 4px solid #2196f3; padding: 15px; margin-bottom: 20px; border-radius: 4px;">
            <h4 style="margin: 0 0 10px 0;">🔍 Active Filters</h4>
            <div>{' | '.join(active_filters)}</div>
        </div>
        """
    
    html = f"""
    <div style="max-width: 1200px; margin: 0 auto;">
        {filter_summary}
    """
    
    for i, example in enumerate(examples, 1):
        # Convert response to OpenAI format for proper display
        response_data = example['response']
        if response_data != 'N/A':
            openai_conversation = convert_to_openai_format(response_data)
            conversation_html = display_openai_conversation_html(openai_conversation, use_accordion=use_accordion, pretty_print_dicts=pretty_print_dicts)
        else:
            conversation_html = "<p style='color: #dc3545; font-style: italic;'>No response data available</p>"
        
        # Determine cluster info
        cluster_info = ""
        if example['fine_cluster_label'] != 'N/A':
            cluster_info = f"""
            <div style="margin-top: 10px; font-size: 13px; color: #666;">
                <strong>🏷️ Cluster:</strong> {example['fine_cluster_label']} (ID: {example['fine_cluster_id']})
            </div>
            """
        
        # Score display for summary
        score_badge = ""
        if example['score'] != 'N/A':
            try:
                score_val = float(example['score'])
                score_color = '#28a745' if score_val >= 0 else '#dc3545'
                score_badge = f"""
                <span style="
                    background: {score_color}; 
                    color: white; 
                    padding: 4px 8px; 
                    border-radius: 12px; 
                    font-size: 12px; 
                    font-weight: bold;
                    margin-left: 10px;
                ">
                    Score: {score_val:.3f}
                </span>
                """
            except:
                pass
        
        # Create short preview of prompt for summary
        prompt_preview = example['prompt'][:80] + "..." if len(example['prompt']) > 80 else example['prompt']
        
        # Create expandable example card
        # First example is expanded by default
        open_attr = "open" if i == 1 else ""
        
        html += f"""
        <details {open_attr} style="
            margin: 15px 0; 
            border: 1px solid #ddd; 
            border-radius: 8px; 
            overflow: hidden; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <summary style="
                padding: 15px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; 
                cursor: pointer; 
                font-weight: 600;
                font-size: 16px;
                user-select: none;
                list-style: none;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <span style="color: white;">
                    Example {i} - {example['model']}
                    {score_badge}
                </span>
                <span style="font-size: 12px; opacity: 0.8; color: white;">
                    {prompt_preview}
                </span>
            </summary>
            
            <div style="padding: 20px; background: #f8f9fa;">
                <div style="margin-bottom: 10px; font-size: 13px; color: #666;">
                    <strong>ID:</strong> {example['id']}
                    {f' | <strong>Score:</strong> {example["score"]}' if example["score"] not in ["N/A", "None"] else ""}
                </div>
                
                <div style="margin-bottom: 10px;">
                    <h5 style="margin: 0 0 8px 0; color: #333; font-size: 14px;">🏷️ Property Description</h5>
                    <div style="
                        background: #fff8e1; 
                        padding: 10px; 
                        border-radius: 6px; 
                        border-left: 4px solid #ffa000;
                        font-size: 13px;
                        font-style: italic;
                    ">
                        {example['property_description']}
                    </div>
                    {cluster_info}
                </div>
                
                <!-- Additional Property Attributes -->
                <div style="margin-bottom: 15px;">
                    <h5 style="margin: 0 0 8px 0; color: #333; font-size: 14px;">📋 Property Details</h5>
                    <div style="
                        background: #f8f9fa; 
                        padding: 15px; 
                        border-radius: 6px; 
                        border-left: 4px solid #6c757d;
                        font-size: 13px;
                    ">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                            {f'<div><strong>Category:</strong> {example["category"]}</div>' if example["category"] not in ["N/A", "None"] else ""}
                            {f'<div><strong>Type:</strong> {example["type"]}</div>' if example["type"] not in ["N/A", "None"] else ""}
                            {f'<div><strong>Impact:</strong> {example["impact"]}</div>' if example["impact"] not in ["N/A", "None"] else ""}
                            {f'<div><strong>Behavior Type:</strong> {example["behavior_type"]}</div>' if example["behavior_type"] not in ["N/A", "None"] else ""}
                            {f'<div><strong>Contains Errors:</strong> {example["contains_errors"]}</div>' if example["contains_errors"] not in ["N/A", "None"] else ""}
                            {f'<div><strong>Unexpected Behavior:</strong> {example["unexpected_behavior"]}</div>' if example["unexpected_behavior"] not in ["N/A", "None"] else ""}
                        </div>
                        {f'<div style="margin-top: 10px;"><strong>Reason:</strong> {example["reason"]}</div>' if example["reason"] not in ["N/A", "None"] else ""}
                        {f'<div style="margin-top: 10px;"><strong>Evidence:</strong> {example["evidence"]}</div>' if example["evidence"] not in ["N/A", "None"] else ""}
                    </div>
                </div>

                <div style="margin-bottom: 15px;">
                    <h5 style="margin: 0 0 8px 0; color: #333; font-size: 14px;">💬 Conversation</h5>
                    <div style="border-radius: 6px; font-size: 13px; line-height: 1.5;">
                        {conversation_html}
                    </div>
                </div>
            </div>
        </details>
        """
    
    html += "</div>"
    return html 

# ---------------------------------------------------------------------------
# Legacy function aliases (backward compatibility)
# ---------------------------------------------------------------------------

def compute_model_rankings(*args, **kwargs):
    """Legacy alias → forwards to compute_model_rankings_new."""
    return compute_model_rankings_new(*args, **kwargs)


def create_model_summary_card(*args, **kwargs):
    """Legacy alias → forwards to create_model_summary_card_new."""
    return create_model_summary_card_new(*args, **kwargs) 


def get_total_clusters_count(metrics: Dict[str, Any]) -> int:
    """Get the total number of clusters from the metrics data."""
    cluster_scores = metrics.get("cluster_scores", {})
    return len(cluster_scores)


def get_light_color_for_cluster(cluster_name: str, index: int) -> str:
    """Generate a light dusty blue background for cluster boxes.
    
    Returns a consistent light dusty blue color for all clusters.
    """
    return "#f0f4f8"  # Very light dusty blue 

__all__ = [
    "get_model_clusters",
    "get_all_models", 
    "get_all_clusters",
    "format_confidence_interval",
    "get_confidence_interval_width",
    "has_confidence_intervals",
    "extract_quality_score",
    "get_top_clusters_for_model",
    "compute_model_rankings_new",
    "create_model_summary_card_new",
    "format_cluster_dataframe",
    "truncate_cluster_name",
    "create_frequency_comparison_table",
    "create_frequency_comparison_plots",
    "search_clusters_by_text",
    "create_interactive_cluster_viewer",
    "get_cluster_statistics",
    "get_unique_values_for_dropdowns",
    "get_example_data",
    "format_examples_display",
    "compute_model_rankings",
    "create_model_summary_card",
    "get_total_clusters_count",
] 
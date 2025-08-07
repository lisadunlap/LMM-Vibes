"""Helpers for the **View Clusters** tab – both the interactive HTML and
fallback dataframe view."""
from typing import List

import pandas as pd

from .state import app_state
from .utils import (
    search_clusters_by_text,
    create_interactive_cluster_viewer,
    get_cluster_statistics,
    format_cluster_dataframe,
)

__all__ = ["view_clusters_interactive", "view_clusters_table"]


# ---------------------------------------------------------------------------
# Interactive HTML view
# ---------------------------------------------------------------------------

def view_clusters_interactive(
    selected_models: List[str],
    cluster_level: str,
    search_term: str = "",
) -> str:
    if app_state["clustered_df"] is None:
        return (
            "<p style='color: #e74c3c; padding: 20px;'>❌ Please load data first "
            "using the 'Load Data' tab</p>"
        )

    df = app_state["clustered_df"].copy()

    # Apply search filter first
    if search_term and search_term.strip():
        df = search_clusters_by_text(df, search_term.strip())

    # Build interactive viewer
    cluster_html = create_interactive_cluster_viewer(df, selected_models, cluster_level)

    # Statistics summary at the top
    stats = get_cluster_statistics(df, selected_models)
    if not stats:
        return (
            "<p style='color: #e74c3c; padding: 20px;'>❌ No cluster data available</p>"
        )

    # Get additional metrics from cluster_scores
    cluster_scores = app_state.get("metrics", {}).get("cluster_scores", {})
    
    # Calculate average quality scores and frequency
    total_frequency = 0
    quality_scores_list = []
    
    for cluster_name, cluster_data in cluster_scores.items():
        total_frequency += cluster_data.get("proportion", 0) * 100
        quality_scores = cluster_data.get("quality", {})
        if quality_scores:
            quality_scores_list.extend(quality_scores.values())
    
    avg_quality = sum(quality_scores_list) / len(quality_scores_list) if quality_scores_list else 0

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
            <div>
                <div style="font-size: 24px; font-weight: bold;">{avg_quality:.3f}</div>
                <div style="opacity: 0.9;">Avg Quality Score</div>
            </div>
    """

    if cluster_level == "fine" and "fine_clusters" in stats:
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
    elif cluster_level == "coarse" and "coarse_clusters" in stats:
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
    
    # Add a note if coarse clusters were requested but not available
    if cluster_level == "coarse" and "coarse_clusters" not in stats and "fine_clusters" in stats:
        stats_html += """
        <div style="
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        ">
            ⚠️ <strong>Note:</strong> Coarse clusters not available in this dataset. Showing fine clusters instead.
        </div>
        """

    # Additional filter chips
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


# ---------------------------------------------------------------------------
# Dataframe fallback view
# ---------------------------------------------------------------------------

def view_clusters_table(
    selected_models: List[str],
    cluster_level: str,
    search_term: str = "",
    max_rows: int = 500,
) -> pd.DataFrame:
    if app_state["clustered_df"] is None:
        return pd.DataFrame({"Message": ["Please load data first using the 'Load Data' tab"]})

    df = app_state["clustered_df"].copy()

    if search_term and search_term.strip():
        df = search_clusters_by_text(df, search_term.strip())

    formatted_df = format_cluster_dataframe(df, selected_models, cluster_level, max_rows)

    if formatted_df.empty:
        if search_term and search_term.strip():
            return pd.DataFrame({"Message": [f"No results found for search term '{search_term}'. Try a different search term."]})
        elif selected_models:
            available_models = df["model"].unique().tolist() if "model" in df.columns else []
            return pd.DataFrame({"Message": [
                f"No data found for selected models: {', '.join(selected_models)}. "
                f"Available models: {', '.join(available_models)}"
            ]})
        else:
            return pd.DataFrame({"Message": [
                "No data available. Please check your data files and try reloading."
            ]})

    return formatted_df 
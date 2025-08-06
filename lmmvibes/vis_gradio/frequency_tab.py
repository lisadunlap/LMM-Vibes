"""Logic for the **Frequency Comparison** tab."""
from typing import List, Tuple, Dict, Any

import pandas as pd

from .state import app_state


# ---------------------------------------------------------------------------
# NOTE: app_state currently stores metrics under the legacy key 'model_stats'.
# During later cleanup this module will switch to 'metrics'. For now we treat
# the value as already being the new FunctionalMetrics dict.
# ---------------------------------------------------------------------------

__all__ = ["create_frequency_comparison", "create_frequency_plots"]


def create_frequency_comparison(
    selected_models: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Create frequency comparison tables for the 3 functional metrics tables."""
    if not app_state["model_stats"]:
        empty_df = pd.DataFrame({"Message": ["Please load data first"]})
        return empty_df, empty_df, empty_df, ""

    if not selected_models:
        empty_df = pd.DataFrame({"Message": ["Please select at least one model"]})
        return empty_df, empty_df, empty_df, ""

    # Get the functional metrics data
    metrics_data = app_state["model_stats"]
    
    # Debug: Print data structure info
    print(f"DEBUG: Creating frequency comparison tables")
    print(f"  - Selected models: {selected_models}")
    print(f"  - Available keys in metrics_data: {list(metrics_data.keys())}")
    
    if "model_cluster_scores" in metrics_data:
        model_cluster_scores = metrics_data["model_cluster_scores"]
        print(f"  - Model cluster scores keys: {list(model_cluster_scores.keys())}")
        for model in selected_models:
            if model in model_cluster_scores:
                clusters = model_cluster_scores[model]
                print(f"  - {model}: {len(clusters)} clusters")
            else:
                print(f"  - {model}: NOT FOUND in model_cluster_scores")
    
    if "cluster_scores" in metrics_data:
        cluster_scores = metrics_data["cluster_scores"]
        print(f"  - Cluster scores: {len(cluster_scores)} clusters")
    
    if "model_scores" in metrics_data:
        model_scores = metrics_data["model_scores"]
        print(f"  - Model scores: {list(model_scores.keys())}")
    
    # Create the three tables
    model_cluster_df = create_model_cluster_table(metrics_data, selected_models)
    cluster_df = create_cluster_table(metrics_data, selected_models)
    model_df = create_model_table(metrics_data, selected_models)
    
    print(f"  - Created tables with rows: Model-Cluster={len(model_cluster_df)}, Cluster={len(cluster_df)}, Model={len(model_df)}")
    
    info_text = f"**Model-Cluster Scores:** {len(model_cluster_df)} rows | **Cluster Scores:** {len(cluster_df)} rows | **Model Scores:** {len(model_df)} rows"
    return model_cluster_df, cluster_df, model_df, info_text


def create_model_cluster_table(metrics_data: Dict[str, Any], selected_models: List[str]) -> pd.DataFrame:
    """Create table for model-cluster scores."""
    model_cluster_scores = metrics_data.get("model_cluster_scores", {})
    
    print(f"DEBUG: Creating model-cluster table")
    print(f"  - Available models in model_cluster_scores: {list(model_cluster_scores.keys())}")
    print(f"  - Selected models: {selected_models}")
    
    rows = []
    for model_name, clusters in model_cluster_scores.items():
        if model_name not in selected_models:
            print(f"  - Skipping {model_name} (not in selected_models)")
            continue
            
        print(f"  - Processing {model_name} with {len(clusters)} clusters")
        for cluster_name, metrics in clusters.items():
            # Basic metrics
            size = metrics.get("size", 0)
            proportion = metrics.get("proportion", 0) * 100  # Convert to percentage
            proportion_delta = metrics.get("proportion_delta", 0) * 100  # Convert to percentage
            
            # Quality metrics - show each metric separately
            quality = metrics.get("quality", {})
            quality_delta = metrics.get("quality_delta", {})
            
            # Create base row
            row = {
                "Model": model_name,
                "Cluster": cluster_name,
                "Size": size,
                "Proportion (%)": f"{proportion:.1f}",
                "Proportion Delta (%)": f"{proportion_delta:.1f}",
                # "Examples": len(metrics.get("examples", []))
            }
            
            # Add quality metrics for each individual metric
            for metric_name, quality_val in quality.items():
                row[f"Quality_{metric_name.title()}"] = f"{quality_val:.3f}"
            
            for metric_name, delta_val in quality_delta.items():
                row[f"Quality_Delta_{metric_name.title()}"] = f"{delta_val:+.3f}"
            
            # Confidence intervals
            proportion_ci = metrics.get("proportion_ci", {})
            proportion_delta_ci = metrics.get("proportion_delta_ci", {})
            
            # Significance flags
            proportion_delta_significant = metrics.get("proportion_delta_significant", False)
            quality_delta_significant = metrics.get("quality_delta_significant", {})
            
            # Format confidence intervals
            proportion_ci_str = format_ci(proportion_ci)
            proportion_delta_ci_str = format_ci(proportion_delta_ci)
            
            # Add confidence intervals and significance
            row.update({
                "Proportion CI": proportion_ci_str,
                "Proportion Delta CI": proportion_delta_ci_str,
                "Proportion Delta Significant": "Yes" if proportion_delta_significant else "No",
            })
            
            # Add quality delta significance for each metric
            for metric_name, is_significant in quality_delta_significant.items():
                row[f"Quality_Delta_{metric_name.title()}_Significant"] = "Yes" if is_significant else "No"
            
            rows.append(row)
    
    print(f"  - Created {len(rows)} rows for model-cluster table")
    return pd.DataFrame(rows)


def create_cluster_table(metrics_data: Dict[str, Any], selected_models: List[str]) -> pd.DataFrame:
    """Create table for cluster scores (aggregated across all models)."""
    cluster_scores = metrics_data.get("cluster_scores", {})
    
    print(f"DEBUG: Creating cluster table")
    print(f"  - Available clusters: {list(cluster_scores.keys())}")
    print(f"  - Number of clusters: {len(cluster_scores)}")
    
    rows = []
    for cluster_name, metrics in cluster_scores.items():
        # Basic metrics
        size = metrics.get("size", 0)
        proportion = metrics.get("proportion", 0) * 100  # Convert to percentage
        
        # Quality metrics - show each metric separately
        quality = metrics.get("quality", {})
        quality_delta = metrics.get("quality_delta", {})
        
        # Create base row
        row = {
            "Cluster": cluster_name,
            "Size": size,
            "Proportion (%)": f"{proportion:.1f}",
            # "Examples": len(metrics.get("examples", []))
        }
        
        # Add quality metrics for each individual metric
        for metric_name, quality_val in quality.items():
            row[f"Quality_{metric_name.title()}"] = f"{quality_val:.3f}"
        
        for metric_name, delta_val in quality_delta.items():
            row[f"Quality_Delta_{metric_name.title()}"] = f"{delta_val:+.3f}"
        
        # Confidence intervals
        proportion_ci = metrics.get("proportion_ci", {})
        quality_ci = metrics.get("quality_ci", {})
        quality_delta_ci = metrics.get("quality_delta_ci", {})
        
        # Significance flags
        quality_delta_significant = metrics.get("quality_delta_significant", {})
        
        # Format confidence intervals
        proportion_ci_str = format_ci(proportion_ci)
        quality_ci_str = format_ci(quality_ci)
        quality_delta_ci_str = format_ci(quality_delta_ci)
        
        # Add confidence intervals and significance
        row.update({
            "Proportion CI": proportion_ci_str,
        })
        
        # Add quality CI and significance for each metric
        for metric_name in quality.keys():
            if metric_name in quality_ci:
                ci = quality_ci[metric_name]
                row[f"Quality_{metric_name.title()}_CI"] = format_ci(ci)
        
        for metric_name in quality_delta.keys():
            if metric_name in quality_delta_ci:
                ci = quality_delta_ci[metric_name]
                row[f"Quality_Delta_{metric_name.title()}_CI"] = format_ci(ci)
            row[f"Quality_Delta_{metric_name.title()}_Significant"] = "Yes" if quality_delta_significant.get(metric_name, False) else "No"
        
        rows.append(row)
    
    print(f"  - Created {len(rows)} rows for cluster table")
    return pd.DataFrame(rows)


def create_model_table(metrics_data: Dict[str, Any], selected_models: List[str]) -> pd.DataFrame:
    """Create table for model scores (aggregated across all clusters)."""
    model_scores = metrics_data.get("model_scores", {})
    
    print(f"DEBUG: Creating model table")
    print(f"  - Available models in model_scores: {list(model_scores.keys())}")
    print(f"  - Selected models: {selected_models}")
    
    rows = []
    for model_name, metrics in model_scores.items():
        # Filter by selected models
        if model_name not in selected_models:
            print(f"  - Skipping {model_name} (not in selected_models)")
            continue
            
        print(f"  - Processing {model_name}")
        # Basic metrics
        size = metrics.get("size", 0)
        proportion = metrics.get("proportion", 0) * 100  # Convert to percentage
        
        # Quality metrics - show each metric separately
        quality = metrics.get("quality", {})
        quality_delta = metrics.get("quality_delta", {})
        
        # Create base row
        row = {
            "Model": model_name,
            "Size": size,
            # "Proportion (%)": f"{proportion:.1f}",
            # "Examples": len(metrics.get("examples", []))
        }
        
        # Add quality metrics for each individual metric
        for metric_name, quality_val in quality.items():
            row[f"Quality_{metric_name.title()}"] = f"{quality_val:.3f}"
        
        # for metric_name, delta_val in quality_delta.items():
        #     row[f"Quality_Delta_{metric_name.title()}"] = f"{delta_val:+.3f}"
        
        # Confidence intervals
        proportion_ci = metrics.get("proportion_ci", {})
        quality_ci = metrics.get("quality_ci", {})
        quality_delta_ci = metrics.get("quality_delta_ci", {})
        
        # Significance flags
        quality_delta_significant = metrics.get("quality_delta_significant", {})
        
        # Format confidence intervals
        proportion_ci_str = format_ci(proportion_ci)
        
        # Add confidence intervals and significance
        row.update({
            "Proportion CI": proportion_ci_str,
        })
        
        # Add quality CI and significance for each metric
        for metric_name in quality.keys():
            if metric_name in quality_ci:
                ci = quality_ci[metric_name]
                row[f"Quality_{metric_name.title()}_CI"] = format_ci(ci)
        
        # for metric_name in quality_delta.keys():
        #     if metric_name in quality_delta_ci:
        #         ci = quality_delta_ci[metric_name]
        #         row[f"Quality_Delta_{metric_name.title()}_CI"] = format_ci(ci)
        #     row[f"Quality_Delta_{metric_name.title()}_Significant"] = "Yes" if quality_delta_significant.get(metric_name, False) else "No"
        
        rows.append(row)
    
    print(f"  - Created {len(rows)} rows for model table")
    return pd.DataFrame(rows)


def format_ci(ci_dict: Dict[str, Any]) -> str:
    """Format confidence interval dictionary to string."""
    if not ci_dict or not isinstance(ci_dict, dict):
        return "N/A"
    
    lower = ci_dict.get("lower")
    upper = ci_dict.get("upper")
    mean = ci_dict.get("mean")
    
    if lower is not None and upper is not None:
        return f"[{lower:.3f}, {upper:.3f}]"
    elif mean is not None:
        return f"Mean: {mean:.3f}"
    else:
        return "N/A"


def create_frequency_plots(*_args, **_kwargs):
    """Removed for now – kept as a stub for backward compatibility."""
    return None, None 
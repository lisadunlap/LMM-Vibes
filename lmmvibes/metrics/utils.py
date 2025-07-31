from typing import Dict, List
import json
import tempfile
import os
import wandb
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def wandb_logging(model_stats: Dict[str, Dict[str, List[Dict]]]):
    """Log model stats to wandb.
    
    Logs:
    1. JSON artifact of the complete model stats
    2. Tables for each model showing fine/coarse cluster performance
    3. Plotly visualization of global stats across all models (if plotly is available)
    
    Args:
        model_stats: Dictionary with model names as keys and stats dictionaries as values.
                    Each model's stats contain "fine", "coarse", and "stats" keys.
    """
    if not wandb.run:
        print("⚠️  No active wandb run found. Skipping wandb logging.")
        return
        
    print("📊 Logging model stats to wandb...")
    
    try:
        # 1. Log JSON artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(model_stats, f, indent=2)
            temp_path = f.name
        
        try:
            artifact = wandb.Artifact("model_stats", type="model_metrics")
            artifact.add_file(temp_path, name="model_stats.json")
            wandb.log_artifact(artifact)
            print("✅ Logged model_stats.json artifact to wandb")
        finally:
            os.unlink(temp_path)  # Clean up temp file
        
        # 2. Create tables for each model
        for model_name, model_data in model_stats.items():
            # Create fine cluster table
            if "fine" in model_data and model_data["fine"]:
                fine_data = []
                for stat in model_data["fine"]:
                    row = {
                        "cluster_description": stat.get("property_description", ""),
                        "score": stat.get("score", 0),
                        "cluster_size": stat.get("cluster_size_global", 0),
                        "model_examples": stat.get("size", 0),
                        "proportion": stat.get("proportion", 0),
                    }
                    
                    # Add confidence intervals for score if available
                    has_score_ci = False
                    if "score_ci" in stat and stat["score_ci"]:
                        ci_lower = stat["score_ci"].get("lower", None)
                        ci_upper = stat["score_ci"].get("upper", None)
                        ci_average = stat["score_ci"].get("average", None)
                        if ci_lower is not None and ci_upper is not None:
                            row["score_ci_lower"] = float(ci_lower)
                            row["score_ci_upper"] = float(ci_upper)
                            row["score_ci_average"] = float(ci_average) if ci_average is not None else float((ci_lower + ci_upper) / 2)
                            row["score_ci_formatted"] = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
                            row["score_ci_width"] = float(ci_upper - ci_lower)
                            has_score_ci = True
                    
                    if not has_score_ci:
                        row["score_ci_lower"] = None
                        row["score_ci_upper"] = None
                        row["score_ci_average"] = None
                        row["score_ci_formatted"] = "N/A"
                        row["score_ci_width"] = None
                    
                    row["has_score_ci"] = has_score_ci
                    
                    # Add statistical significance if available
                    if "score_statistical_significance" in stat:
                        row["score_statistical_significance"] = stat["score_statistical_significance"]
                    else:
                        row["score_statistical_significance"] = None
                    
                    # Add quality scores if available
                    if "quality_score" in stat and stat["quality_score"]:
                        for key, value in stat["quality_score"].items():
                            row[f"quality_{key}"] = value
                            
                    # Add quality score confidence intervals if available
                    quality_ci_keys = []
                    if "quality_score_ci" in stat and stat["quality_score_ci"]:
                        for key, ci_data in stat["quality_score_ci"].items():
                            if isinstance(ci_data, dict):
                                ci_lower = ci_data.get("lower", None)
                                ci_upper = ci_data.get("upper", None)
                                ci_average = ci_data.get("average", None)
                                if ci_lower is not None and ci_upper is not None:
                                    row[f"quality_{key}_ci_lower"] = float(ci_lower)
                                    row[f"quality_{key}_ci_upper"] = float(ci_upper)
                                    row[f"quality_{key}_ci_average"] = float(ci_average) if ci_average is not None else float((ci_lower + ci_upper) / 2)
                                    row[f"quality_{key}_ci_formatted"] = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
                                    row[f"quality_{key}_ci_width"] = float(ci_upper - ci_lower)
                                    row[f"quality_{key}_has_ci"] = True
                                    quality_ci_keys.append(key)
                                else:
                                    row[f"quality_{key}_ci_lower"] = None
                                    row[f"quality_{key}_ci_upper"] = None
                                    row[f"quality_{key}_ci_average"] = None
                                    row[f"quality_{key}_ci_formatted"] = "N/A"
                                    row[f"quality_{key}_ci_width"] = None
                                    row[f"quality_{key}_has_ci"] = False
                            else:
                                row[f"quality_{key}_ci_lower"] = None
                                row[f"quality_{key}_ci_upper"] = None
                                row[f"quality_{key}_ci_average"] = None
                                row[f"quality_{key}_ci_formatted"] = "N/A"
                                row[f"quality_{key}_ci_width"] = None
                                row[f"quality_{key}_has_ci"] = False
                    
                    # Add quality score statistical significance if available
                    if "quality_score_statistical_significance" in stat and stat["quality_score_statistical_significance"]:
                        for key, significance in stat["quality_score_statistical_significance"].items():
                            row[f"quality_{key}_statistical_significance"] = significance
                    
                    # Add summary of available CIs
                    row["quality_ci_available_keys"] = ", ".join(quality_ci_keys) if quality_ci_keys else "None"
                    row["total_ci_metrics"] = len(quality_ci_keys) + (1 if has_score_ci else 0)
                    
                    fine_data.append(row)
                
                if fine_data:  # Only create table if we have data
                    fine_table = wandb.Table(dataframe=pd.DataFrame(fine_data))
                    wandb.log({f"Metrics/{model_name}_fine_clusters": fine_table})
            
            # Create coarse cluster table  
            if "coarse" in model_data and model_data["coarse"]:
                coarse_data = []
                for stat in model_data["coarse"]:
                    row = {
                        "cluster_description": stat.get("property_description", ""),
                        "score": stat.get("score", 0),
                        "total_conversations": stat.get("cluster_size_global", 0), 
                        "num_model_conversations": stat.get("size", 0),
                        "proportion_of_model_conversations": stat.get("proportion", 0),
                    }
                    
                    # Add confidence intervals for score if available
                    has_score_ci = False
                    if "score_ci" in stat and stat["score_ci"]:
                        ci_lower = stat["score_ci"].get("lower", None)
                        ci_upper = stat["score_ci"].get("upper", None)
                        ci_average = stat["score_ci"].get("average", None)
                        if ci_lower is not None and ci_upper is not None:
                            row["score_ci_lower"] = float(ci_lower)
                            row["score_ci_upper"] = float(ci_upper)
                            row["score_ci_average"] = float(ci_average) if ci_average is not None else float((ci_lower + ci_upper) / 2)
                            row["score_ci_formatted"] = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
                            row["score_ci_width"] = float(ci_upper - ci_lower)
                            has_score_ci = True
                    
                    if not has_score_ci:
                        row["score_ci_lower"] = None
                        row["score_ci_upper"] = None
                        row["score_ci_average"] = None
                        row["score_ci_formatted"] = "N/A"
                        row["score_ci_width"] = None
                    
                    row["has_score_ci"] = has_score_ci
                    
                    # Add statistical significance if available
                    if "score_statistical_significance" in stat:
                        row["score_statistical_significance"] = stat["score_statistical_significance"]
                    else:
                        row["score_statistical_significance"] = None
                    
                    # Add quality scores if available
                    if "quality_score" in stat and stat["quality_score"]:
                        for key, value in stat["quality_score"].items():
                            row[f"quality_{key}"] = value
                            
                    # Add quality score confidence intervals if available
                    quality_ci_keys = []
                    if "quality_score_ci" in stat and stat["quality_score_ci"]:
                        for key, ci_data in stat["quality_score_ci"].items():
                            if isinstance(ci_data, dict):
                                ci_lower = ci_data.get("lower", None)
                                ci_upper = ci_data.get("upper", None)
                                ci_average = ci_data.get("average", None)
                                if ci_lower is not None and ci_upper is not None:
                                    row[f"quality_{key}_ci_lower"] = float(ci_lower)
                                    row[f"quality_{key}_ci_upper"] = float(ci_upper)
                                    row[f"quality_{key}_ci_average"] = float(ci_average) if ci_average is not None else float((ci_lower + ci_upper) / 2)
                                    row[f"quality_{key}_ci_formatted"] = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
                                    row[f"quality_{key}_ci_width"] = float(ci_upper - ci_lower)
                                    row[f"quality_{key}_has_ci"] = True
                                    quality_ci_keys.append(key)
                                else:
                                    row[f"quality_{key}_ci_lower"] = None
                                    row[f"quality_{key}_ci_upper"] = None
                                    row[f"quality_{key}_ci_average"] = None
                                    row[f"quality_{key}_ci_formatted"] = "N/A"
                                    row[f"quality_{key}_ci_width"] = None
                                    row[f"quality_{key}_has_ci"] = False
                            else:
                                row[f"quality_{key}_ci_lower"] = None
                                row[f"quality_{key}_ci_upper"] = None
                                row[f"quality_{key}_ci_average"] = None
                                row[f"quality_{key}_ci_formatted"] = "N/A"
                                row[f"quality_{key}_ci_width"] = None
                                row[f"quality_{key}_has_ci"] = False
                    
                    # Add quality score statistical significance if available
                    if "quality_score_statistical_significance" in stat and stat["quality_score_statistical_significance"]:
                        for key, significance in stat["quality_score_statistical_significance"].items():
                            row[f"quality_{key}_statistical_significance"] = significance
                    
                    # Add summary of available CIs
                    row["quality_ci_available_keys"] = ", ".join(quality_ci_keys) if quality_ci_keys else "None"
                    row["total_ci_metrics"] = len(quality_ci_keys) + (1 if has_score_ci else 0)
                    
                    coarse_data.append(row)
                
                if coarse_data:  # Only create table if we have data
                    coarse_table = wandb.Table(dataframe=pd.DataFrame(coarse_data))
                    wandb.log({f"Metrics/{model_name}_coarse_clusters": coarse_table})
                        
        print(f"✅ Logged cluster tables for {len(model_stats)} models")
        
        # Check if any confidence intervals were logged
        has_any_ci = False
        for model_name, model_data in model_stats.items():
            for level in ["fine", "coarse"]:
                if level in model_data and model_data[level]:
                    for stat in model_data[level]:
                        if (stat.get("score_ci") or stat.get("quality_score_ci")):
                            has_any_ci = True
                            break
                    if has_any_ci:
                        break
            if has_any_ci:
                break
        
        if has_any_ci:
            print("📊 Confidence intervals included in wandb tables:")
            print("   • score_ci_lower/upper/average: Distinctiveness score confidence intervals")
            print("   • quality_{key}_ci_lower/upper/average: Quality score confidence intervals for each metric")
            print("   • *_ci_formatted: Human-readable CI format [lower, upper]")
            print("   • *_ci_width: Confidence interval width (upper - lower)")
            print("   • *_ci_average: Midpoint of confidence interval ((lower + upper) / 2)")
            print("   • has_score_ci: Boolean indicating if score CI is available")
            print("   • quality_{key}_has_ci: Boolean indicating if quality CI is available")
            
            # Add debugging for statistical significance
            print("📈 Statistical significance fields:")
            print("   • score_statistical_significance: Boolean indicating if distinctiveness is significantly > 1")
            print("   • quality_score_statistical_significance: Dict indicating if quality scores are significantly ≠ 0")
        else:
            print("ℹ️  No confidence intervals computed (set compute_confidence_intervals=True to enable)")

        # 3. Create plotly visualization of global stats (if plotly is available)
        global_stats_data = []
        models = []
        
        for model_name, model_data in model_stats.items():
            if "stats" in model_data and model_data["stats"]:
                models.append(model_name)
                global_stats_data.append(model_data["stats"])
        
        if global_stats_data and models:
            # Convert to DataFrame for easier plotting
            global_df = pd.DataFrame(global_stats_data, index=models)
            
            if not global_df.empty and len(global_df.columns) > 0:
                # Create subplots for each metric
                metrics = global_df.columns.tolist()
                n_metrics = len(metrics)
                
                if n_metrics == 1:
                    fig = go.Figure()
                    metric = metrics[0]
                    
                    fig.add_trace(go.Bar(
                        x=models,
                        y=global_df[metric],
                        name=metric,
                        text=[f"{val:.3f}" for val in global_df[metric]],
                        textposition='auto',
                    ))
                    fig.update_layout(
                        title=f"Global Model Performance: {metric}",
                        xaxis_title="Models",
                        yaxis_title=metric,
                        showlegend=False
                    )
                else:
                    # Multiple metrics - create subplots
                    cols = min(2, n_metrics)
                    rows = (n_metrics + cols - 1) // cols
                    
                    fig = make_subplots(
                        rows=rows, 
                        cols=cols,
                        subplot_titles=[f"Global {metric}" for metric in metrics],
                        vertical_spacing=0.15,
                        horizontal_spacing=0.1
                    )
                    
                    for i, metric in enumerate(metrics):
                        row = (i // cols) + 1
                        col = (i % cols) + 1
                        
                        fig.add_trace(
                            go.Bar(
                                x=models,
                                y=global_df[metric],
                                name=metric,
                                text=[f"{val:.3f}" for val in global_df[metric]],
                                textposition='auto',
                                showlegend=False,
                            ),
                            row=row, 
                            col=col
                        )
                        
                        fig.update_xaxes(title_text="Models", row=row, col=col)
                        fig.update_yaxes(title_text=metric, row=row, col=col)
                    
                    fig.update_layout(
                        title="Global Model Performance Across All Metrics",
                        height=300 * rows,
                        showlegend=False
                    )
                
                # Log the plotly figure
                wandb.log({"global_model_performance": fig})
                
                # Also create a heatmap if we have multiple metrics
                if n_metrics > 1:
                    heatmap_fig = go.Figure(data=go.Heatmap(
                        z=global_df.values.T,
                        x=models,
                        y=metrics,
                        colorscale='RdYlBu_r',
                        text=[[f"{val:.3f}" for val in row] for row in global_df.values.T],
                        texttemplate="%{text}",
                        textfont={"size": 10},
                        hoverongaps=False
                    ))
                    
                    heatmap_fig.update_layout(
                        title="Global Model Performance Heatmap",
                        xaxis_title="Models",
                        yaxis_title="Metrics"
                    )
                    
                    wandb.log({"global_performance_heatmap": heatmap_fig})
                
                print("✅ Logged global performance visualizations to wandb")
        else:
            print("⚠️  No global stats found across models")

        print("🎉 Wandb logging completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during wandb logging: {e}")
        raise
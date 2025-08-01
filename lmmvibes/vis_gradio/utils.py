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


def extract_quality_score(quality_score) -> float:
    """
    Extract a float quality score from quality_score field.
    
    Args:
        quality_score: Either a float, int, dictionary, or nested structure with score keys
        
    Returns:
        float: The quality score value, always guaranteed to be a float
    """
    if quality_score is None:
        return 0.0
    elif isinstance(quality_score, (int, float)):
        return float(quality_score)
    elif isinstance(quality_score, dict):
        # Handle dictionary cases
        if not quality_score:  # Empty dict
            return 0.0
        
        # Try to extract a numeric value from the dictionary
        for key, value in quality_score.items():
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, dict):
                # Handle nested dictionaries recursively
                nested_result = extract_quality_score(value)
                if nested_result != 0.0:  # Found a valid value
                    return nested_result
        
        # If no numeric values found, return 0.0
        return 0.0
    else:
        # For any other type, try to convert to float, fallback to 0.0
        try:
            return float(quality_score)
        except (ValueError, TypeError):
            return 0.0


def format_confidence_interval(score_ci: dict, confidence_level: float = 0.95) -> str:
    """
    Format confidence interval for display.
    
    Args:
        score_ci: Dict with "lower" and "upper" keys, or None
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        str: Formatted confidence interval string
    """
    if not score_ci or not isinstance(score_ci, dict):
        return "N/A"
    
    lower = score_ci.get("lower")
    upper = score_ci.get("upper")
    
    if lower is None or upper is None:
        return "N/A"
    
    ci_percent = int(confidence_level * 100)
    return f"[{lower:.3f}, {upper:.3f}] ({ci_percent}% CI)"


def has_confidence_intervals(cluster_stat: dict) -> bool:
    """
    Check if a cluster statistic has confidence intervals.
    
    Args:
        cluster_stat: Cluster statistic dictionary
        
    Returns:
        bool: True if confidence intervals are available
    """
    score_ci = cluster_stat.get('score_ci')
    return (isinstance(score_ci, dict) and 
            score_ci.get('lower') is not None and 
            score_ci.get('upper') is not None)


def get_confidence_interval_width(score_ci: dict) -> float:
    """
    Calculate the width of a confidence interval.
    
    Args:
        score_ci: Dict with "lower" and "upper" keys
        
    Returns:
        float: Width of the interval
    """
    if not score_ci or not isinstance(score_ci, dict):
        return 0.0
    
    lower = score_ci.get("lower")
    upper = score_ci.get("upper")
    
    if lower is None or upper is None:
        return 0.0
        
    return upper - lower


def compute_model_rankings(model_stats: Dict[str, Any]) -> List[tuple]:
    """Compute model rankings by average score"""
    model_scores = {}
    for model, stats in model_stats.items():
        fine_scores = [stat['score'] for stat in stats.get('fine', [])]
        if fine_scores:
            model_scores[model] = {
                'avg_score': np.mean(fine_scores),
                'median_score': np.median(fine_scores),
                'num_clusters': len(fine_scores),
                'top_score': max(fine_scores),
                'std_score': np.std(fine_scores)
            }
        else:
            model_scores[model] = {
                'avg_score': 0, 'median_score': 0, 'num_clusters': 0, 
                'top_score': 0, 'std_score': 0
            }
    
    return sorted(model_scores.items(), key=lambda x: x[1]['avg_score'], reverse=True)


def get_top_clusters_for_model(model_stats: Dict[str, Any], model_name: str, 
                              level: str = 'fine', top_n: int = 10) -> List[Dict[str, Any]]:
    """Get top N clusters for a specific model"""
    model_data = model_stats.get(model_name, {})
    clusters = model_data.get(level, [])
    return sorted(clusters, key=lambda x: x['score'], reverse=True)[:top_n]


def create_model_summary_card(model_name: str, model_stats: Dict[str, Any], 
                             cluster_level: str = 'fine', top_n: int = 3) -> str:
    """Create HTML summary card for a model."""
    top_clusters = get_top_clusters_for_model(model_stats, model_name, cluster_level, 5)
    
    if not top_clusters:
        return f"""
        <div style="padding: 20px; border: 1px solid #ddd; border-radius: 8px; margin: 10px 0;">
            <h3 style="margin: 0 0 10px 0; color: #333;">{model_name}</h3>
            <p>No cluster data available</p>
        </div>
        """
    
    # Calculate total battles by summing cluster sizes
    total_battles = sum(cluster.get('size', 0) for cluster in top_clusters)
    
    # Create cluster cards
    cluster_cards = ""
    for idx, cluster in enumerate(top_clusters[:top_n]):
        cluster_desc = cluster['property_description']
        frequency = cluster.get('proportion', 0) * 100  # Convert to percentage
        cluster_size = cluster.get('size', 0)
        cluster_size_global = cluster.get('cluster_size_global', 0)
        quality_score = extract_quality_score(cluster.get('quality_score', 0))
        distinctiveness = cluster.get('score', 1.0)
        
        # Get confidence intervals
        score_ci = cluster.get('score_ci')
        has_ci = has_confidence_intervals(cluster)
        ci_display = ""
        if has_ci:
            ci_display = f"<br><span style='font-size: 12px; color: #666;'>CI: {format_confidence_interval(score_ci)}</span>"
        
        cluster_cards += f"""
        <div style="margin: 8px 0; padding: 10px; border-left: 3px solid #3182ce; background-color: #f8f9fa; position: relative;">
            <div style="font-weight: 600; font-size: 14px; margin-bottom: 5px;">
                {cluster_desc}
            </div>
            <div style="font-size: 13px; color: #666;">
                <strong>{frequency:.1f}% frequency</strong> ({cluster_size} out of {cluster_size_global} total)
            </div>
            <div style="font-size: 12px; color: #3182ce;">
                {distinctiveness:.1f}x more distinctive{ci_display}
            </div>
            <div style="position: absolute; bottom: 8px; right: 10px; font-size: 12px; font-weight: 600; color: {'#28a745' if quality_score >= 0 else '#dc3545'};">
                Quality: {quality_score:.3f}
            </div>
        </div>
        """
    
    return f"""
    <div style="padding: 20px; border: 1px solid #ddd; border-radius: 8px; margin: 10px 0; background: white;">
        <div style="border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 12px;">
            <h3 style="margin: 0; color: #1a202c; font-weight: 500;">{model_name}</h3>
        </div>
        <p style="margin: 0 0 10px 0; color: #666; font-size: 13px;">
            {total_battles:,} battles • Top clusters by frequency ⬇️
        </p>
        {cluster_cards}
    </div>
    """


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
        cols = ['question_id', 'model', 'property_description', id_col, label_col, 'score']
    else:
        id_col = 'property_description_coarse_cluster_id'
        label_col = 'property_description_coarse_cluster_label'
        cols = ['question_id', 'model', 'property_description', id_col, label_col, 'score']
    
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
                                    cluster_level: str = 'fine',
                                    top_n: int = 50,
                                    selected_model: str = None,
                                    selected_quality_metric: str = None) -> pd.DataFrame:
    """Create a simplified comparison table with cluster, frequency, quality, and significance data."""
    if not selected_models:
        return pd.DataFrame()
    
    # Collect all clusters across all models for the chart
    all_clusters_data = []
    for model_name, model_data in model_stats.items():
        if model_name not in selected_models:
            continue
            
        clusters = model_data.get(cluster_level, [])
        for cluster in clusters:
            # Get confidence intervals for quality scores if available
            quality_score_ci = cluster.get('quality_score_ci', {})
            has_quality_ci = bool(quality_score_ci)
            
            # Get distinctiveness score confidence intervals
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
                'has_quality_ci': has_quality_ci,
                'quality_score': cluster.get('quality_score', {}),
                'quality_score_ci': quality_score_ci,
                # Significance flags from metrics computation
                'score_significance': cluster.get('score_statistical_significance', False),
                'quality_significance': any(cluster.get('quality_score_statistical_significance', {}).values()) if isinstance(cluster.get('quality_score_statistical_significance'), dict) else False
            })
    
    if not all_clusters_data:
        return pd.DataFrame()
        
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
        return pd.DataFrame()
    
    # Create a simplified table with the requested columns
    table_data = []
    for cluster_desc in top_clusters:
        # Get frequency data for all models
        cluster_freq_data = chart_data[chart_data['property_description'] == cluster_desc]
        
        # Calculate average frequency across all models
        avg_frequency = cluster_freq_data['frequency'].mean()
        
        # Calculate average confidence intervals for frequency
        freq_ci_lower = []
        freq_ci_upper = []
        for _, row in cluster_freq_data.iterrows():
            if (row.get('has_ci', False) and 
                row.get('ci_lower') is not None and 
                row.get('ci_upper') is not None):
                
                # Convert distinctiveness score CIs to frequency uncertainty
                distinctiveness_ci_width = row['ci_upper'] - row['ci_lower']
                freq_uncertainty = distinctiveness_ci_width * row['frequency'] * 0.1
                ci_lower = max(0, row['frequency'] - freq_uncertainty)
                ci_upper = row['frequency'] + freq_uncertainty
                freq_ci_lower.append(ci_lower)
                freq_ci_upper.append(ci_upper)
        
        # Calculate average frequency CI
        avg_freq_ci_lower = np.mean(freq_ci_lower) if freq_ci_lower else None
        avg_freq_ci_upper = np.mean(freq_ci_upper) if freq_ci_upper else None
        
        # Get quality score and confidence intervals for this cluster
        quality_score = None
        quality_score_ci = {}
        
        # Collect quality scores and CIs from all models for this cluster
        cluster_quality_scores = {}
        cluster_quality_cis = {}
        
        for model_name, model_data in model_stats.items():
            clusters = model_data.get(cluster_level, [])
            for cluster in clusters:
                if cluster['property_description'] == cluster_desc:
                    # Get quality score for this model
                    model_quality_score = cluster.get('quality_score', {})
                    if isinstance(model_quality_score, dict):
                        for score_key, score_value in model_quality_score.items():
                            if score_key not in cluster_quality_scores:
                                cluster_quality_scores[score_key] = []
                            cluster_quality_scores[score_key].append(score_value)
                    
                    # Get quality CIs for this model
                    model_quality_ci = cluster.get('quality_score_ci', {})
                    if isinstance(model_quality_ci, dict):
                        for score_key, ci_data in model_quality_ci.items():
                            if isinstance(ci_data, dict):
                                ci_lower = ci_data.get('lower')
                                ci_upper = ci_data.get('upper')
                                if ci_lower is not None and ci_upper is not None:
                                    if score_key not in cluster_quality_cis:
                                        cluster_quality_cis[score_key] = {'lower': [], 'upper': []}
                                    cluster_quality_cis[score_key]['lower'].append(ci_lower)
                                    cluster_quality_cis[score_key]['upper'].append(ci_upper)
                    break
        
        # Calculate average quality score and CI based on the selected metric
        quality_score = None
        quality_ci = None

        if selected_quality_metric:
            # A specific metric is selected
            scores = cluster_quality_scores.get(selected_quality_metric, [])
            if scores:
                quality_score = np.mean(scores)
            
            ci_data = cluster_quality_cis.get(selected_quality_metric, {})
            if ci_data.get('lower') and ci_data.get('upper'):
                avg_lower = np.mean(ci_data['lower'])
                avg_upper = np.mean(ci_data['upper'])
                quality_ci = {'lower': avg_lower, 'upper': avg_upper}
        else:
            # "All Metrics" is selected - average across all available metric scores
            all_scores = [score for scores in cluster_quality_scores.values() for score in scores]
            if all_scores:
                quality_score = np.mean(all_scores)

            # Average the CIs across all metrics
            all_ci_lowers = [l for ci_data in cluster_quality_cis.values() for l in ci_data.get('lower', [])]
            all_ci_uppers = [u for ci_data in cluster_quality_cis.values() for u in ci_data.get('upper', [])]
            if all_ci_lowers and all_ci_uppers:
                quality_ci = {'lower': np.mean(all_ci_lowers), 'upper': np.mean(all_ci_uppers)}
        
        # Get significance data, respecting the filter
        score_significance = cluster_freq_data['score_significance'].any() if 'score_significance' in cluster_freq_data.columns else False
        
        quality_significance = False
        if selected_quality_metric:
            # Check significance for the specific metric across all models in this cluster
            all_sigs = []
            for _, row in cluster_freq_data.iterrows():
                sig_dict = row.get('quality_score_statistical_significance', {})
                if isinstance(sig_dict, dict):
                    all_sigs.append(sig_dict.get(selected_quality_metric, False))
            quality_significance = any(all_sigs)
        else:
            # For "All metrics", check if any metric is significant for any model
            quality_significance = cluster_freq_data['quality_significance'].any() if 'quality_significance' in cluster_freq_data.columns else False

        # Create row with simplified structure
        row = {
            'Cluster': cluster_desc,
            'Frequency (%)': f"{avg_frequency:.1f}",
            'Freq CI': f"[{avg_freq_ci_lower:.1f}, {avg_freq_ci_upper:.1f}]" if avg_freq_ci_lower is not None and avg_freq_ci_upper is not None else "N/A",
            'Quality': f"{quality_score:.3f}" if quality_score is not None else "N/A",
            'Quality CI': f"[{quality_ci['lower']:.3f}, {quality_ci['upper']:.3f}]" if quality_ci else "N/A"
        }
        
        # Add significance columns
        row['Score Significance'] = "Yes" if score_significance else "No"
        row['Quality Significance'] = "Yes" if quality_significance else "No"
        
        table_data.append(row)
    
    # Create DataFrame and return
    if table_data:
        table_df = pd.DataFrame(table_data)
        return table_df
    else:
        return pd.DataFrame()


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
    
    # Use the actual column names from the pipeline output (matching Streamlit version)
    if cluster_level == 'fine':
        id_col = 'property_description_fine_cluster_id'
        label_col = 'property_description_fine_cluster_label'
    else:
        id_col = 'property_description_coarse_cluster_id'  
        label_col = 'property_description_coarse_cluster_label'
    
    # Check if required columns exist and provide helpful debug info
    if id_col not in df.columns or label_col not in df.columns:
        available_cols = list(df.columns)
        return f"""
        <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
            <h4>❌ Missing {cluster_level} cluster columns in data</h4>
            <p><strong>Expected:</strong> {id_col}, {label_col}</p>
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
    
    for _, row in cluster_groups.iterrows():
        cluster_id = row[id_col]
        cluster_label = row[label_col]
        cluster_size = row['size']
        property_descriptions = row['property_descriptions']
        models_in_cluster = row['models']
        
        # Create expandable cluster card
        html += f"""
        <details style="margin: 15px 0; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">
            <summary style="
                padding: 15px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; 
                cursor: pointer; 
                font-weight: 600;
                font-size: 16px;
                user-select: none;
                list-style: none;
            ">
                📊 {cluster_label} 
                <span style="
                    background: rgba(255,255,255,0.2); 
                    padding: 4px 8px; 
                    border-radius: 12px; 
                    font-size: 14px; 
                    margin-left: 10px;
                ">
                    {cluster_size} properties
                </span>
            </summary>
            
            <div style="padding: 20px; background: #f8f9fa;">
                <div style="margin-bottom: 15px;">
                    <strong>Cluster ID:</strong> {cluster_id}<br>
                    <strong>Size:</strong> {cluster_size} properties<br>
                    <strong>Models:</strong> {', '.join(models_in_cluster)}
                </div>
                
                <h4 style="color: #333; margin: 15px 0 10px 0;">
                    Property Descriptions ({len(property_descriptions)})
                </h4>
        """
        
        # Add property descriptions
        for i, desc in enumerate(property_descriptions, 1):
            html += f"""
            <div style="
                margin: 8px 0;
                padding: 12px;
                background: white;
                border-left: 4px solid #3182ce;
                border-radius: 4px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            ">
                <span style="
                    display: inline-block;
                    background: #3182ce;
                    color: white;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 12px;
                    margin-right: 8px;
                ">
                    {i}
                </span>
                {desc}
            </div>
            """
        
        html += """
            </div>
        </details>
        """
    
    html += "</div>"
    
    # Add CSS for better interactivity
    html = f"""
    <style>
    details > summary {{
        transition: all 0.3s ease;
    }}
    details > summary:hover {{
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    details[open] > summary {{
        background: linear-gradient(135deg, #4c51bf 0%, #553c9a 100%) !important;
    }}
    details > summary::-webkit-details-marker {{
        display: none;
    }}
    details > summary::marker {{
        display: none;
    }}
    </style>
    {html}
    """
    
    return html


def get_cluster_statistics(clustered_df: pd.DataFrame, 
                         selected_models: Optional[List[str]] = None) -> Dict[str, Any]:
    """Get statistics about clusters for display."""
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
    
    # Fine cluster statistics using correct column names from pipeline
    fine_id_col = 'property_description_fine_cluster_id'
    if fine_id_col in df.columns:
        stats['fine_clusters'] = df[fine_id_col].nunique()
        cluster_sizes = df.groupby(fine_id_col).size()
        stats['min_properties_per_fine_cluster'] = cluster_sizes.min() if not cluster_sizes.empty else 0
        stats['max_properties_per_fine_cluster'] = cluster_sizes.max() if not cluster_sizes.empty else 0
        stats['avg_properties_per_fine_cluster'] = cluster_sizes.mean() if not cluster_sizes.empty else 0
    
    # Coarse cluster statistics using correct column names from pipeline
    coarse_id_col = 'property_description_coarse_cluster_id'
    if coarse_id_col in df.columns:
        stats['coarse_clusters'] = df[coarse_id_col].nunique()
        cluster_sizes = df.groupby(coarse_id_col).size()
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
    
    # Use fine cluster labels instead of property descriptions
    properties = []
    if 'property_description_fine_cluster_label' in clustered_df.columns:
        unique_properties = clustered_df['property_description_fine_cluster_label'].dropna().unique().tolist()
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


def convert_to_openai_format(response_data):
    """Convert various response formats to OpenAI format (Gradio version)"""
    
    if isinstance(response_data, list):
        # Already in OpenAI format
        return response_data
    elif isinstance(response_data, str):
        # Try to parse as Python literal first (handles single quotes)
        try:
            import ast
            parsed = ast.literal_eval(response_data)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        
        # Try to parse as JSON
        try:
            parsed = json.loads(response_data)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Fallback: treat as plain text
        return [{"role": "assistant", "content": response_data}]
    else:
        # Fallback for other types
        return [{"role": "assistant", "content": str(response_data)}]


def display_openai_conversation_html(conversation_data, use_accordion=True):
    """Convert OpenAI conversation format to HTML (Gradio version)
    
    Args:
        conversation_data: List of conversation messages
        use_accordion: If True, group info and system messages in collapsible accordions
    """
    
    if not conversation_data:
        return "<p>No conversation data available</p>"
    
    # Define colors for different roles
    role_colors = {
        "system": "#ff6b6b",      # Red
        "info": "#4ecdc4",        # Teal 
        "assistant": "#45b7d1",   # Blue
        "tool": "#96ceb4",        # Green
        "user": "#feca57"         # Yellow
    }
    
    html = ""
    
    if use_accordion:
        # Group messages by type for accordion
        system_messages = []
        info_messages = []
        other_messages = []
        
        for message in conversation_data:
            if not isinstance(message, dict):
                continue
                
            role = message.get("role", "unknown").lower()
            content = message.get("content", "")

            if type(content) == dict and "text" in content:
                content = content["text"]
            
            if role in ["system", "info"]:
                if role == "system":
                    system_messages.append((role, content))
                else:
                    info_messages.append((role, content))
            else:
                other_messages.append((role, content))
        
        # Create accordion for system messages
        if system_messages:
            html += """
            <details style="margin: 8px 0; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">
                <summary style="
                    padding: 12px 15px; 
                    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
                    color: white; 
                    cursor: pointer; 
                    font-weight: 600;
                    font-size: 14px;
                    user-select: none;
                    list-style: none;
                ">
                    🔧 System Messages ({len(system_messages)})
                </summary>
                <div style="padding: 15px; background: #fff5f5;">
            """
            
            for role, content in system_messages:
                color = role_colors.get(role, "#95a5a6")
                
                # Format content with markdown
                if isinstance(content, dict):
                    content_html = f"<pre style='background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto;'>{json.dumps(content, indent=2)}</pre>"
                elif isinstance(content, str):
                    content_html = markdown.markdown(content, extensions=['nl2br', 'fenced_code'])
                elif content is None:
                    content_html = "<em>(No content)</em>"
                else:
                    content_html = str(content)
                
                html += f"""
                <div style="
                    border-left: 4px solid {color};
                    margin: 8px 0;
                    background-color: #f8f9fa;
                    padding: 12px;
                    border-radius: 0 8px 8px 0;
                ">
                    <div style="
                        font-weight: 600;
                        color: {color};
                        margin-bottom: 8px;
                        text-transform: capitalize;
                        font-size: 14px;
                    ">
                        {role}
                    </div>
                    <div style="
                        color: #333;
                        line-height: 1.6;
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    ">
                        {content_html}
                    </div>
                </div>
                """
            
            html += """
                </div>
            </details>
            """
        
        # Create accordion for info messages
        if info_messages:
            html += """
            <details style="margin: 8px 0; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">
                <summary style="
                    padding: 12px 15px; 
                    background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
                    color: white; 
                    cursor: pointer; 
                    font-weight: 600;
                    font-size: 14px;
                    user-select: none;
                    list-style: none;
                ">
                    ℹ️ Info Messages ({len(info_messages)})
                </summary>
                <div style="padding: 15px; background: #f0fdfa;">
            """
            
            for role, content in info_messages:
                color = role_colors.get(role, "#95a5a6")
                
                # Format content with markdown
                if isinstance(content, dict):
                    content_html = f"<pre style='background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto;'>{json.dumps(content, indent=2)}</pre>"
                elif isinstance(content, str):
                    content_html = markdown.markdown(content, extensions=['nl2br', 'fenced_code'])
                elif content is None:
                    content_html = "<em>(No content)</em>"
                else:
                    content_html = str(content)
                
                html += f"""
                <div style="
                    border-left: 4px solid {color};
                    margin: 8px 0;
                    background-color: #f8f9fa;
                    padding: 12px;
                    border-radius: 0 8px 8px 0;
                ">
                    <div style="
                        font-weight: 600;
                        color: {color};
                        margin-bottom: 8px;
                        text-transform: capitalize;
                        font-size: 14px;
                    ">
                        {role}
                    </div>
                    <div style="
                        color: #333;
                        line-height: 1.6;
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    ">
                        {content_html}
                    </div>
                </div>
                """
            
            html += """
                </div>
            </details>
            """
        
        # Display other messages normally
        for role, content in other_messages:
            color = role_colors.get(role, "#95a5a6")
            
            # Format content with markdown
            if isinstance(content, dict):
                content_html = f"<pre style='background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto;'>{json.dumps(content, indent=2)}</pre>"
            elif isinstance(content, str):
                content_html = markdown.markdown(content, extensions=['nl2br', 'fenced_code'])
            elif content is None:
                content_html = "<em>(No content)</em>"
            else:
                content_html = str(content)
            
            html += f"""
            <div style="
                border-left: 4px solid {color};
                margin: 8px 0;
                background-color: #f8f9fa;
                padding: 12px;
                border-radius: 0 8px 8px 0;
            ">
                <div style="
                    font-weight: 600;
                    color: {color};
                    margin-bottom: 8px;
                    text-transform: capitalize;
                    font-size: 14px;
                ">
                    {role}
                </div>
                <div style="
                    color: #333;
                    line-height: 1.6;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                ">
                    {content_html}
                </div>
            </div>
            """
    
    else:
        # Original behavior - display all messages normally
        for message in conversation_data:
            if not isinstance(message, dict):
                continue
                
            role = message.get("role", "unknown")
            content = message.get("content", "")

            if type(content) == dict and "text" in content:
                content = content["text"]
            
            # Get color for this role, default to gray
            color = role_colors.get(role.lower(), "#95a5a6")
            
            # Format content with markdown
            if isinstance(content, dict):
                content_html = f"<pre style='background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto;'>{json.dumps(content, indent=2)}</pre>"
            elif isinstance(content, str):
                # Convert markdown to HTML properly
                content_html = markdown.markdown(content, extensions=['nl2br', 'fenced_code'])
            elif content is None:
                content_html = "<em>(No content)</em>"
            else:
                content_html = str(content)
            
            # Create message HTML
            html += f"""
            <div style="
                border-left: 4px solid {color};
                margin: 8px 0;
                background-color: #f8f9fa;
                padding: 12px;
                border-radius: 0 8px 8px 0;
            ">
                <div style="
                    font-weight: 600;
                    color: {color};
                    margin-bottom: 8px;
                    text-transform: capitalize;
                    font-size: 14px;
                ">
                    {role}
                </div>
                <div style="
                    color: #333;
                    line-height: 1.6;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                ">
                    {content_html}
                </div>
            </div>
            """
    
    # Add CSS for accordion styling
    if use_accordion:
        html = f"""
        <style>
        details > summary {{
            transition: all 0.3s ease;
        }}
        details > summary:hover {{
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        details > summary::-webkit-details-marker {{
            display: none;
        }}
        details > summary::marker {{
            display: none;
        }}
        </style>
        {html}
        """
    
    return html


def get_example_data(clustered_df: pd.DataFrame, 
                    selected_prompt: str = None,
                    selected_model: str = None, 
                    selected_property: str = None,
                    max_examples: int = 5) -> List[Dict[str, Any]]:
    """Get example data based on filter criteria."""
    if clustered_df.empty:
        return []
    
    df = clustered_df.copy()
    
    # Apply filters
    if selected_model and selected_model != "All Models":
        df = df[df['model'] == selected_model]
    
    if selected_prompt and selected_prompt != "All Prompts":
        # Handle truncated prompt selection by matching the beginning of the text
        if selected_prompt.endswith("..."):
            prompt_prefix = selected_prompt[:-3]  # Remove the "..."
            # Check different possible prompt column names
            if 'prompt' in df.columns:
                df = df[df['prompt'].str.startswith(prompt_prefix, na=False)]
            elif 'question' in df.columns:
                df = df[df['question'].str.startswith(prompt_prefix, na=False)]
            elif 'input' in df.columns:
                df = df[df['input'].str.startswith(prompt_prefix, na=False)]
            elif 'user_prompt' in df.columns:
                df = df[df['user_prompt'].str.startswith(prompt_prefix, na=False)]
        else:
            # Exact match for non-truncated prompts
            if 'prompt' in df.columns:
                df = df[df['prompt'] == selected_prompt]
            elif 'question' in df.columns:
                df = df[df['question'] == selected_prompt]
            elif 'input' in df.columns:
                df = df[df['input'] == selected_prompt]
            elif 'user_prompt' in df.columns:
                df = df[df['user_prompt'] == selected_prompt]
    
    if selected_property and selected_property != "All Clusters":
        # Handle truncated property selection - now using fine cluster labels
        if selected_property.endswith("..."):
            property_prefix = selected_property[:-3]  # Remove the "..."
            if 'property_description_fine_cluster_label' in df.columns:
                df = df[df['property_description_fine_cluster_label'].str.startswith(property_prefix, na=False)]
            else:
                # Fallback to property descriptions
                df = df[df['property_description'].str.startswith(property_prefix, na=False)]
        else:
            # Exact match for non-truncated properties
            if 'property_description_fine_cluster_label' in df.columns:
                df = df[df['property_description_fine_cluster_label'] == selected_property]
            else:
                # Fallback to property descriptions
                df = df[df['property_description'] == selected_property]
    
    # Limit results
    df = df.head(max_examples)
    
    # Convert to list of dictionaries for easy display
    examples = []
    for _, row in df.iterrows():
        # Get prompt from various possible columns
        prompt = (row.get('prompt') or 
                 row.get('question') or 
                 row.get('input') or 
                 row.get('user_prompt') or 
                 'N/A')
        
        # Get response from various possible columns (matching Streamlit version)
        response = (row.get('model_response') or 
                   row.get('model_a_response') or 
                   row.get('model_b_response') or 
                   row.get('responses') or
                   row.get('response') or 
                   row.get('output') or 
                   'N/A')
        
        example = {
            'id': row.get('id', 'N/A'),
            'model': row.get('model', 'N/A'),
            'prompt': prompt,
            'response': response,
            'property_description': row.get('property_description', 'N/A'),
            'score': row.get('score', 'N/A'),
            'fine_cluster_id': row.get('property_description_fine_cluster_id', 'N/A'),
            'fine_cluster_label': row.get('property_description_fine_cluster_label', 'N/A'),
            'coarse_cluster_id': row.get('property_description_coarse_cluster_id', 'N/A'),
            'coarse_cluster_label': row.get('property_description_coarse_cluster_label', 'N/A'),
        }
        examples.append(example)
    
    return examples


def format_examples_display(examples: List[Dict[str, Any]], 
                          selected_prompt: str = None,
                          selected_model: str = None,
                          selected_property: str = None,
                          use_accordion: bool = True) -> str:
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
            conversation_html = display_openai_conversation_html(openai_conversation, use_accordion=use_accordion)
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
                </div>
                
                <div style="margin-bottom: 15px;">
                    <div style="border-radius: 6px; font-size: 13px; line-height: 1.5;">
                        {conversation_html}
                    </div>
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
            </div>
        </details>
        """
    
    html += "</div>"
    return html 
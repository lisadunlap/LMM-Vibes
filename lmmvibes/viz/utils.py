"""
Utility functions for pipeline results app.

This module contains common utility functions used across different tabs.
"""

import numpy as np
from typing import Dict, List, Any


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
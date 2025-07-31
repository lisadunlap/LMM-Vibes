"""
Data loading utilities for Gradio pipeline results app.

This module contains functions for loading pipeline results data without Streamlit dependencies.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache


class DataCache:
    """Simple caching mechanism for data loading."""
    _cache = {}
    
    @classmethod
    def get(cls, key: str):
        return cls._cache.get(key)
    
    @classmethod
    def set(cls, key: str, value):
        cls._cache[key] = value
    
    @classmethod
    def clear(cls):
        cls._cache.clear()


def load_pipeline_results(results_dir: str) -> Tuple[pd.DataFrame, Dict[str, Any], Path]:
    """Load pipeline outputs optimized for large datasets"""
    cache_key = f"pipeline_results_{results_dir}"
    cached = DataCache.get(cache_key)
    if cached:
        return cached
    
    results_path = Path(results_dir)
    
    # Load model statistics
    model_stats_path = results_path / "model_stats.json"
    if not model_stats_path.exists():
        raise FileNotFoundError(f"model_stats.json not found in {results_dir}")
        
    with open(model_stats_path) as f:
        model_stats = json.load(f)
    
    # Load clustered results
    clustered_path = results_path / "clustered_results.jsonl"
    
    if not clustered_path.exists():
        raise FileNotFoundError(f"clustered_results.jsonl not found in {results_dir}")
    
    # Load essential columns for performance
    try:
        clustered_df = pd.read_json(clustered_path, lines=True)
    except Exception as e:
        raise ValueError(f"Could not load clustered results: {e}")
    
    result = (clustered_df, model_stats, results_path)
    DataCache.set(cache_key, result)
    return result


def load_property_examples(results_path: Path, property_ids: List[str]) -> pd.DataFrame:
    """Load specific property examples on-demand"""
    if not property_ids:
        return pd.DataFrame()
    
    cache_key = f"examples_{results_path}_{hash(tuple(sorted(property_ids)))}"
    cached = DataCache.get(cache_key)
    if cached is not None:
        return cached
        
    # Load full dataset to get prompt/response details
    clustered_path = results_path / "clustered_results.jsonl"
    
    if not clustered_path.exists():
        raise FileNotFoundError("Could not load example data - clustered_results.jsonl not found")
    
    try:
        full_df = pd.read_json(clustered_path, lines=True)
        result = full_df[full_df['id'].isin(property_ids)]
        DataCache.set(cache_key, result)
        return result
    except Exception as e:
        raise ValueError(f"Failed to load examples: {e}")


def scan_for_result_subfolders(base_dir: str) -> List[str]:
    """Scan a directory for subfolders containing pipeline results"""
    base_path = Path(base_dir)
    if not base_path.exists() or not base_path.is_dir():
        return []
    
    valid_subfolders = []
    
    # Check if the base directory itself contains results
    if (base_path / "model_stats.json").exists() or (base_path / "clustered_results.jsonl").exists():
        valid_subfolders.append(".")  # Current directory
    
    # Check subdirectories
    try:
        for item in base_path.iterdir():
            if item.is_dir():
                # Check if this subdirectory contains pipeline results
                has_model_stats = (item / "model_stats.json").exists()
                has_clustered_results = (item / "clustered_results.jsonl").exists()
                
                if has_model_stats or has_clustered_results:
                    valid_subfolders.append(item.name)
    except PermissionError:
        pass  # Skip directories we can't read
    
    return sorted(valid_subfolders)


def get_available_models(model_stats: Dict[str, Any]) -> List[str]:
    """Get list of available models from model stats."""
    return list(model_stats.keys())


def validate_results_directory(results_dir: str) -> Tuple[bool, str]:
    """Validate if a directory contains valid pipeline results.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not results_dir:
        return False, "Please provide a results directory"
    
    results_path = Path(results_dir)
    if not results_path.exists():
        return False, f"Directory does not exist: {results_dir}"
    
    if not results_path.is_dir():
        return False, f"Path is not a directory: {results_dir}"
    
    # Check for required files
    model_stats_path = results_path / "model_stats.json"
    clustered_path = results_path / "clustered_results.jsonl"
    
    if not model_stats_path.exists() and not clustered_path.exists():
        # Check for subfolders
        subfolders = scan_for_result_subfolders(results_dir)
        if not subfolders:
            return False, f"No pipeline results found in {results_dir}. Expected model_stats.json and clustered_results.jsonl"
    
    return True, "" 
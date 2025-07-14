"""HELM BigCodeBench dataset for LMM-Vibes.

This module provides a dataset loader for the HELM BigCodeBench results.
"""

from __future__ import annotations

from typing import Tuple, Callable
import pandas as pd
import json
from pathlib import Path


def load_single_model_data(args) -> Tuple[pd.DataFrame, Callable, str]:
    """Load the HELM BigCodeBench dataset."""
    print("Loading HELM BigCodeBench dataset...")
    
    # Path to the HELM dataset
    helm_path = Path(__file__).parent.parent.parent / "data" / "helm" / "helm_bigcodebench_results.jsonl"
    
    if not helm_path.exists():
        raise FileNotFoundError(f"HELM dataset not found at {helm_path}")
    
    # Read the JSONL file
    data = []
    with open(helm_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add required columns for compatibility with pipeline
    df['model_response'] = df['result']
    df['id'] = [f'helm_example_{i:06d}' for i in range(len(df))]
    df['question_id'] = [f'helm_question_{i:06d}' for i in range(len(df))]
    
    print(f"Loaded {len(df)} examples from HELM BigCodeBench dataset")
    print(f"Model: {df['model'].iloc[0]}")
    
    def _extract_content_helm(conversation):
        """Extract content for HELM data."""
        # For HELM data, we just return the prompt and response
        return conversation['prompt'], conversation['result']
    
    return df, _extract_content_helm, "helm_system_prompt" 
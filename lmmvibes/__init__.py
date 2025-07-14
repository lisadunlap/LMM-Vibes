"""
LMM-Vibes: Language Model Model Vibes Analysis

A toolkit for analyzing and understanding model behavior patterns through
property extraction, clustering, and metrics computation.
"""

from .public import explain, explain_side_by_side, explain_single_model, explain_with_custom_pipeline, compute_metrics_only
from .prompts import get_default_system_prompt


__version__ = "0.1.0"
__all__ = [
    "explain",
    "explain_side_by_side", 
    "explain_single_model",
    "explain_with_custom_pipeline",
    "compute_metrics_only",
    "get_default_system_prompt"
] 
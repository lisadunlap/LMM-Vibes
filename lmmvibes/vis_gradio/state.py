"""
Shared application state for the LMM-Vibes Gradio viewer.

This module centralises mutable globals so they can be imported from any other
sub-module without circular-import problems.
"""
from typing import Any, Dict, Optional

# Global runtime state – mutable and shared across all tabs
app_state: Dict[str, Any] = {
    "clustered_df": None,
    "model_stats": None,
    "results_path": None,
    "available_models": [],
    "current_results_dir": None,
}

# Base directory that contains experiment result folders. Can be changed at
# runtime via launch_app(results_dir=…).  A value of None means "not set".
BASE_RESULTS_DIR: Optional[str] = "results" 
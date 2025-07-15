"""
LMM-Vibes: Language Model Model Vibes Analysis

A toolkit for analyzing and understanding model behavior patterns through
property extraction, clustering, and metrics computation.
"""

from .public import explain, explain_side_by_side, explain_single_model, explain_with_custom_pipeline, compute_metrics_only
from .prompts import (
    get_default_system_prompt,
    # Standard model comparison prompts
    sbs_w_metrics_system_prompt,
    one_sided_system_prompt_no_examples,
    search_enabled_system_prompt_no_examples,
    # Web development prompts
    webdev_system_prompt,
    webdev_system_prompt_no_examples,
    webdev_single_model_system_prompt,
    # Single model prompts
    single_model_no_score_system_prompt,
    single_model_system_prompt,
    # Agent-specific prompts for agentic environments
    taubench_system_prompt,
    taubench_comparison_system_prompt,
    agentic_swe_system_prompt,
    agentic_tool_focused_prompt,
    agentic_reasoning_focused_prompt,
    agentic_reward_hacking_focused_prompt
)


__version__ = "0.1.0"
__all__ = [
    "explain",
    "explain_side_by_side", 
    "explain_single_model",
    "explain_with_custom_pipeline",
    "compute_metrics_only",
    "get_default_system_prompt",
    # Standard model comparison prompts
    "sbs_w_metrics_system_prompt",
    "one_sided_system_prompt_no_examples",
    "search_enabled_system_prompt_no_examples",
    # Web development prompts
    "webdev_system_prompt",
    "webdev_system_prompt_no_examples",
    "webdev_single_model_system_prompt",
    # Single model prompts
    "single_model_no_score_system_prompt", 
    "single_model_system_prompt",
    # Agent prompts
    "taubench_system_prompt",
    "taubench_comparison_system_prompt",
    "agentic_swe_system_prompt", 
    "agentic_tool_focused_prompt",
    "agentic_reasoning_focused_prompt",
    "agentic_reward_hacking_focused_prompt"
] 
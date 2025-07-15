"""
Prompts module for LMM-Vibes.

This module contains system prompts and prompt utilities for property extraction.
"""

from .extractor_prompts import (
    sbs_w_metrics_system_prompt,
    one_sided_system_prompt_no_examples,
    search_enabled_system_prompt_no_examples,
    webdev_system_prompt,
    webdev_system_prompt_no_examples,
    webdev_single_model_system_prompt,
    single_model_no_score_system_prompt,
    single_model_system_prompt,
)

# Import agent-specific prompts for agentic environments
from .agents import (
    taubench_comparison_system_prompt,
    taubench_system_prompt,
    agentic_swe_system_prompt,
    agentic_tool_focused_prompt,
    agentic_reasoning_focused_prompt,
    agentic_reward_hacking_focused_prompt
)


def get_default_system_prompt(method: str, contains_score: bool = True) -> str:
    """
    Get the default system prompt based on the method and whether the data contains scores.
    
    Args:
        method: The analysis method ("side_by_side" or "single_model")
        contains_score: Whether the data contains score/preference information
        
    Returns:
        The appropriate default system prompt
        
    Raises:
        ValueError: If method is not recognized
    """
    if method == "side_by_side":
        if contains_score:
            return sbs_w_metrics_system_prompt
        else:
            return one_sided_system_prompt_no_examples
    elif method == "single_model":
        if contains_score:
            return single_model_system_prompt
        else:
            return single_model_no_score_system_prompt
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods: 'side_by_side', 'single_model'")


__all__ = [
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
    # Agent-specific prompts for agentic environments
    "taubench_comparison_system_prompt",
    "taubench_system_prompt",
    "agentic_swe_system_prompt",
    "agentic_tool_focused_prompt",
    "agentic_reasoning_focused_prompt"
] 
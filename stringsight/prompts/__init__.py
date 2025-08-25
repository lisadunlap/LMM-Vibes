"""
Prompts module for LMM-Vibes.

This module contains system prompts and prompt utilities for property extraction.
"""

from .extractor_prompts import (
    sbs_system_prompt,
    single_model_system_prompt,
    single_model_system_prompt_custom,
)

# Import agent-specific prompts for agentic environments
from .agents import (
    agent_system_prompt,
    taubench_comparison_system_prompt,
    agentic_swe_system_prompt,
    agentic_tool_focused_prompt,
    agentic_reasoning_focused_prompt,
    agentic_reward_hacking_focused_prompt
)


# Import fixed-axis prompts
from .fixed_axes import (
    fixed_axis_prompt,
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
            return sbs_system_prompt
        else:
            return sbs_system_prompt
    elif method == "single_model":
        if contains_score:
            return single_model_system_prompt
        else:
            return single_model_system_prompt
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods: 'side_by_side', 'single_model'")


__all__ = [
    "get_default_system_prompt",
    # Extraction prompts
    "sbs_system_prompt",
    "single_model_system_prompt",
    "single_model_system_prompt_custom",
    # Agent-specific prompts for agentic environments
    "agent_system_prompt",
    "taubench_comparison_system_prompt",
    "agentic_swe_system_prompt",
    "agentic_tool_focused_prompt",
    "agentic_reasoning_focused_prompt",
    "agentic_reward_hacking_focused_prompt",
    # Fixed-axis prompts
    "fixed_axis_prompt",

] 
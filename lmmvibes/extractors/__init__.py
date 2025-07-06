"""
Property extraction stages for LMM-Vibes.

This module contains stages that extract behavioral properties from model responses.
"""

from typing import Callable, Optional
from ..core.stage import PipelineStage


def get_extractor(
    model_name: str = "gpt-4o-mini",
    system_prompt: str = "one_sided_system_prompt",
    prompt_builder: Optional[Callable] = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 16000,
    max_workers: int = 16,
    **kwargs
) -> PipelineStage:
    """
    Factory function to get the appropriate extractor based on model name.
    
    Args:
        model_name: Name of the LLM to use for extraction
        system_prompt: System prompt for property extraction
        prompt_builder: Optional custom prompt builder function
        temperature: Temperature for LLM
        top_p: Top-p for LLM  
        max_tokens: Max tokens for LLM
        max_workers: Max parallel workers for API calls
        **kwargs: Additional configuration
        
    Returns:
        Configured extractor stage
    """
    
    if model_name.lower().startswith("gpt"):
        from .openai import OpenAIExtractor
        return OpenAIExtractor(
            model=model_name,
            system_prompt=system_prompt,
            prompt_builder=prompt_builder,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_workers=max_workers,
            **kwargs
        )
    else:
        from .vllm import VLLMExtractor
        return VLLMExtractor(
            model=model_name,
            system_prompt=system_prompt,
            prompt_builder=prompt_builder,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )


# Re-export key classes
from .openai import OpenAIExtractor
from .batch import BatchExtractor

__all__ = [
    "get_extractor",
    "OpenAIExtractor", 
    "BatchExtractor"
] 
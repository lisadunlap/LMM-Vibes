"""
Metrics computation stages for LMM-Vibes.

This module contains stages that compute statistics and rankings for model behavior patterns.
"""

from ..core.stage import PipelineStage


def get_metrics(
    method: str = "side_by_side",
    **kwargs
) -> PipelineStage:
    """
    Factory function to get the appropriate metrics stage.
    
    Args:
        method: Method type ("side_by_side" or "single_model")
        **kwargs: Additional configuration
        
    Returns:
        Configured metrics stage
    """
    
    if method == "side_by_side":
        from .side_by_side import SideBySideMetrics
        return SideBySideMetrics(**kwargs)
    elif method == "single_model":
        from .single_model import SingleModelMetrics
        return SingleModelMetrics(**kwargs)
    else:
        raise ValueError(f"Unknown metrics method: {method}")


__all__ = [
    "get_metrics"
] 
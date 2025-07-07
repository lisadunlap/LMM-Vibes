"""Metrics computation modules.

Currently available:

* :pyclass:`lmmvibes.metrics.side_by_side.SideBySideMetrics` – metrics for the
  Arena‐style side-by-side dataset where each question is answered by multiple
  models.
"""

from importlib import import_module as _imp
from typing import Dict, Any

__all__: list[str] = [
    "SideBySideMetrics",
    "SingleModelMetrics",
    "get_metrics",
]

# Lazy import to keep import time low
SideBySideMetrics = _imp("lmmvibes.metrics.side_by_side").SideBySideMetrics
SingleModelMetrics = _imp("lmmvibes.metrics.single_model").SingleModelMetrics


def get_metrics(method: str, **kwargs) -> "PipelineStage":
    """
    Factory function for metrics stages.
    
    Args:
        method: "side_by_side" or "single_model"
        **kwargs: Additional configuration for the metrics stage
        
    Returns:
        Configured metrics stage
    """
    if method == "side_by_side":
        return SideBySideMetrics(**kwargs)
    elif method == "single_model":
        return SingleModelMetrics(**kwargs)
    else:
        raise ValueError(f"Unknown metrics method: {method}") 
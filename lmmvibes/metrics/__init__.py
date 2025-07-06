"""Metrics computation modules.

Currently available:

* :pyclass:`lmmvibes.metrics.side_by_side.SideBySideMetrics` – metrics for the
  Arena‐style side-by-side dataset where each question is answered by multiple
  models.
"""

from importlib import import_module as _imp

__all__: list[str] = [
    "SideBySideMetrics",
]

# Lazy import to keep import time low
SideBySideMetrics = _imp("lmmvibes.metrics.side_by_side").SideBySideMetrics 
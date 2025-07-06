"""
LMM-Vibes: Explain Large Language Model Behavior Patterns

A pipeline for extracting, clustering, and analyzing behavioral properties
of large language models from conversation data.
"""

from .public import explain
from .core.data_objects import PropertyDataset, ConversationRecord, Property

__version__ = "0.1.0"
__all__ = ["explain", "PropertyDataset", "ConversationRecord", "Property"] 
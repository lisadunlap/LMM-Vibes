"""
Clustering stages for LMM-Vibes.

This module contains stages that cluster properties into coherent groups.
"""

from typing import Union
from ..core.stage import PipelineStage


def get_clusterer(
    method: str = "hdbscan",
    min_cluster_size: int = 30,
    embedding_model: str = "openai",
    hierarchical: bool = False,
    assign_outliers: bool = False,
    include_embeddings: bool = True,
    **kwargs
) -> PipelineStage:
    """
    Factory function to get the appropriate clusterer.
    
    Args:
        method: Clustering method ("hdbscan", "hdbscan_native", "hierarchical")
        min_cluster_size: Minimum cluster size
        embedding_model: Embedding model to use
        hierarchical: Whether to create hierarchical clusters
        assign_outliers: Whether to assign outliers to nearest clusters
        include_embeddings: Whether to include embeddings in output
        **kwargs: Additional configuration
        
    Returns:
        Configured clusterer stage
    """
    
    if method == "hdbscan":
        from .hdbscan import HDBSCANClusterer
        return HDBSCANClusterer(
            min_cluster_size=min_cluster_size,
            embedding_model=embedding_model,
            hierarchical=hierarchical,
            assign_outliers=assign_outliers,
            include_embeddings=include_embeddings,
            **kwargs
        )
    elif method == "hdbscan_native":
        from .hdbscan import HDBSCANNativeClusterer
        return HDBSCANNativeClusterer(
            min_cluster_size=min_cluster_size,
            embedding_model=embedding_model,
            include_embeddings=include_embeddings,
            **kwargs
        )
    elif method == "hierarchical":
        from .hierarchical import HierarchicalClusterer
        return HierarchicalClusterer(
            min_cluster_size=min_cluster_size,
            embedding_model=embedding_model,
            include_embeddings=include_embeddings,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown clustering method: {method}")


__all__ = [
    "get_clusterer"
] 
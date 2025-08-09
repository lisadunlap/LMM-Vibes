"""
HDBSCAN-based clustering stages.

This module migrates the clustering logic from clustering/hierarchical_clustering.py
into pipeline stages.
"""

from typing import List
import pandas as pd

from .base import BaseClusterer
from ..core.data_objects import PropertyDataset, Cluster
from ..core.mixins import LoggingMixin, TimingMixin, WandbMixin


class HDBSCANClusterer(BaseClusterer):
    """
    HDBSCAN clustering stage.
    
    This stage migrates the hdbscan_cluster_categories function from
    clustering/hierarchical_clustering.py into the pipeline architecture.
    """
    
    def __init__(
        self,
        min_cluster_size: int = 30,
        embedding_model: str = "openai", 
        hierarchical: bool = False,
        assign_outliers: bool = False,
        include_embeddings: bool = True,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        max_coarse_clusters: int = 25,
        output_dir: str | None = None,
        **kwargs
    ):
        """Initialize the HDBSCAN clusterer.
        
        Args:
            min_cluster_size: Minimum cluster size
            embedding_model: Embedding model to use
            hierarchical: Whether to create hierarchical clusters
            assign_outliers: Whether to assign outliers
            include_embeddings: Whether to include embeddings in output
            use_wandb: Whether to use wandb for logging
            wandb_project: wandb project name
            output_dir: Directory to save clustering results (optional)
            **kwargs: Additional configuration
        """
        super().__init__(
            output_dir=output_dir,
            include_embeddings=include_embeddings,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            hierarchical=hierarchical,
            **kwargs,
        )
        self.min_cluster_size = min_cluster_size
        self.embedding_model = embedding_model
        self.assign_outliers = assign_outliers
        self.max_coarse_clusters = max_coarse_clusters
        # Store config for save_clustered_results
        self.config = type('Config', (), {
            'min_cluster_size': min_cluster_size,
            'embedding_model': embedding_model,
            'hierarchical': hierarchical,
            'assign_outliers': assign_outliers,
            'use_wandb': use_wandb,
            'wandb_project': wandb_project,
            'max_coarse_clusters': max_coarse_clusters,
            'disable_dim_reduction': False,
            'min_samples': min(min_cluster_size, max(5, min_cluster_size // 2)),
            'cluster_selection_epsilon': 0.0,
        })()

    def cluster(self, data: PropertyDataset, column_name: str) -> pd.DataFrame:
        """Run HDBSCAN clustering and return a standardized DataFrame."""
        try:
            from lmmvibes.clusterers.hierarchical_clustering import (
                hdbscan_cluster_categories,
                ClusterConfig,
            )
        except ImportError:
            from .hierarchical_clustering import (  # type: ignore
                hdbscan_cluster_categories,
                ClusterConfig,
            )

        cfg = ClusterConfig(
            min_cluster_size=self.min_cluster_size,
            embedding_model=self.embedding_model,
            hierarchical=self.hierarchical,
            assign_outliers=self.assign_outliers,
            include_embeddings=self.include_embeddings,
            verbose=False,
            max_coarse_clusters=self.max_coarse_clusters,
        )
        # Ensure base uses the same config for saving/logging
        self.config = cfg

        clustered_df = hdbscan_cluster_categories(
            data.to_dataframe(type="properties"),
            column_name=column_name,
            config=cfg,
        )
        return clustered_df

    def postprocess_clustered_df(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """Assign clusters smaller than min_cluster_size to Outliers."""
        cfg = self.get_config()
        fine_label_col = f"{column_name}_fine_cluster_label"
        fine_id_col = f"{column_name}_fine_cluster_id"
        label_counts = df[fine_label_col].value_counts()
        too_small_labels = label_counts[label_counts < int(getattr(cfg, "min_cluster_size", 1))].index
        for label in too_small_labels:
            mask = df[fine_label_col] == label
            cid = df.loc[mask, fine_id_col].iloc[0] if not df.loc[mask].empty else None
            self.log(f"Assigning cluster {cid} (label '{label}') to Outliers because it has {label_counts[label]} items")
            df.loc[mask, fine_label_col] = "Outliers"
            df.loc[mask, fine_id_col] = -1
        return df
        
        
class LLMOnlyClusterer(HDBSCANClusterer):
    """
    HDBSCAN clustering stage.
    
    This stage migrates the hdbscan_cluster_categories function from
    clustering/hierarchical_clustering.py into the pipeline architecture.
    """
    
    def run(self, data: PropertyDataset, column_name: str = "property_description") -> PropertyDataset:
        """Cluster properties using HDBSCAN (delegates to base)."""
        return super().run(data, column_name)
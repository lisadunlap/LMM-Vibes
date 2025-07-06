"""
HDBSCAN-based clustering stages.

This module migrates the clustering logic from clustering/hierarchical_clustering.py
into pipeline stages.
"""

from typing import List
from ..core.stage import PipelineStage
from ..core.data_objects import PropertyDataset, Cluster
from ..core.mixins import LoggingMixin, TimingMixin, WandbMixin


class HDBSCANClusterer(PipelineStage, LoggingMixin, TimingMixin, WandbMixin):
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
        wandb_project: str = None,
        **kwargs
    ):
        """
        Initialize the HDBSCAN clusterer.
        
        Args:
            min_cluster_size: Minimum cluster size
            embedding_model: Embedding model to use
            hierarchical: Whether to create hierarchical clusters
            assign_outliers: Whether to assign outliers
            include_embeddings: Whether to include embeddings in output
            use_wandb: Whether to use wandb for logging
            wandb_project: wandb project name
            **kwargs: Additional configuration
        """
        super().__init__(use_wandb=use_wandb, wandb_project=wandb_project, **kwargs)
        self.min_cluster_size = min_cluster_size
        self.embedding_model = embedding_model
        self.hierarchical = hierarchical
        self.assign_outliers = assign_outliers
        self.include_embeddings = include_embeddings
                
    def run(self, data: PropertyDataset, column_name: str = "property_description") -> PropertyDataset:
        """
        Cluster properties using HDBSCAN.
        
        Args:
            data: PropertyDataset with properties to cluster
            
        Returns:
            PropertyDataset with populated clusters
        """
        self.log(f"Clustering {len(data.properties)} properties using HDBSCAN")
        
        # ------------------------------------------------------------------
        # Build a DataFrame of property descriptions
        # ------------------------------------------------------------------
        import pandas as pd
        descriptions = [p.property_description for p in data.properties if p.property_description]
        if not descriptions:
            self.log("No property descriptions to cluster – skipping stage")
            return data

        df_props = pd.DataFrame({
            column_name: descriptions
        })

        # ------------------------------------------------------------------
        # Run clustering using the original helper
        # ------------------------------------------------------------------
        try:
            from lmmvibes.clusterers.hierarchical_clustering import hdbscan_cluster_categories, ClusterConfig
        except ImportError:
            # Fallback to relative import if package layout differs when installed
            from .hierarchical_clustering import hdbscan_cluster_categories, ClusterConfig  # type: ignore

        cfg = ClusterConfig(
            min_cluster_size=self.min_cluster_size,
            embedding_model=self.embedding_model,
            hierarchical=self.hierarchical,
            assign_outliers=self.assign_outliers,
            include_embeddings=self.include_embeddings,
            verbose=False,
        )

        clustered_df = hdbscan_cluster_categories(
            data.to_dataframe(type="properties"),
            column_name=column_name,
            config=cfg
        )
        print("clustered_df ", clustered_df.columns)

        # ------------------------------------------------------------------
        # Convert clustering result into a simple summary dict
        # ------------------------------------------------------------------
        fine_label_col = f'{column_name}_fine_cluster_label'
        fine_id_col    = f'{column_name}_fine_cluster_id'
        coarse_label_col = f'{column_name}_coarse_cluster_label'
        coarse_id_col    = f'{column_name}_coarse_cluster_id'

        clusters: List[Cluster] = []

        for cid, group in clustered_df.groupby(fine_id_col):
            label = group[fine_label_col].iloc[0]
            clusters.append(Cluster(id=int(cid), label=label, size=len(group), property_descriptions=group[column_name].tolist(), question_ids=group["question_id"].tolist()))

        # assign coarse cluster labels to existing fine clusters
        if self.hierarchical and coarse_id_col in clustered_df.columns:
            for c in clusters:
                coarse_clusters = clustered_df[clustered_df[fine_id_col] == c.id][coarse_id_col].tolist()
                assert len(coarse_clusters) == 1, f"Expected 1 coarse cluster for fine cluster {c.id}, got {len(coarse_clusters)}"
                c.parent_id = coarse_clusters[0]
                c.parent_label = clustered_df[clustered_df[coarse_id_col] == coarse_clusters[0]][coarse_label_col].iloc[0]

        self.log(f"Created {len(clusters)} fine clusters")
        if self.hierarchical:
            self.log(f"Created {len(clusters)} coarse clusters")

        # Attach cluster id/label back onto Property objects (optional: first occurrence)
        desc_to_fine_id = dict(zip(clustered_df[column_name], clustered_df[fine_id_col]))
        desc_to_fine_label = dict(zip(clustered_df[column_name], clustered_df[fine_label_col]))
        for p in data.properties:
            if p.property_description in desc_to_fine_id:
                setattr(p, 'fine_cluster_id', int(desc_to_fine_id[p.property_description]))
                setattr(p, 'fine_cluster_label', desc_to_fine_label[p.property_description])

        # --- Wandb logging ---
        if self.use_wandb:
            self.init_wandb(project=self.wandb_project)
            try:
                import wandb
                log_df = pd.DataFrame([c.to_dict() for c in clusters]).astype(str)
                self.log_wandb({
                    "hdbscan_clustered_table": wandb.Table(dataframe=log_df)
                })
                import json
                self.log_wandb({
                    "hdbscan_clusters_json": wandb.Html(f'<pre>{json.dumps(clusters, indent=2)}</pre>')
                })
            except Exception as e:
                self.log(f"Failed to log to wandb: {e}", level="warning")
        # --- End wandb logging ---

        return PropertyDataset(
            conversations=data.conversations,
            properties=data.properties,
            clusters=clusters,
            model_stats=data.model_stats
        )

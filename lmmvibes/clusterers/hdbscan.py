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
        max_coarse_clusters: int = 25,
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
        self.max_coarse_clusters = max_coarse_clusters
                
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

        # remove bad properties
        valid_properties = data.get_valid_properties()

        descriptions = [p.property_description for p in valid_properties]
        if not descriptions:
            self.log("No property descriptions to cluster – skipping stage")
            return data

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
            max_coarse_clusters=self.max_coarse_clusters
        )

        clustered_df = hdbscan_cluster_categories(
            data.to_dataframe(type="properties"),
            column_name=column_name,
            config=cfg
        )

        # ------------------------------------------------------------------
        # Convert clustering result into a simple summary dict
        # ------------------------------------------------------------------
        fine_label_col = f'{column_name}_fine_cluster_label'
        fine_id_col    = f'{column_name}_fine_cluster_id'
        coarse_label_col = f'{column_name}_coarse_cluster_label'
        coarse_id_col    = f'{column_name}_coarse_cluster_id'

        clusters: List[Cluster] = []

        clustered_df.to_csv("clustered_df.csv")

        # ------------------------------------------------------------------
        # Convert clustering result into a simple summary dict
        # ------------------------------------------------------------------
        if self.hierarchical:
            for cid, group in clustered_df.groupby(fine_id_col):
                cid_group = group[group[fine_id_col] == cid]
                label = cid_group[fine_label_col].iloc[0]
                coarse_label = cid_group[coarse_label_col].unique().tolist()
                assert len(coarse_label) == 1, f"Expected exactly one coarse label for fine cluster {cid}, but got {coarse_label}"
                coarse_id = cid_group[coarse_id_col].iloc[0]
                clusters.append(Cluster(id=int(cid), label=label, size=len(cid_group), property_descriptions=cid_group[column_name].tolist(), question_ids=cid_group["question_id"].tolist(), parent_id=int(coarse_id), parent_label=coarse_label[0]))
        else:
            for cid, group in clustered_df.groupby(fine_id_col):
                cid_group = group[group[fine_id_col] == cid]
                label = cid_group[fine_label_col].iloc[0]
                clusters.append(Cluster(id=int(cid), label=label, size=len(cid_group), property_descriptions=cid_group[column_name].tolist(), question_ids=cid_group["question_id"].tolist()))

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
                    "hdbscan_clusters_json": wandb.Html(f'<pre>{json.dumps([c.to_dict() for c in clusters], indent=2)}</pre>')
                })
            except Exception as e:
                self.log(f"Failed to log to wandb: {e}", level="warning")
        # --- End wandb logging ---

        return PropertyDataset(
            conversations=data.conversations,
            all_models=data.all_models,
            properties=data.properties,
            clusters=clusters,
            model_stats=data.model_stats
        )

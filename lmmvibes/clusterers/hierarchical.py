"""
Traditional hierarchical clustering stage.
"""

from typing import List
from ..core.stage import PipelineStage
from ..core.data_objects import PropertyDataset, Cluster
from ..core.mixins import LoggingMixin, TimingMixin, WandbMixin


class HierarchicalClusterer(PipelineStage, LoggingMixin, TimingMixin, WandbMixin):
    """
    Traditional hierarchical clustering stage.
    
    This stage migrates the hierarchical_cluster_categories function from
    clustering/hierarchical_clustering.py into the pipeline architecture.
    Uses sklearn's AgglomerativeClustering for both coarse and fine clustering.
    """
    
    def __init__(
        self,
        n_coarse_clusters: int = 10,
        n_fine_clusters: int = 50,
        embedding_model: str = "openai",
        include_embeddings: bool = True,
        use_wandb: bool = False,
        wandb_project: str = None,
        **kwargs
    ):
        """
        Initialize the hierarchical clusterer.
        
        Args:
            n_coarse_clusters: Number of coarse clusters to create
            n_fine_clusters: Number of fine clusters to create
            embedding_model: Embedding model to use
            include_embeddings: Whether to include embeddings in output
            use_wandb: Whether to use wandb for logging
            wandb_project: wandb project name
            **kwargs: Additional configuration
        """
        super().__init__(use_wandb=use_wandb, wandb_project=wandb_project, **kwargs)
        self.n_coarse_clusters = n_coarse_clusters
        self.n_fine_clusters = n_fine_clusters
        self.embedding_model = embedding_model
        self.include_embeddings = include_embeddings
        
    def run(self, data: PropertyDataset, column_name: str = "property_description") -> PropertyDataset:
        """
        Cluster properties using traditional hierarchical clustering.
        
        Args:
            data: PropertyDataset with properties to cluster
            column_name: Column name to cluster on
            
        Returns:
            PropertyDataset with populated clusters
        """
        self.log(f"Clustering {len(data.properties)} properties using hierarchical clustering")
        
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
            from lmmvibes.clusterers.hierarchical_clustering import hierarchical_cluster_categories, ClusterConfig
        except ImportError:
            # Fallback to relative import if package layout differs when installed
            from .hierarchical_clustering import hierarchical_cluster_categories, ClusterConfig  # type: ignore

        cfg = ClusterConfig(
            max_coarse_clusters=self.n_coarse_clusters,
            min_cluster_size=self.n_fine_clusters,  # This gets used to calculate n_fine_clusters
            embedding_model=self.embedding_model,
            include_embeddings=self.include_embeddings,
            verbose=False
        )

        clustered_df = hierarchical_cluster_categories(
            data.to_dataframe(type="properties"),
            column_name=column_name,
            config=cfg
        )

        # ------------------------------------------------------------------
        # Convert clustering result into Cluster objects
        # ------------------------------------------------------------------
        fine_label_col = f'{column_name}_fine_cluster_label'
        fine_id_col = f'{column_name}_fine_cluster_id'
        coarse_label_col = f'{column_name}_coarse_cluster_label'
        coarse_id_col = f'{column_name}_coarse_cluster_id'

        clusters: List[Cluster] = []

        clustered_df.to_csv("hierarchical_clustered_df.csv")

        # Create fine clusters with parent information
        for cid, group in clustered_df.groupby(fine_id_col):
            cid_group = group[group[fine_id_col] == cid]
            label = cid_group[fine_label_col].iloc[0]
            
            # Get parent cluster info
            coarse_labels = cid_group[coarse_label_col].unique().tolist()
            assert len(coarse_labels) == 1, f"Expected exactly one coarse label for fine cluster {cid}, but got {coarse_labels}"
            coarse_label = coarse_labels[0]
            coarse_id = cid_group[coarse_id_col].iloc[0]
            
            clusters.append(Cluster(
                id=int(cid),
                label=label,
                size=len(cid_group),
                property_descriptions=cid_group[column_name].tolist(),
                question_ids=cid_group["question_id"].tolist(),
                parent_id=int(coarse_id),
                parent_label=coarse_label
            ))

        self.log(f"Created {len(clusters)} fine clusters")
        
        # Count unique coarse clusters
        coarse_cluster_count = len(clustered_df[coarse_id_col].unique())
        self.log(f"Created {coarse_cluster_count} coarse clusters")

        # Attach cluster id/label back onto Property objects
        desc_to_fine_id = dict(zip(clustered_df[column_name], clustered_df[fine_id_col]))
        desc_to_fine_label = dict(zip(clustered_df[column_name], clustered_df[fine_label_col]))
        desc_to_coarse_id = dict(zip(clustered_df[column_name], clustered_df[coarse_id_col]))
        desc_to_coarse_label = dict(zip(clustered_df[column_name], clustered_df[coarse_label_col]))
        
        for p in data.properties:
            if p.property_description in desc_to_fine_id:
                setattr(p, 'fine_cluster_id', int(desc_to_fine_id[p.property_description]))
                setattr(p, 'fine_cluster_label', desc_to_fine_label[p.property_description])
                setattr(p, 'coarse_cluster_id', int(desc_to_coarse_id[p.property_description]))
                setattr(p, 'coarse_cluster_label', desc_to_coarse_label[p.property_description])

        # --- Wandb logging ---
        if self.use_wandb:
            self.init_wandb(project=self.wandb_project)
            try:
                import wandb
                log_df = pd.DataFrame([c.to_dict() for c in clusters]).astype(str)
                self.log_wandb({
                    "hierarchical_clustered_table": wandb.Table(dataframe=log_df)
                })
                import json
                self.log_wandb({
                    "hierarchical_clusters_json": wandb.Html(f'<pre>{json.dumps([c.to_dict() for c in clusters], indent=2)}</pre>')
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
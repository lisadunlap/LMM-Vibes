"""
HDBSCAN-based clustering stages.

This module migrates the clustering logic from clustering/hierarchical_clustering.py
into pipeline stages.
"""

from typing import List
import pandas as pd
import os

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
        output_dir: str = None,
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
            output_dir: Directory to save clustering results (optional)
            **kwargs: Additional configuration
        """
        super().__init__(use_wandb=use_wandb, wandb_project=wandb_project, **kwargs)
        self.min_cluster_size = min_cluster_size
        self.embedding_model = embedding_model
        self.hierarchical = hierarchical
        self.assign_outliers = assign_outliers
        self.include_embeddings = include_embeddings
        self.max_coarse_clusters = max_coarse_clusters
        self.output_dir = output_dir
        # Store config for save_clustered_results
        self.config = type('Config', (), {
            'min_cluster_size': min_cluster_size,
            'embedding_model': embedding_model,
            'hierarchical': hierarchical,
            'assign_outliers': assign_outliers,
            'use_wandb': use_wandb,
            'wandb_project': wandb_project,
            'max_coarse_clusters': max_coarse_clusters,
            'disable_dim_reduction': False,  # Default value
            'min_samples': min(min_cluster_size, max(5, min_cluster_size // 2)),  # Default calculation
            'cluster_selection_epsilon': 0.0  # Default value
        })()

    def _save(self, clustered_df: pd.DataFrame, clusters: List[Cluster]):
        """
        Save the clustered results to a file.
        """
        from .clustering_utils import save_clustered_results
        
        # Generate base filename from output directory
        base_filename = os.path.basename(self.output_dir.rstrip('/'))
        
        # Save clustered results using the enhanced function
        save_results = save_clustered_results(
            df=clustered_df,
            base_filename=base_filename,
            include_embeddings=self.include_embeddings,
            config=self.config,
            output_dir=self.output_dir
        )
        
        self.log(f"✅ Auto-saved clustering results to: {self.output_dir}")
        for key, path in save_results.items():
            if path:
                self.log(f"  • {key}: {path}")

        # --- Wandb logging ---
        if self.use_wandb:
            self.init_wandb(project=self.wandb_project)
            import wandb
            log_df = pd.DataFrame([c.to_dict() for c in clusters]).astype(str)
            self.log_wandb({
                "Clustering/hdbscan_clustered_table": wandb.Table(dataframe=log_df)
            })
        # --- End wandb logging ---

    def run_embedding_clustering(self, data: PropertyDataset, column_name: str = "property_description") -> PropertyDataset:
        """
        Cluster properties using HDBSCAN.
        """
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

        return clustered_df, cfg
        
                
    def run(self, data: PropertyDataset, column_name: str = "property_description") -> PropertyDataset:
        """
        Cluster properties using HDBSCAN.
        
        Args:
            data: PropertyDataset with properties to cluster
            
        Returns:
            PropertyDataset with populated clusters
        """
        self.log(f"Clustering {len(data.properties)} properties using HDBSCAN")
        
        # remove bad properties
        valid_properties = data.get_valid_properties()

        descriptions = [p.property_description for p in valid_properties]
        if not descriptions:
            raise ValueError("No property descriptions to cluster")

        # ------------------------------------------------------------------
        # Run HDBSCAN clustering
        # ------------------------------------------------------------------
        clustered_df, cfg = self.run_embedding_clustering(data, column_name)

        # assign any clusters smaller than min_cluster_size to Outliers
        fine_label_col = f'{column_name}_fine_cluster_label'
        fine_id_col = f'{column_name}_fine_cluster_id'
        # Get counts for each fine cluster label
        label_counts = clustered_df[fine_label_col].value_counts()
        # Find labels that are too small
        too_small_labels = label_counts[label_counts < cfg.min_cluster_size].index
        # For each too-small label, assign its rows to Outliers
        for label in too_small_labels:
            mask = clustered_df[fine_label_col] == label
            cid = clustered_df.loc[mask, fine_id_col].iloc[0] if not clustered_df.loc[mask].empty else None
            print(f"Assigning cluster {cid} (label '{label}') to Outliers because it has {label_counts[label]} items")
            clustered_df.loc[mask, fine_label_col] = "Outliers"
            clustered_df.loc[mask, fine_id_col] = -1

        # ------------------------------------------------------------------
        # Convert clustering result into a simple summary dict
        # ------------------------------------------------------------------
        fine_label_col = f'{column_name}_fine_cluster_label'
        fine_id_col    = f'{column_name}_fine_cluster_id'
        coarse_label_col = f'{column_name}_coarse_cluster_label'
        coarse_id_col    = f'{column_name}_coarse_cluster_id'

        clusters: List[Cluster] = []
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

        # CHANGE: Create a "No properties" cluster for conversations without properties
        # This ensures all conversations are considered in metrics calculation
        conversations_with_properties = set()
        for prop in data.properties:
            conversations_with_properties.add((prop.question_id, prop.model))
        
        conversations_without_properties = []
        for conv in data.conversations:
            if isinstance(conv.model, str):
                if (conv.question_id, conv.model) not in conversations_with_properties:
                    conversations_without_properties.append((conv.question_id, conv.model))
            elif isinstance(conv.model, list):
                for model in conv.model:
                    if (conv.question_id, model) not in conversations_with_properties:
                        conversations_without_properties.append((conv.question_id, model))
        
        if conversations_without_properties:
            self.log(f"Found {len(conversations_without_properties)} conversations without properties - creating 'No properties' cluster")
            
            # Create the "No properties" cluster
            no_props_cluster = Cluster(
                id=-2,  # Use -2 since -1 is for outliers
                label="No properties",
                size=len(conversations_without_properties),
                property_descriptions=["No properties"] * len(conversations_without_properties),
                question_ids=[qid for qid, _ in conversations_without_properties],
                parent_id=-2,  # Same ID for coarse cluster
                parent_label="No properties"
            )
            clusters.append(no_props_cluster)
            
            self.log(f"Created 'No properties' cluster with {len(conversations_without_properties)} conversations")
        else:
            self.log("All conversations have properties - no 'No properties' cluster needed")

        # Attach cluster id/label back onto Property objects (optional: first occurrence)
        desc_to_fine_id = dict(zip(clustered_df[column_name], clustered_df[fine_id_col]))
        desc_to_fine_label = dict(zip(clustered_df[column_name], clustered_df[fine_label_col]))
        for p in data.properties:
            if p.property_description in desc_to_fine_id:
                setattr(p, 'fine_cluster_id', int(desc_to_fine_id[p.property_description]))
                setattr(p, 'fine_cluster_label', desc_to_fine_label[p.property_description])

        # ------------------------------------------------------------------
        # Auto-save clustering results if output_dir is provided
        # ------------------------------------------------------------------
        self._save(clustered_df, clusters)

        return PropertyDataset(
            conversations=data.conversations,
            all_models=data.all_models,
            properties=data.properties,
            clusters=clusters,
            model_stats=data.model_stats
        )
    
class LLMOnlyClusterer(HDBSCANClusterer):
    """
    HDBSCAN clustering stage.
    
    This stage migrates the hdbscan_cluster_categories function from
    clustering/hierarchical_clustering.py into the pipeline architecture.
    """
    
    def run(self, data: PropertyDataset, column_name: str = "property_description") -> PropertyDataset:
        """
        Cluster properties using HDBSCAN.
        """
        return super().run(data, column_name)
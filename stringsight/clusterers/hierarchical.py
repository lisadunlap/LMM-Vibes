"""
Hierarchical clustering stages.

This module migrates the hierarchical clustering logic from clustering/hierarchical_clustering.py
into pipeline stages.
"""

from typing import List, Optional
from ..core.stage import PipelineStage
from ..core.data_objects import PropertyDataset, Cluster
from ..core.mixins import LoggingMixin, TimingMixin, WandbMixin

# Unified config
try:
    from .config import ClusterConfig
except ImportError:
    from config import ClusterConfig


class HierarchicalClusterer(PipelineStage, LoggingMixin, TimingMixin, WandbMixin):
    """
    Hierarchical clustering stage.

    This stage migrates the hierarchical_cluster_categories function from
    clustering/hierarchical_clustering.py into the pipeline architecture.
    """

    def __init__(
        self,
        min_cluster_size: int = 30,
        embedding_model: str = "openai",
        hierarchical: bool = True,
        assign_outliers: bool = False,
        include_embeddings: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        max_coarse_clusters: int = 25,
        output_dir: Optional[str] = None,
        # Optional explicit parameters
        verbose: bool = True,
        **kwargs,
    ):
        """
        Initialize the hierarchical clusterer.
        """
        super().__init__(use_wandb=use_wandb, wandb_project=wandb_project, **kwargs)
        self.output_dir = output_dir

        # Build consistent config
        self.config = ClusterConfig(
            min_cluster_size=min_cluster_size,
            embedding_model=embedding_model,
            hierarchical=hierarchical,
            assign_outliers=assign_outliers,
            include_embeddings=include_embeddings,
            verbose=verbose,
            max_coarse_clusters=max_coarse_clusters,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
        )

    def run(self, data: PropertyDataset, column_name: str = "property_description") -> PropertyDataset:
        """
        Cluster properties using hierarchical clustering.
        """
        self.log(f"Clustering {len(data.properties)} properties using hierarchical clustering")

        import pandas as pd

        # remove bad properties
        valid_properties = data.get_valid_properties()

        descriptions = [p.property_description for p in valid_properties]
        if not descriptions:
            self.log("No property descriptions to cluster – skipping stage")
            return data

        try:
            from stringsight.clusterers.hierarchical_clustering import hierarchical_cluster_categories
        except ImportError:
            from .hierarchical_clustering import hierarchical_cluster_categories  # type: ignore

        clustered_df = hierarchical_cluster_categories(
            data.to_dataframe(type="properties"),
            column_name=column_name,
            config=self.config,
        )

        # Convert clustering result into Cluster objects
        fine_label_col = f"{column_name}_fine_cluster_label"
        fine_id_col = f"{column_name}_fine_cluster_id"
        coarse_label_col = f"{column_name}_coarse_cluster_label"
        coarse_id_col = f"{column_name}_coarse_cluster_id"

        clusters: List[Cluster] = []

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

        # Create a "No properties" cluster for conversations without properties
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
                id=-2,
                label="No properties",
                size=len(conversations_without_properties),
                property_descriptions=["No properties"] * len(conversations_without_properties),
                question_ids=[qid for qid, _ in conversations_without_properties],
                parent_id=-2,
                parent_label="No properties"
            )
            clusters.append(no_props_cluster)
        else:
            self.log("All conversations have properties - no 'No properties' cluster needed")

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

        # Auto-save clustering results if output_dir is provided
        from .clustering_utils import save_clustered_results
        import os

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            base_filename = os.path.basename(self.output_dir.rstrip('/'))
            save_results = save_clustered_results(
                df=clustered_df,
                base_filename=base_filename,
                include_embeddings=self.config.include_embeddings,
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
            try:
                import wandb
                log_df = pd.DataFrame([c.to_dict() for c in clusters]).astype(str)
                self.log_wandb({
                    "hierarchical_clustered_table": wandb.Table(dataframe=log_df)
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
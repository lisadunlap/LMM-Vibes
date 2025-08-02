from __future__ import annotations

from typing import List, Dict, Any, Optional
import pandas as pd
import os
from ..core.stage import PipelineStage
from ..core.mixins import LoggingMixin, TimingMixin
from ..core.data_objects import PropertyDataset, Cluster


class DummyClusterer(LoggingMixin, TimingMixin, PipelineStage):
    """A **no-op** clustering stage used for fixed-taxonomy pipelines.

    Every unique ``property_description`` becomes its own *fine* cluster.  No
    embeddings or distance computations are performed.

    Parameters
    ----------
    allowed_labels:
        List of labels that are present in the user-supplied taxonomy.
    unknown_label:
        Name assigned to properties whose description is *not* in
        ``allowed_labels`` (default: ``"Other"``).
    output_dir:
        Directory to save clustering results (optional)
    include_embeddings:
        Whether to include embeddings in output (default: False since dummy clustering doesn't use embeddings)
    """

    def __init__(
        self, 
        allowed_labels: List[str], 
        unknown_label: str = "Other", 
        output_dir: str = None,
        include_embeddings: bool = False,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.allowed_labels = set(allowed_labels)
        self.unknown_label = unknown_label
        self.output_dir = output_dir
        self.include_embeddings = include_embeddings
        # Create a dummy config for save_clustered_results
        self.config = type('Config', (), {
            'min_cluster_size': 1,
            'embedding_model': 'dummy',
            'hierarchical': False,
            'assign_outliers': False,
            'use_wandb': False,
            'wandb_project': None,
            'max_coarse_clusters': len(allowed_labels),
            'disable_dim_reduction': True,
            'min_samples': 1,
            'cluster_selection_epsilon': 0.0
        })()

    def _save(self, clustered_df: pd.DataFrame, clusters: List[Cluster]):
        """
        Save the clustered results to a file using the same format as HDBSCANClusterer.
        """
        if not self.output_dir:
            return
            
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

    # ------------------------------------------------------------------
    # PipelineStage interface
    # ------------------------------------------------------------------
    def run(self, data: PropertyDataset) -> PropertyDataset:
        self.log(f"💠 DummyClusterer: mapping {len(data.properties)} properties → clusters (|allowed|={len(self.allowed_labels)})")

        # ------------------------------------------------------------------
        # 1.  Sanitize property descriptions
        # ------------------------------------------------------------------
        for prop in data.properties:
            desc = (prop.property_description or "").strip()
            if desc not in self.allowed_labels:
                prop.property_description = self.unknown_label

        # ------------------------------------------------------------------
        # 2.  Build clusters – one per description
        # ------------------------------------------------------------------
        clusters: List[Cluster] = []
        desc_to_props: Dict[str, List] = {}
        for prop in data.properties:
            desc_to_props.setdefault(prop.property_description, []).append(prop)

        # Keep deterministic ordering: allowed labels first, then 'Other'
        all_labels_order = list(self.allowed_labels) + ([self.unknown_label] if self.unknown_label not in self.allowed_labels else [])

        for idx, desc in enumerate(all_labels_order):
            props = desc_to_props.get(desc, [])

            clusters.append(
                Cluster(
                    id=idx,
                    label=desc,
                    size=len(props),
                    property_descriptions=[p.property_description for p in props],
                    question_ids=[p.question_id for p in props],
                )
            )

            # Attach id/label back onto each Property that matched this description
            for p in props:
                setattr(p, "fine_cluster_id", idx)
                setattr(p, "fine_cluster_label", desc)

        self.log(f"💠 DummyClusterer: created {len(clusters)} clusters (including empty ones)")

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

        # ------------------------------------------------------------------
        # 3. Create clustered DataFrame for saving (if output_dir provided)
        # ------------------------------------------------------------------
        if self.output_dir:
            # Use the same approach as HDBSCAN - get the merged conversation+property data
            clustered_df = data.to_dataframe(type="properties")
            
            # Add cluster information to the DataFrame
            if not clustered_df.empty:
                # Create mappings from property attributes to cluster info
                prop_to_cluster_id = {}
                prop_to_cluster_label = {}
                
                for prop in data.properties:
                    prop_id = prop.id if hasattr(prop, 'id') and prop.id else None
                    if prop_id:
                        prop_to_cluster_id[prop_id] = getattr(prop, 'fine_cluster_id', -1)
                        prop_to_cluster_label[prop_id] = getattr(prop, 'fine_cluster_label', 'Unknown')
                
                # Add cluster columns to the DataFrame
                if 'id' in clustered_df.columns:
                    clustered_df['property_description_fine_cluster_id'] = clustered_df['id'].map(prop_to_cluster_id).fillna(-1)
                    clustered_df['property_description_fine_cluster_label'] = clustered_df['id'].map(prop_to_cluster_label).fillna('Unknown')
                    clustered_df['fine_cluster_id'] = clustered_df['property_description_fine_cluster_id']
                    clustered_df['fine_cluster_label'] = clustered_df['property_description_fine_cluster_label']
            
            # Save using the same function as HDBSCANClusterer
            self._save(clustered_df, clusters)

        return PropertyDataset(
            conversations=data.conversations,
            all_models=data.all_models,
            properties=data.properties,
            clusters=clusters,
            model_stats=data.model_stats,
        ) 
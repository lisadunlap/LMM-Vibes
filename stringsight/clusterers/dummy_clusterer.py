from __future__ import annotations

from typing import List, Dict, Any
import pandas as pd

from .base import BaseClusterer
from ..core.data_objects import PropertyDataset

# Unified config
try:
    from .config import ClusterConfig
except ImportError:
    from config import ClusterConfig


class DummyClusterer(BaseClusterer):
    """A no-op clustering stage used for fixed-taxonomy pipelines.

    Every unique `property_description` becomes its own fine cluster. No
    embeddings or distance computations are performed.

    Parameters
    ----------
    allowed_labels:
        List of labels that are present in the user-supplied taxonomy.
    unknown_label:
        Name assigned to properties whose description is not in
        `allowed_labels` (default: "Other").
    output_dir:
        Directory to save clustering results (optional)
    include_embeddings:
        Whether to include embeddings in output (default: False)
    """

    def __init__(
        self,
        allowed_labels: List[str],
        unknown_label: str = "Other",
        output_dir: str | None = None,
        include_embeddings: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            output_dir=output_dir,
            include_embeddings=include_embeddings,
            hierarchical=False,
            **kwargs,
        )
        self.allowed_labels = set(allowed_labels)
        self.unknown_label = unknown_label
        # Minimal config for saving/logging and consistency
        self.config = ClusterConfig(
            min_cluster_size=1,
            embedding_model="dummy",
            hierarchical=False,
            prettify_labels=False,
            assign_outliers=False,
            use_wandb=False,
            wandb_project=None,
            max_coarse_clusters=len(allowed_labels),
            disable_dim_reduction=True,
            include_embeddings=include_embeddings,
        )

    def cluster(self, data: PropertyDataset, column_name: str) -> pd.DataFrame:
        """Map properties to a fixed taxonomy and return a standardized DataFrame."""
        # 1) Sanitize property descriptions in-memory for clustering
        for prop in data.properties:
            desc = (getattr(prop, column_name, "") or "").strip()
            if desc not in self.allowed_labels:
                setattr(prop, column_name, self.unknown_label)

        # 2) Build a properties DataFrame
        df = data.to_dataframe(type="properties")
        if df.empty:
            return df

        # 3) Compute deterministic ordering: allowed labels first, then 'Other' if not included
        ordered_labels = list(self.allowed_labels)
        if self.unknown_label not in self.allowed_labels:
            ordered_labels.append(self.unknown_label)
        label_to_id: Dict[str, int] = {label: idx for idx, label in enumerate(ordered_labels)}

        # 4) Add standardized fine cluster columns
        fine_id_col = f"{column_name}_fine_cluster_id"
        fine_label_col = f"{column_name}_fine_cluster_label"
        df[fine_label_col] = df[column_name].map(lambda x: x if x in label_to_id else self.unknown_label)
        df[fine_id_col] = df[fine_label_col].map(label_to_id)
        # Add base convenience aliases used elsewhere
        df["fine_cluster_id"] = df[fine_id_col]
        df["fine_cluster_label"] = df[fine_label_col]

        return df 
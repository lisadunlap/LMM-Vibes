from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import pandas as pd

from ..core.stage import PipelineStage
from ..core.data_objects import PropertyDataset, Cluster
from ..core.mixins import LoggingMixin, TimingMixin, WandbMixin


class BaseClusterer(LoggingMixin, TimingMixin, WandbMixin, PipelineStage, ABC):
    """Abstract base class for clustering stages.

    This class defines a minimal, unified contract for clustering steps
    in the pipeline. Subclasses implement the clustering strategy while
    reusing shared orchestration, saving, and metadata-handling provided
    by the base class.

    Responsibilities
    ----------------
    - Define a single entry point (`run`) for the clustering pipeline
    - Define an abstract `cluster` method that returns a standardized
      DataFrame schema
    - Provide hooks for post-processing, configuration, saving, and
      converting DataFrames into `Cluster` objects

    Standardized DataFrame Contract
    --------------------------------
    Implementations of `cluster` must return a DataFrame containing:
    - `question_id`
    - `{column_name}` (by default `property_description`)
    - `{column_name}_fine_cluster_id` (int)
    - `{column_name}_fine_cluster_label` (str)
    - If hierarchical is used:
      - `{column_name}_coarse_cluster_id`
      - `{column_name}_coarse_cluster_label`
    """

    def __init__(
        self,
        *,
        output_dir: Optional[str] = None,
        include_embeddings: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        hierarchical: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the clusterer with common options.

        Parameters
        ----------
        output_dir:
            Directory where clustering artifacts should be saved.
        include_embeddings:
            Whether embedding columns should be included in saved artifacts.
        use_wandb:
            Enable Weights & Biases logging for clustering outputs.
        wandb_project:
            W&B project name to log under when enabled.
        hierarchical:
            Whether the clusterer expects/produces a hierarchical schema.
        kwargs:
            Additional implementation-specific options for derived classes.
        """
        super().__init__(use_wandb=use_wandb, wandb_project=wandb_project, **kwargs)
        self.output_dir = output_dir
        self.include_embeddings = include_embeddings
        self.hierarchical = hierarchical

    @abstractmethod
    def cluster(self, data: PropertyDataset, column_name: str) -> pd.DataFrame:
        """Produce a standardized clustered DataFrame from the dataset.

        Implementations may compute embeddings or use heuristic rules, but
        should not mutate `data`. The returned DataFrame must follow the
        standardized column naming contract described in the class docstring.

        Parameters
        ----------
        data:
            The input `PropertyDataset` containing conversations and properties.
        column_name:
            The name of the textual feature column to cluster (default
            expected value is "property_description").

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the standardized cluster columns.
        """
        ...

    def postprocess_clustered_df(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """Optional hook to modify the clustered DataFrame.

        Called after `cluster` and before converting to `Cluster` objects.
        Typical use cases include:
        - Reassigning small clusters to an "Outliers" label
        - Thresholding cluster assignments or cleaning labels

        Parameters
        ----------
        df:
            The clustered DataFrame produced by `cluster`.
        column_name:
            The feature column used for clustering.

        Returns
        -------
        pd.DataFrame
            The potentially modified DataFrame. Default: return `df` unchanged.
        """
        return df

    def get_config(self) -> Any:
        """Return a configuration object for saving/logging.

        The returned object should expose attributes referenced by saving and
        logging routines (e.g., `min_cluster_size`, `embedding_model`,
        `hierarchical`, `assign_outliers`, `use_wandb`, `wandb_project`,
        `max_coarse_clusters`, `disable_dim_reduction`, `min_samples`,
        `cluster_selection_epsilon`).

        Returns
        -------
        Any
            A configuration object or namespace-like structure.
        """
        if hasattr(self, "config") and self.config is not None:
            return self.config
        # Provide a minimal default to avoid attribute errors downstream
        self.config = type("Config", (), {
            "min_cluster_size": 1,
            "embedding_model": "unknown",
            "hierarchical": bool(self.hierarchical),
            "assign_outliers": False,
            "use_wandb": bool(self.use_wandb),
            "wandb_project": getattr(self, "wandb_project", None),
            "max_coarse_clusters": 0,
            "disable_dim_reduction": False,
            "min_samples": 1,
            "cluster_selection_epsilon": 0.0,
        })()
        return self.config

    def run(self, data: PropertyDataset, column_name: str = "property_description") -> PropertyDataset:
        """Execute the clustering pipeline and return an updated dataset.

        Expected orchestration steps:
        1. Create a standardized clustered DataFrame via `cluster(...)`.
        2. Optionally post-process via `postprocess_clustered_df(...)`.
        3. Convert groups to `Cluster` domain objects.
        4. Add a synthetic "No properties" cluster via `add_no_properties_cluster(...)`
           to cover conversations without properties (when desired).
        5. Attach `fine_cluster_id` and `fine_cluster_label` to each
           `Property` in the input dataset when possible.
        6. Persist artifacts via `save(...)` if an `output_dir` is provided.
        7. Return a new `PropertyDataset` that includes the clusters and any
           property annotations.
        """
        self.log(f"Clustering {len(data.properties)} properties using {self.__class__.__name__}")

        if not data.properties:
            raise ValueError("No properties to cluster")

        clustered_df = self.cluster(data, column_name)
        clustered_df = self.postprocess_clustered_df(clustered_df, column_name)

        clusters = self._build_clusters_from_df(clustered_df, column_name)
        self.add_no_properties_cluster(data, clusters)
        self._attach_cluster_attrs_to_properties(data, clustered_df, column_name)

        self.save(clustered_df, clusters)

        return PropertyDataset(
            conversations=data.conversations,
            all_models=data.all_models,
            properties=data.properties,
            clusters=clusters,
            model_stats=data.model_stats,
        )

    def save(self, df: pd.DataFrame, clusters: List[Cluster]) -> Dict[str, str]:
        """Persist clustering artifacts to disk (and optionally external loggers).

        Implementations should leverage a common saving utility to ensure
        consistent artifact formats across clusterers.
        """
        if not self.output_dir:
            return {}

        from .clustering_utils import save_clustered_results

        base_filename = os.path.basename(self.output_dir.rstrip("/"))
        config = self.get_config()

        paths = save_clustered_results(
            df=df,
            base_filename=base_filename,
            include_embeddings=bool(self.include_embeddings),
            config=config,
            output_dir=self.output_dir,
        )

        self.log(f"✅ Auto-saved clustering results to: {self.output_dir}")
        for key, path in paths.items():
            if path:
                self.log(f"  • {key}: {path}")

        if self.use_wandb:
            self.init_wandb(project=self.wandb_project)
            import wandb
            log_df = pd.DataFrame([c.to_dict() for c in clusters]).astype(str)
            self.log_wandb({
                f"Clustering/{self.__class__.__name__}_clustered_table": wandb.Table(dataframe=log_df)
            })

        return paths

    def _build_clusters_from_df(self, df: pd.DataFrame, column_name: str) -> List[Cluster]:
        """Construct `Cluster` objects from a standardized DataFrame.

        Group rows by fine cluster id, extract labels and collect
        `question_id` and `{column_name}` values for each cluster. For
        hierarchical clustering, ensure that each fine cluster is associated
        with exactly one coarse cluster id/label.
        """
        fine_label_col = f"{column_name}_fine_cluster_label"
        fine_id_col = f"{column_name}_fine_cluster_id"
        coarse_label_col = f"{column_name}_coarse_cluster_label"
        coarse_id_col = f"{column_name}_coarse_cluster_id"

        config = self.get_config()
        is_hierarchical = bool(getattr(config, "hierarchical", False))

        clusters: List[Cluster] = []
        for cid, group in df.groupby(fine_id_col):
            cid_group = group[group[fine_id_col] == cid]
            label = str(cid_group[fine_label_col].iloc[0])

            if is_hierarchical:
                coarse_labels = cid_group[coarse_label_col].unique().tolist()
                assert len(coarse_labels) == 1, (
                    f"Expected exactly one coarse label for fine cluster {cid}, but got {coarse_labels}"
                )
                coarse_id = int(cid_group[coarse_id_col].iloc[0])
                clusters.append(
                    Cluster(
                        id=int(cid),
                        label=label,
                        size=len(cid_group),
                        property_descriptions=cid_group[column_name].tolist(),
                        question_ids=cid_group["question_id"].tolist(),
                        parent_id=int(coarse_id),
                        parent_label=coarse_labels[0],
                    )
                )
            else:
                clusters.append(
                    Cluster(
                        id=int(cid),
                        label=label,
                        size=len(cid_group),
                        property_descriptions=cid_group[column_name].tolist(),
                        question_ids=cid_group["question_id"].tolist(),
                    )
                )

        self.log(f"Created {len(clusters)} fine clusters")
        if is_hierarchical:
            coarse_ids = df[coarse_id_col].dropna().unique().tolist()
            self.log(f"Created {len(coarse_ids)} coarse clusters")

        return clusters

    def _attach_cluster_attrs_to_properties(self, data: PropertyDataset, df: pd.DataFrame, column_name: str) -> None:
        """Attach cluster annotations to properties in the dataset.

        For each `Property` whose `{column_name}` value appears in the
        standardized DataFrame, set `fine_cluster_id` and `fine_cluster_label`
        on the property instance.
        """
        fine_id_map = dict(zip(df[column_name], df[f"{column_name}_fine_cluster_id"]))
        fine_label_map = dict(zip(df[column_name], df[f"{column_name}_fine_cluster_label"]))

        for prop in data.properties:
            value = getattr(prop, column_name, None)
            if value in fine_id_map:
                setattr(prop, "fine_cluster_id", int(fine_id_map[value]))
                setattr(prop, "fine_cluster_label", fine_label_map[value])

    def add_no_properties_cluster(self, data: PropertyDataset, clusters: List[Cluster]) -> None:
        """Append a synthetic "No properties" cluster when applicable.

        Detect conversations that lack any associated properties and create a
        dedicated cluster entry so they are represented in downstream metrics
        and visualizations. Subclasses may override to disable or customize
        this behavior.
        """
        conversations_with_properties = set()
        for prop in data.properties:
            conversations_with_properties.add((prop.question_id, prop.model))

        conversations_without_properties: List[tuple] = []
        for conv in data.conversations:
            if isinstance(conv.model, str):
                key = (conv.question_id, conv.model)
                if key not in conversations_with_properties:
                    conversations_without_properties.append(key)
            elif isinstance(conv.model, list):
                for model in conv.model:
                    key = (conv.question_id, model)
                    if key not in conversations_with_properties:
                        conversations_without_properties.append(key)

        if not conversations_without_properties:
            self.log("All conversations have properties - no 'No properties' cluster needed")
            return

        self.log(
            f"Found {len(conversations_without_properties)} conversations without properties - creating 'No properties' cluster"
        )

        no_props_cluster = Cluster(
            id=-2,
            label="No properties",
            size=len(conversations_without_properties),
            property_descriptions=["No properties"] * len(conversations_without_properties),
            question_ids=[qid for qid, _ in conversations_without_properties],
            parent_id=-2,
            parent_label="No properties",
        )
        clusters.append(no_props_cluster)
        self.log(
            f"Created 'No properties' cluster with {len(conversations_without_properties)} conversations"
        )


__all__ = ["BaseClusterer"] 
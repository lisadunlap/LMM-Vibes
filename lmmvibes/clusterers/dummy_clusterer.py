from __future__ import annotations

from typing import List, Dict, Any, Optional
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
    """

    def __init__(self, allowed_labels: List[str], unknown_label: str = "Other", **kwargs: Any):
        super().__init__(**kwargs)
        self.allowed_labels = set(allowed_labels)
        self.unknown_label = unknown_label

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

        return PropertyDataset(
            conversations=data.conversations,
            all_models=data.all_models,
            properties=data.properties,
            clusters=clusters,
            model_stats=data.model_stats,
        ) 
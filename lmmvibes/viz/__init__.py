"""Utility helpers for interactive visualisation of clustering results.

This module provides thin wrappers that transform the core `PropertyDataset`
object into structures that are easy to consume in a UI (e.g. Streamlit).
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple

import pandas as pd  # noqa: F401 – kept for potential future helpers

from lmmvibes.core.data_objects import (
    PropertyDataset,
    Cluster,
    ConversationRecord,
    Property,
)

# ---------------------------------------------------------------------------
# Cached dataset loading -----------------------------------------------------
# ---------------------------------------------------------------------------
def load_dataset(path: str | Path) -> PropertyDataset:
    """Load a :class:`PropertyDataset` from *path* (JSON or pickle).

    If the file originates from an **older schema** that didn't include the
    ``model`` field inside ``ConversationRecord``, we patch the dictionaries on
    the fly for backwards-compatibility so visualisation continues to work.
    """
    try:
        return PropertyDataset.load(str(path))
    except TypeError as err:
        # Fallback for legacy files lacking the 'model' key.
        if "missing 1 required positional argument: 'model'" not in str(err):
            raise  # Different problem – bubble up

        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # --- Conversations --------------------------------------------------
        convs: list[ConversationRecord] = []
        for conv in data.get("conversations", []):
            if "model" not in conv:
                # Try reconstructing from legacy keys
                model_a = conv.pop("model_a", None)
                model_b = conv.pop("model_b", None)

                if model_a is not None and model_b is not None:
                    conv["model"] = (model_a, model_b)
                elif model_a is not None:
                    conv["model"] = model_a
                else:
                    conv["model"] = "unknown"

            convs.append(ConversationRecord(**conv))

        # --- Properties & clusters -----------------------------------------
        props = [Property(**p) for p in data.get("properties", [])]
        clusters_raw = data.get("clusters", [])
        # Handle both list[dict] and list[Cluster] variants
        clusters: list[Cluster] = [c if isinstance(c, Cluster) else Cluster(**c) for c in clusters_raw]

        model_stats = data.get("model_stats", {})

        return PropertyDataset(
            conversations=convs,
            properties=props,
            clusters=clusters,
            model_stats=model_stats,
        )


# ---------------------------------------------------------------------------
# Hierarchy helpers ----------------------------------------------------------
# ---------------------------------------------------------------------------

def build_hierarchy(dataset: PropertyDataset) -> tuple[list[Cluster], dict[int | str, list[Cluster]]]:
    """Split *dataset.clusters* into coarse and fine levels.

    Returns
    -------
    coarse : list[Cluster]
        All clusters that have no ``parent_id`` (top-level).
    fine_map : dict[parent_id, list[Cluster]]
        Mapping from coarse cluster id → list of its children.
    """
    # -------------------------------------------------------------------
    # Primary path – use pre-computed Cluster objects if they look valid
    # -------------------------------------------------------------------
    if dataset.clusters and all(c.property_descriptions for c in dataset.clusters):
        coarse: List[Cluster] = []
        fine_map: Dict[int | str, List[Cluster]] = {}

        for cluster in dataset.clusters:
            if cluster.parent_id is None:
                coarse.append(cluster)
            else:
                fine_map.setdefault(cluster.parent_id, []).append(cluster)

        # Sort for deterministic display
        coarse.sort(key=lambda c: c.size, reverse=True)
        for child_list in fine_map.values():
            child_list.sort(key=lambda c: c.size, reverse=True)

        return coarse, fine_map

    # -------------------------------------------------------------------
    # Fallback – reconstruct clusters from the *DataFrame* representation
    # -------------------------------------------------------------------
    df = dataset.to_dataframe(type="clusters")
    if df.empty:
        # No clustering info; return empty structures
        return [], {}

    # Fill NaNs for grouping
    df = df.fillna({"coarse_cluster_id": -1, "coarse_cluster_label": "No coarse"})

    # Build fine clusters first
    fine_clusters: List[Cluster] = []
    for (fid, flabel, cid, clabel), grp in df.groupby([
        "fine_cluster_id",
        "fine_cluster_label",
        "coarse_cluster_id",
        "coarse_cluster_label",
    ]):
        prop_descs = grp["property_description"].dropna().tolist()
        fine_clusters.append(
            Cluster(
                id=fid,
                label=flabel,
                size=len(grp),
                parent_id=cid if cid != -1 else None,
                parent_label=clabel if cid != -1 else None,
                property_descriptions=prop_descs,
                question_ids=grp["question_id"].dropna().astype(str).tolist()
                if "question_id" in grp.columns else [],
            )
        )

    # Split into coarse / fine mapping
    coarse: List[Cluster] = []
    fine_map: Dict[int | str, List[Cluster]] = {}
    for fc in fine_clusters:
        if fc.parent_id is None:
            coarse.append(fc)
        else:
            fine_map.setdefault(fc.parent_id, []).append(fc)

    # Sort for deterministic order
    coarse.sort(key=lambda c: c.size, reverse=True)
    for child_list in fine_map.values():
        child_list.sort(key=lambda c: c.size, reverse=True)

    # Cache reconstructed clusters back onto dataset so next call is fast
    if not dataset.clusters:
        dataset.clusters = fine_clusters

    return coarse, fine_map


# ---------------------------------------------------------------------------
# DataFrame-only helpers -----------------------------------------------------
# ---------------------------------------------------------------------------

def load_clusters_dataframe(path: str | Path):  # → pd.DataFrame
    """Load a clusters DataFrame saved as JSON Lines, JSON, CSV, or Parquet.

    The file **must** contain at least the columns produced by
    ``Cluster.to_dataframe()``: ``fine_cluster_id``, ``fine_cluster_label``,
    ``property_description``.  If hierarchical information is present, the
    columns ``coarse_cluster_id`` and ``coarse_cluster_label`` should be
    included as well.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix in {".jsonl", ".json"}:
        try:
            df = pd.read_json(path, orient="records", lines=True)
        except ValueError:
            # Probably a regular JSON list – try again without lines=True
            df = pd.read_json(path, orient="records")
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    return df


def build_hierarchy_from_df(df: "pd.DataFrame") -> tuple[list[Cluster], dict[int | str, list[Cluster]]]:
    """Create coarse / fine Cluster objects directly from a *clusters* DataFrame."""
    if df.empty:
        return [], {}

    # Ensure necessary columns exist
    required_cols = {"fine_cluster_id", "fine_cluster_label", "property_description"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing)}")

    df = df.copy()
    # Fill NaNs for grouping safety
    df = df.fillna({"coarse_cluster_id": -1, "coarse_cluster_label": "No coarse"})

    # Step 1: Create coarse clusters by grouping by coarse_cluster_id and coarse_cluster_label
    coarse_clusters: list[Cluster] = []
    if "coarse_cluster_id" in df.columns and "coarse_cluster_label" in df.columns:
        for (cid, clabel), grp in df.groupby(["coarse_cluster_id", "coarse_cluster_label"]):
            if cid != -1:  # Skip outliers for coarse clusters
                prop_descs = grp["property_description"].dropna().tolist()
                coarse_clusters.append(
                    Cluster(
                        id=cid,
                        label=clabel,
                        size=len(grp),
                        parent_id=None,  # Coarse clusters have no parent
                        parent_label=None,
                        property_descriptions=prop_descs,
                        question_ids=grp.get("question_id", pd.Series(dtype=str)).dropna().astype(str).tolist(),
                    )
                )

    # Step 2: Create fine clusters by grouping by fine_cluster_id and fine_cluster_label
    fine_clusters: list[Cluster] = []
    for (fid, flabel, cid, clabel), grp in df.groupby([
        "fine_cluster_id",
        "fine_cluster_label",
        "coarse_cluster_id",
        "coarse_cluster_label",
    ]):
        prop_descs = grp["property_description"].dropna().tolist()
        fine_clusters.append(
            Cluster(
                id=fid,
                label=flabel,
                size=len(grp),
                parent_id=cid if cid != -1 else None,
                parent_label=clabel if cid != -1 else None,
                property_descriptions=prop_descs,
                question_ids=grp.get("question_id", pd.Series(dtype=str)).dropna().astype(str).tolist(),
            )
        )

    # Step 3: Build hierarchy mapping
    fine_map: dict[int | str, list[Cluster]] = {}
    for fc in fine_clusters:
        if fc.parent_id is not None:
            fine_map.setdefault(fc.parent_id, []).append(fc)

    # Step 4: If no coarse clusters were created (flat clustering), use top-level fine clusters
    if not coarse_clusters:
        coarse_clusters = [fc for fc in fine_clusters if fc.parent_id is None]

    # Sort for deterministic order
    coarse_clusters.sort(key=lambda c: c.size, reverse=True)
    for child_list in fine_map.values():
        child_list.sort(key=lambda c: c.size, reverse=True)

    return coarse_clusters, fine_map


__all__: tuple[str, ...] = ("load_dataset", "build_hierarchy", "load_clusters_dataframe", "build_hierarchy_from_df") 
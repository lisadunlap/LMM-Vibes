"""lmmvibes.metrics.single_model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute comprehensive metrics for every property cluster, analyzing **how strongly each model 
exhibits specific behaviors compared to its peers** in an Arena-style side-by-side dataset.

Metric definitions
-----------------

For a given model *m* and cluster *c*:

**Representation Score:**
```
prop(m, c)   =   #questions where m shows c   /   #questions answered by m
score(m, c)  =   prop(m, c) / median_{m'} prop(m', c)
```

**Quality Score:**
For each score key *k* in the multi-dimensional score dictionary:
```
quality_score(k, c) = avg_{m in c} (avg_score(m, c, k) / avg_score(m, global, k))
```

Where:
- `avg_score(m, c, k)` = average score for model *m* in cluster *c* for key *k*
- `avg_score(m, global, k)` = average score for model *m* across all data for key *k*

**Interpretation:**
- `score > 1`: Model is **over-represented** in that cluster
- `score < 1`: Model is **under-represented** in that cluster  
- `quality_score > 1`: Models in this cluster perform better than their global average
- `quality_score < 1`: Models in this cluster perform worse than their global average

The metrics support multi-dimensional scoring where each question can have multiple
evaluation criteria (e.g., accuracy, helpfulness, harmlessness) stored as dictionary keys.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..core.stage import PipelineStage
from ..core.mixins import LoggingMixin, TimingMixin
from ..core.data_objects import PropertyDataset, ModelStats


class SingleModelMetrics(PipelineStage, LoggingMixin, TimingMixin):
    """Metrics stage for side-by-side data."""

    def __init__(self, output_dir: str | Path | None = None, **kwargs):
        super().__init__(**kwargs)
        self.output_dir = Path(output_dir) if output_dir else None

    # ------------------------------------------------------------------
    @staticmethod
    def _example_props(group: pd.DataFrame, n: int = 3) -> List[str]:
        return (
            group["id"].dropna().unique().tolist()[:n]
            if "id" in group.columns else []
        )

    # ------------------------------------------------------------------
    def run(self, data: PropertyDataset) -> PropertyDataset:  # noqa: D401
        """Compute fine & coarse metrics and attach them to *data*."""

        self.log("⚖️  Computing metrics …")

        df = data.to_dataframe(type="clusters")
        if df.empty:
            self.log("No cluster info found; skipping metrics stage.")
            return data

        # Required columns check
        req = {
            "model",
            "question_id",
            "fine_cluster_id",
            "fine_cluster_label",
        }
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {', '.join(missing)}")

        df = df.copy()
        df["coarse_cluster_id"] = df["coarse_cluster_id"].fillna(-1)
        df["coarse_cluster_label"] = df["coarse_cluster_label"].fillna("<flat>")

        # ------------------------------------------------------------------
        # Handle duplicate columns produced by DataFrame merges (…_x / …_y)
        # ------------------------------------------------------------------
        for base in [
            "fine_cluster_id",
            "fine_cluster_label",
            "coarse_cluster_id",
            "coarse_cluster_label",
        ]:
            if base not in df.columns:
                # Try to collapse _x / _y variants
                x_col, y_col = f"{base}_x", f"{base}_y"
                if x_col in df.columns or y_col in df.columns:
                    df[base] = df.get(x_col).combine_first(df.get(y_col))
        # Drop helper cols to avoid confusion
        df = df[[c for c in df.columns if not c.endswith(("_x", "_y"))]]

        # Validate that all model names are strings (not lists/tuples)
        invalid_models = df[df["model"].apply(lambda x: not isinstance(x, str))]
        if not invalid_models.empty:
            self.log(f"⚠️  Found {len(invalid_models)} properties with invalid model names:", level="warning")
            for _, row in invalid_models.iterrows():
                self.log(f"  - Question {row['question_id']}: model = {row['model']} (type: {type(row['model'])})", level="warning")
            # Filter out properties with invalid model names
            df = df[df["model"].apply(lambda x: isinstance(x, str))]
            self.log(f"Filtered out {len(invalid_models)} properties with invalid model names")

        self.log(f"Models used for metrics: {df['model'].unique()}")

        # ------------------------------------------------------------------
        # Calculate denominator: unique questions answered per model
        # ------------------------------------------------------------------
        # Using DataFrame.value_counts on a single column returns a MultiIndex
        # whose keys are *tuples* like ("gpt-4o",).  That propagates through
        # the downstream metrics and we end up with dict keys that look like
        # ('gpt-4o',) instead of plain strings.

        total_q_series = (
            df[["model", "question_id"]]
            .drop_duplicates()
            .groupby("model")
            .size()
        )
        total_q = total_q_series.to_dict()

        # calcualte the average score per model
        # If score is a dict with multiple keys, compute the average for each key per model
        import collections

        # First, get all possible keys in the score dicts
        all_score_keys = set()
        for s in df["score"]:
            if isinstance(s, dict):
                all_score_keys.update(s.keys())
        all_score_keys = sorted(all_score_keys)

        # For each key, compute the average score per model
        avg_scores_per_key = {}
        for key in all_score_keys:
            # Extract the score for this key (0 if missing or not a dict)
            df[f"score_{key}"] = df["score"].apply(
                lambda x: x.get(key, 0) if isinstance(x, dict) else 0
            )
            avg_scores_per_key[key] = df.groupby("model")[f"score_{key}"].mean()
            print(f"Average score for key '{key}':")
            print(avg_scores_per_key[key])

        # Optionally, you could aggregate these into a DataFrame:
        self.avg_score_per_model = pd.DataFrame(avg_scores_per_key)
        print("Average scores per model (all keys):")
        print(self.avg_score_per_model)

        stats: Dict[str, Dict[str, List[ModelStats]]] = defaultdict(lambda: {"fine": [], "coarse": []})

        # ---------------- Fine clusters ----------------
        for cid, group in df.groupby("fine_cluster_id"):
            lbl = group["fine_cluster_label"].iloc[0]
            self._compute(group, cid, lbl, "fine", total_q, stats)

        # ---------------- Coarse clusters --------------
        if (df["coarse_cluster_id"] != -1).any():
            for cid, group in df.groupby("coarse_cluster_id"):
                if cid == -1:
                    continue
                lbl = group["coarse_cluster_label"].iloc[0]
                self._compute(group, cid, lbl, "coarse", total_q, stats)

        # Sort stats lists by score descending for deterministic order
        for m in stats:
            for lvl in ("fine", "coarse"):
                stats[m][lvl].sort(key=lambda s: s.score, reverse=True)

        data.model_stats = stats  # type: ignore[assignment]

        if self.output_dir:
            self._dump(stats)

        self.log(f"✅ Metrics computed for {len(stats)} models")
        return data
    
    def _compute_quality_score(self, group: pd.DataFrame) -> dict:
        """
        Compute the quality score of a cluster for each score key:
        For each key, compute (average score in cluster for that model) / (average score for that model overall),
        for each model present in the cluster. Exclude models not present in the cluster.
        Returns a dictionary where keys are the score keys.
        """
        # Get all score keys from the first non-empty score dict
        score_keys = []
        for s in group["score"]:
            if isinstance(s, dict) and s:
                score_keys = list(s.keys())
                break
        if not score_keys:
            return {}

        # Compute per-model average for each key in the cluster
        result = {}
        models_in_cluster = group["model"].unique()
        for key in score_keys:
            ratios = []
            for model in models_in_cluster:
                model_rows = group[group["model"] == model]
                # Average score for this model in this cluster
                cluster_avg = model_rows["score"].apply(
                    lambda x: x.get(key, 0) if isinstance(x, dict) else 0
                ).mean()
                # Average score for this model overall (from self.avg_score_per_model)
                try:
                    model_avg = self.avg_score_per_model.loc[model, key]
                except Exception:
                    model_avg = None
                if model_avg and model_avg != 0:
                    ratios.append(cluster_avg / model_avg)
            # If no valid ratios, skip
            if ratios:
                result[key] = sum(ratios) / len(ratios)
            else:
                result[key] = 0
        return result

    # ------------------------------------------------------------------
    def _compute(
        self,
        group: pd.DataFrame,
        cid: int | str,
        label: str,
        level: str,
        total_q: Dict[str, int],
        out: Dict[str, Dict[str, List[ModelStats]]],
    ) -> None:
        """Compute metrics for one cluster and add to *out*."""

        # Counts per model inside this cluster
        counts_series = (
            group[["model", "question_id"]]
            .drop_duplicates()
            .groupby("model")
            .size()
        )
        counts = counts_series.to_dict()

        props = {m: counts.get(m, 0) / total_q[m] for m in total_q}
        median_prop = np.median([v for v in props.values() if v > 0]) or 1e-9

        for model, prop in props.items():
            score = prop / median_prop if median_prop else 0.0
            quality_score = self._compute_quality_score(group)
            ms = ModelStats(
                property_description=str(label),
                model_name=str(model),
                score=float(score),
                quality_score=quality_score,
                cluster_size_global=int(len(group["id"].unique())),
                size=int(counts.get(model, 0)),
                proportion=float(prop),
                examples=self._example_props(group[group.model == model]),
                metadata={
                    "cluster_id": cid,
                    "level": level,
                },
            )
            out[model][level].append(ms)

    # ------------------------------------------------------------------
    def _sanitize_name(self, name: str) -> str:
        """Make *name* safe for filenames (remove brackets, spaces, commas)."""
        import re
        return re.sub(r"[^A-Za-z0-9._-]", "_", name)

    def _dump(self, stats: Dict[str, Dict[str, List[ModelStats]]]):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for model, by_level in stats.items():
            model_str = model if isinstance(model, str) else str(model)
            model_safe = self._sanitize_name(model_str.strip("()[]"))
            for level, items in by_level.items():
                path = self.output_dir / f"{model_safe}_{level}_metrics.jsonl"
                pd.DataFrame([i.to_dict() for i in items]).to_json(
                    path, orient="records", lines=True, force_ascii=False
                )
                self.log(f"📄 wrote {path}") 
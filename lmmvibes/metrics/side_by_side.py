"""lmmvibes.metrics.side_by_side
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute, for every property cluster, **how strongly each model exhibits the
behaviour compared to its peers** in an Arena-style side-by-side dataset.

Metric definition
-----------------

For a given model *m* and cluster *c*

```
prop(m, c)   =   #questions where m shows c   /   #questions answered by m
score(m, c)  =   prop(m, c) / median_{m'} prop(m', c)
```

Thus ``score > 1`` means the model is **over-represented** in that cluster,
``score < 1`` under-represented.
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


class SideBySideMetrics(PipelineStage, LoggingMixin, TimingMixin):
    """Metrics stage for side-by-side data."""

    def __init__(self, output_dir: str | Path | None = None):
        super().__init__()
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

        self.log("⚖️  Computing side-by-side metrics …")

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

        # Normalize model column – flatten singleton lists / tuples
        def _norm_model(val):
            if isinstance(val, (list, tuple)) and len(val) == 1:
                return val[0]
            return val

        df["model"] = df["model"].apply(_norm_model)

        # Denominator: questions answered per model
        total_q = (
            df[["model", "question_id"]].drop_duplicates().value_counts(subset=["model"]).to_dict()
        )

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
        counts = (
            group[["model", "question_id"]]
            .drop_duplicates()
            .value_counts(subset=["model"])
            .to_dict()
        )

        props = {m: counts.get(m, 0) / total_q[m] for m in total_q}
        median_prop = np.median([v for v in props.values() if v > 0]) or 1e-9

        for model, prop in props.items():
            score = prop / median_prop if median_prop else 0.0
            ms = ModelStats(
                property_description=str(label),
                model_name=str(model),
                score=float(score),
                size=int(counts.get(model, 0)),
                proportion=float(prop),
                examples=self._example_props(group[group.model == model]),
                metadata={
                    "cluster_id": cid,
                    "level": level,
                    "cluster_size_global": int(len(group["id"].unique())),
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
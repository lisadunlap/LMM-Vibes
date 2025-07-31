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

from .base_metrics import BaseMetrics
from ..core.data_objects import ModelStats


class SideBySideMetrics(BaseMetrics):
    """Metrics stage for side-by-side data."""

    def _extract_score_for_key(self, row: pd.Series, key: str) -> float:
        """Extract score with special handling for winner scores."""
        if key == "winner":
            return self.extract_winner_score(row)
        else:
            return row["score"].get(key, 0) if isinstance(row["score"], dict) else 0
    
    def extract_winner_score(self, row: pd.Series) -> float:
        """Convert winner/loser/tie to 1/-1/0."""
        winner = row["score"]["winner"]
        if winner == row["model"]:
            return 1
        elif "tie" in winner:
            return 0
        else:
            return -1

    def _compute(
        self,
        group: pd.DataFrame,
        cid: int | str,
        label: str,
        level: str,
        total_q: Dict[str, int],
        total_questions_global: int,
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

        # Compute quality score once for the whole cluster
        quality_score = self._compute_normalized_quality_score(group)

        for model, prop in props.items():
            score = prop / median_prop if median_prop else 0.0
            
            # Get confidence intervals for this model
            score_ci = None
            quality_score_ci = {}
            
            if self.compute_confidence_intervals:
                # Get distinctiveness CI
                distinctiveness_ci = self.distinctiveness_cis.get(level, {}).get(cid, {}).get(model, (0.0, 0.0))
                if distinctiveness_ci != (0.0, 0.0):
                    score_ci = {
                        "lower": distinctiveness_ci[0],
                        "upper": distinctiveness_ci[1]
                    }
                
                # Get quality score CIs
                for score_key in quality_score.keys():
                    model_ci = self.quality_score_cis.get(level, {}).get(cid, {}).get(score_key, {}).get(model, (0.0, 0.0))
                    if model_ci != (0.0, 0.0):
                        quality_score_ci[score_key] = {"lower": model_ci[0], "upper": model_ci[1]}

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
                score_ci=score_ci,
                quality_score_ci=quality_score_ci if quality_score_ci else None,
            )
            out[model][level].append(ms) 
"""
lmmvibes.metrics.single_model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module computes comprehensive metrics for every property cluster, analyzing **how strongly each model 
exhibits specific behaviors compared to its peers** in an Arena-style or single-model dataset.

Metric definitions
------------------

For a given model *m* and cluster *c*:

**Representation Score:**
    prop(m, c)   =   (# questions where m shows c) / (# questions answered by m)
    score(m, c)  =   prop(m, c) / median_{m'} prop(m', c)

**Global Cluster Frequency:**
    proportion_global(c) = (# questions in cluster c) / (# total questions across all models)

**Quality Score:**
For each score key *k* in the multi-dimensional score dictionary:
    quality_score(m, c, k) = avg_score(m, c, k) / avg_score(m, global, k)

Where:
- `avg_score(m, c, k)` = average score for model *m* in cluster *c* for key *k*
- `avg_score(m, global, k)` = average score for model *m* across all data for key *k*

**Interpretation:**
- `score > 1`: Model is **over-represented** in that cluster
- `score < 1`: Model is **under-represented** in that cluster  
- `proportion_global`: Global frequency of this cluster across all questions
- `quality_score > 1`: Model performs better in this cluster than its global average for key *k*
- `quality_score < 1`: Model performs worse in this cluster than its global average for key *k*

The metrics support multi-dimensional scoring where each question can have multiple
evaluation criteria (e.g., accuracy, helpfulness, harmlessness) stored as dictionary keys.

**Updates:**
- Confidence intervals for both representation and quality scores are computed via bootstrapping if enabled.
- Statistical significance is determined for each score and quality score key based on confidence intervals.
- Results are saved in a JSON artifact and optionally logged to wandb, including per-model and per-cluster statistics.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .base_metrics import BaseMetrics
from ..core.data_objects import ModelStats


class SingleModelMetrics(BaseMetrics):
    """Metrics stage for single-model data."""

    def _extract_score_for_key(self, row: pd.Series, key: str) -> float:
        """Extract score directly from score dict."""
        return row["score"].get(key, 0) if isinstance(row["score"], dict) else 0

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

        # Calculate proportion of questions answered by each model in this cluster
        # This is (# questions where m shows c) / (# questions answered by m)
        # This is the numerator for the representation score
        prop_global_per_model = {}
        for model in counts:
            if total_q.get(model, 0) > 0:
                prop_global_per_model[model] = counts[model] / total_q[model]
            else:
                prop_global_per_model[model] = 0.0

        # Calculate the median of these proportions for normalization
        median_prop_global = np.median([v for v in prop_global_per_model.values() if v > 0]) or 1e-9

        # Cluster size for information (not stored in ModelStats anymore)
        cluster_size_global = len(group["question_id"].unique())

        # Compute quality score once for the whole cluster
        quality_score = self._compute_normalized_quality_score(group)

        for model, prop_global in prop_global_per_model.items():
            # Skip models with no examples in this cluster
            if counts.get(model, 0) == 0:
                continue
            
            # Get confidence intervals for this model
            score_ci = None
            quality_score_ci = {}
            
            # Initialize score and quality_score
            score = prop_global / median_prop_global if median_prop_global else 0.0
            quality_score_for_model = quality_score.copy()  # Make a copy to modify per model
            
            if self.compute_confidence_intervals:
                # Get distinctiveness CI
                distinctiveness_ci = self.distinctiveness_cis.get(level, {}).get(cid, {}).get(model, (0.0, 0.0))
                if distinctiveness_ci != (0.0, 0.0):
                    score_ci = {
                        "lower": distinctiveness_ci[0],
                        "upper": distinctiveness_ci[1],
                        "average": (distinctiveness_ci[0] + distinctiveness_ci[1]) / 2
                    }
                    # Use bootstrap average as the point estimate
                    bootstrap_average = self.distinctiveness_averages.get(level, {}).get(cid, {}).get(model, None)
                    if bootstrap_average is not None:
                        score = bootstrap_average
                    else:
                        score = score_ci["average"]
                
                # Get quality score CIs and update quality_score_for_model
                for score_key in quality_score.keys():
                    model_ci = self.quality_score_cis.get(level, {}).get(cid, {}).get(score_key, {}).get(model, (0.0, 0.0))
                    if model_ci != (0.0, 0.0):
                        quality_score_ci[score_key] = {
                            "lower": model_ci[0], 
                            "upper": model_ci[1],
                            "average": (model_ci[0] + model_ci[1]) / 2
                        }
                        # Use bootstrap average as the point estimate for this quality score key
                        bootstrap_average = self.quality_score_averages.get(level, {}).get(cid, {}).get(score_key, {}).get(model, None)
                        if bootstrap_average is not None:
                            quality_score_for_model[score_key] = bootstrap_average
                        else:
                            quality_score_for_model[score_key] = quality_score_ci[score_key]["average"]

            ms = ModelStats(
                property_description=str(label),
                model_name=str(model),
                score=float(score),
                quality_score=quality_score_for_model,
                size=int(counts.get(model, 0)),
                proportion=float(prop_global),
                cluster_size=cluster_size_global,
                
                examples=self._example_props(group[group.model == model]),
                metadata={
                    "cluster_id": cid,
                    "level": level,
                },
                score_ci=score_ci,
                quality_score_ci=quality_score_ci if quality_score_ci else None,
            )
            out[model][level].append(ms) 
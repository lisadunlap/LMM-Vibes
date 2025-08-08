"""lmmvibes.metrics.side_by_side
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Side-by-side metrics implemented on top of the functional metrics pipeline.

This adapts the Arena-style pairwise inputs by expanding each conversation into
per-model rows and converting the 'winner' field into a numeric score per model
(+1 winner, -1 loser, 0 tie). Other numeric quality metrics in the score dict
are preserved as-is if present.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .functional_metrics import FunctionalMetrics


class SideBySideMetrics(FunctionalMetrics):
    """Metrics stage for side-by-side data using functional metrics.

    The output artifacts and wandb logging are identical to `FunctionalMetrics`.
    """

    def __init__(
        self,
        output_dir: str | None = None,
        compute_bootstrap: bool = True,
        bootstrap_samples: int = 100,
        log_to_wandb: bool = True,
        generate_plots: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            output_dir=output_dir,
            compute_bootstrap=compute_bootstrap,
            bootstrap_samples=bootstrap_samples,
            log_to_wandb=log_to_wandb,
            generate_plots=generate_plots,
            **kwargs,
        )

    def _prepare_data(self, data) -> pd.DataFrame:
        """Prepare SxS data: expand each pair into two rows (one per model).

        Produces the same schema expected by FunctionalMetrics:
        columns: [conversation_id, conversation_metadata, property_metadata, model, cluster, property_description, scores]
        """
        # Extract clusters and properties data
        if not data.clusters:
            return pd.DataFrame()

        properties = pd.DataFrame([cluster.to_dict() for cluster in data.clusters])

        # Explode property_descriptions and question_ids
        properties = properties.explode(["property_descriptions", "question_ids"]).drop_duplicates(
            subset=["property_descriptions", "question_ids"]
        )
        properties = properties.dropna(subset=["property_descriptions", "question_ids"]).rename(
            {"question_ids": "question_id", "property_descriptions": "property_description"}, axis=1
        )

        # Expand conversations: one row per model with per-model scores
        expanded_rows: List[Dict[str, Any]] = []
        for conv in data.conversations:
            qid = conv.question_id
            meta = conv.meta
            scores = conv.scores or {}

            # Side-by-side: conv.model is a list/tuple of two models
            if isinstance(conv.model, (list, tuple)) and len(conv.model) == 2:
                model_a, model_b = conv.model[0], conv.model[1]
                expanded_rows.append(
                    {
                        "question_id": qid,
                        "scores": self._transform_scores_for_model(scores, model_a, model_b),
                        "meta": meta,
                        "model": model_a,
                    }
                )
                expanded_rows.append(
                    {
                        "question_id": qid,
                        "scores": self._transform_scores_for_model(scores, model_b, model_a),
                        "meta": meta,
                        "model": model_b,
                    }
                )
            else:
                # Fallback to single-model row
                model_name = conv.model if isinstance(conv.model, str) else str(conv.model)
                expanded_rows.append(
                    {
                        "question_id": qid,
                        "scores": scores,
                        "meta": meta,
                        "model": model_name,
                    }
                )

        conversations = pd.DataFrame(expanded_rows)

        # Join conversations with properties
        properties = properties.merge(conversations, on="question_id", how="left").rename(
            {"meta": "conversation_metadata", "label": "cluster", "question_id": "conversation_id"},
            axis=1,
        )
        properties["property_metadata"] = properties["property_description"].apply(
            lambda x: {"property_description": x}
        )

        important_columns = [
            "conversation_id",
            "conversation_metadata",
            "property_metadata",
            "model",
            "cluster",
            "property_description",
            "scores",
        ]
        properties = properties[important_columns]
        return properties

    @staticmethod
    def _transform_scores_for_model(all_scores: Dict[str, Any], this_model: str, other_model: str) -> Dict[str, float]:
        """Convert the side-by-side score dict into per-model numeric scores.

        - "winner": +1 if this_model won, -1 if lost, 0 if tie
        - Preserve other numeric keys as floats when possible
        """
        result: Dict[str, float] = {}
        if isinstance(all_scores, dict):
            # Winner conversion
            winner = all_scores.get("winner")
            if isinstance(winner, str):
                if winner == this_model:
                    result["winner"] = 1.0
                elif "tie" in winner:
                    result["winner"] = 0.0
                else:
                    result["winner"] = -1.0

            # Copy other numeric metrics if present
            for k, v in all_scores.items():
                if k == "winner":
                    continue
                if isinstance(v, (int, float)):
                    result[k] = float(v)
        return result 
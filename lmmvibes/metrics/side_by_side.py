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
        
        # Ensure conversation_metadata exists - fill missing values with empty dict
        if "conversation_metadata" not in properties.columns:
            properties["conversation_metadata"] = {}
        else:
            properties["conversation_metadata"] = properties["conversation_metadata"].fillna({})
        
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
        
        # Ensure all required columns exist before filtering
        for col in important_columns:
            if col not in properties.columns:
                if col == "scores":
                    properties[col] = {}
                elif col == "model":
                    properties[col] = "unknown"
                else:
                    properties[col] = ""
        
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

    # --- Robust metrics computation for SxS to handle empty bootstrap subsets ---
    def _infer_metric_keys(self, df: pd.DataFrame) -> List[str]:
        """Infer score metric keys from any available non-empty scores dict in df."""
        if df is None or df.empty or "scores" not in df.columns:
            return []
        for val in df["scores"]:
            if isinstance(val, dict) and val:
                return list(val.keys())
        return []

    def compute_cluster_metrics(self, df: pd.DataFrame, clusters: List[str] | str, models: List[str] | str) -> Dict[str, Any]:
        """Override to avoid indexing into empty DataFrames during bootstrap.

        Mirrors FunctionalMetrics.compute_cluster_metrics but with guards for
        empty model subsets and key alignment without assertions.
        """
        if isinstance(clusters, str):
            clusters = [clusters]
        if isinstance(models, str):
            models = [models]

        model_df = df[df["model"].isin(models)]
        if model_df.empty:
            metric_keys = self._infer_metric_keys(df)
            return self.empty_metrics(metric_keys)

        cluster_model_df = model_df[model_df["cluster"].isin(clusters)]

        # Determine metric keys from available rows
        metric_keys = self._infer_metric_keys(model_df)
        if not metric_keys:
            metric_keys = self._infer_metric_keys(df)

        if len(cluster_model_df) == 0:
            return self.empty_metrics(metric_keys)

        # Compute sizes and raw quality scores
        model_size, model_scores = self.compute_size_and_score(model_df)
        cluster_model_size, cluster_model_scores = self.compute_size_and_score(cluster_model_df)

        # Align keys without asserting strict equality
        all_keys = set(metric_keys) | set(model_scores.keys()) | set(cluster_model_scores.keys())
        for k in all_keys:
            if k not in model_scores:
                model_scores[k] = 0.0
            if k not in cluster_model_scores:
                cluster_model_scores[k] = 0.0

        quality_delta = self.compute_relative_quality(cluster_model_scores, model_scores)
        proportion = cluster_model_size / model_size if model_size != 0 else 0

        return {
            "size": cluster_model_size,
            "proportion": proportion,
            "quality": cluster_model_scores,
            "quality_delta": quality_delta,
            "examples": list(
                zip(
                    cluster_model_df["conversation_id"],
                    cluster_model_df["conversation_metadata"],
                    cluster_model_df["property_metadata"],
                )
            ),
        } 
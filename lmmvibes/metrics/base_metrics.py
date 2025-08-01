"""
lmmvibes.metrics.base_metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Base class for metrics computation, defining the core logic for calculating
per-model and aggregate statistics for behavioral property clusters.

Metric Definitions
------------------

Metrics are computed for each model and aggregated globally under an "all" key.

**1. Per-Model Metrics:**

For a given model *m* and cluster *c*:

- **Distinctiveness Score:**
  - `prop(m, c) = (# unique questions where m exhibits c) / (# total unique questions for m)`
  - `score(m, c) = prop(m, c) / median_{m'} prop(m', c)`
  - A score > 1 indicates the model is **over-represented** in that cluster.

- **Quality Score:**
  - For each quality metric *k* (e.g., 'BERTScore-F'), the score is the average
    value of *k* for all conversations belonging to model *m* within cluster *c*.
  - This is a direct average, not normalized by the model's global average.

- **Proportion:**
  - `proportion(m, c) = (# properties from m in c) / (total # properties from m)`

- **Confidence Intervals & Significance:**
  - If enabled, 95% confidence intervals are computed for both distinctiveness
    and quality scores via bootstrapping.
  - Statistical significance is determined based on these CIs (e.g., for
    distinctiveness, if the CI is entirely above 1.0).

**2. Aggregate ("all") Metrics:**

For each cluster *c*, statistics are aggregated across all models:

- **Distinctiveness Score:**
  - The **maximum** distinctiveness score observed for cluster *c* across all
    individual models is chosen. The CI and significance flag from that top-
    performing model are carried over.

- **Quality Score:**
  - A **weighted average** of the quality score for each metric *k* is computed,
    weighted by the number of properties (`size`) each model contributed to the cluster.

- **Size & Proportion:**
  - `size` is the total count of all properties in the cluster from all models.
  - `proportion` is `size / (total properties across all models)`.

- **Examples:**
  - A combined list of all example property IDs from all models in the cluster.

The final output is a `model_stats.json` file containing a dictionary where keys
are model names, plus an additional key `"all"` for the aggregate statistics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from ..core.stage import PipelineStage
from ..core.mixins import LoggingMixin, TimingMixin
from ..core.data_objects import PropertyDataset, ModelStats
from .utils import wandb_logging


class BaseMetrics(PipelineStage, LoggingMixin, TimingMixin, ABC):
    """Base class for metrics computation with shared functionality."""

    def __init__(self, output_dir: str | Path | None = None, compute_confidence_intervals: bool = False, bootstrap_samples: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.output_dir = Path(output_dir) if output_dir else None
        self.compute_confidence_intervals = compute_confidence_intervals
        self.bootstrap_samples = bootstrap_samples

    # ------------------------------------------------------------------
    @staticmethod
    def _example_props(group: pd.DataFrame, n: int = 3) -> List[str]:
        return (
            group["id"].dropna().unique().tolist()[:n]
            if "id" in group.columns else []
        )

    # ------------------------------------------------------------------
    @abstractmethod
    def _extract_score_for_key(self, row: pd.Series, key: str) -> float:
        """Extract score for a given key from a row. To be implemented by subclasses."""
        pass

    # ------------------------------------------------------------------
    def run(self, data: PropertyDataset) -> PropertyDataset:
        """Compute fine & coarse metrics and attach them to *data*."""

        self.log("⚖️  Computing metrics …")

        df = data.to_dataframe(type="clusters")
        if df.empty:
            self.log("No cluster info found; skipping metrics stage.")
            return data

        # Store the full DataFrame for use in normalization
        self.full_df = df.copy()

        # Print diagnostic information about models
        print(f"\n🔍 Metrics stage diagnostic:")
        print(f"   • Input dataset has {len(data.all_models)} models: {sorted(data.all_models)}")
        print(f"   • DataFrame has {len(df)} rows")
        if 'model' in df.columns:
            df_models = df['model'].unique()
            print(f"   • DataFrame has {len(df_models)} unique models: {sorted(df_models)}")
            
            # Check for models that are missing from the DataFrame
            missing_models = set(data.all_models) - set(df_models)
            if missing_models:
                print(f"   • ⚠️  Missing models from DataFrame: {sorted(missing_models)}")
                print(f"   • This suggests these models had no valid properties after extraction/validation")
                
                # Raise ValueError if entire models are missing - this indicates a serious pipeline issue
                raise ValueError(
                    f"\n" + "="*60 + "\n"
                    f"ERROR: Entire models missing from metrics computation!\n"
                    f"="*60 + "\n"
                    f"The following models were completely filtered out before reaching the metrics stage:\n"
                    f"  {sorted(missing_models)}\n\n"
                    f"="*60
                )

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
            
            # Print diagnostic about the filtering
            print(f"   • After filtering invalid model names: {len(df)} rows")
            if 'model' in df.columns:
                remaining_models = df['model'].unique()
                print(f"   • Remaining models: {sorted(remaining_models)}")

        self.log(f"Models used for metrics: {df['model'].unique()}")
        
        # Final diagnostic summary
        print(f"   • Final models for metrics computation: {sorted(df['model'].unique())}")
        print()

        # ------------------------------------------------------------------
        # Calculate denominator: unique questions answered per model
        # ------------------------------------------------------------------
        total_q_series = (
            df[["model", "question_id"]]
            .drop_duplicates()
            .groupby("model")
            .size()
        )
        total_q = total_q_series.to_dict()

        # Calculate total number of unique questions across all models
        total_questions_global = len(df["question_id"].unique())

        # Calculate the average score per model
        # If score is a dict with multiple keys, compute the average for each key per model
        # First, get all possible keys in the score dicts
        all_score_keys = set()
        for s in df["score"]:
            if isinstance(s, dict):
                all_score_keys.update(s.keys())
        all_score_keys = sorted(all_score_keys)

        # For each key, compute the average score per model
        avg_scores_per_key = {}
        for key in all_score_keys:
            # Extract the score for this key using the subclass-specific method
            df[f"score_{key}"] = df.apply(lambda row: self._extract_score_for_key(row, key), axis=1)
            avg_scores_per_key[key] = df.groupby("model")[f"score_{key}"].mean()
            print(f"Average score for key '{key}':")
            print(avg_scores_per_key[key])

        # Optionally, you could aggregate these into a DataFrame:
        self.avg_score_per_model = pd.DataFrame(avg_scores_per_key)
        print("Average scores per model (all keys):")
        print(self.avg_score_per_model)

        # Compute global min/max values for each score key (for normalization)
        self.global_score_stats = {}
        for key in all_score_keys:
            # Use vectorized operations instead of iterrows()
            scores_series = df.apply(lambda row: self._extract_score_for_key(row, key), axis=1)
            
            # Filter out zeros (which might be missing values)
            valid_scores = scores_series[scores_series != 0]
            
            if len(valid_scores) > 0:
                global_min = valid_scores.min()
                global_max = valid_scores.max()
                # Apply the hack: if min > 0, set min = 0
                if global_min > 0:
                    global_min = 0
                self.global_score_stats[key] = {
                    "min": global_min,
                    "max": global_max,
                    "range": global_max - global_min
                }
            else:
                self.global_score_stats[key] = {
                    "min": 0,
                    "max": 1,
                    "range": 1
                }

        stats: Dict[str, Dict[str, List[ModelStats]]] = defaultdict(lambda: {"fine": [], "coarse": []})

        # Compute confidence intervals once for the entire dataset (if enabled)
        if self.compute_confidence_intervals:
            self.distinctiveness_cis, self.quality_score_cis, self.distinctiveness_averages, self.quality_score_averages = self._bootstrap_whole_dataset_confidence_intervals(df, total_q)
        else:
            self.distinctiveness_cis, self.quality_score_cis, self.distinctiveness_averages, self.quality_score_averages = {}, {}, {}, {}

        # ---------------- Fine clusters ----------------
        fine_clusters = list(df.groupby("fine_cluster_id"))
        self.log(f"Computing metrics for {len(fine_clusters)} fine clusters...")
        
        for i, (cid, group) in enumerate(fine_clusters):
            self.log(f"  Processing fine cluster {i+1}/{len(fine_clusters)}: {cid}")
            lbl = group["fine_cluster_label"].iloc[0]
            self._compute(group, cid, lbl, "fine", total_q, total_questions_global, stats)

        # ---------------- Coarse clusters --------------
        if (df["coarse_cluster_id"] != -1).any():
            coarse_clusters = [(cid, group) for cid, group in df.groupby("coarse_cluster_id") if cid != -1]
            self.log(f"Computing metrics for {len(coarse_clusters)} coarse clusters...")
            
            for i, (cid, group) in enumerate(coarse_clusters):
                self.log(f"  Processing coarse cluster {i+1}/{len(coarse_clusters)}: {cid}")
                lbl = group["coarse_cluster_label"].iloc[0]
                self._compute(group, cid, lbl, "coarse", total_q, total_questions_global, stats)

        # Sort stats lists by score descending for deterministic order
        for m in stats:
            for lvl in ("fine", "coarse"):
                stats[m][lvl].sort(key=lambda s: s.score, reverse=True)

        # Compute global stats for each model (this is now optimized)
        global_stats = self._compute_global_stats(df)
        
        # Add global stats to each model's stats
        for model in stats:
            stats[model]["stats"] = global_stats.get(model, {})

        # ------------------------------------------------------------------
        # NEW: Compute "all" (global) metrics across every model
        # ------------------------------------------------------------------
        all_stats = self._compute_all_metrics(stats)
        stats["all"] = all_stats
        
        data.model_stats = stats  # type: ignore[assignment]

        if self.compute_confidence_intervals:
            self.log(f"📊 Confidence intervals computed using {self.bootstrap_samples} bootstrap samples")
            self._compute_statistical_significance(stats)

        # Save metrics results if output_dir is provided
        if self.output_dir:
            self._dump(stats)

        self.log(f"✅ Metrics computed for {len(stats)} models")
        
        # Print top clusters for each model
        self._print_top_clusters(stats)
        
        return data

    # ------------------------------------------------------------------
    @abstractmethod
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
        """Compute metrics for one cluster and add to *out*. To be implemented by subclasses."""
        pass

    # ------------------------------------------------------------------
    def _compute_quality_score(self, group: pd.DataFrame) -> dict:
        """
        Compute the quality score of a cluster for each score key and model:
        For each key, compute (average score in cluster for that model) / (average score for that model overall),
        for each model present in the cluster. Exclude models not present in the cluster.
        Returns a dictionary where keys are the score keys and values are dictionaries of model -> ratio.
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
            model_ratios = {}
            for model in models_in_cluster:
                model_rows = group[group["model"] == model]
                # Average score for this model in this cluster
                cluster_avg = model_rows.apply(lambda row: self._extract_score_for_key(row, key), axis=1).mean()
                # Average score for this model overall (from self.avg_score_per_model)
                try:
                    model_avg = self.avg_score_per_model.loc[model, key]
                except Exception:
                    model_avg = None
                if model_avg and model_avg != 0:
                    model_ratios[model] = cluster_avg / model_avg
                else:
                    model_ratios[model] = 0
            result[key] = model_ratios
        return result

    def _compute_normalized_quality_score(self, group: pd.DataFrame) -> dict:
        """
        Compute normalized quality scores that are centered around zero.
        
        **Normalization Method:**
        For each score key, we compute:
        1. Cluster average: avg(score for all examples in this cluster)
        2. Global average for each model: avg(score for that model across all data)
        3. Normalized score: (cluster_avg - model_global_avg) / global_range
        
        **Interpretation:**
        - Values > 0: Model performs better in this cluster than its global average
        - Values < 0: Model performs worse in this cluster than its global average
        - Values = 0: Model performs exactly at its global average in this cluster
        
        **Benefits:**
        - Zero-centered scale makes it easy to see relative performance
        - Positive values indicate above-average performance in this cluster
        - Negative values indicate below-average performance in this cluster
        - Normalization by global range keeps scores comparable across metrics
        
        Returns a dictionary where keys are score keys and values are normalized scores.
        """
        # Get all score keys from the first non-empty score dict
        score_keys = []
        for s in group["score"]:
            if isinstance(s, dict) and s:
                score_keys = list(s.keys())
                break
        if not score_keys:
            return {}

        result = {}
        for key in score_keys:
            # Use vectorized operations instead of iterrows()
            scores_series = group.apply(lambda row: self._extract_score_for_key(row, key), axis=1)
            
            # Calculate cluster average using vectorized operations
            cluster_avg = scores_series.mean()
            
            # Calculate model-specific global averages for models in this cluster
            models_in_cluster = group["model"].unique()
            model_global_avgs = []
            
            for model in models_in_cluster:
                try:
                    model_global_avg = self.avg_score_per_model.loc[model, key]
                    model_global_avgs.append(model_global_avg)
                except Exception:
                    # If model not found, skip
                    continue
            
            # Use the average of model global averages as the baseline
            if model_global_avgs:
                baseline_avg = np.mean(model_global_avgs)
            else:
                baseline_avg = 0
            
            # Use pre-computed global stats for normalization
            if key in self.global_score_stats:
                global_range = self.global_score_stats[key]["range"]
                
                # Normalize using the new formula
                if global_range > 0:
                    normalized_score = (cluster_avg - baseline_avg) / global_range
                else:
                    normalized_score = 0  # If all values are the same, no difference
            else:
                normalized_score = 0  # Fallback if key not found
            
            result[key] = normalized_score
        
        return result

    def _compute_global_stats(self, df: pd.DataFrame) -> dict:
        """
        Compute global statistics for each model across all examples.
        Returns a dictionary with model names as keys and average scores for each metric as values.
        """
        global_stats = {}
        
        # Get all score keys
        all_score_keys = set()
        for s in df["score"]:
            if isinstance(s, dict):
                all_score_keys.update(s.keys())
        all_score_keys = sorted(all_score_keys)
        
        # Use vectorized operations instead of iterrows()
        for key in all_score_keys:
            # Extract scores for this key using vectorized operations
            scores_series = df.apply(lambda row: self._extract_score_for_key(row, key), axis=1)
            
            # Group by model and compute mean
            model_means = scores_series.groupby(df["model"]).mean()
            
            # Store results
            for model in model_means.index:
                if model not in global_stats:
                    global_stats[model] = {}
                global_stats[model][key] = model_means[model]
        
        return global_stats

    def _bootstrap_whole_dataset_confidence_intervals(
        self, 
        df: pd.DataFrame,
        total_q: Dict[str, int],
        confidence_level: float = 0.95
    ) -> tuple[Dict[str, Dict[str, Dict[str, tuple[float, float]]]], Dict[str, Dict[str, Dict[str, Dict[str, tuple[float, float]]]]], Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]:
        """
        Compute confidence intervals by bootstrapping the entire dataset.
        
        This is much more efficient than the previous approach because:
        1. We sample the entire dataset once per bootstrap iteration
        2. We recalculate all metrics for all clusters in that sample
        3. We collect statistics across all bootstrap samples
        
        Returns:
            Tuple of (distinctiveness_cis, quality_score_cis, distinctiveness_averages, quality_score_averages) where:
            - distinctiveness_cis[level][cluster_id][model] = (lower, upper)
            - quality_score_cis[level][cluster_id][score_key][model] = (lower, upper)
            - distinctiveness_averages[level][cluster_id][model] = average
            - quality_score_averages[level][cluster_id][score_key][model] = average
        """
        if not self.compute_confidence_intervals:
            return {}, {}, {}, {}
        
        self.log(f"🔄 Computing confidence intervals using whole-dataset bootstrapping with {self.bootstrap_samples} samples...")
        
        # Store bootstrap results
        bootstrap_distinctiveness = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # [level][cluster_id][model] = [scores]
        bootstrap_quality_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))  # [level][cluster_id][score_key][model] = [scores]
        
        # Progress tracking
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=self.bootstrap_samples, desc="Bootstrap samples", unit="samples")
        except ImportError:
            progress_bar = None
            self.log(f"Computing {self.bootstrap_samples} bootstrap samples...")
        
        for i in range(self.bootstrap_samples):
            # Sample the entire dataset with replacement
            bootstrap_indices = np.random.choice(len(df), size=len(df), replace=True)
            bootstrap_df = df.iloc[bootstrap_indices].copy()
            bootstrap_df["qid_bootstrap"] = bootstrap_df["question_id"].astype(str) + "_" + bootstrap_df.groupby("question_id").cumcount().astype(str)
            
            # Recalculate total questions per model for this bootstrap sample
            bootstrap_total_q = (
                bootstrap_df[["model", "qid_bootstrap"]]
                .drop_duplicates()
                .groupby("model")
                .size()
                .to_dict()
            )
            
            # Process fine clusters
            for cid, group in bootstrap_df.groupby("fine_cluster_id"):
                if len(group) == 0:
                    continue
                    
                # Calculate distinctiveness scores using the same logic as _compute
                counts_series = (
                    group[["model", "qid_bootstrap"]]
                    .drop_duplicates()
                    .groupby("model")
                    .size()
                )
                counts = counts_series.to_dict()
                
                # Calculate proportion of questions answered by each model in this cluster
                prop_global_per_model = {}
                for model in counts:
                    if bootstrap_total_q.get(model, 0) > 0:
                        prop_global_per_model[model] = counts[model] / bootstrap_total_q[model]
                    else:
                        prop_global_per_model[model] = 0.0

                # Calculate the median of these proportions for normalization
                median_prop_global = np.median([v for v in prop_global_per_model.values() if v > 0]) or 1e-9
                
                for model, prop_global in prop_global_per_model.items():
                    if model in bootstrap_total_q:  # Only include models present in this bootstrap sample
                        score = prop_global / median_prop_global if median_prop_global else 0.0
                        bootstrap_distinctiveness["fine"][cid][model].append(score)
                
                # Calculate quality scores
                quality_scores = self._compute_normalized_quality_score(group)
                for score_key, normalized_score in quality_scores.items():
                    models_in_cluster = group["model"].unique()
                    for model in models_in_cluster:
                        if model in bootstrap_total_q:  # Only include models present in this bootstrap sample
                            bootstrap_quality_scores["fine"][cid][score_key][model].append(normalized_score)
            
            # Process coarse clusters
            if (bootstrap_df["coarse_cluster_id"] != -1).any():
                for cid, group in bootstrap_df.groupby("coarse_cluster_id"):
                    if cid == -1 or len(group) == 0:
                        continue
                        
                    # Calculate distinctiveness scores using the same logic as _compute
                    counts_series = (
                        group[["model", "qid_bootstrap"]]
                        .drop_duplicates()
                        .groupby("model")
                        .size()
                    )
                    counts = counts_series.to_dict()
                    
                    # Calculate proportion of questions answered by each model in this cluster
                    prop_global_per_model = {}
                    for model in counts:
                        if bootstrap_total_q.get(model, 0) > 0:
                            prop_global_per_model[model] = counts[model] / bootstrap_total_q[model]
                        else:
                            prop_global_per_model[model] = 0.0

                    # Calculate the median of these proportions for normalization
                    median_prop_global = np.median([v for v in prop_global_per_model.values() if v > 0]) or 1e-9
                    
                    for model, prop_global in prop_global_per_model.items():
                        if model in bootstrap_total_q:  # Only include models present in this bootstrap sample
                            score = prop_global / median_prop_global if median_prop_global else 0.0
                            bootstrap_distinctiveness["coarse"][cid][model].append(score)
                    
                    # Calculate quality scores
                    quality_scores = self._compute_normalized_quality_score(group)
                    for score_key, normalized_score in quality_scores.items():
                        models_in_cluster = group["model"].unique()
                        for model in models_in_cluster:
                            if model in bootstrap_total_q:  # Only include models present in this bootstrap sample
                                bootstrap_quality_scores["coarse"][cid][score_key][model].append(normalized_score)
            
            # Update progress
            if progress_bar:
                progress_bar.update(1)
            elif i % 100 == 0:
                self.log(f"Bootstrap progress: {i}/{self.bootstrap_samples} samples ({i/self.bootstrap_samples*100:.1f}%)")
        
        if progress_bar:
            progress_bar.close()
        
        # Calculate confidence intervals and averages from bootstrap distributions
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        # Convert to final format
        distinctiveness_cis = {}
        quality_score_cis = {}
        distinctiveness_averages = {}
        quality_score_averages = {}
        
        for level in bootstrap_distinctiveness:
            distinctiveness_cis[level] = {}
            distinctiveness_averages[level] = {}
            for cluster_id in bootstrap_distinctiveness[level]:
                distinctiveness_cis[level][cluster_id] = {}
                distinctiveness_averages[level][cluster_id] = {}
                for model in bootstrap_distinctiveness[level][cluster_id]:
                    scores = bootstrap_distinctiveness[level][cluster_id][model]
                    if len(scores) > 0:
                        lower = np.percentile(scores, lower_percentile)
                        upper = np.percentile(scores, upper_percentile)
                        average = np.mean(scores)
                        distinctiveness_cis[level][cluster_id][model] = (float(lower), float(upper))
                        distinctiveness_averages[level][cluster_id][model] = float(average)
                    else:
                        distinctiveness_cis[level][cluster_id][model] = (0.0, 0.0)
                        distinctiveness_averages[level][cluster_id][model] = 0.0
        
        for level in bootstrap_quality_scores:
            quality_score_cis[level] = {}
            quality_score_averages[level] = {}
            for cluster_id in bootstrap_quality_scores[level]:
                quality_score_cis[level][cluster_id] = {}
                quality_score_averages[level][cluster_id] = {}
                for score_key in bootstrap_quality_scores[level][cluster_id]:
                    quality_score_cis[level][cluster_id][score_key] = {}
                    quality_score_averages[level][cluster_id][score_key] = {}
                    for model in bootstrap_quality_scores[level][cluster_id][score_key]:
                        scores = bootstrap_quality_scores[level][cluster_id][score_key][model]
                        if len(scores) > 0:
                            lower = np.percentile(scores, lower_percentile)
                            upper = np.percentile(scores, upper_percentile)
                            average = np.mean(scores)
                            quality_score_cis[level][cluster_id][score_key][model] = (float(lower), float(upper))
                            quality_score_averages[level][cluster_id][score_key][model] = float(average)
                        else:
                            quality_score_cis[level][cluster_id][score_key][model] = (0.0, 0.0)
                            quality_score_averages[level][cluster_id][score_key][model] = 0.0
        
        self.log(f"✅ Confidence intervals computed for {len(distinctiveness_cis)} levels")
        return distinctiveness_cis, quality_score_cis, distinctiveness_averages, quality_score_averages
    
    def _compute_statistical_significance(self, model_stats: Dict[str, Dict[str, List[ModelStats]]]) -> None:
        """
        Compute statistical significance between models for each score key.
        For quality, if the confidence interval does not contain 0, then the model is statistically significantly better than the others.
        For distinctiveness, if the confidence interval is above 1, then the model is statistically significantly better than the others.
        """
        print("--------------------------------")
        print("Computing statistical significance")
        print("--------------------------------")
        print(f"🔍 Computing statistical significance for {len(model_stats)} models...")
        
        for model_name, model_data in model_stats.items():
            for level in ["fine", "coarse"]:
                if level in model_data and model_data[level]:
                    print(f"  Processing {level} clusters for {model_name} ({len(model_data[level])} clusters)")
                    for stat in model_data[level]:
                        print(stat)
                        # Handle quality score statistical significance
                        quality_statistical_significance = {}
                        print(f"    Quality CIs available for {len(stat.quality_score_ci)} keys")
                        for key, ci_data in stat.quality_score_ci.items():
                            ci_lower = ci_data['lower']
                            ci_upper = ci_data['upper']
                            # if the interval does not contain 0, then the model is statistically significantly different
                            if (ci_lower < 0 and ci_upper < 0) or (ci_lower > 0 and ci_upper > 0):
                                print(f"    {stat.property_description} is statistically significantly different")
                                quality_statistical_significance[key] = True
                            else:
                                quality_statistical_significance[key] = False
                        stat.quality_score_statistical_significance = quality_statistical_significance

                        # Handle distinctiveness score statistical significance
                        ci_lower = stat.score_ci['lower']
                        ci_upper = stat.score_ci['upper']
                        # if the confidence interval is entirely above 1, then the model is statistically significantly over-represented
                        if ci_lower > 1 and ci_upper > 1:
                            print(f"    {model_name} is statistically significantly over-represented")
                            stat.score_statistical_significance = True
                        else:
                            stat.score_statistical_significance = False
                        
                        # Debug: Print the final statistical significance values
                        print(f"    Final score_statistical_significance: {stat.score_statistical_significance}")
                        print(f"    Final quality_score_statistical_significance: {stat.quality_score_statistical_significance}")

    # ------------------------------------------------------------------
    def _sanitize_name(self, name: str) -> str:
        """Make *name* safe for filenames (remove brackets, spaces, commas)."""
        import re
        return re.sub(r"[^A-Za-z0-9._-]", "_", name)

    def _dump(self, stats: Dict[str, Dict[str, List[ModelStats]]]):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save a single model_stats.json file for backward compatibility
        import json
        
        # Convert ModelStats objects to dictionaries for JSON serialization
        stats_for_json = {}
        for model_name, model_stats in stats.items():
            stats_for_json[str(model_name)] = {
                "fine": [stat.to_dict() for stat in model_stats["fine"]]
            }
            if "coarse" in model_stats:
                stats_for_json[str(model_name)]["coarse"] = [stat.to_dict() for stat in model_stats["coarse"]]
            # Add global stats if they exist - handle different data types
            if "stats" in model_stats:
                stats_data = model_stats["stats"]
                # Ensure stats is serializable - convert to dict if needed
                if hasattr(stats_data, 'to_dict'):
                    stats_for_json[str(model_name)]["stats"] = stats_data.to_dict()
                elif isinstance(stats_data, dict):
                    stats_for_json[str(model_name)]["stats"] = stats_data
                else:
                    # For other types, convert to string representation
                    stats_for_json[str(model_name)]["stats"] = str(stats_data)
        
        # Save the combined model_stats.json file
        model_stats_path = self.output_dir / "model_stats.json"
        with open(model_stats_path, 'w') as f:
            json.dump(stats_for_json, f, indent=2)
        self.log(f"📄 wrote {model_stats_path}") 

        # log the model_stats.json file to wandb
        wandb_logging(stats_for_json)

    def _print_top_clusters(self, stats: Dict[str, Dict[str, List[ModelStats]]]):
        """Prints the top cluster for each model based on their score."""
        print("\n🏆 Top clusters for each model:")
        for model, by_level in stats.items():
            model_str = model if isinstance(model, str) else str(model)
            model_safe = self._sanitize_name(model_str.strip("()[]"))
            print(f"\n--- {model_safe} ---")
            
            # Find the top cluster for each level
            for level in ("fine", "coarse"):
                if by_level[level]:
                    top_cluster = by_level[level][0]
                    print(f"  Top {level} cluster: {top_cluster.property_description} (Score: {top_cluster.score:.2f})")
                else:
                    print(f"  No {level} clusters found for {model_str}") 

    # ------------------------------------------------------------------
    def _compute_all_metrics(self, per_model_stats: Dict[str, Dict[str, List[ModelStats]]]) -> Dict[str, List[ModelStats]]:
        """Aggregate per-model stats to produce a global "all" entry.

        For each cluster we merge all examples, average quality metrics, sum sizes, and
        choose the *max* distinctiveness score among models (carrying its CI & significance).
        """
        # Collect clusters across all models → key: (level, property_description)
        clusters: Dict[tuple[str, str], List[ModelStats]] = {}
        for model, by_level in per_model_stats.items():
            for level, stats_list in by_level.items():
                if level not in ("fine", "coarse"):
                    continue
                for ms in stats_list:
                    key = (level, ms.property_description)
                    clusters.setdefault(key, []).append(ms)

        out: Dict[str, List[ModelStats]] = {"fine": [], "coarse": []}

        for (level, prop_desc), stats_list in clusters.items():
            # Combine sizes & examples
            total_size = sum(s.size for s in stats_list)
            cluster_size = stats_list[0].cluster_size
            all_examples: List[str] = []
            for s in stats_list:
                all_examples.extend(s.examples)

            # Frequency proportion: based on sum of model totals. Need total properties across all models
            # We approximate by summing total properties for each model (size / proportion)
            total_props_global = 0
            for s in stats_list:
                if s.proportion > 0:
                    total_props_global += int(round(s.size / s.proportion))
            proportion = total_size / total_props_global if total_props_global else 0.0

            # Quality averages across all examples (weighted by size)
            quality_keys = set()
            for s in stats_list:
                quality_keys.update(s.quality_score.keys())

            quality_score_avg: Dict[str, float] = {}
            for key in quality_keys:
                # weighted average by size
                numer = sum(s.quality_score.get(key, 0) * s.size for s in stats_list)
                quality_score_avg[key] = numer / total_size if total_size else 0.0

            # Combine quality CI: simple min/max of bounds for now
            quality_ci_combined: Dict[str, Dict[str, float]] = {}
            for key in quality_keys:
                lowers_raw = [s.quality_score_ci.get(key, {}).get("lower") for s in stats_list if s.quality_score_ci]
                uppers_raw = [s.quality_score_ci.get(key, {}).get("upper") for s in stats_list if s.quality_score_ci]
                lowers = [v for v in lowers_raw if v is not None]
                uppers = [v for v in uppers_raw if v is not None]
                if lowers and uppers:
                    quality_ci_combined[key] = {"lower": min(lowers), "upper": max(uppers)}

            # Distinctiveness – choose max score
            max_stat = max(stats_list, key=lambda s: s.score)
            score = max_stat.score
            score_ci = max_stat.score_ci
            score_sig = max_stat.score_statistical_significance

            # Quality significance: any true across models
            quality_sig: Dict[str, bool] = {}
            for key in quality_keys:
                quality_sig[key] = any(s.quality_score_statistical_significance and s.quality_score_statistical_significance.get(key, False) for s in stats_list)

            all_ms = ModelStats(
                property_description=prop_desc,
                model_name="all",
                score=score,
                quality_score=quality_score_avg,
                size=total_size,
                proportion=proportion,
                cluster_size=cluster_size,
                examples=all_examples,
                metadata={"level": level},
                score_ci=score_ci,
                quality_score_ci=quality_ci_combined if quality_ci_combined else None,
                score_statistical_significance=score_sig,
                quality_score_statistical_significance=quality_sig if quality_sig else None,
            )

            out[level].append(all_ms)

        # Sort each level by score descending
        for level in ("fine", "coarse"):
            out[level].sort(key=lambda s: s.score, reverse=True)

        return out 
"""
Data preparation utilities for sampling and filtering datasets before running the pipeline.

This module centralizes dataset subsampling logic so that the CLI and dashboard share
the same behavior.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd


def _infer_models(df: pd.DataFrame, method: str) -> list[str]:
    """Infer the list of models present in the dataset.

    Args:
        df: Input dataframe.
        method: Either "single_model" or "side_by_side".

    Returns:
        List of unique model names across the dataset.
    """
    if method == "single_model":
        if "model" in df.columns:
            return sorted([m for m in df["model"].dropna().unique().tolist()])
        return []
    elif method == "side_by_side":
        models: set[str] = set()
        if "model_a" in df.columns:
            models.update([m for m in df["model_a"].dropna().unique().tolist()])
        if "model_b" in df.columns:
            models.update([m for m in df["model_b"].dropna().unique().tolist()])
        return sorted(list(models))
    else:
        return []


def sample_prompts_evenly(
    df: pd.DataFrame,
    *,
    sample_size: Optional[int],
    method: str = "single_model",
    prompt_column: str = "prompt",
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample prompts evenly across models.

    When the dataset is balanced for single_model (each prompt has all models, one row
    per model), automatically sample prompts evenly across models. In that case we select
    K = int(N / M) prompts, where N is the desired total sample size and M is the number
    of unique models. The returned dataframe is then restricted to rows whose prompt is
    in this sampled set. If the dataset is not balanced (or method != single_model),
    this falls back to simple row-level sampling.

    Args:
        df: Input dataframe containing at least a ``prompt`` column and model columns as
            determined by ``method``.
        sample_size: Desired total number of rows across all models. If None or greater
            than the dataset size, the input dataframe is returned unchanged.
        method: Either "single_model" or "side_by_side".
        prompt_column: Name of the prompt column to use for grouping.
        random_state: Random seed for reproducible sampling.

    Returns:
        A dataframe filtered to the sampled prompts. If ``sample_size`` is None or not
        smaller than the input size, the input dataframe is returned.

    Raises:
        ValueError: If the inferred number of models is greater than 0 and
            int(sample_size / num_models) equals 0 (i.e., sample_size is too small).
    """
    if sample_size is None or sample_size <= 0 or sample_size >= len(df):
        return df

    if prompt_column not in df.columns:
        # Fallback to row-level sampling if no prompt column exists
        return df.sample(n=int(sample_size), random_state=random_state)

    models = _infer_models(df, method)
    num_models = max(1, len(models))

    # Check balanced condition for single_model: each prompt has all models, once
    is_balanced = False
    if method == "single_model" and "model" in df.columns and num_models > 0:
        coverage = df.groupby(prompt_column)["model"].nunique()
        per_pair_counts = df.groupby([prompt_column, "model"]).size()
        has_full_coverage = (coverage.min() == num_models) if len(coverage) > 0 else False
        has_single_entry_per_pair = (per_pair_counts.max() == 1) if len(per_pair_counts) > 0 else False
        is_balanced = bool(has_full_coverage and has_single_entry_per_pair)

    if not is_balanced:
        # Fall back to row-level sampling
        return df.sample(n=int(sample_size), random_state=random_state)

    prompts_per_model = int(sample_size // num_models)
    if prompts_per_model <= 0:
        raise ValueError(
            "Requested sample_size is smaller than the number of models; "
            "int(N/M) == 0 prompts. Increase sample_size or reduce models."
        )

    # Balanced: all prompts have all models → choose prompts uniformly at random
    all_prompts = df[prompt_column].dropna().unique().tolist()
    if len(all_prompts) == 0:
        return df.head(0)

    k = min(prompts_per_model, len(all_prompts))
    sampled_prompts = pd.Series(all_prompts).sample(n=k, random_state=random_state).tolist()
    return df[df[prompt_column].isin(sampled_prompts)]



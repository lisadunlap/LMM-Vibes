"""Logic for the **Frequency Comparison** tab."""
from typing import List, Tuple

import pandas as pd

from .state import app_state
from .utils import create_frequency_comparison_table

__all__ = ["create_frequency_comparison", "create_frequency_plots"]


def create_frequency_comparison(
    selected_models: List[str],
    cluster_level: str,
    top_n: int = 50,
    selected_model: str | None = None,
    selected_quality_metric: str | None = None,
) -> Tuple[pd.DataFrame, str]:
    if not app_state["model_stats"]:
        return pd.DataFrame({"Message": ["Please load data first"]}), ""

    if not selected_models:
        return pd.DataFrame({"Message": ["Please select at least one model"]}), ""

    model_filter = None if selected_model == "All Models" else selected_model
    quality_filter = None if selected_quality_metric == "All Metrics" else selected_quality_metric

    comparison_df = create_frequency_comparison_table(
        app_state["model_stats"],
        selected_models,
        cluster_level,
        top_n,
        model_filter,
        quality_filter,
    )

    if comparison_df.empty:
        return pd.DataFrame({"Message": ["No data available for comparison"]}), "Rows: 0"

    info_text = f"**Displaying {len(comparison_df)} rows.**"
    return comparison_df, info_text


def create_frequency_plots(*_args, **_kwargs):
    """Removed for now – kept as a stub for backward compatibility."""
    return None, None 
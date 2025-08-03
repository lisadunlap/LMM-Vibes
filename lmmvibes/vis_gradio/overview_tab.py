"""Logic helpers for the **Overview** tab."""
from typing import List

from .state import app_state
from .utils import compute_model_rankings, create_model_summary_card

__all__ = ["create_overview"]


def create_overview(
    selected_models: List[str],
    cluster_level: str,
    top_n: int,
    score_significant_only: bool = False,
    quality_significant_only: bool = False,
    sort_by: str = "score_desc",
) -> str:
    """Return the HTML snippet that summarises model performance."""
    if not app_state["model_stats"]:
        return "Please load data first using the 'Load Data' tab."

    if not selected_models:
        return "Please select at least one model to display."

    # 1. Compute global rankings and filter to selection
    model_rankings = compute_model_rankings(app_state["model_stats"])
    filtered_rankings = [
        (name, stats) for name, stats in model_rankings if name in selected_models
    ]

    if not filtered_rankings:
        return "No data available for selected models."

    # 2. Assemble HTML
    overview_html = """
    <div style="max-width: 1200px; margin: 0 auto;">
        <h2>🔍 Model Performance Overview</h2>
        <p style="color: #666; margin-bottom: 30px;">
            Top distinctive clusters where each model shows unique behavioral patterns.
            Frequency shows what percentage of a model's battles resulted in that behavioral pattern.
        </p>
    """

    for model_name, _ in filtered_rankings:
        card_html = create_model_summary_card(
            model_name,
            app_state["model_stats"],
            cluster_level,
            top_n,
            score_significant_only=score_significant_only,
            quality_significant_only=quality_significant_only,
            sort_by=sort_by,
        )
        overview_html += card_html

    overview_html += "</div>"
    return overview_html 
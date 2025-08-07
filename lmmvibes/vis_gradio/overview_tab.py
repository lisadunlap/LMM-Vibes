"""Logic helpers for the **Overview** tab."""
from typing import List

from .state import app_state
from .utils import compute_model_rankings_new, create_model_summary_card_new

__all__ = ["create_overview"]


def create_overview(
    selected_models: List[str],
    top_n: int,
    score_significant_only: bool = False,
    quality_significant_only: bool = False,
    sort_by: str = "quality_asc",
    min_cluster_size: int = 1,
) -> str:
    """Return the HTML snippet that summarises model performance."""
    if not app_state["metrics"]:
        return "Please load data first using the 'Load Data' tab."

    if not selected_models:
        return "Please select at least one model to display."

    # 1. Compute global rankings and filter to selection
    model_rankings = compute_model_rankings_new(app_state["metrics"])
    filtered_rankings = [
        (name, stats) for name, stats in model_rankings if name in selected_models
    ]

    # Sort so "all" appears first, then the rest by their rankings
    all_models = [(name, stats) for name, stats in filtered_rankings if name == "all"]
    other_models = [(name, stats) for name, stats in filtered_rankings if name != "all"]
    filtered_rankings = all_models + other_models

    if not filtered_rankings:
        return "No data available for selected models."

    # 2. Assemble HTML
    overview_html = """
    <div style="max-width: 1200px; margin: 0 auto;">
        <p style="color: #666; margin-bottom: 10px;">
            Top distinctive clusters where each model shows unique behavioural patterns.
            Frequency shows what percentage of a model's battles resulted in that behavioural pattern.
        </p>

        <details style="margin-bottom:25px;">
            <summary style="cursor:pointer; color:#4c6ef5; font-weight:600;">ℹ️  What do "× more distinctive" and "Quality Δ" mean?</summary>
            <div style="margin-top:12px; font-size:14px; line-height:1.5; color:#333;">
                <strong>Distinctiveness (× factor)</strong><br>
                For each cluster we compute how often <em>this model</em> appears in that cluster compared with the average across all models.<br>
                • A value &gt; 1 (e.g. <code>1.8×</code>) means the model hits the behaviour more often than average.<br>
                • A value &lt; 1 (e.g. <code>0.9×</code>) means it appears less often.<br>
                It is derived from the&nbsp;<code>proportion_delta</code>&nbsp;field in <code>model_cluster_scores.json</code>.<br><br>
                <strong>Quality Δ</strong><br>
                The difference between the cluster's quality score(s) for this model and the model's <em>overall</em> quality baseline, shown for each individual metric (e.g., helpfulness, accuracy).<br>
                Positive values (green) indicate the model performs better than its average in that behaviour; negative values (red) indicate worse.<br>
                This is derived from the <code>quality_delta</code> metric dictionary in <code>model_cluster_scores.json</code>.
            </div>
        </details>
    """

    for model_name, _ in filtered_rankings:
        card_html = create_model_summary_card_new(
            model_name,
            app_state["metrics"],
            # top_n etc.
            top_n,
            score_significant_only=score_significant_only,
            quality_significant_only=quality_significant_only,
            sort_by=sort_by,
            min_cluster_size=min_cluster_size,
        )
        overview_html += card_html

    overview_html += "</div>"
    return overview_html 
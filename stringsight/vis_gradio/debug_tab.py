"""Logic for the **Debug Data** tab."""
from __future__ import annotations

from .state import app_state

__all__ = ["debug_data_structure"]


def debug_data_structure() -> str:
    if app_state["clustered_df"] is None:
        return "<p style='color: #e74c3c;'>❌ No data loaded</p>"

    df = app_state["clustered_df"]

    n_rows = len(df)
    n_cols = len(df.columns)

    # Check for both naming patterns
    has_fine_clusters = ("property_description_fine_cluster_id" in df.columns or 
                        "fine_cluster_id" in df.columns)
    has_coarse_clusters = ("property_description_coarse_cluster_id" in df.columns or 
                          "coarse_cluster_id" in df.columns)

    sample_rows = min(3, len(df))
    sample_data = df.head(sample_rows).to_html(
        escape=False,
        classes="table table-striped",
        table_id="debug-table",
    )

    html = f"""
    <div style="max-width: 1200px; margin: 0 auto;">
        <h3>🐛 Data Structure Debug Info</h3>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h4>Basic Statistics</h4>
            <ul>
                <li><strong>Rows:</strong> {n_rows:,}</li>
                <li><strong>Columns:</strong> {n_cols}</li>
                <li><strong>Fine Clusters Available:</strong> {'✅ Yes' if has_fine_clusters else '❌ No'}</li>
                <li><strong>Coarse Clusters Available:</strong> {'✅ Yes' if has_coarse_clusters else '❌ No'}</li>
            </ul>
        </div>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h4>Available Columns</h4>
            <div style="max-height: 200px; overflow-y: auto; background: white; padding: 10px; border-radius: 4px;">
                <ul>
    """

    for col in sorted(df.columns):
        unique_values = df[col].nunique() if df[col].dtype == "object" else "N/A"
        html += f"<li><code>{col}</code> - {df[col].dtype} (unique values: {unique_values})</li>"

    html += f"""
                </ul>
            </div>
        </div>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h4>Sample Data (First {sample_rows} rows)</h4>
            <div style="max-height: 400px; overflow: auto; background: white; padding: 10px; border-radius: 4px;">
                {sample_data}
            </div>
        </div>
    </div>

    <style>
    #debug-table {{
        font-size: 12px;
        width: 100%;
    }}
    #debug-table th, #debug-table td {{
        padding: 4px 8px;
        border: 1px solid #ddd;
    }}
    #debug-table th {{
        background: #f1f1f1;
    }}
    </style>
    """

    return html 
from __future__ import annotations

"""lmmvibes.metrics.helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Small, focused helpers that remove repeated dataframe manipulations and
arithmetic dotted around the *metrics* package.  Importing from here keeps
`base_metrics.py` and its subclasses easier to read and maintain.

Only lightweight utility functions live here – nothing with heavy external
imports.  This avoids circular-import issues and keeps testability high.
"""

from typing import List
import pandas as pd

__all__ = [
    "score_keys",
    # metric computation functions were moved into BaseMetrics
    "extract_scores",
    "sanitize_cluster_column",
]


# ---------------------------------------------------------------------------
# Score-dictionary helpers
# ---------------------------------------------------------------------------

def score_keys(df: pd.DataFrame, column: str = "score") -> List[str]:
    """Return the union of all keys appearing in *df[column]* (dictionaries).

    We stop at the first non-empty dict for speed, but *only* if it contains
    all keys present elsewhere.  Fallback is a full scan if dictionaries can
    be heterogeneous.
    """
    # Quick path – assume homogeneous keys and grab the first non-empty dict
    for v in df[column]:
        if isinstance(v, dict) and v:
            return list(v.keys())
    return []


def extract_scores(df: pd.DataFrame, key: str, extractor) -> pd.Series:
    """Vectorised convenience wrapper around *extractor(row, key)*."""
    return df.apply(lambda row: extractor(row, key), axis=1)


# ---------------------------------------------------------------------------
# DataFrame clean-up helpers
# ---------------------------------------------------------------------------

def sanitize_cluster_column(df: pd.DataFrame, base: str) -> None:
    """Collapse *_x / *_y* artefacts produced by pandas merges in-place.

    If *base* already exists, nothing is done.  Otherwise, the function looks
    for ``{base}_x`` or ``{base}_y`` and fills *base* with the first non-null
    values found between them.
    """
    if base in df.columns:
        return

    x_col, y_col = f"{base}_x", f"{base}_y"
    if x_col in df.columns or y_col in df.columns:
        df[base] = df.get(x_col).combine_first(df.get(y_col)) 
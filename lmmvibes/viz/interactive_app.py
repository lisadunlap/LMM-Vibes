"""Streamlit application for exploring clustering results produced by LMM-Vibes.

Run with either of the following commands (depending on your packaging setup):

1. If you installed the package in editable mode (`pip install -e .`):

       streamlit run lmmvibes/viz/interactive_app.py -- --dataset path/to/dataset.json

2. Or via the module loader (requires `python -m` before *streamlit*):

       python -m streamlit run lmmvibes/viz/interactive_app.py -- --dataset path/to/dataset.json

The *dataset* must be a file previously saved via :py:meth:`PropertyDataset.save`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import streamlit as st
import plotly.express as px

from lmmvibes.core.data_objects import Cluster
from lmmvibes.viz import (
    load_dataset,
    build_hierarchy,
    load_clusters_dataframe,
    build_hierarchy_from_df,
)

# ---------------------------------------------------------------------------
# CLI args (Streamlit forwards everything after "--")
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=False, help="Path to PropertyDataset (json / pkl / parquet)")
    # Parse known to avoid Streamlit's own flags
    args, _ = parser.parse_known_args()
    return args

args = _parse_args()

# ---------------------------------------------------------------------------
# Sidebar – dataset selection -------------------------------------------------
# ---------------------------------------------------------------------------

st.set_page_config(page_title="LMM-Vibes Cluster Explorer", layout="wide")
st.title("checkin' in on our clusters")

# Allow both CLI arg and file-uploader fallback
if args.dataset:
    dataset_path = Path(args.dataset)
else:
    st.sidebar.info("Upload a PropertyDataset file (.json, .pkl, .parquet)")
    uploaded = st.sidebar.file_uploader("Dataset", type=["json", "pkl", "pickle", "parquet"])
    if uploaded is None:
        st.stop()
    # Save to a temporary file so that PropertyDataset.load can open it
    import tempfile, shutil
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp_dir.name) / uploaded.name
    with tmp_path.open("wb") as tmp_f:
        shutil.copyfileobj(uploaded, tmp_f)
    dataset_path = tmp_path

# ---------------------------------------------------------------------------
# Data loading (dataset **or** clusters DataFrame) ---------------------------
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading data …")
def _load(path: Path):
    # 1) Try full PropertyDataset first
    try:
        ds = load_dataset(path)
        return "dataset", ds
    except Exception:
        # 2) Fallback – attempt clusters DataFrame
        df = load_clusters_dataframe(path)
        return "df", df

kind, data_obj = _load(dataset_path)

if kind == "dataset":
    coarse_clusters, fine_map = build_hierarchy(data_obj)  # type: ignore[arg-type]
    all_clusters = data_obj.clusters
else:
    coarse_clusters, fine_map = build_hierarchy_from_df(data_obj)  # type: ignore[arg-type]
    all_clusters = [*coarse_clusters, *[c for lst in fine_map.values() for c in lst]]
    # After reconstruction we can still expose dataset-like attrs for UX
    class _Dummy:
        properties: list = []

    data_obj = _Dummy()  # type: ignore[assignment]

st.sidebar.success(f"Loaded **{dataset_path.name}** as {kind.upper()}")

# ---------------------------------------------------------------------------
# Global overview: accordion of all clusters ---------------------------------
# ---------------------------------------------------------------------------

with st.expander("📚 Cluster overview (click to expand)"):
    for c in coarse_clusters:
        child_list = [cl for cl in all_clusters if cl.parent_id == c.id]
        label = f"Coarse {c.id} – {c.label}  [{c.size}]"
        with st.expander(label):
            if child_list:
                st.markdown("**Fine clusters:**")
                for f in child_list:
                    st.markdown(f"- {f.label} [{f.size}]")
            else:
                st.markdown("*(No hierarchical children – flat cluster)*")

# ---------------------------------------------------------------------------
# Level 1 – Coarse clusters ---------------------------------------------------
# ---------------------------------------------------------------------------

coarse_labels = [f"[{c.size:>3}] {c.label}" for c in coarse_clusters]
sel_coarse_idx = st.sidebar.selectbox("Coarse clusters", range(len(coarse_clusters)), format_func=lambda i: coarse_labels[i])
coarse: Cluster = coarse_clusters[sel_coarse_idx]

# Bar chart of fine cluster sizes
with st.sidebar.expander("Fine-cluster distribution", expanded=False):
    children = [c for c in all_clusters if c.parent_id == coarse.id]
    if children:
        fig = px.bar(x=[c.label for c in children], y=[c.size for c in children], labels={"x": "Fine cluster", "y": "Size"}, height=250)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No hierarchical children – flat clustering.")

# ---------------------------------------------------------------------------
# Level 2 – Fine clusters -----------------------------------------------------
# ---------------------------------------------------------------------------

# Derive children directly if fine_map is incomplete
children = [c for c in all_clusters if c.parent_id == coarse.id]
if not children:
    # Fallback: treat coarse as fine when no explicit children
    children = [coarse]

# Display coarse cluster info in a more compact way
st.markdown(f"#### 🇽🇰 Coarse Cluster #{coarse.id}")
st.markdown(f"#### {coarse.label}")
st.markdown(f"**{coarse.size} properties** • **{len(children)} fine clusters**")

fine_labels = [f"[{c.size:>3}] {c.label}" for c in children]
sel_fine_idx = st.selectbox("Fine clusters", range(len(children)), format_func=lambda i: fine_labels[i])
fine: Cluster = children[sel_fine_idx]

# ---------------------------------------------------------------------------
# Level 3 – Property descriptions -------------------------------------------
# ---------------------------------------------------------------------------

# st.markdown("---")  # Add visual separator
st.markdown(f"#### 🇫🇮 Fine Cluster #{fine.id}")
st.markdown(f"**{fine.size} properties** • *{fine.label}*")

# Optional word-cloud visualisation
with st.expander("☁️ Word-cloud", expanded=False):
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        wc = WordCloud(width=800, height=300, background_color="white").generate(" ".join(fine.property_descriptions))
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    except ModuleNotFoundError:
        st.info("Run `pip install wordcloud matplotlib` for word-cloud view.")

st.markdown("**📋 Property descriptions**")

for i, desc in enumerate(fine.property_descriptions, 1):
    st.markdown(f"**{i}.** {desc}") 
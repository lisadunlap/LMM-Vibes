# LMM-Vibes Gradio Viewer

This sub-package contains the standalone Gradio application that lets you
explore the behavioural-property results produced by the LMM-Vibes pipeline.
The code has been split into small, focused modules to keep maintenance easy.

## Directory layout

```
vis_gradio/
├── app.py                # Entry-point that builds & launches the interface
├── state.py              # Shared mutable globals (app_state, BASE_RESULTS_DIR)
│
├── load_data_tab.py      # "Load Data" tab – data ingestion helpers
├── overview_tab.py       # "Overview" tab – model summary cards
├── clusters_tab.py       # "View Clusters" tab – interactive cluster explorer
├── examples_tab.py       # "View Examples" tab – individual prompt/response view
├── frequency_tab.py      # "Frequency Comparison" tab – cross-model stats
├── debug_tab.py          # "Debug Data" tab – quick sanity checks
│
└── data_loader.py        # Low-level JSON/JSONL reading utilities (unchanged)
```

Only `app.py` is imported by external callers (`python -m lmmvibes.vis_gradio.launcher`); it wires together the tab builders from the per-tab modules.

## Expected input files

Each *experiment* lives in its own directory and **must contain at least**

* `model_stats.json`  – aggregated per-model statistics per cluster.
* `clustered_results.jsonl`  – one JSON line per evaluated prompt/response.

The default launch script assumes a *base results directory* (`results/` by
default).  Every sub-folder inside it that contains the two files above is
treated as an experiment and appears in the **Select Experiment** dropdown.

### `model_stats.json`
Roots of this file:
```
{
  "<model_name>": {
      "fine": [ { …cluster summary… }, … ],
      "coarse": [ { …cluster summary… }, … ]
  },
  …
}
```
Each *cluster summary* object may include – but is not limited to – the keys
below (unused keys are ignored):

| key                | type          | description                                        |
|--------------------|--------------|----------------------------------------------------|
| `cluster_id`       | int          | numerical id of the cluster                        |
| `cluster_label`    | str          | human-readable label                               |
| `frequency`        | float        | fraction of battles that fall in this cluster      |
| `score`            | float        | distinctiveness score                              |
| `score_pvalue`     | float        | p-value for `score`                                |
| `quality_score`    | dict[str, float] | optional quality metrics per evaluator       |

### `clustered_results.jsonl`
Each line is a dictionary with, at minimum, the columns used by the viewer:

* `question_id` – unique identifier for the prompt instance
* `prompt` – the full prompt text
* `model` – name of the responding model (must match keys in `model_stats.json`)
* `property_description` – textual description of the behavioural property
* `fine_cluster_id` / `coarse_cluster_id` – numeric ids (optional but needed
  for the corresponding cluster level)

Optional but recommended columns:

* `fine_cluster_label`, `coarse_cluster_label` – readable names
* `score` – per-row distinctiveness score
* Any additional metadata you’d like to surface in the **Debug Data** tab

## How the data is processed

1. **Loading** – `load_data_tab.load_data()` validates the directory, resolves
   nested sub-folders (produced by sweeps), reads both files and stores the
   resulting dataframes/dicts in `state.app_state`.
2. **Global state** – every tab imports `app_state` to access the shared data.
3. **Per-tab logic** – each `*_tab.py` exposes the Gradio event callbacks that
   create HTML or `DataFrame` objects based on the current view parameters.

The viewer does *not* mutate the raw JSON/JSONL; everything is computed in
memory.  When in doubt, open the **Debug Data** tab to confirm your columns are
present and spelled correctly.

## Launching

```bash
python -m lmmvibes.vis_gradio.launcher --results_dir /path/to/results
```

If the directory contains exactly one experiment it is auto-loaded; otherwise
pick one from the dropdown.

For more deployment options (port binding, public sharing, etc.) see the root
project **README**. 
# LMM-Vibes Gradio Visualization

A Gradio-based web interface for exploring LMM-Vibes pipeline results. This provides an alternative to the Streamlit interface with a clean, intuitive UI for analyzing model performance, cluster analysis, and detailed examples.

## Features

🔍 **Model Overview**: Interactive cards showing top distinctive clusters for each model
📊 **Cluster Analysis**: Searchable tables of cluster data with filtering
📈 **Frequency Comparison**: Side-by-side comparison of cluster frequencies across models  
🔎 **Search Examples**: Text search across all data fields to find specific examples
📁 **Easy Data Loading**: Simple interface for loading pipeline results

## Installation

Make sure you have Gradio installed:

```bash
pip install gradio
```

## Usage

### Method 1: Python API

```python
from lmmvibes.vis_gradio import launch_app

# Launch with auto-loaded data
launch_app(results_dir="/path/to/your/results")

# Launch without pre-loading (load data in the UI)
launch_app()

# Launch with custom settings
launch_app(
    results_dir="/path/to/results",
    share=True,  # Create public link
    server_port=8080
)
```

### Method 2: Command Line

```bash
# Launch with auto-loaded data
python -m lmmvibes.vis_gradio.launcher --results_dir /path/to/results

# Launch with public sharing
python -m lmmvibes.vis_gradio.launcher --results_dir /path/to/results --share

# Launch on custom port
python -m lmmvibes.vis_gradio.launcher --results_dir /path/to/results --port 8080

# Launch without pre-loading data
python -m lmmvibes.vis_gradio.launcher
```

### Method 3: Direct Script

```bash
python lmmvibes/vis_gradio/launcher.py --results_dir /path/to/results
```

## Expected Data Format

Your results directory should contain:

- `model_stats.json` - Model performance statistics and cluster data
- `clustered_results.jsonl` - Detailed clustering results with property descriptions

The app will automatically detect subfolders containing these files if they're not in the root directory.

## Interface Overview

### 📁 Load Data Tab
- Enter the path to your pipeline results directory
- View data summary and select models for analysis
- Automatic validation and subfolder detection

### 📊 Overview Tab  
- Model performance cards with top distinctive clusters
- Configurable cluster level (fine/coarse) and number of clusters
- Interactive frequency and distinctiveness metrics

### 📋 View Clusters Tab
- Searchable table of all cluster data
- Filter by selected models and cluster level
- Export-friendly format

### 📈 Frequency Comparison Tab
- Side-by-side comparison of cluster frequencies across models
- Sortable by average frequency
- Both frequency percentages and raw scores

### 🔎 Search Examples Tab
- Full-text search across all data fields
- Filter by selected models
- Configurable result limits

## Comparison with Streamlit Version

| Feature | Gradio | Streamlit |
|---------|--------|-----------|
| Model Overview | ✅ Interactive cards | ✅ Detailed cards |
| Cluster Tables | ✅ Searchable | ✅ Advanced filtering |
| Frequency Analysis | ✅ Comparison tables | ✅ Charts + tables |
| Search Functionality | ✅ Simple text search | ✅ Vector search |
| Data Loading | ✅ Simple UI | ✅ Advanced options |
| Deployment | ✅ Easy sharing | ✅ Streamlit Cloud |
| Customization | ✅ Programmatic API | ✅ Widget ecosystem |

## Performance Notes

- Data is cached automatically for fast interactions
- Large datasets are limited to reasonable display sizes
- Search results are capped at 100 items for performance

## Troubleshooting

### "Please load data first" messages
Make sure you've successfully loaded data using the Load Data tab and that there are no error messages.

### Empty tables/results
Check that:
- Your results directory contains the required files
- You have selected at least one model for analysis
- Your search terms match existing data

### Performance issues
For very large datasets:
- Reduce the "Max Rows" slider values
- Use more specific search terms
- Select fewer models for comparison

## Development

The Gradio interface is built with modularity in mind:

- `app.py` - Main application and UI logic
- `data_loader.py` - Data loading utilities (no Streamlit dependencies)  
- `utils.py` - Analysis and formatting utilities
- `launcher.py` - CLI interface

To extend functionality, add new functions to `utils.py` and wire them up in `app.py`. 
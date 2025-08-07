# Quick Start

Get up and running with LMM-Vibes in minutes.

## Installation

```bash
pip install lmm-vibes
# or for development
pip install -e .
```

Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Basic Usage

### 1. Import LMM-Vibes

```python
import pandas as pd
from lmmvibes import explain
```

### 2. Prepare Your Data

**Side-by-side comparison format:**
```python
df = pd.DataFrame([
    {
        "question_id": "q1",
        "model_a": "gpt-4", 
        "model_b": "claude-3",
        "model_a_response": "The answer is 4.",
        "model_b_response": "2 + 2 equals 4.",
        "winner": "tie"  # optional
    }
])
```

**Single model format:**
```python
df = pd.DataFrame([
    {
        "question_id": "q1",
        "model": "gpt-4",
        "model_response": "The answer is 4.",
        "score": 8.5  # optional
    }
])
```

### 3. Analyze Model Behavior

```python
# Side-by-side comparison
clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    min_cluster_size=30,
    output_dir="results/"
)

# Single model analysis  
clustered_df, model_stats = explain(
    df,
    method="single_model",
    min_cluster_size=20,
    output_dir="results/"
)
```

### 4. Explore Results

```python
# View behavioral clusters
print(clustered_df[['property_description', 'property_description_coarse_cluster_label']].head())

# Check model rankings
print(model_stats)
```

### 5. Interactive Visualization

```bash
streamlit run lmmvibes/viz/interactive_app.py -- --dataset results/clustered_results.parquet
```

## Advanced Usage

### Custom Prompts

```python
custom_prompt = """
Analyze this conversation and identify the key behavioral difference.
Focus on: reasoning style, factual accuracy, helpfulness.
"""

clustered_df, model_stats = explain(
    df,
    method="side_by_side", 
    system_prompt=custom_prompt,
    model_name="gpt-4o",
    temperature=0.1
)
```

### Hierarchical Clustering

```python
clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    hierarchical=True,  # Creates both fine and coarse clusters
    max_coarse_clusters=15,
    embedding_model="openai"  # Higher quality embeddings
)
```

### Pipeline Configuration

```python
from lmmvibes.pipeline import PipelineBuilder
from lmmvibes.extractors import OpenAIExtractor
from lmmvibes.clusterers import HDBSCANClusterer

pipeline = (PipelineBuilder("Custom Pipeline")
    .extract_properties(OpenAIExtractor(model="gpt-4o-mini"))
    .cluster_properties(HDBSCANClusterer(min_cluster_size=15))
    .configure(use_wandb=True)
    .build())

# Use custom pipeline
clustered_df, model_stats = explain(
    df, 
    custom_pipeline=pipeline
)
```

## Data Formats

### Required Columns

**Side-by-side:**
- `question_id` - Unique identifier for each question
- `model_a`, `model_b` - Model names
- `model_a_response`, `model_b_response` - Model responses
- `winner` (optional) - "model_a", "model_b", or "tie"

**Single model:**  
- `question_id` - Unique identifier 
- `model` - Model name
- `model_response` - Model response
- `score` (optional) - Numeric quality score

### Output Files

When you specify `output_dir`, LMM-Vibes saves:
- `clustered_results.parquet` - Complete results with clusters
- `model_stats.json` - Model performance statistics  
- `full_dataset.json` - Complete dataset for reuse
- `summary.txt` - Human-readable summary

## Next Steps

- Learn about the [explain() and label() functions](../user-guide/basic-usage.md)
- Understand the [output files](../user-guide/configuration.md)
- Explore the [API Reference](../api/core.md) 
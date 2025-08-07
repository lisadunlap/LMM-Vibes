# Explain and Label Functions

Learn how to use the two main functions in LMM-Vibes for analyzing model behavior.

## Core Functions

LMM-Vibes provides two primary functions:

- **`explain()`**: Discovers behavioral patterns through clustering 
- **`label()`**: Classifies behavior using predefined taxonomies

Both functions analyze conversation data and return clustered results with model statistics.

## The `explain()` Function

The `explain()` function automatically discovers behavioral patterns in model responses through property extraction and clustering.

### Basic Usage

```python
import pandas as pd
from lmmvibes import explain

# Load your conversation data
df = pd.read_csv("model_conversations.csv")

# Analyze side-by-side comparisons
clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    min_cluster_size=30,
    output_dir="results/"
)

# Analyze single model responses
clustered_df, model_stats = explain(
    df,
    method="single_model", 
    min_cluster_size=20,
    output_dir="results/"
)
```

### Parameters

**Core Parameters:**
- `df`: Input DataFrame with conversation data
- `method`: `"side_by_side"` or `"single_model"`
- `system_prompt`: Custom prompt for property extraction (optional)
- `output_dir`: Directory to save results

**Extraction Parameters:**
- `model_name`: LLM for property extraction (default: `"gpt-4o"`)
- `temperature`: Temperature for LLM calls (default: `0.7`)
- `max_workers`: Parallel workers for API calls (default: `16`)

**Clustering Parameters:**
- `clusterer`: Clustering method (`"hdbscan"`, `"hierarchical"`)
- `min_cluster_size`: Minimum cluster size (default: `30`)
- `embedding_model`: `"openai"` or sentence-transformer model
- `hierarchical`: Create both fine and coarse clusters (default: `False`)

### Examples

**Custom System Prompt:**
```python
custom_prompt = """
Analyze this conversation and identify behavioral differences.
Focus on: reasoning approach, factual accuracy, response style.
Return a JSON object with 'property_description' and 'property_evidence'.
"""

clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    system_prompt=custom_prompt
)
```

**Hierarchical Clustering:**
```python
clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    hierarchical=True,
    max_coarse_clusters=15,
    min_cluster_size=20
)
```

## The `label()` Function

The `label()` function classifies model behavior using a predefined taxonomy rather than discovering patterns.

### Basic Usage

```python
from lmmvibes import label

# Define your evaluation taxonomy
taxonomy = {
    "accuracy": "Is the response factually correct?",
    "helpfulness": "Does the response address the user's needs?", 
    "clarity": "Is the response clear and well-structured?",
    "safety": "Does the response avoid harmful content?"
}

# Classify responses
clustered_df, model_stats = label(
    df,
    taxonomy=taxonomy,
    model_name="gpt-4o-mini",
    output_dir="results/"
)
```

### Parameters

**Core Parameters:**
- `df`: Input DataFrame (must be single-model format)
- `taxonomy`: Dictionary mapping labels to descriptions
- `model_name`: LLM for classification (default: `"gpt-4o-mini"`)
- `output_dir`: Directory to save results

**Other Parameters:**
- `temperature`: Temperature for classification (default: `0.0`)
- `max_workers`: Parallel workers (default: `8`)
- `verbose`: Print progress information (default: `True`)

### Example

**Quality Assessment:**
```python
quality_taxonomy = {
    "excellent": "Response is comprehensive, accurate, and well-structured",
    "good": "Response is mostly accurate with minor issues",
    "fair": "Response has some accuracy or clarity problems", 
    "poor": "Response has significant issues or inaccuracies"
}

clustered_df, model_stats = label(
    df,
    taxonomy=quality_taxonomy,
    temperature=0.0,  # Deterministic classification
    output_dir="quality_results/"
)
```

## Data Formats

### Side-by-side Format (for `explain()` only)

```python
df = pd.DataFrame([
    {
        "question_id": "q1",
        "model_a": "gpt-4",
        "model_b": "claude-3", 
        "model_a_response": "Response from model A...",
        "model_b_response": "Response from model B...",
        "winner": "tie"  # optional: "model_a", "model_b", or "tie"
    }
])
```

### Single Model Format (for both functions)

```python
df = pd.DataFrame([
    {
        "question_id": "q1",
        "model": "gpt-4",
        "model_response": "The model's response...",
        "score": 8.5  # optional: numeric quality score
    }
])
```

## Understanding Results

### Output DataFrames

Both functions return a DataFrame with added columns:

```python
# Original columns plus:
print(clustered_df.columns)
# ['question_id', 'model', 'model_response', 
#  'property_description', 'property_evidence',
#  'property_description_cluster_id', 'property_description_cluster_label']
```

### Model Statistics

```python
print(model_stats.keys())
# Contains performance metrics, cluster distributions, and rankings
```

### Saved Files

When `output_dir` is specified, both functions save:
- `clustered_results.parquet` - Complete results with clusters
- `model_stats.json` - Model performance statistics
- `full_dataset.json` - Complete dataset for reanalysis  
- `summary.txt` - Human-readable summary

## When to Use Each Function

**Use `explain()` when:**
- You want to discover unknown behavioral patterns
- You're comparing multiple models
- You need flexible, data-driven analysis
- You want to understand what makes models different

**Use `label()` when:**
- You have specific criteria to evaluate
- You need consistent scoring across datasets
- You're building evaluation pipelines  
- You want controlled, taxonomy-based analysis

## Next Steps

- Understand the [output files](configuration.md) in detail
- Explore [configuration options](configuration.md)
- Learn about the [pipeline architecture](../api/core.md) 
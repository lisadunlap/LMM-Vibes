# Quick Start

Get up and running with LMM-Vibes in minutes.

## Installation

```bash
# Basic installation
pip install stringsight

# With optional dependencies for better performance
pip install stringsight[full]  # includes sentence-transformers, wandb
```

**Requirements:**
- Python 3.8+
- OpenAI API key for property extraction and embeddings

Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Basic Usage

### 1. Import LMM-Vibes

```python
import pandas as pd
from stringsight import explain
```

### 2. Prepare Your Data

LMM-Vibes analyzes model conversations to extract behavioral properties. You need either single model responses or side-by-side comparisons.

**Single model format (for analyzing individual model behavior):**
```python
df = pd.DataFrame({
    "prompt": ["What is machine learning?", "Explain quantum computing", "Write a poem about AI"],
    "model": ["gpt-4", "gpt-4", "gpt-4"],
    "model_response": ["Machine learning involves...", "QC leverages quantum...", "Silicon dreams awaken..."],
    "score": [{"accuracy": 1, "helpfulness": 4.2}, {"accuracy": 0, "helpfulness": 3.8}, {"accuracy": 1, "helpfulness": 4.5}]
})
```

**Side-by-side comparison format (for comparing two models):**
```python
df = pd.DataFrame({
    "prompt": ["What is machine learning?", "Explain quantum computing", "Write a poem about AI"],
    "model_a": ["gpt-4", "gpt-4", "gpt-4"],
    "model_b": ["claude-3", "claude-3", "claude-3"],
    "model_a_response": ["ML is a subset of AI...", "Quantum computing uses...", "In circuits of light..."],
    "model_b_response": ["Machine learning involves...", "QC leverages quantum...", "Silicon dreams awaken..."],
    "score": [{"winner": "gpt-4", "helpfulness": 4.2}, {"winner": "gpt-4", "helpfulness": 3.8}, {"winner": "claude-3", "helpfulness": 4.5}]
})
```

The `score` column is optional but helps provide context for behavioral analysis.

### 3. Analyze Model Behavior

The `explain()` function runs a 4-stage pipeline: **Property Extraction** → **Post-processing** → **Clustering** → **Metrics & Analysis**

```python
# For single model analysis
clustered_df, model_stats = explain(
    df,
    method="single_model",
    min_cluster_size=10,  # Minimum size for behavior clusters
    output_dir="results/test"  # Save all results here
)

# For side-by-side comparison
clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    min_cluster_size=10,
    output_dir="results/test"
)
```

**What happens during analysis:**

1. **Property Extraction**: GPT-4 analyzes each response and extracts behavioral properties like "Provides step-by-step reasoning" or "Uses creative examples"

2. **Clustering**: Similar properties are grouped together using embeddings (e.g., "explains clearly" and "shows work" → "Reasoning Transparency" cluster)

3. **Metrics Calculation**: Computes which models excel at which behavioral patterns, with quality scores and statistical significance

4. **Results Saving**: All intermediate and final results are saved to your output directory

### 4. Explore Results

```python
# View the extracted behavioral properties and their clusters
print("Sample behavioral properties found:")
print(clustered_df[['property_description', 'property_description_fine_cluster_label']].head())

# Check which behaviors each model excels at
for model, stats in model_stats.items():
    print(f"\n{model} excels at:")
    # Show top behavioral clusters for this model
    for behavior in stats["fine"][:3]:  # top 3 behaviors
        print(f"  • {behavior.property_description} (score: {behavior.score:.2f})")
```

### 5. Interactive Visualization

Launch the Gradio web interface to explore your results interactively:

```bash
# View clusters, examples, and metrics in a web interface
python -m stringsight.dashboard.launcher --share
```

This opens a web interface where you can:
- Browse behavioral clusters and their examples
- Compare model performance across different behaviors  
- Explore the extracted properties and metrics
- Visualize model rankings and statistical analysis

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

### Pipeline Configuration

```python
from stringsight.pipeline import PipelineBuilder
from stringsight.extractors import OpenAIExtractor
from stringsight.clusterers import HDBSCANClusterer

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

### Expected Columns

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
- `clustered_results.jsonl` - Complete results with clusters
- `model_stats.json` - Model performance statistics  
- `full_dataset.json` - Complete dataset for reuse
- `summary.txt` - Human-readable summary

## Next Steps

- Learn about the [explain() and label() functions](../user-guide/basic-usage.md)
- Understand the [output files](../user-guide/configuration.md)
- Explore the [API Reference](../api/core.md) 
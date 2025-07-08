# Welcome to whatever this project is called

*Potential name options: Axiom, Tesseract, Quill, Mentat, Autopilot, TempCheck, explAIner, Marcel (i just like that name, no relation to anything this does)*

**Extract, cluster, and analyze behavioral properties from Large Language Models**

We help you understand how different generative models behave by automatically extracting behavioral properties from their responses, grouping similar behaviors together, and quantifying how important these behvaiors are (damn claude you are not really selling this). 

## Installation

```bash
# Basic installation
pip install lmmvibes

# With optional dependencies
pip install lmmvibes[full]  # includes sentence-transformers, wandb
```

*Requirements:*

- Python 3.8+
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)
- Optional: Weights & Biases account for experiment tracking

## Quick Start

```python
import pandas as pd
from lmmvibes import explain

# Your data with model responses
df = pd.DataFrame({
    "question_id": ["q1", "q2", "q3"],
    "prompt": ["What is machine learning?", "Explain quantum computing", "Write a poem about AI"],
    "model_a": ["gpt-4", "gpt-4", "gpt-4"],
    "model_b": ["claude-3", "claude-3", "claude-3"],
    "model_a_response": ["ML is a subset of AI...", "Quantum computing uses...", "In circuits of light..."],
    "model_b_response": ["Machine learning involves...", "QC leverages quantum...", "Silicon dreams awaken..."],
    "winner": ["model_a", "model_b", "tie"]
})

# Extract and cluster behavioral properties
clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    min_cluster_size=10,
    output_dir="results/"
)
```

## What You Get

**`clustered_df`** - Your original data plus:
- Extracted behavioral properties (`property_description`, `category`, `impact`, `type`)
- Cluster assignments (`property_description_fine_cluster_id`, `property_description_coarse_cluster_id`)
- Cluster labels (`property_description_fine_cluster_label`, `property_description_coarse_cluster_label`)

**`model_stats`** - Per-model behavioral analysis:
- Which behaviors each model shows most/least
- Relative scores compared to other models
- Example responses for each behavior cluster

## Interactive Visualization

Explore your results visually:

```bash
streamlit run lmmvibes/viz/interactive_app.py -- --dataset results/clustered_results.parquet
```

Browse clusters, see examples, and understand model differences through an interactive interface.

## Input Data Requirements

LMM-Vibes supports two primary analysis methods with specific data requirements:

### Side-by-Side Comparisons
For comparing two models head-to-head (like Arena-style battles):

**Required columns:**
- `question_id` - Unique identifier for each conversation/question
- `prompt` (or `user_prompt`) - The question or prompt given to models
- `model_a` - Name of the first model
- `model_b` - Name of the second model  
- `model_a_response` - Response from the first model
- `model_b_response` - Response from the second model

**Optional columns:**
- `score` - Dictionary of metrics: `{"winner": "model_a", "rating": 4.5, "helpfulness": 0.8}`
- Any additional metadata columns (language, category, difficulty, etc.)

```python
df = pd.DataFrame({
    "question_id": ["q1", "q2"],
    "prompt": ["Question text"],
    "model_a": ["gpt-4", "gpt-4"],
    "model_b": ["claude-3", "claude-3"],
    "model_a_response": ["Response from model A"],
    "model_b_response": ["Response from model B"],
    "score": [{"winner": "model_a"}, {"winner": "model_b", "confidence": 0.9}],  # optional
    "language": ["en", "en"],  # optional metadata
})
```

### Single Model Analysis
For analyzing individual model responses:

**Required columns:**
- `question_id` - Unique identifier for each conversation/question
- `prompt` (or `user_prompt`) - The question or prompt given to the model
- `model` - Name of the model
- `model_response` (or `response`) - The model's response

**Optional columns:**
- `score` - Dictionary of metrics: `{"rating": 4.2, "accuracy": 0.85, "helpfulness": 0.9}`
- Any additional metadata columns

```python
df = pd.DataFrame({
    "question_id": ["q1", "q2"],
    "prompt": ["Question text"],
    "model": ["gpt-4", "gpt-4"],
    "model_response": ["Model response"],
    "score": [{"rating": 4.2, "accuracy": 0.8}, {"rating": 4.5, "accuracy": 0.9}],  # optional
    "category": ["reasoning", "creative"],  # optional metadata
})
```

## Pipeline Components

LMM-Vibes follows a 4-stage pipeline architecture. Here's what each component does:

### 1. Property Extraction (`lmmvibes.extractors`)
**Goal**: Identify specific behavioral properties from model responses using LLM analysis.

Takes each conversation and asks an LLM (like GPT-4) to extract behavioral properties such as:
- "Provides step-by-step reasoning"
- "Uses formal/informal tone" 
- "Includes creative examples"
- "Shows uncertainty appropriately"

**Available extractors:**
- `OpenAIExtractor` - Uses OpenAI API (GPT models)
- `VLLMExtractor` - Uses local models via vLLM (I had claude YOLO this in the refactor so 20% it works)
- `BatchExtractor` - Makes a file for the batch API (i know this one isnt done, need to port over from pre-refactor repo)

### 2. Post-processing (`lmmvibes.postprocess`)
**Goal**: Parse and validate the extracted properties into structured data.

- `LLMJsonParser` - Converts raw LLM responses into structured property objects
- `PropertyValidator` - Ensures properties meet quality standards and required fields

### 3. Clustering (`lmmvibes.clusterers`) 
**Goal**: Group similar behavioral properties into coherent clusters for analysis.

Takes individual properties like "explains step-by-step" and "shows work clearly" and groups them into clusters like "Reasoning Transparency". Creates both fine-grained and coarse-grained cluster hierarchies.

**Available clusterers:**
- `HDBSCANClusterer` - Density-based clustering (recommended for >10k samples)
- `HierarchicalClusterer` - Traditional hierarchical clustering with LLM-powered naming (i thiiiink this is also yet to be ported over from pre-refactor)

### 4. Metrics & Analysis (`lmmvibes.metrics`)
**Goal**: Calculate model performance statistics and behavioral rankings.

Computes which models excel at which behavioral patterns:
- Model scores per behavior cluster
- Relative strengths/weaknesses between models  
- Statistical significance of differences
- Example responses for each cluster

**Available metrics:**
- `SideBySideMetrics` - For model comparison data
- `SingleModelMetrics` - For individual model analysis

## Configuration Options

```python
clustered_df, model_stats = explain(
    df,
    method="side_by_side",              # or "single_model"
    system_prompt="one_sided_system_prompt",  # custom extraction prompt
    min_cluster_size=30,                # minimum cluster size
    embedding_model="openai",           # or any sentence-transformer model
    hierarchical=True,                  # create coarse clusters
    output_dir="results/",              # save outputs here
    use_wandb=True,                     # log to Weights & Biases
    wandb_project="my-lmm-analysis"
)
```

## Output Files

When you specify `output_dir`, LMM-Vibes automatically saves:

- `clustered_results.parquet` - Full dataset with properties and clusters
- `full_dataset.json` - Complete dataset in JSON format
- `model_stats.json` - Per-model behavioral statistics
- `summary.txt` - Human-readable analysis summary

## Examples

### Analyzing Model Competition Data
```python
# Load your Arena-style competition data
df = pd.read_parquet("arena_data.parquet")

# Extract behavioral differences
clustered_df, stats = explain(df, method="side_by_side")

# See which behaviors each model excels at
for model, model_stats in stats.items():
    print(f"\n{model} excels at:")
    for behavior in model_stats["fine"][:3]:  # top 3 behaviors
        print(f"  • {behavior.property_description} (score: {behavior.score:.2f})")
```

### Understanding Model Capabilities
```python
# Single model analysis
df = pd.DataFrame({
    "question_id": range(100),
    "prompt": ["Explain quantum physics"] * 100,
    "model": ["gpt-4"] * 100,
    "response": responses  # your model responses
})

clustered_df, stats = explain(df, method="single_model")
```

### Viewing Results in Streamlit
```bash
# View your clustered results interactively
streamlit run lmmvibes/viz/interactive_app.py -- --dataset results/clustered_results.parquet

# Or view just the clusters DataFrame
streamlit run lmmvibes/viz/interactive_app.py -- --dataset results/property_with_clusters.csv
```

## Advanced: Running Pipeline Components

For more control, you can run each pipeline stage separately:

```python
from lmmvibes.core import PropertyDataset
from lmmvibes.extractors import OpenAIExtractor
from lmmvibes.postprocess import LLMJsonParser, PropertyValidator
from lmmvibes.clusterers import HDBSCANClusterer
from lmmvibes.metrics import SideBySideMetrics

# 1. Load your data
dataset = PropertyDataset.from_dataframe(df, method="side_by_side")

# 2. Extract properties
extractor = OpenAIExtractor(
    system_prompt="one_sided_system_prompt",
    model="gpt-4o-mini",
    temperature=0.6
)
dataset = extractor(dataset)

# 3. Parse and validate properties
parser = LLMJsonParser()
dataset = parser(dataset)

validator = PropertyValidator()
dataset = validator(dataset)

# 4. Cluster properties
clusterer = HDBSCANClusterer(
    min_cluster_size=30,
    embedding_model="openai",
    hierarchical=True
)
dataset = clusterer(dataset)

# 5. Compute metrics
metrics = SideBySideMetrics()
dataset = metrics(dataset)

# 6. Save results
dataset.save("results/full_pipeline_output.json")
```

## Contributing

So uh, I'm still building this out a lot so maybe contribute when i have something more stable... but hey if you really wanna submit a PR i'll review it. 

If you want to know more about the nitty gritty abstractions and code structure, check out the [design doc](README_ABSTRACTION.md).

---

**Need help?** Check out the [documentation](https://lmm-vibes.readthedocs.io) or open an issue on GitHub. (JK this doesnt exist claude assumes i am more organized than i actually am)
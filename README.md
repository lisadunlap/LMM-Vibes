<div align="center">

# Whatever this is
### *Extract, cluster, and analyze behavioral properties from Large Language Models*

*Potential name options: VibeCheck, ReAgent, MindPalace, Autopilot, TempCheck, explAIner, Marcel (i just like that name, no relation to anything this does)*

---

**Help you understand how different generative models behave by automatically extracting behavioral properties from their responses, grouping similar behaviors together, and quantifying how important these behaviors are.**

</div>

## Installation

```bash
# Basic installation
pip install lmmvibes

# With optional dependencies
pip install lmmvibes[full]  # includes sentence-transformers, wandb
```

### Requirements

- Python 3.8+
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)
- Optional: Weights & Biases account for experiment tracking

## Quick Start

```python
import pandas as pd
from lmmvibes import explain

# Your data with model responses (can contain multiple models)
df = pd.DataFrame({
    "question_id": ["q1", "q2", "q3"], # unique id that specifies a uniqeu conversation (useful if there are duplicate prompts or many models)
    "prompt": ["What is machine learning?", "Explain quantum computing", "Write a poem about AI"],
    "model": ["gpt-4", "gpt-4", "gpt-4"],
    "model_response": ["Machine learning involves...", "QC leverages quantum...", "Silicon dreams awaken..."],
    "score": [{"accuracy": 1, "helpfulness": 4.2}, {"accuracy": 0, "helpfulness": 3.8}, {"accuracy": 1, "helpfulness": 4.5}]
})

# Extract and cluster behavioral properties
clustered_df, model_stats = explain(
    df,
    method="single_model",
    min_cluster_size=10,
    output_dir="results/test"
)

# Your data with model responses (for side-by-side comparrison)
df = pd.DataFrame({
    "question_id": ["q1", "q2", "q3"],
    "prompt": ["What is machine learning?", "Explain quantum computing", "Write a poem about AI"],
    "model_a": ["gpt-4", "gpt-4", "gpt-4"],
    "model_b": ["claude-3", "claude-3", "claude-3"],
    "model_a_response": ["ML is a subset of AI...", "Quantum computing uses...", "In circuits of light..."],
    "model_b_response": ["Machine learning involves...", "QC leverages quantum...", "Silicon dreams awaken..."],
    "score": [{"winner": "gpt-4", "helpfulness": 4.2}, {"winner": "gpt-4", "helpfulness": 3.8}, {"winner": "claude-3", "helpfulness": 4.5}]
})

# Extract and cluster behavioral properties
clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    min_cluster_size=10,
    output_dir="results/test"
)
```

### Viewing Results in Gradio
```bash
# View clusters, examples, and metrics
python -m lmmvibes.vis_gradio.launcher --share
```

## Outputs

After running `explain`, you receive two main outputs:

### `clustered_df`
A DataFrame containing your original data plus new columns for extracted and clustered behavioral properties:

- **property_description**: A short natural language description of a behavioral trait found in a model response (e.g., "Provides step-by-step reasoning").
- **category**: A higher-level grouping for the property (e.g., "Reasoning", "Creativity").
- **impact**: The estimated effect or importance of the property (e.g., "positive", "negative", or a numeric score).
- **type**: The kind of property (e.g., "format", "content", "style").
- **property_description_fine_cluster_label**: Human-readable label for the fine-grained cluster (e.g., "Step-by-step Reasoning").
- **property_description_coarse_cluster_label**: Human-readable label for the coarse-grained cluster (e.g., "Reasoning Transparency").

### `model_stats`
A dictionary or DataFrame with per-model behavioral analysis, including:
- Which behaviors each model exhibits most or least frequently.
- Relative scores for each model on different behavioral clusters.
- Example responses from each model for each behavior cluster.
- **Quality scores**: How well each model performs within specific behavioral clusters compared to their overall performance.
- **Normalized quality scores**: Consistent 0-1 scale metrics across different evaluation criteria.

This allows you to see not just which model "won" overall, but *why*—by surfacing the behavioral patterns and strengths/weaknesses of each model.

## Input Data Requirements

LMM-Vibes supports two analysis methods, each with specific data format requirements. The system automatically converts simple string responses to OpenAI format for processing.

### Single Model Analysis
Analyze behavioral patterns from individual model responses.

**Required Columns:**
| Column | Description | Example |
|--------|-------------|---------|
| `question_id` | Unique identifier for each question | `"q1"`, `"math_problem_001"` |
| `prompt` | The question or prompt given to the model | `"What is machine learning?"` |
| `model` | Name of the model being analyzed | `"gpt-4"`, `"claude-3-opus"` |
| `model_response` | The model's complete response (string or OAI format) | `"Machine learning is a subset..."` or OAI conversation format |

**Optional Columns:**
| Column | Description | Example |
|--------|-------------|---------|
| `score` | Dictionary of evaluation metrics | `{"accuracy": 0.85, "helpfulness": 4.2, "harmlessness": 4.8}` |
| `category` | Question category/topic | `"math"`, `"coding"`, `"creative"` |
| `difficulty` | Question difficulty level | `"easy"`, `"medium"`, `"hard"` |

**Example DataFrame:**
```python
df = pd.DataFrame({
    "question_id": ["q1", "q2", "q3"],
    "prompt": ["What is machine learning?", "Explain quantum computing", "Write a poem about AI"],
    "model": ["gpt-4", "gpt-4", "gpt-4"],
    "model_response": ["Machine learning involves...", "QC leverages quantum...", "Silicon dreams awaken..."],
    "score": [{"accuracy": 1, "helpfulness": 4.5}, {"accuracy": 0, "helpfulness": 3.8}, {"accuracy": 1, "helpfulness": 4.2}]
})
```

### Side-by-Side Comparisons
Compare two models head-to-head (Arena-style battles).

**Required Columns:**
| Column | Description | Example |
|--------|-------------|---------|
| `question_id` | Unique identifier for each question | `"q1"`, `"math_problem_001"` |
| `prompt` | The question given to both models | `"What is machine learning?"` |
| `model_a` | Name of the first model | `"gpt-4"`, `"claude-3-opus"` |
| `model_b` | Name of the second model | `"gpt-3.5-turbo"`, `"llama-2"` |
| `model_a_response` | Response from the first model (string or OAI format) | `"Machine learning is a subset..."` or OAI conversation format |
| `model_b_response` | Response from the second model (string or OAI format) | `"ML involves training algorithms..."` or OAI conversation format |

**Optional Columns:**
| Column | Description | Example |
|--------|-------------|---------|
| `score` | Dictionary with winner and metrics | `{"winner": "model_a", "helpfulness": 4.5, "accuracy": 1}` |

**Example DataFrame:**
```python
df = pd.DataFrame({
    "question_id": ["q1", "q2", "q3"],
    "prompt": ["What is machine learning?", "Explain quantum computing", "Write a poem about AI"],
    "model_a": ["gpt-4", "gpt-4", "gpt-4"],
    "model_b": ["claude-3", "claude-3", "claude-3"],
    "model_a_response": ["ML is a subset of AI...", "Quantum computing uses...", "In circuits of light..."],
    "model_b_response": ["Machine learning involves...", "QC leverages quantum...", "Silicon dreams awaken..."],
    "score": [{"winner": "gpt-4", "helpfulness": 4.2}, {"winner": "gpt-4", "helpfulness": 3.8}, {"winner": "claude-3", "helpfulness": 4.5}]
})
```

**Format Conversion:**
The system automatically detects and converts simple string responses to OpenAI conversation format. If your `model_response` (or `model_a_response`/`model_b_response`) is a string, it will be converted to:
```python
[
    {"role": "user", "content": "your_prompt"},
    {"role": "assistant", "content": "your_response"}
]
```

If your response is already in OpenAI format (a list of message dictionaries), it will be used as-is.

## Pipeline Components

LMM-Vibes follows a 4-stage pipeline architecture:

```
Data Input → Property Extraction → Post-processing → Clustering → Metrics & Analysis
```

<details>
<summary><strong>1. Property Extraction</strong> (<code>lmmvibes.extractors</code>)</summary>

**Goal**: Identify specific behavioral properties from model responses using LLM analysis.

Takes each conversation and asks an LLM (like GPT-4) to extract behavioral properties such as:
- "Provides step-by-step reasoning"
- "Uses formal/informal tone" 
- "Includes creative examples"
- "Shows uncertainty appropriately"

**Available extractors:**
- `OpenAIExtractor` - Uses OpenAI API (GPT models)
- `VLLMExtractor` - Uses local models via vLLM
- `BatchExtractor` - Makes a file for the batch API

</details>

<details>
<summary><strong>2. Post-processing</strong> (<code>lmmvibes.postprocess</code>)</summary>

**Goal**: Parse and validate the extracted properties into structured data.

- `LLMJsonParser` - Converts raw LLM responses into structured property objects
- `PropertyValidator` - Ensures properties meet quality standards and required fields

</details>

<details>
<summary><strong>3. Clustering</strong> (<code>lmmvibes.clusterers</code>)</summary>

**Goal**: Group similar behavioral properties into coherent clusters for analysis.

Takes individual properties like "explains step-by-step" and "shows work clearly" and groups them into clusters like "Reasoning Transparency". Creates both fine-grained and coarse-grained cluster hierarchies.

**Available clusterers:**
- `HDBSCANClusterer` - Density-based clustering (recommended for >10k samples)
- `HierarchicalClusterer` - Traditional hierarchical clustering with LLM-powered naming

</details>

<details>
<summary><strong>4. Metrics & Analysis</strong> (<code>lmmvibes.metrics</code>)</summary>

**Goal**: Calculate model performance statistics and behavioral rankings.

Computes which models excel at which behavioral patterns:
- Model scores per behavior cluster
- Relative strengths/weaknesses between models  
- Statistical significance of differences
- Example responses for each cluster
- **Quality scores**: How well each model performs within specific behavioral clusters compared to their overall performance
- **Normalized quality scores**: Consistent 0-1 scale metrics across different evaluation criteria
- **Multi-dimensional scoring**: Support for multiple evaluation metrics (accuracy, helpfulness, harmlessness, etc.)

**Available metrics:**
- `SideBySideMetrics` - For model comparison data with winner determination
- `SingleModelMetrics` - For individual model analysis with performance metrics

**Metric Definitions:**

For a given model *m* and cluster *c*:

**Representation Score:**
```
prop(m, c)   =   #questions where m shows c   /   #questions answered by m
score(m, c)  =   prop(m, c) / median_{m'} prop(m', c)
```

**Quality Score:**
For each score key *k* in the multi-dimensional score dictionary:
```
quality_score(k, c) = avg_{m in c} (avg_score(m, c, k) / avg_score(m, global, k))
```

**Normalized Quality Score:**
```
normalized_score(k, c) = (cluster_avg(k, c) - global_min(k)) / (global_max(k) - global_min(k))
```

**Interpretation:**
- `score > 1`: Model is **over-represented** in that cluster
- `score < 1`: Model is **under-represented** in that cluster  
- `quality_score > 1`: Models in this cluster perform better than their global average
- `quality_score < 1`: Models in this cluster perform worse than their global average
- `normalized_score` close to 1.0: Cluster performs near the best observed performance
- `normalized_score` close to 0.0: Cluster performs near the worst observed performance

</details>

## Configuration Options

```python
clustered_df, model_stats = explain(
    df,
    method="side_by_side",              # or "single_model"
    system_prompt=None # custom extraction prompt (you can also omit this and we will use our default)
    min_cluster_size=30,                # minimum cluster size for a property 
    embedding_model="openai",           # or any sentence-transformer model
    hierarchical=True,                  # create coarse clusters
    output_dir="results/",              # save outputs here
    use_wandb=True,                     # log to Weights & Biases
    wandb_project="my-lmm-analysis"
)
```

## Output Files

When you specify `output_dir`, LMM-Vibes automatically saves:

| File | Description |
|------|-------------|
| `clustered_results.parquet` | Full dataset with properties and clusters |
| `full_dataset.json` | Complete dataset in JSON format |
| `model_stats.json` | Per-model behavioral statistics with quality scores |
| `summary.txt` | Human-readable analysis summary |

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
        if hasattr(behavior, 'quality_scores'):
            print(f"    Quality scores: {behavior.quality_scores}")
```

### Understanding Model Capabilities with Multi-dimensional Scoring
```python
# Single model analysis with multiple evaluation criteria
df = pd.DataFrame({
    "question_id": range(100),
    "prompt": ["Explain quantum physics"] * 100,
    "model": ["gpt-4"] * 100,
    "response": responses,  # your model responses
    "score": [
        {"accuracy": 1, "helpfulness": 4.2, "harmlessness": 4.8},
        {"accuracy": 0, "helpfulness": 3.5, "harmlessness": 4.9},
        # ... more scores
    ]
})

clustered_df, stats = explain(df, method="single_model")

# Analyze quality scores across different metrics
for model, model_stats in stats.items():
    print(f"\n{model} performance by behavioral cluster:")
    for behavior in model_stats["fine"][:3]:
        print(f"  • {behavior.property_description}")
        if hasattr(behavior, 'normalized_quality_scores'):
            for metric, score in behavior.normalized_quality_scores.items():
                print(f"    {metric}: {score:.3f}")
```

## Advanced: Running Pipeline Components

<details>
<summary><strong>For more control, you can run each pipeline stage separately:</strong></summary>

```python
from lmmvibes.core import PropertyDataset
from lmmvibes.extractors import OpenAIExtractor
from lmmvibes.postprocess import LLMJsonParser, PropertyValidator
from lmmvibes.clusterers import HDBSCANClusterer
from lmmvibes.metrics import SideBySideMetrics, SingleModelMetrics

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
metrics = SideBySideMetrics()  # or SingleModelMetrics()
dataset = metrics(dataset)

# 6. Save results
dataset.save("results/full_pipeline_output.json")
```

</details>

## Contributing

So uh, I'm still building this out a lot so maybe contribute when i have something more stable... but hey if you really wanna submit a PR i'll review it. 

If you want to know more about the nitty gritty abstractions and code structure, check out the [design doc](README_ABSTRACTION.md).

---

## TODO
- Add auto conversation conversion (both from input -> string and string -> unified OAI format for website vis)
- Change so model response is whatever, then we save `converted_response` and use model response for vis purposes
- Add image support (should be somewhat easy)
- Change quality score to certer around either 0 or 1 (maybe we just take the difference between cluster score and avg score)
- Add functionality for MAST/providing your own taxonomy

<div align="center">

**Need help?** Check out the [documentation](https://lmm-vibes.readthedocs.io) or open an issue on GitHub. 

*(JK this doesnt exist claude assumes i am more organized than i actually am)*

</div>
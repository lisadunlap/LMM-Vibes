# LMM-Vibes – Abstraction README

_A high-level design & user guide for the LMM-Vibes package._

---

## 0. Quick-start (public API)

```python
from lmmvibes import explain          # pip install lmmvibes

clustered_df, model_stats = explain(
    df,                                 # pandas DataFrame with conversation data
    method="side_by_side",              # or "single_model"
    system_prompt=None,                 # auto-determined based on data format
    # Extraction parameters
    model_name="gpt-4o-mini",          # LLM for property extraction
    temperature=0.7,
    max_workers=16,
    # Clustering parameters  
    clusterer="hdbscan",                # or "hdbscan_native", "hierarchical"
    min_cluster_size=30,
    embedding_model="openai",           # or any sentence-transformer model
    hierarchical=False,                 # create coarse clusters
    # Output & logging
    output_dir="results/",              # automatically save results
    use_wandb=True,                     # log to Weights & Biases
    verbose=True
)
```

* **`clustered_df`** – original rows plus  
  • extracted properties (`property_description`, `category`, `impact`, `type`, `reason`, `evidence`)  
  • cluster ids & labels (`property_description_fine_cluster_id`, `property_description_coarse_cluster_id`, etc.)
  • embeddings (optional, `property_description_embedding`)

* **`model_stats`** – `{model_name → {"fine": [ModelStats...], "coarse": [ModelStats...]}}` sorted by score.

That's it—no pipeline objects, no loaders—just vibes.

---

## 1. Why abstract?

1. **Separation of concerns** – loading, extraction, clustering, metrics each live in their own module.  
2. **Plug-and-play** – add new extractors / clusterers / metrics without touching the rest.  
3. **Single data contract** – every stage passes a `PropertyDataset` object.  
4. **User-friendly facade** – `explain` hides all complexity but the rich pipeline remains for power users.
5. **Migration path** – wrap existing code into stages with consistent interfaces.

---

## 2. Core data objects (`lmmvibes.core`)

The core data objects define the data contract that flows between pipeline stages:

```python
@dataclass
class ConversationRecord:
    """A single conversation with prompt, responses, and metadata."""
    question_id: str
    prompt: str
    model: str | List[str]  # model name(s) - single string or list for side-by-side comparisons
    responses: str | List[str] # responses matching model format
    scores: Dict[str, Any]     # {score_name: score_value}
    meta: Dict[str, Any] = field(default_factory=dict)  # winner, language, etc.

@dataclass  
class Property:
    """An extracted behavioral property from a model response."""
    id: str
    question_id: str
    model: str
    # Parsed fields (filled by LLMJsonParser)
    property_description: Optional[str] = None
    category: Optional[str] = None
    type: Optional[str] = None  # "Context-Specific" or "General"
    impact: Optional[str] = None  # "High", "Medium", "Low"
    reason: Optional[str] = None
    evidence: Optional[str] = None
    user_preference_direction: Optional[str] = None # Capability-focused|Experience-focused|Neutral|Negative
    raw_response: Optional[str] = None
    contains_errors: Optional[bool] = None
    unexpected_behavior: Optional[bool] = None

@dataclass
class Cluster:
    """A cluster of properties."""
    id: str # fine cluster id
    label: str # fine cluster label
    size: int # fine cluster size
    parent_id: str | None = None # coarse cluster id
    parent_label: str | None = None # coarse cluster label
    property_descriptions: List[str] = field(default_factory=list) # property descriptions in the cluster
    question_ids: List[str] = field(default_factory=list) # ids of the conversations in the cluster

@dataclass
class ModelStats:
    """Model statistics for a cluster."""
    property_description: str # name of property cluster (either fine or coarse)
    model_name: str # name of model we are comparing
    cluster_size_global: int # number of properties in the cluster
    score: float # score of the property cluster
    quality_score: Dict[str, Any] # quality score of the property cluster
    size: int # number of properties in the cluster for this model
    proportion: float # proportion of model's responses showing this behavior
    examples: List[str] # example property id's in the cluster
    metadata: Dict[str, Any] = field(default_factory=dict) # all other metadata

@dataclass
class PropertyDataset:
    """Container for all data flowing through the pipeline."""
    conversations: List[ConversationRecord] = field(default_factory=list)
    all_models: List[str] = field(default_factory=list)
    properties: List[Property] = field(default_factory=list)
    clusters: List[Cluster] = field(default_factory=list)
    model_stats: Dict[str, Any] = field(default_factory=dict)
```

Key features:
- **ConversationRecord**: Handles both single-model and side-by-side comparisons
- **Property**: Rich metadata about extracted behavioral properties
- **Cluster**: Supports hierarchical clustering with parent/child relationships
- **ModelStats**: Computed statistics for model performance in each cluster
- **PropertyDataset**: The core data contract passed between pipeline stages

The dataset provides rich DataFrame conversion and persistence methods:

```python
# Load/save in multiple formats
dataset = PropertyDataset.from_dataframe(df, method="side_by_side")
dataset.save("results.json")  # or .parquet, .pickle
dataset = PropertyDataset.load("results.json")

# Convert to DataFrame with different views
df = dataset.to_dataframe(type="all")  # or "base", "properties", "clusters"
```

---

## 3. Stage interface (internal)

```python
class PipelineStage(ABC):
    @abstract
    def run(self, data: PropertyDataset) -> PropertyDataset: ...
```

Concrete subclasses:

* **Extractors** – `OpenAIExtractor`, `BatchExtractor`, `VLLMExtractor`
* **Post-processors** – `LLMJsonParser`, `PropertyValidator`
* **Clusterers** – `HDBSCANClusterer`, `HDBSCANNativeClusterer`, `HierarchicalClusterer`
* **Metrics** – `SideBySideMetrics`, `SingleModelMetrics`

Mix-ins (`LoggingMixin`, `TimingMixin`, `ErrorHandlingMixin`, `WandbMixin`, `CacheMixin`) add cross-cutting behaviour (logging, timing, tqdm, wandb, caching).  
They are always listed **before** `PipelineStage` in the inheritance list, e.g.

```python
class OpenAIExtractor(LoggingMixin, TimingMixin, ErrorHandlingMixin,
                      WandbMixin, PipelineStage):
    ...
```

Each mix-in initialises only its own state and then calls `super().__init__()` **without keyword arguments**.  This "mixin-first / co-operative super" pattern prevents the classic

```
TypeError: object.__init__() takes exactly one argument
```

while still letting every mix-in in the MRO chain run its setup code.

---

## 4. How `explain` works under the hood

```
User DataFrame  ──▶ PropertyDataset wrapper
                  │
                  ├─ OpenAIExtractor(system_prompt, model="gpt-4o-mini")
                  ├─ LLMJsonParser()  # Parse JSON responses, handle errors
                  ├─ PropertyValidator()  # Clean/validate extracted properties
                  ├─ HDBSCANClusterer(min_cluster_size=30, embedding_model="openai")
                  └─ SideBySideMetrics()  # Calculate frequency metrics
                  ▼
        (clustered_df, model_stats)
```

All glued together by an internal `Pipeline` class, but invisible to the casual user.

Signature (complete):

```python
def explain(
        df: pd.DataFrame,
        method: str = "single_model",                # or "side_by_side"
        system_prompt: str = None,                   # auto-determined if None
        prompt_builder: Callable[[pd.Series], str] | None = None,
        *,
        # Extraction parameters
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 16000,
        max_workers: int = 16,
        # Clustering parameters  
        clusterer: str | Clusterer = "hdbscan",
        min_cluster_size: int = 30,
        embedding_model: str = "openai",
        hierarchical: bool = False,
        assign_outliers: bool = False,
        max_coarse_clusters: int = 25,
        # Metrics parameters
        metrics_kwargs: dict | None = None,
        # Caching & logging
        use_wandb: bool = True,
        wandb_project: str | None = None,
        include_embeddings: bool = True,
        verbose: bool = True,
        # Output parameters
        output_dir: str | None = None,
        # Pipeline configuration
        custom_pipeline: Pipeline | None = None,
        # Cache configuration
        extraction_cache_dir: str | None = None,
        clustering_cache_dir: str | None = None,
        metrics_cache_dir: str | None = None,
        **kwargs
) -> tuple[pd.DataFrame, dict]:
    ...
```

Advanced users can pass:

* Custom `Clusterer` instances with specific configurations
* `metrics_kwargs` for alternative scoring formulas
* Full control over LLM parameters and caching
* Custom pipeline for complete control

---

## 5. Factory functions and stage configuration

### 5.1 Extractors (`lmmvibes.extractors`)

```python
from lmmvibes.extractors import get_extractor, OpenAIExtractor, BatchExtractor

# Factory function for automatic extractor selection
extractor = get_extractor(
    model_name="gpt-4o-mini",          # Auto-selects OpenAIExtractor
    system_prompt="one_sided_system_prompt",
    temperature=0.7,
    max_workers=16
)

# Direct instantiation
extractor = OpenAIExtractor(
    model="gpt-4o-mini",
    system_prompt="one_sided_system_prompt",
    temperature=0.7,
    max_workers=16
)
```

**Available extractors:**
- `OpenAIExtractor` - Uses OpenAI API (GPT models)
- `VLLMExtractor` - Uses local models via vLLM
- `BatchExtractor` - Creates batch files for batch API processing

### 5.2 Post-processors (`lmmvibes.postprocess`)

```python
from lmmvibes.postprocess import LLMJsonParser, PropertyValidator

# Parse raw LLM responses into structured properties
parser = LLMJsonParser()

# Validate and clean extracted properties
validator = PropertyValidator()
```

### 5.3 Clusterers (`lmmvibes.clusterers`)

```python
from lmmvibes.clusterers import get_clusterer

# Factory function for automatic clusterer selection
clusterer = get_clusterer(
    method="hdbscan",                   # or "hdbscan_native", "hierarchical"
    min_cluster_size=30,
    embedding_model="openai",
    hierarchical=True,
    assign_outliers=False
)
```

**Available clusterers:**
- `HDBSCANClusterer` - Density-based clustering (recommended for >10k samples)
- `HDBSCANNativeClusterer` - Native HDBSCAN implementation
- `HierarchicalClusterer` - Traditional hierarchical clustering with LLM-powered naming

### 5.4 Metrics (`lmmvibes.metrics`)

```python
from lmmvibes.metrics import get_metrics

# Factory function for automatic metrics selection
metrics = get_metrics(
    method="side_by_side",              # or "single_model"
    output_dir="outputs/metrics"
)
```

**Available metrics:**
- `SideBySideMetrics` - For model comparison data (Arena-style)
- `SingleModelMetrics` - For individual model analysis

---

## 6. Pipeline orchestration (advanced use)

```python
from lmmvibes.pipeline import Pipeline, PipelineBuilder
from lmmvibes.extractors import get_extractor
from lmmvibes.postprocess import LLMJsonParser, PropertyValidator
from lmmvibes.clusterers import get_clusterer
from lmmvibes.metrics import get_metrics

# Method 1: Using PipelineBuilder (recommended)
builder = PipelineBuilder(name="Custom-Pipeline")
pipeline = (builder
    .extract_properties(get_extractor(model_name="gpt-4o-mini"))
    .parse_properties(LLMJsonParser())
    .add_stage(PropertyValidator())
    .cluster_properties(get_clusterer(method="hdbscan", hierarchical=True))
    .compute_metrics(get_metrics(method="side_by_side"))
    .configure(use_wandb=True, verbose=True)
    .build()
)

# Method 2: Direct Pipeline construction
pipeline = Pipeline("Custom-Pipeline", [
    get_extractor(model_name="gpt-4o-mini"),
    LLMJsonParser(),
    PropertyValidator(),
    get_clusterer(method="hdbscan", hierarchical=True),
    get_metrics(method="side_by_side")
], use_wandb=True, verbose=True)

# Run the pipeline
dataset = PropertyDataset.from_dataframe(df, method="side_by_side")
result = pipeline.run(dataset)

# Convert back to DataFrame format
clustered_df = result.to_dataframe()
model_stats = result.model_stats
```

---

## 7. System prompts and auto-detection

The `explain` function automatically determines the appropriate system prompt based on your data:

```python
from lmmvibes.prompts import get_default_system_prompt

# Auto-detection based on method and data format
system_prompt = get_default_system_prompt(
    method="side_by_side", 
    contains_score=True  # auto-detected from DataFrame
)
```

**Available system prompts:**
- `sbs_w_metrics_system_prompt` - Side-by-side with score/preference data
- `one_sided_system_prompt_no_examples` - Side-by-side without scores
- `single_model_system_prompt` - Single model with scores
- `single_model_no_score_system_prompt` - Single model without scores
- `webdev_system_prompt` - Web development specific prompts
- Agent-specific prompts for agentic environments

---

## 8. Output files and persistence

When you specify `output_dir`, LMM-Vibes automatically saves:

| File | Description |
|------|-------------|
| `clustered_results.json` | Complete DataFrame with clusters (JSON format) |
| `clustered_results.parquet` | Complete DataFrame with clusters (parquet format) |
| `full_dataset.json` | Complete PropertyDataset object (JSON format) |
| `full_dataset.parquet` | Complete PropertyDataset object (parquet format) |
| `model_stats.json` | Per-model behavioral statistics |
| `parsing_failures.json` | Detailed parsing failure information |
| `summary.txt` | Human-readable summary with dataset counts and leaderboard |

These filenames are **stable** so downstream notebooks and dashboards can rely on them.

---

## 9. Package layout

```
lmmvibes/
│
├─ core/                    # data objects & stage base classes
│  ├─ __init__.py
│  ├─ data_objects.py       # PropertyDataset, ConversationRecord, Property, Cluster, ModelStats
│  ├─ stage.py              # PipelineStage ABC
│  └─ mixins.py             # LoggingMixin, CacheMixin, etc.
│
├─ extractors/              # property extraction
│  ├─ __init__.py           # get_extractor factory function
│  ├─ openai.py             # OpenAIExtractor
│  ├─ batch.py              # BatchExtractor
│  └─ vllm.py               # VLLMExtractor
│
├─ postprocess/             # post-processing stages
│  ├─ __init__.py
│  ├─ parser.py             # LLMJsonParser
│  └─ validator.py          # PropertyValidator
│
├─ clusterers/              # clustering
│  ├─ __init__.py           # get_clusterer factory function
│  ├─ hdbscan.py            # HDBSCANClusterer, HDBSCANNativeClusterer
│  ├─ hierarchical.py       # HierarchicalClusterer
│  ├─ clustering_prompts.py # LLM prompts for cluster naming
│  └─ clustering_utils.py   # Shared clustering utilities
│
├─ metrics/                 # metrics computation
│  ├─ __init__.py           # get_metrics factory function
│  ├─ side_by_side.py       # SideBySideMetrics
│  └─ single_model.py       # SingleModelMetrics
│
├─ prompts/                 # prompt templates
│  ├─ __init__.py           # get_default_system_prompt
│  ├─ extractor_prompts.py  # Property extraction prompts
│  └─ agents.py             # Agent-specific prompts
│
├─ viz/                     # optional Streamlit dashboards
│  ├─ __init__.py
│  └─ interactive_app.py    # summary visualization
│
├─ public.py                # explain() function lives here
├─ pipeline.py              # Pipeline & PipelineBuilder classes
└─ __init__.py              # re-export explain
```

No circular imports—subpackages only depend on `core` or lower layers.

---

## 10. Interactive visualisation (Streamlit)

The package ships with a zero-config **Streamlit** app that lets you *browse clusters visually*:

```bash
# View clusters, examples, and metrics
streamlit run lmmvibes/viz/interactive_app.py -- --dataset results/clustered_results.parquet
```

Key features:

1. Sidebar selector for **coarse → fine** clusters.
2. Click-through to see **all property descriptions** in a cluster.
3. Global *overview accordion* that lists every coarse cluster and its fine children.
4. Optional bar-chart and word-cloud summaries.
5. Accepts either a full `PropertyDataset` file *(json / pkl / parquet)* **or** a clusters DataFrame saved as JSONL / CSV / Parquet.

---

## 11. Advanced features

### 11.1 Caching

```python
# Enable caching for expensive operations
clustered_df, model_stats = explain(
    df,
    extraction_cache_dir="cache/extraction/",
    clustering_cache_dir="cache/clustering/",
    metrics_cache_dir="cache/metrics/"
)
```

### 11.2 Custom pipelines

```python
from lmmvibes import explain_with_custom_pipeline

# Use a completely custom pipeline
clustered_df, model_stats = explain_with_custom_pipeline(
    df,
    pipeline=my_custom_pipeline,
    method="side_by_side"
)
```

### 11.3 Metrics-only computation

```python
from lmmvibes import compute_metrics_only

# Recompute metrics on existing pipeline results
clustered_df, model_stats = compute_metrics_only(
    input_path="results/previous_run/full_dataset.json",
    method="side_by_side",
    output_dir="results/metrics_only"
)
```

### 11.4 Convenience functions

```python
from lmmvibes import explain_side_by_side, explain_single_model

# Convenience functions for common use cases
clustered_df, model_stats = explain_side_by_side(df, min_cluster_size=20)
clustered_df, model_stats = explain_single_model(df, hierarchical=True)
```

---

## 12. Extending the ecosystem

| Goal | What to implement | Example |
|------|-------------------|---------|
| New extraction technique | `class MyExtractor(PipelineStage)` | Wrap your existing LLM calling code |
| Alternative clustering  | `class MyClusterer(PipelineStage)` | Wrap existing clustering functions |
| Different metric        | `class MyMetrics(PipelineStage)` | Implement new metric calculations |
| New prompt library      | Add to `prompts/` module | Add domain-specific prompts |
| Custom post-processing  | `class MyValidator(PipelineStage)` | Wrap existing cleaning logic |
| Batch API support       | `class BatchExtractor(PipelineStage)` | Migrate batch logic |

All new stages become instantly available to `explain` via factory functions or passed as instances.

---

## 13. Migration status & TODO

### ✅ Already implemented in the new package

* Core data objects (`ConversationRecord`, `Property`, `Cluster`, `ModelStats`, `PropertyDataset`)
* `PipelineStage` interface & `Pipeline` / `PipelineBuilder`
* Mixin stack (Logging, Timing, ErrorHandling, Wandb, Cache) using the new *mixin-first* pattern
* Working stages
  * `OpenAIExtractor` (rewritten from `generate_differences.py`)
  * `LLMJsonParser` (from `post_processing.py`)
  * `PropertyValidator`
  * `HDBSCANClusterer` (from `clustering/hierarchical_clustering.py`)
  * `SideBySideMetrics` (computes model behavior scores for each cluster)
  * `SingleModelMetrics` (for individual model analysis)
* End-to-end `explain()` public API with wandb logging
* Factory functions for automatic stage selection
* Auto-detection of system prompts based on data format
* Comprehensive output file generation
* Tests: extraction → parsing → clustering → metrics integration
* **Interactive Streamlit viewer** (`lmmvibes.viz.interactive_app`)
* Caching support for expensive operations
* Custom pipeline support
* Metrics-only computation mode
* Convenience functions for common use cases

### 🔜 Still to integrate from the original notebooks / scripts

| Component | Source file | Target module |
|-----------|-------------|---------------|
| Native / hierarchical clustering | `clustering/hierarchical_clustering.py` | `lmmvibes.clusterers.hdbscan_native_hierarchical` |
| Batch & vLLM extractors | `generate_differences.py` / `vllm` | `lmmvibes.extractors.batch`, `vllm` |
| Data-loader helpers | `data_loader.py` | `lmmvibes.datasets.*` (arena, webdev, etc.) |
| CLI + YAML runner | n/a | `lmmvibes.cli` |
| Comprehensive test suite | n/a | `tests/` |

Nice-to-have polish after migration:

* Outlier reassignment & hierarchical summaries
* Documentation examples & badges
* PyPI packaging scripts

Feel free to tick items off this list as they land. 🎉 

---

## 14. Validation & Testing

Each pipeline stage has dedicated tests to verify its behavior in isolation and as part of the full pipeline:

```python
def test_hdbscan_clusterer_basic():
    """Test basic HDBSCAN clustering functionality."""
    # Load test dataset
    dataset = PropertyDataset.load("tests/outputs/arena_first50_dataset.json")
    
    # Skip if not enough data
    valid_props = [p for p in dataset.properties if p.property_description]
    if len(valid_props) < 3:
        pytest.skip("Not enough parsed properties to cluster.")

    # Run clustering
    clusterer = HDBSCANClusterer(
        min_cluster_size=2,  # Small for test data
        hierarchical=False,
        include_embeddings=False
    )
    result = clusterer(dataset)

    # Verify results
    assert len(result.clusters) > 0, "Should create at least one cluster"
    
    # Check cluster assignments
    for p in result.properties:
        if p.property_description:
            assert hasattr(p, "fine_cluster_id"), "Properties should have cluster IDs"
            
    # Save results for inspection
    result.save("tests/outputs/hdbscan_clustered_results.parquet", format="parquet")

def test_explain():
    """Test the full pipeline with clustering."""
    df = pd.DataFrame({
        "question_id": ["q1", "q2", "q3"],
        "prompt": ["What is X?", "How does Y work?", "Compare A and B"],
        "model_a": ["gpt-4", "gpt-4", "gpt-4"],
        "model_b": ["claude-2", "claude-2", "claude-2"],
        "model_a_response": ["X is...", "Y works by...", "A is better because..."],
        "model_b_response": ["X means...", "Y uses...", "B has advantages in..."],
        "winner": ["model_a", "model_b", "tie"]
    })
    
    # Run full pipeline
    clustered_df, model_stats = explain(
        df,
        method="side_by_side",
        clusterer="hdbscan",
        min_cluster_size=2,
        embedding_model="openai"
    )
    
    # Verify outputs
    assert "property_description_fine_cluster_label" in clustered_df.columns
    assert len(model_stats) > 0
```

The test suite:
1. Tests each stage in isolation with small, controlled inputs
2. Verifies the full pipeline with realistic data
3. Saves intermediate outputs for manual inspection
4. Uses pytest fixtures to manage test data and dependencies

---

## 15. Road-map

1. **Phase 1-2**: Refactor existing scripts into stage subclasses (~2-3 weeks) ✅
2. **Phase 3**: Implement public API and CLI (~1 week) ✅  
3. **Phase 4**: Polish, test, and publish (~1 week) 🔜
4. **Future**: Advanced features (distributed processing, more clusterers, richer metrics)

With this README you have both:  
• a dead-simple **`explain` one-liner** for everyday users, and  
• a robust, extensible pipeline that **directly migrates your existing code** for researchers and power users. 

---

## 16. Automatic output files

When you pass `output_dir="some/path"` to `explain()` the library automatically writes a tidy bundle of artefacts:

| File | What it contains |
|------|------------------|
| `clustered_results.json` | Full DataFrame with conversations, extracted properties and cluster columns (JSON format). |
| `clustered_results.parquet` | Same DataFrame in efficient columnar parquet format. |
| `full_dataset.json` | Complete `PropertyDataset` serialised as JSON. |
| `full_dataset.parquet` | Same dataset in efficient columnar parquet format. |
| `model_stats.json` | Per-model statistics (fine & coarse clusters, scores, examples). |
| `parsing_failures.json` | Detailed parsing failure information for debugging. |
| `summary.txt` | Human-readable summary with dataset counts and a leaderboard. |

These filenames are **stable** so downstream notebooks and dashboards can rely on them without additional configuration. 
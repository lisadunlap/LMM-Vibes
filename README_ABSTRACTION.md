# LMM-Vibes – Abstraction README

_A high-level design & user guide for the forthcoming pip package._

---

## 0. Quick-start (public API)

```python
from lmmvibes import explain          # pip install lmmvibes

clustered_df, model_stats = explain(
    df,                                 # pandas DataFrame with columns: prompt, responses, score (+ any metadata)
    method="side_by_side",              # or "single_model"
    system_prompt="one_sided_system_prompt", # systems prompt to extract properties, optional
    prompt_builder=build_pair_prompt,   # your own function; optional
    clusterer="hdbscan",                # or "hdbscan_native", "hierarchical"
    min_cluster_size=30,                # clustering parameters
    embedding_model="openai"            # or any sentence-transformer model
)
```

* **`clustered_df`** – original rows plus  
  • extracted properties (`property_description`, `category`, `impact`, `type`, `reason`, `evidence`)  
  • cluster ids & labels (`property_description_fine_cluster_id`, `property_description_coarse_cluster_id`, etc.)
  • embeddings (optional, `property_description_embedding`, `property_description_fine_cluster_label_embedding`)

* **`model_stats`** – `{model_name → [property_dict …]}` sorted by the chosen metric (frequency, proportion, etc.).

That's it—no pipeline objects, no loaders—just vibes.

---

## 1. Why abstract?

1. **Separation of concerns** – loading, extraction, clustering, metrics each live in their own module.  
2. **Plug-and-play** – add new extractors / clusterers / metrics without touching the rest.  
3. **Single data contract** – every stage passes a `PropertyDataset` object.  
4. **User-friendly facade** – `explain` hides all complexity but the rich pipeline remains for power users.
5. **Migration path** – wrap existing code (`generate_differences.py`, `clustering/hierarchical_clustering.py`) into stages.

---

## 2. Core data objects (`lmmvibes.core`)

The core data objects define the data contract that flows between pipeline stages:

```python
@dataclass
class ConversationRecord:
    """A single conversation with prompt, responses, and metadata."""
    question_id: str
    prompt: str
    model: str | tuple[str, str]  # model name(s) - single string or tuple for side-by-side comparisons
    responses: str | tuple[str, str] # responses matching model format
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
    @abstractmethod
    def run(self, data: PropertyDataset) -> PropertyDataset: ...
```

Concrete subclasses:

* **Extractors** – `OpenAIExtractor`, `BatchExtractor`, `VLLMExtractor`
* **Post-processors** – `LLMJsonParser`, `PropertyValidator`, `NaNPruner`
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
                  ├─ OpenAIExtractor(system_prompt, prompt_builder, model="gpt-4o-mini")
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
        method: str = "side_by_side",                # or "single_model"
        system_prompt: str = "one_sided_system_prompt",
        prompt_builder: Callable[[pd.Series], str] | None = None,
        *,
        # Extraction parameters
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 16000,
        max_workers: int = 16,
        # Clustering parameters  
        clusterer: str | Clusterer = "hdbscan",
        min_cluster_size: int = 30,
        embedding_model: str = "openai",
        hierarchical: bool = False,
        assign_outliers: bool = False,
        # Metrics parameters
        metrics_kwargs: dict | None = None,
        # Caching & logging
        use_wandb: bool = True,
        wandb_project: str | None = None,
        include_embeddings: bool = True,
) -> tuple[pd.DataFrame, dict]:
    ...
```

Advanced users can pass:

* Custom `Clusterer` instances with specific configurations
* `metrics_kwargs` for alternative scoring formulas
* Full control over LLM parameters and caching

---

## 5. Migration from existing code

### 5.1 Extraction Stage (from `generate_differences.py`)

```python
class OpenAIExtractor(PipelineStage, LoggingMixin):
    def __init__(self, system_prompt: str, prompt_builder: Callable, 
                 model: str = "gpt-4o-mini", temperature: float = 0.6, 
                 max_workers: int = 16):
        self.system_prompt = system_prompt
        self.prompt_builder = prompt_builder
        self.model = model
        self.temperature = temperature
        self.max_workers = max_workers
    
    def run(self, data: PropertyDataset) -> PropertyDataset:
        # Migrate your process_openai_batch logic here
        # Format conversations using prompt_builder
        # Extract properties using litellm
        # Return updated PropertyDataset with populated properties
        pass
```

### 5.2 Clustering Stage

The clustering stage uses HDBSCAN to group similar properties and optionally create a hierarchical structure. The implementation is split into configuration and execution:

```python
@dataclass
class ClusterConfig:
    """Configuration for clustering operations."""
    min_cluster_size: int = 30
    embedding_model: str = "openai"
    verbose: bool = True
    include_embeddings: bool = True
    context: Optional[str] = None
    hierarchical: bool = False
    assign_outliers: bool = False
    min_grandparent_size: int = 5
    max_coarse_clusters: int = 15
    # Dimension reduction settings
    dim_reduction_method: str = "adaptive"  # "adaptive", "umap", "pca", "none"
    umap_n_components: int = 100
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    # Wandb logging
    use_wandb: bool = True
    wandb_project: Optional[str] = None

class HDBSCANClusterer(PipelineStage, LoggingMixin, TimingMixin, WandbMixin):
    """HDBSCAN clustering stage."""
    
    def __init__(
        self,
        min_cluster_size: int = 30,
        embedding_model: str = "openai", 
        hierarchical: bool = False,
        assign_outliers: bool = False,
        include_embeddings: bool = True,
        use_wandb: bool = False,
        wandb_project: str = None,
    ):
        super().__init__(use_wandb=use_wandb, wandb_project=wandb_project)
        self.min_cluster_size = min_cluster_size
        self.embedding_model = embedding_model
        self.hierarchical = hierarchical
        self.assign_outliers = assign_outliers
        self.include_embeddings = include_embeddings
```

Usage example:

```python
from lmmvibes.clusterers.hdbscan import HDBSCANClusterer

# Load dataset with extracted properties
dataset = PropertyDataset.load("extracted_properties.json")

# Configure and run clustering
clusterer = HDBSCANClusterer(
    min_cluster_size=30,
    embedding_model="openai",
    hierarchical=True,  # Enable hierarchical clustering
    assign_outliers=True,  # Try to assign outlier points to nearest cluster
    include_embeddings=True  # Keep embeddings in output for visualization
)

# Run clustering
result = clusterer(dataset)

# Inspect results
print(f"Created {len(result.clusters)} clusters")
for cluster in result.clusters:
    print(f"\nCluster {cluster.id}: {cluster.label}")
    print(f"Size: {cluster.size}")
    if cluster.parent_id:
        print(f"Parent cluster: {cluster.parent_label} ({cluster.parent_id})")
    print("Sample properties:")
    for desc in cluster.property_descriptions[:3]:
        print(f"  • {desc}")

# Save results
result.save("clustered_results.parquet", format="parquet")
```

The clustering process:
1. Computes embeddings for property descriptions using the specified model
2. Optionally reduces dimensionality using UMAP (adaptive based on data size)
3. Runs HDBSCAN to create fine-grained clusters
4. If hierarchical=True, creates a second level of coarse clusters
5. Assigns cluster IDs and labels back to the original properties
6. Optionally logs results to Weights & Biases for visualization

### 5.3 Post-processing Stage (from `post_processing.py`)

```python
class LLMJsonParser(PipelineStage):
    def run(self, data: PropertyDataset) -> PropertyDataset:
        # Parse JSON responses from raw differences
        # Handle parsing errors gracefully
        # Convert parsed dicts to Property objects
        # Filter out invalid/incomplete properties
        pass
```

---

## 6. Pipeline orchestration (advanced use)

```python
from lmmvibes.pipeline import Pipeline
from lmmvibes.extractors import OpenAIExtractor, BatchExtractor
from lmmvibes.postprocess import LLMJsonParser, PropertyValidator
from lmmvibes.clusterers import HDBSCANClusterer
from lmmvibes.metrics import SideBySideMetrics

# Custom pipeline with batch API
pipeline = Pipeline([
    BatchExtractor(system_prompt="one_sided_system_prompt"),  # Creates batch files
    # ... manual batch processing step ...
    LLMJsonParser(),  # Parse batch results
    PropertyValidator(),  # Clean and validate
    HDBSCANClusterer(min_cluster_size=20, hierarchical=True),
    SideBySideMetrics()
])

dataset = PropertyDataset.from_dataframe(df, method="side_by_side")
result = pipeline.run(dataset)

# Convert back to DataFrame format
clustered_df = result.to_dataframe()
model_stats = result.model_stats
```

---

## 7. Configuration-driven runs

Provide a YAML and run from CLI:

```bash
python -m lmmvibes run --config config.yaml
```

`config.yaml`

```yaml
# Data loading
dataset_path: arena.parquet
method: side_by_side

# Extraction
extraction:
system_prompt: one_sided_system_prompt
  model: gpt-4o-mini
  temperature: 0.6
  max_workers: 16
  use_batch_api: false

# Clustering  
clustering:
  type: hdbscan
  min_cluster_size: 30
  embedding_model: openai
  hierarchical: true
  assign_outliers: false

# Metrics
metrics:
  type: side_by_side
  
# Logging
wandb:
  project: lmm-vibes
  entity: your-org
  
output_dir: outputs/
```

---

## 8. Package layout

```
lmmvibes/
│
├─ core/                    # data objects & stage base classes
│  ├─ __init__.py
│  ├─ data_objects.py       # PropertyDataset, ConversationRecord, Property
│  ├─ stage.py              # PipelineStage ABC
│  └─ mixins.py             # LoggingMixin, CacheMixin, etc.
│
├─ loaders/                 # data loading (migrate from data_loader.py)
│  ├─ __init__.py
│  ├─ arena.py              # load_arena_data
│  └─ webdev.py             # load_webdev_data
│
├─ extractors/              # property extraction
│  ├─ __init__.py
│  ├─ openai.py             # OpenAIExtractor (from generate_differences.py)
│  ├─ batch.py              # BatchExtractor
│  └─ vllm.py               # VLLMExtractor
│
├─ postprocess/             # post-processing stages
│  ├─ __init__.py
│  ├─ parser.py             # LLMJsonParser (from post_processing.py)
│  └─ validator.py          # PropertyValidator
│
├─ clusterers/              # clustering (migrate from clustering/)
│  ├─ __init__.py
│  ├─ hdbscan.py            # HDBSCANClusterer
│  ├─ hierarchical.py       # HierarchicalClusterer
│  └─ config.py             # ClusterConfig (from clustering/hierarchical_clustering.py)
│
├─ metrics/                 # metrics computation
│  ├─ __init__.py
│  ├─ side_by_side.py       # SideBySideMetrics
│  └─ single_model.py       # SingleModelMetrics
│
├─ viz/                     # optional Streamlit dashboards
│  ├─ __init__.py
│  └─ interactive_app.py    # summary visualization
│
├─ prompts/                 # prompt templates (migrate from prompts.py)
│  ├─ __init__.py
│  ├─ arena.py              # one_sided_system_prompt, etc.
│  └─ webdev.py             # webdev_system_prompt, etc.
│
├─ public.py                # explain lives here
├─ pipeline.py              # Pipeline orchestration
├─ cli.py                   # CLI interface
└─ __init__.py              # re-export explain
```

No circular imports—subpackages only depend on `core` or lower layers.

---

## 9. Interactive visualisation (Streamlit)

The package ships with a zero-config **Streamlit** app that lets you *browse clusters visually*:

```bash
# Full PropertyDataset                   # Or clusters-only DataFrame
streamlit run lmmvibes/viz/interactive_app.py \
    -- --dataset clustered_results.parquet
#                                         streamlit run lmmvibes/viz/interactive_app.py \
#                                         -- --dataset property_with_clusters.jsonl
```

Key features:

1. Sidebar selector for **coarse → fine** clusters.
2. Click-through to see **all property descriptions** in a cluster.
3. Global *overview accordion* that lists every coarse cluster and its fine children.
4. Optional bar-chart and word-cloud summaries.
5. Accepts either a full `PropertyDataset` file *(json / pkl / parquet)* **or** a clusters DataFrame saved as JSONL / CSV / Parquet.

`lmmvibes.viz` also exposes helpers:

```python
from lmmvibes.viz import load_dataset, build_hierarchy,
    load_clusters_dataframe, build_hierarchy_from_df
```

These make it trivial to embed the same visuals in Jupyter, Dash, or a custom web app.

---

## 10. Extending the ecosystem

| Goal | What to implement | Migration from |
|------|-------------------|----------------|
| New extraction technique | `class MyExtractor(PipelineStage)` | Wrap your existing LLM calling code |
| Alternative clustering  | `class MyClusterer(PipelineStage)` | Wrap existing clustering functions |
| Different metric        | `class MyMetrics(PipelineStage)` | Implement new metric calculations |
| New prompt library      | Add to `prompts/` module | Move from existing `prompts.py` |
| Custom post-processing  | `class MyValidator(PipelineStage)` | Wrap existing cleaning logic |
| Batch API support       | `class BatchExtractor(PipelineStage)` | Migrate batch logic from `generate_differences.py` |

All new stages become instantly available to `explain` via keywords or passed as instances.

---

## 11. Migration strategy

### Phase 1: Core Infrastructure
1. Create `core/` module with data objects and stage interface
2. Implement `Pipeline` class and basic mixins
3. Create `PropertyDataset.from_dataframe()` to wrap existing data

### Phase 2: Wrap Existing Code
1. **Extractors**: Wrap `generate_differences.py` logic into `OpenAIExtractor`
2. **Clusterers**: Wrap `clustering/hierarchical_clustering.py` into stage classes
3. **Post-processors**: Wrap `post_processing.py` logic into `LLMJsonParser`
4. **Loaders**: Wrap `data_loader.py` into loader modules

### Phase 3: Public API
1. Implement `explain()` function using pipeline
2. Add CLI interface with YAML config support
3. Create basic visualization app

### Phase 4: Polish & Packaging
1. Add comprehensive tests
2. Create documentation and examples
3. Publish to PyPI
4. Add advanced features (caching, distributed processing, etc.)

---

## 12. Validation & Testing

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
    
    # Create detailed CSV for manual review
    df = result.to_dataframe(type="clusters")
    df.to_csv("tests/outputs/property_with_clusters.csv", index=False)

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

The Streamlit viewer has now been implemented and moved to the ✅ list above.

---

## 13. Road-map

1. **Phase 1-2**: Refactor existing scripts into stage subclasses (~2-3 weeks)
2. **Phase 3**: Implement public API and CLI (~1 week)  
3. **Phase 4**: Polish, test, and publish (~1 week)
4. **Future**: Advanced features (distributed processing, more clusterers, richer metrics)

With this README you have both:  
• a dead-simple **`explain` one-liner** for everyday users, and  
• a robust, extensible pipeline that **directly migrates your existing code** for researchers and power users. 

---

## 14. Migration status & TODO

### ✅ Already implemented in the new package

* Core data objects (`ConversationRecord`, `Property`, `PropertyDataset`)
* `PipelineStage` interface & `Pipeline` / `PipelineBuilder`
* Mixin stack (Logging, Timing, ErrorHandling, Wandb, Cache) using the new *mixin-first* pattern
* Working stages
  * `OpenAIExtractor` (rewritten from `generate_differences.py`)
  * `LLMJsonParser` (from `post_processing.py`)
  * `PropertyValidator`
* End-to-end `explain()` public API with wandb logging
* Tests: extraction → parsing integration, mix-in initialisation
* **Interactive Streamlit viewer** (`lmmvibes.viz.interactive_app`)

### 🔜 Still to integrate from the original notebooks / scripts

| Component | Source file | Target module |
|-----------|-------------|---------------|
| HDBSCAN clustering logic | `clustering/hierarchical_clustering.py` | `lmmvibes.clusterers.hdbscan` |
| Native / hierarchical clustering | same | `lmmvibes.clusterers.hdbscan_native_hierarchical` |
| Metrics computation | ad-hoc in notebooks | `lmmvibes.metrics.side_by_side`, `single_model` |
| Batch & vLLM extractors | `generate_differences.py` / `vllm` | `lmmvibes.extractors.batch`, `vllm` |
| Data-loader helpers | `data_loader.py` | `lmmvibes.datasets.*` (arena, webdev, etc.) |
| CLI + YAML runner | n/a | `lmmvibes.cli` |
| Streamlit / viz | n/a | `lmmvibes.viz` |
| Comprehensive test suite | n/a | `tests/` |

Nice-to-have polish after migration:

* Outlier reassignment & hierarchical summaries
* Documentation examples & badges
* PyPI packaging scripts

Feel free to tick items off this list as they land. 🎉 
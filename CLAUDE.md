# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LMM-Vibes is a comprehensive analysis framework for evaluating and comparing Large Language Model responses. It specializes in analyzing differences between model behaviors and generating statistical rankings using Arena dataset battles and other comparative evaluation datasets.

## Core Architecture

The project follows a **4-step pipeline architecture**:

1. **Property Extraction** → 2. **Post-processing** → 3. **Clustering** → 4. **Metrics & Visualization**

### Key Components

- **Main API**: `lmmvibes.explain()` - Primary entry point for users
- **Pipeline System**: `lmmvibes.pipeline.Pipeline` - Orchestrates sequential stage execution
- **Property Extraction**: `lmmvibes.extractors` - Extract behavioral properties using LLM analysis
- **Clustering**: `lmmvibes.clusterers` - Group similar behaviors using HDBSCAN/hierarchical methods
- **Data Management**: `lmmvibes.core.data_objects` - Core data structures (`PropertyDataset`, `Property`, `ConversationRecord`)
- **Metrics**: `lmmvibes.metrics` - Calculate model performance statistics
- **Visualization**: `lmmvibes.viz` - Interactive Streamlit applications

## Common Development Commands

### Environment Setup
```bash
pip install -e .
# or
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key-here"  # Required for LLM features and embeddings
```

### Running Tests
```bash
python -m pytest tests/
```

### Core Pipeline Usage

**Simple API (Recommended)**
```python
from lmmvibes import explain
import pandas as pd

# Side-by-side comparison
clustered_df, model_stats = explain(
    df,
    method="side_by_side",
    min_cluster_size=30,
    embedding_model="openai",
    hierarchical=True,
    output_dir="results/"
)

# Single model analysis
clustered_df, model_stats = explain(
    df,
    method="single_model",
    system_prompt="single_model_system_prompt",
    clusterer="hdbscan"
)
```

**Advanced Pipeline Construction**
```python
from lmmvibes.pipeline import PipelineBuilder
from lmmvibes.extractors import OpenAIExtractor
from lmmvibes.clusterers import HDBSCANClusterer

builder = PipelineBuilder("Custom Pipeline")
pipeline = (builder
    .extract_properties(OpenAIExtractor(model="gpt-4o-mini"))
    .cluster_properties(HDBSCANClusterer(min_cluster_size=20))
    .configure(use_wandb=True)
    .build())
```

### Interactive Visualization
```bash
streamlit run lmmvibes/viz/interactive_app.py -- --dataset results/clustered_results.parquet
```

## Data Architecture

### Input Data Formats
- **Side-by-Side**: `question_id`, `model_a`, `model_b`, `model_a_response`, `model_b_response`, `winner` (optional)
- **Single Model**: `question_id`, `model`, `response`, `score` (optional)

### Core Data Objects
- `PropertyDataset`: Main container for all pipeline data
- `ConversationRecord`: Individual conversation with prompt/responses
- `Property`: Extracted behavioral property with metadata
- `Cluster`: Grouped properties with labels and statistics
- `ModelStats`: Per-model performance metrics

### Directory Structure
- `/lmmvibes/` - Main package code
- `/tests/` - Test suite
- `/data/` - Raw datasets and processed results
- `/results/` - Output directory for analysis results
- `/legacy/` - Old scripts and implementations

## Package Structure

### Core Framework (`lmmvibes/core/`)
- `data_objects.py` - Core data structures
- `stage.py` - Pipeline stage interface
- `mixins.py` - Logging, timing, error handling mixins
- `pipeline.py` - Pipeline orchestration

### Processing Stages
- `extractors/` - Property extraction (OpenAI, vLLM, batch processing)
- `postprocess/` - JSON parsing and validation
- `clusterers/` - Clustering algorithms (HDBSCAN, hierarchical)
- `metrics/` - Performance calculations
- `viz/` - Interactive visualization apps

### Entry Points
- `public.py` - Main `explain()` function
- `__init__.py` - Package imports

## Development Guidelines

### Adding New Pipeline Stages
1. Inherit from `PipelineStage`
2. Implement `run(data: PropertyDataset) -> PropertyDataset`
3. Add validation and error handling
4. Include logging and metrics

### Testing
- Use `pytest` for all tests
- Place test files in `/tests/`
- Test both unit functionality and full pipeline integration

### Configuration
- Use environment variables for API keys
- Support both string and object configuration
- Enable optional wandb logging for experiment tracking

## Key Dependencies

- **ML/NLP**: scikit-learn, sentence-transformers, hdbscan, umap-learn
- **LLM Integration**: openai, litellm, vllm
- **Data Processing**: pandas, numpy, pydantic, pyarrow
- **Visualization**: streamlit, plotly
- **Experiment Tracking**: wandb

## Performance Considerations

- Use `embedding_model="openai"` for quality, `"all-MiniLM-L6-v2"` for speed
- Enable `hierarchical=True` for better cluster organization
- Use `min_cluster_size=30` for large datasets (>10k samples)
- Enable `use_wandb=True` for experiment tracking
- Set `output_dir` to automatically save results

## Important Notes

- The package uses a pipeline architecture where each stage operates on `PropertyDataset` objects
- All pipeline stages inherit from `PipelineStage` and implement the `run()` method
- The main entry point is `explain()` which handles common use cases
- Results are automatically saved in multiple formats (parquet, JSON, CSV)
- The project supports both side-by-side model comparison and single model analysis
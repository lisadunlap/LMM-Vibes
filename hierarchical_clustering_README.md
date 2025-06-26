# Hierarchical Clustering for Text Data

This module provides scalable hierarchical clustering for text data using semantic embeddings. It's designed to automatically group similar text values into meaningful categories at multiple hierarchical levels with optional LLM-powered cluster naming.

## 🚀 Quick Start

```python
import pandas as pd
from hierarchical_clustering import bertopic_hierarchical_cluster_categories

# Load your data
df = pd.read_json("your_data.jsonl", lines=True)

# Cluster text column into hierarchical topics using BERTopic
df_clustered = bertopic_hierarchical_cluster_categories(
    df, 
    column_name="text_column",
    max_coarse_topics=25,          # Broad topic categories
    max_fine_topics_per_coarse=50, # Subtopics per broad category
    embedding_model="all-MiniLM-L6-v2",  # Local model (faster)
    use_llm_summaries=True,        # Generate human-readable cluster names
    context="properties seen in AI responses"  # Context for better naming
)
```

## 📊 What It Does

**Input**: A DataFrame with a text column containing values like:
- "The response was helpful and accurate"
- "Answer lacks sufficient detail" 
- "Model provided creative solution"
- "Response contains factual errors"

**Output**: The same DataFrame with new columns:
- `text_column_coarse_topic_label`: Broad topic categories (e.g., "helpful and accurate responses")
- `text_column_fine_topic_label`: Specific subtopics (e.g., "uses enthusiastic tone")
- `text_column_coarse_topic_id`: Numeric topic IDs for broad categories
- `text_column_fine_topic_id`: Numeric topic IDs for specific categories
- `text_column_embedding`: Vector embeddings (optional)

## 🎯 Clustering Methods

### 🔥 **BERTopic Hierarchical Clustering (RECOMMENDED)**
```python
df_clustered = bertopic_hierarchical_cluster_categories(
    df, "column_name",
    min_cluster_size=10,           # Minimum points per cluster
    min_topic_size=10,             # Minimum points per topic
    max_coarse_topics=25,          # Maximum broad topics
    max_fine_topics_per_coarse=50, # Maximum subtopics per broad topic
    embedding_model="all-MiniLM-L6-v2",  # or "openai"
    use_llm_summaries=True,        # Generate human-readable names
    use_llm_coarse_clustering=False, # Use LLM to create coarse clusters from fine ones
    cache_embeddings=True,         # Cache embeddings for reuse
    context="properties seen in AI responses"  # Context for LLM naming
)
```

**Parameters:**
- `min_cluster_size` (int, default=10): Minimum cluster size for BERTopic
- `min_topic_size` (int, default=10): Minimum size for fine-level topics
- `max_coarse_topics` (int, default=25): Maximum number of coarse topics
- `max_fine_topics_per_coarse` (int, default=50): Maximum fine topics per coarse topic
- `embedding_model` (str, default="openai"): Embedding method ("openai", "all-MiniLM-L6-v2", "all-mpnet-base-v2")
- `include_embeddings` (bool, default=True): Include embeddings in output
- `cache_embeddings` (bool, default=True): Save/load embeddings from disk
- `use_llm_summaries` (bool, default=False): Use LLM to generate human-readable cluster names
- `context` (str, optional): Context for LLM summaries (e.g., "properties seen in AI responses")
- `use_llm_coarse_clustering` (bool, default=False): Use LLM-only approach to create coarse clusters from fine cluster names
- `max_coarse_clusters` (int, default=15): Maximum coarse clusters when using LLM coarse clustering
- `input_model_name` (str, optional): Name of the input model being analyzed (for cache differentiation)

### ⚡ **HDBSCAN Clustering (FAST & SCALABLE)**
```python
df_clustered = hdbscan_cluster_categories(
    df, "column_name",
    min_cluster_size=30,           # Minimum cluster size
    embedding_model="all-MiniLM-L6-v2",
    hierarchical=True,             # Enable hierarchical clustering
    assign_outliers=True,          # Assign outliers to nearest clusters
    enable_dim_reduction=True,     # Enable UMAP for large datasets
    use_llm_summaries=True,        # Generate human-readable names
    use_llm_coarse_clustering=True # Create coarse clusters with LLM
)
```

**Parameters:**
- `min_cluster_size` (int, default=30): Minimum cluster size
- `embedding_model` (str, default="openai"): Embedding method
- `include_embeddings` (bool, default=True): Include embeddings in output
- `use_llm_summaries` (bool, default=False): Use LLM to generate cluster summaries
- `context` (str, optional): Context for LLM summaries
- `precomputed_embeddings` (array/dict, optional): Precomputed embeddings
- `enable_dim_reduction` (bool, default=False): Enable UMAP dimensionality reduction
- `assign_outliers` (bool, default=False): Assign HDBSCAN outliers to nearest clusters
- `hierarchical` (bool, default=False): Enable hierarchical clustering (cluster the clusters)
- `min_grandparent_size` (int, default=5): Minimum size for grandparent clusters
- `use_llm_coarse_clustering` (bool, default=False): Use LLM-only approach for coarse clusters
- `max_coarse_clusters` (int, default=15): Maximum coarse clusters when using LLM coarse clustering
- `input_model_name` (str, optional): Name of input model for cache differentiation

### 🎯 **Traditional Hierarchical Clustering**
```python
df_clustered = hierarchical_cluster_categories(
    df, "column_name",
    n_coarse_clusters=10,          # Number of coarse clusters
    n_fine_clusters=50,            # Number of fine clusters
    embedding_model="all-MiniLM-L6-v2",
    use_llm_summaries=True
)
```

**Parameters:**
- `n_coarse_clusters` (int, default=10): Number of coarse clusters
- `n_fine_clusters` (int, default=50): Number of fine clusters
- `embedding_model` (str, default="openai"): Embedding method
- `include_embeddings` (bool, default=True): Include embeddings in output
- `use_llm_summaries` (bool, default=False): Use LLM to generate cluster summaries
- `context` (str, default='properties seen in AI responses'): Context for LLM summaries
- `input_model_name` (str, optional): Name of input model for cache differentiation

## ⚙️ Advanced Features

### 🤖 **LLM-Powered Cluster Naming**
Generate human-readable cluster names using LLM analysis:

```python
df_clustered = bertopic_hierarchical_cluster_categories(
    df, "feedback_text",
    use_llm_summaries=True,        # Enable LLM naming
    context="customer feedback about AI responses",  # Provide context
    embedding_model="openai"       # Higher quality embeddings
)
```

**Example output:**
- Instead of: "topic_0", "topic_1"
- You get: "enthusiastic and positive tone", "lacks specific technical details"

### 🔗 **LLM Coarse Clustering**
Create high-level categories by having LLM analyze fine cluster names:

```python
df_clustered = hdbscan_cluster_categories(
    df, "properties",
    use_llm_coarse_clustering=True,  # Enable LLM coarse clustering
    max_coarse_clusters=15,          # Maximum high-level categories
    context="properties of AI model responses"
)
```

### 💾 **Embedding Caching**
Automatically cache embeddings to speed up repeated runs:

```python
df_clustered = bertopic_hierarchical_cluster_categories(
    df, "text_column",
    cache_embeddings=True,         # Enable caching (default)
    input_model_name="gpt-4",      # Differentiate cache by model
    embedding_model="openai"
)
```

### 📊 **Precomputed Embeddings**
Use existing embeddings to skip computation:

```python
# Load precomputed embeddings
embeddings = load_precomputed_embeddings("embeddings.pkl")

df_clustered = hdbscan_cluster_categories(
    df, "text_column",
    precomputed_embeddings=embeddings,
    hierarchical=True
)
```

## 🚀 Command Line Usage

```bash
# Basic HDBSCAN clustering
python hierarchical_clustering.py \
    --file data.jsonl \
    --column property_description \
    --method hdbscan \
    --min-cluster-size 15

# Advanced hierarchical HDBSCAN with LLM features
python hierarchical_clustering.py \
    --file data.jsonl \
    --column text_column \
    --method hdbscan \
    --min-cluster-size 10 \
    --hierarchical \
    --assign-outliers \
    --enable-dim-reduction \
    --embedding-model all-MiniLM-L6-v2 \
    --use-llm-coarse-clustering \
    --max-coarse-clusters 20 \
    --context "properties seen in AI responses"

# BERTopic with LLM summaries
python hierarchical_clustering.py \
    --file data.jsonl \
    --column feedback_text \
    --method bertopic \
    --max-coarse-topics 30 \
    --max-fine-topics 60 \
    --embedding-model openai \
    --context "customer feedback" \
    --output customer_feedback_clusters
```

### Command Line Parameters

- `--file, -f`: Path to input JSONL file (required)
- `--column, -c`: Column name to cluster on (default: property_description)
- `--method, -m`: Clustering method (bertopic, hdbscan, hierarchical)
- `--min-cluster-size`: Minimum cluster size
- `--embedding-model`: Embedding model (openai, all-MiniLM-L6-v2, all-mpnet-base-v2)
- `--output, -o`: Output filename prefix
- `--no-embeddings`: Exclude embeddings from output
- `--no-llm-summaries`: Disable LLM-based cluster summaries
- `--context`: Context for LLM summaries
- `--hierarchical`: Enable hierarchical HDBSCAN clustering
- `--assign-outliers`: Assign outliers to nearest clusters
- `--enable-dim-reduction`: Enable UMAP dimensionality reduction
- `--use-llm-coarse-clustering`: Use LLM for coarse clustering
- `--max-coarse-clusters`: Maximum coarse clusters for LLM clustering
- `--precomputed-embeddings`: Path to precomputed embeddings file
- `--input-model-name`: Name of input model for cache differentiation

## 📈 Performance Guide

| Dataset Size | Recommended Method | Embedding Model | Features | Expected Time |
|-------------|-------------------|------------------|----------|---------------|
| <1k values | `bertopic_hierarchical_cluster_categories` | `"all-MiniLM-L6-v2"` | LLM summaries | <1 minute |
| 1k-10k values | `bertopic_hierarchical_cluster_categories` | `"all-MiniLM-L6-v2"` | LLM summaries | 2-5 minutes |
| 10k-50k values | `hdbscan_cluster_categories` | `"all-MiniLM-L6-v2"` | Hierarchical, LLM coarse | 5-15 minutes |
| 50k+ values | `hdbscan_cluster_categories` | `"all-MiniLM-L6-v2"` | Dim reduction, hierarchical | 15-45 minutes |

**Optimization Tips:**
- Use `"all-MiniLM-L6-v2"` for speed, `"openai"` for quality
- Enable `cache_embeddings=True` for repeated runs
- Use `enable_dim_reduction=True` for datasets >10k values
- Use `precomputed_embeddings` when running multiple clustering experiments

## 💾 Saving and Loading Results

The module automatically saves results in multiple formats:

```python
# Results are saved to cluster_results/{output_prefix}/
save_clustered_results(df_clustered, "my_analysis", include_embeddings=True)
```

**Output files:**
- `{prefix}_with_embeddings.parquet`: Full results with vector embeddings
- `{prefix}_lightweight.parquet`: Results without embeddings (smaller)
- `{prefix}.csv.gz`: Compressed CSV format
- `{prefix}.jsonl`: JSON lines format

**Loading results:**
```python
from hierarchical_clustering import load_clustered_results
df = load_clustered_results("cluster_results/my_analysis/my_analysis_with_embeddings.parquet")
```

## 🛠️ Requirements

```bash
# Core requirements
pip install pandas scikit-learn numpy

# For BERTopic and HDBSCAN (recommended)
pip install bertopic sentence-transformers hdbscan umap-learn

# For LLM features (optional but recommended)
pip install litellm

# Set OpenAI API key for embeddings and LLM features
export OPENAI_API_KEY="your-api-key-here"
```

## 📋 Example Output

After clustering with LLM summaries, your DataFrame will have these new columns:

```
Original: "The model gave a helpful and accurate response"
→ Coarse topic: "helpful and accurate responses" 
→ Fine topic: "uses enthusiastic and positive tone"
→ Coarse ID: 2
→ Fine ID: 15
```

**Key Advantages**:
- **LLM-powered naming**: Clusters get human-readable names like "technical accuracy issues" instead of "cluster_3"
- **Automatic hierarchy**: Fine topics are discovered within coarse topics
- **Context awareness**: Provide context to get domain-specific cluster names
- **Flexible coarse clustering**: Use LLM to create higher-level categories from fine clusters
- **Caching system**: Speeds up repeated experiments with same data
- **Multiple output formats**: Results saved in parquet, CSV, and JSON formats 
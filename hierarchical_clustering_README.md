# Hierarchical Clustering for Text Data

This module provides scalable hierarchical clustering for text data using semantic embeddings. It's designed to automatically group similar text values into meaningful categories at multiple hierarchical levels.

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
    max_fine_topics_per_coarse=20, # Subtopics per broad category
    embedding_model="all-MiniLM-L6-v2"  # Local model (faster)
)
```

## 📊 What It Does

**Input**: A DataFrame with a text column containing values like:
- "The response was helpful and accurate"
- "Answer lacks sufficient detail" 
- "Model provided creative solution"
- "Response contains factual errors"

**Output**: The same DataFrame with new columns:
- `text_column_coarse_topic_label`: Broad topic categories with automatic labels (e.g., "helpful_accurate_response")
- `text_column_fine_topic_label`: Specific subtopics with automatic labels (e.g., "Subtopic: helpful, accurate, clear")
- `text_column_coarse_topic_id`: Numeric topic IDs for broad categories
- `text_column_fine_topic_id`: Numeric topic IDs for specific categories
- `text_column_embedding`: Vector embeddings (optional)

## 🎯 Clustering Methods

### 🔥 **BERTopic Hierarchical Clustering (RECOMMENDED)**
```python
df_clustered = bertopic_hierarchical_cluster_categories(
    df, "column_name",
    min_cluster_size=20,
    max_coarse_topics=25,
    embedding_model="all-MiniLM-L6-v2"  # or "openai"
)
```
- **Algorithm**: BERTopic with HDBSCAN clustering
- **Speed**: Fast and scalable
- **Best for**: Automatic topic discovery, natural hierarchies, interpretable labels
- **Advantages**: 
  - Automatic topic labeling from keywords
  - Natural hierarchical structure by clustering within topics
  - Uses BERT embeddings for better semantic understanding
  - Handles noise and outliers well

### Small Datasets (<10k unique values)
```python
df_clustered = hierarchical_cluster_categories(
    df, "column_name",
    embedding_method="local"  # Fastest for small datasets
)
```
- **Algorithm**: Agglomerative clustering
- **Speed**: Fast
- **Best for**: Exploratory analysis, small datasets

### Medium Datasets (10k-100k unique values)  
```python
df_clustered = scalable_hierarchical_cluster_categories(
    df, "column_name",
    min_cluster_size=20,
    embedding_method="batch_api"
)
```
- **Algorithm**: HDBSCAN (density-based)
- **Speed**: Very fast (O(n log n))
- **Best for**: Natural cluster discovery, handles noise well

### Large Datasets (50k+ unique values)
```python
df_clustered = two_stage_hierarchical_cluster_categories(
    df, "column_name", 
    n_initial_clusters=1000,
    embedding_method="batch_api"
)
```
- **Algorithm**: Two-stage (MiniBatch K-means → Hierarchical)
- **Speed**: Scales to millions of data points
- **Best for**: Preserving hierarchical structure at scale

## ⚙️ Configuration Options

### BERTopic Parameters
- `min_cluster_size`: Minimum points per cluster (default: 10)
- `min_topic_size`: Minimum points per topic (default: 10)
- `max_coarse_topics`: Maximum broad topics to extract (default: 25)
- `max_fine_topics_per_coarse`: Maximum subtopics per broad topic (default: 20)

### Embedding Methods
- `"all-MiniLM-L6-v2"`: Local sentence transformer (fastest, good quality)
- `"all-mpnet-base-v2"`: Local sentence transformer (better quality, slower)
- `"openai"`: OpenAI API embeddings (highest quality, requires API key)
- `"local"`: Use local sentence-transformers (for legacy functions)
- `"batch_api"`: OpenAI API in batches (for legacy functions)

### Other Parameters
- `context`: Optional description to improve topic modeling
- `include_embeddings=True`: Include vector embeddings in output
- `verbose=True`: Print progress and topic summaries

## 💾 Saving Results

```python
# Full results with embeddings (use for ML/analysis)
df.to_parquet("results_with_embeddings.parquet")

# Lightweight results (use for quick analysis)  
df_light = df.drop(columns=[col for col in df.columns if 'embedding' in col])
df_light.to_parquet("results_lightweight.parquet")
df_light.to_csv("results.csv")
```

**File Size Comparison**: Parquet provides 5-10x compression vs CSV/JSON and preserves data types.

## 🔧 Advanced Usage

### Using OpenAI Embeddings for Higher Quality
```python
df_clustered = bertopic_hierarchical_cluster_categories(
    df, "feedback_text",
    embedding_model="openai",  # Requires OPENAI_API_KEY
    max_coarse_topics=30,
    verbose=True
)
```

### Fine-tuning Topic Granularity
```python
df_clustered = bertopic_hierarchical_cluster_categories(
    df, "text_column", 
    min_cluster_size=5,              # Smaller = more fine-grained topics
    min_topic_size=10,               # Minimum viable topic size
    max_coarse_topics=15,            # Fewer broad categories
    max_fine_topics_per_coarse=30    # More subtopics per category
)
```

### Load Previously Clustered Results
```python
from hierarchical_clustering import load_clustered_results

df = load_clustered_results("results_with_embeddings.parquet")
```

## 📈 Performance Guide

| Dataset Size | Recommended Method | Embedding Model | Expected Time |
|-------------|-------------------|------------------|---------------|
| <1k values | `bertopic_hierarchical_cluster_categories` | `"all-MiniLM-L6-v2"` | <1 minute |
| 1k-10k values | `bertopic_hierarchical_cluster_categories` | `"all-MiniLM-L6-v2"` | 2-5 minutes |
| 10k-50k values | `bertopic_hierarchical_cluster_categories` | `"all-MiniLM-L6-v2"` | 5-15 minutes |
| 50k+ values | `bertopic_hierarchical_cluster_categories` | `"all-MiniLM-L6-v2"` | 15-45 minutes |

For higher quality (but slower), use `"all-mpnet-base-v2"` or `"openai"` embeddings.

## 🛠️ Requirements

```bash
# Core requirements
pip install pandas scikit-learn numpy

# For BERTopic (recommended)
pip install bertopic sentence-transformers hdbscan umap-learn

# For legacy methods
pip install sentence-transformers hdbscan
pip install litellm  # For API embeddings
```

## 📋 Example Output

After BERTopic clustering, your DataFrame will have these new columns:
```
Original: "The model gave a helpful and accurate response"
→ Coarse topic: "helpful_accurate_response" 
→ Fine topic: "Subtopic: helpful, accurate, clear"
→ Coarse ID: 2
→ Fine ID: 15
```

**BERTopic Advantages**:
- **Automatic labeling**: Topics are labeled with representative keywords
- **Natural hierarchy**: Fine topics are discovered within coarse topics
- **Better quality**: Uses advanced BERT embeddings and UMAP dimensionality reduction
- **Noise handling**: Outliers are automatically identified
- **Interpretability**: Topic words show what each cluster represents

This enables analysis like:
- Automatically discover themes in customer feedback
- Create hierarchical taxonomies with interpretable labels
- Identify outliers and noise in text data
- Build features for downstream ML tasks with semantic meaning 
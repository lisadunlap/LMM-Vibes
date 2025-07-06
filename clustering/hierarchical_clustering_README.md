# Hierarchical Clustering for Text Data

This module provides scalable hierarchical clustering for text data using semantic embeddings. It's designed to automatically group similar text values into meaningful categories at multiple hierarchical levels with optional LLM-powered cluster naming.

## 🚀 Quick Start

### **⚠️ IMPORTANT: Running the Script**

The clustering script can be run from different directories. Here are the recommended approaches:

#### **Option 1: Run from Parent Directory (Recommended)**
```bash
# Navigate to the parent directory (LMM-Vibes)
cd /path/to/LMM-Vibes

# Run the script with full path
python clustering/hierarchical_clustering.py \
    --file your_data.jsonl \
    --method hdbscan \
    --min-cluster-size 15 \
    --hierarchical
```

#### **Option 2: Run from Clustering Directory**
```bash
# Navigate to the clustering directory
cd clustering

# Run the script directly (imports are now fixed to handle this)
python hierarchical_clustering.py \
    --file ../your_data.jsonl \
    --method hdbscan \
    --min-cluster-size 15 \
    --hierarchical
```

#### **Option 3: Use Python Module Syntax**
```bash
# From parent directory, use module syntax
python -m clustering.hierarchical_clustering \
    --file your_data.jsonl \
    --method hdbscan \
    --min-cluster-size 15 \
    --hierarchical
```

### **🆕 NEW: Simplified Configuration Approach**

```python
import pandas as pd
from hierarchical_clustering import ClusterConfig, hdbscan_cluster_categories

# Load your data
df = pd.read_json("your_data.jsonl", lines=True)

# Create a configuration object (NEW APPROACH - RECOMMENDED)
config = ClusterConfig(
    min_cluster_size=15,
    embedding_model="all-MiniLM-L6-v2",
    use_llm_summaries=True,
    hierarchical=True,
    context="properties seen in AI responses"
)

# Run clustering with config
df_clustered = hdbscan_cluster_categories(df, "text_column", config=config)
```

### **📦 Legacy Parameter Style (Still Supported)**

```python
# Old approach still works for backward compatibility
df_clustered = hdbscan_cluster_categories(
    df, "text_column",
    min_cluster_size=15,
    embedding_model="all-MiniLM-L6-v2", 
    use_llm_summaries=True,
    hierarchical=True,
    context="properties seen in AI responses"
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

### ⚡ **HDBSCAN Clustering (RECOMMENDED)**

#### **🆕 NEW: Configuration Object Approach**
```python
from hierarchical_clustering import ClusterConfig, hdbscan_cluster_categories

# Create config with all parameters in one place
config = ClusterConfig(
    min_cluster_size=30,
    embedding_model="all-MiniLM-L6-v2",
    hierarchical=True,             # Enhanced LLM-powered hierarchical clustering
    assign_outliers=True,          # Assign outliers to nearest clusters
    enable_dim_reduction=True,     # Enable UMAP for large datasets
    use_llm_summaries=True,        # Generate human-readable names
    max_coarse_clusters=15,        # Maximum coarse clusters for hierarchical mode
    context="properties seen in AI responses",
    cache_embeddings=True
)

# Run clustering with simple function call
df_clustered = hdbscan_cluster_categories(df, "column_name", config=config)
```

#### **📦 Legacy Parameter Style** 
```python
# Old approach still works for backward compatibility
df_clustered = hdbscan_cluster_categories(
    df, "column_name",
    min_cluster_size=30,           # Minimum cluster size
    embedding_model="all-MiniLM-L6-v2",
    hierarchical=True,             # Enable enhanced LLM-powered hierarchical clustering
    assign_outliers=True,          # Assign outliers to nearest clusters
    enable_dim_reduction=True,     # Enable UMAP for large datasets
    use_llm_summaries=True,        # Generate human-readable names
    max_coarse_clusters=15         # Maximum coarse clusters for hierarchical mode
)
```

### 🎯 **Traditional Hierarchical Clustering**

#### **🆕 Configuration Object Approach**
```python
config = ClusterConfig(
    min_cluster_size=10,           # For n_coarse_clusters compatibility
    embedding_model="all-MiniLM-L6-v2",
    use_llm_summaries=True,
    context="properties seen in AI responses"
)

df_clustered = hierarchical_cluster_categories(df, "column_name", config=config)
```

### **⚙️ ClusterConfig Parameters**

All clustering methods now use the unified `ClusterConfig` class:

- `min_cluster_size` (int, default=30): Minimum cluster size
- `embedding_model` (str, default="openai"): Embedding method ("openai", "all-MiniLM-L6-v2", etc.)
- `verbose` (bool, default=True): Print progress information
- `include_embeddings` (bool, default=True): Include embeddings in output
- `use_llm_summaries` (bool, default=False): Use LLM to generate cluster summaries
- `context` (str, optional): Context for LLM summaries
- `precomputed_embeddings` (array/dict, optional): Precomputed embeddings
- `enable_dim_reduction` (bool, default=False): Enable UMAP dimensionality reduction
- `assign_outliers` (bool, default=False): Assign HDBSCAN outliers to nearest clusters
- `hierarchical` (bool, default=False): **Enhanced LLM-powered hierarchical clustering**
- `max_coarse_clusters` (int, default=15): Maximum coarse clusters for hierarchical mode
- `use_llm_coarse_clustering` (bool, default=False): Legacy LLM coarse clustering
- `input_model_name` (str, optional): Name of input model for cache differentiation
- `min_samples` (int, optional): HDBSCAN min_samples parameter (auto-calculated if None)
- `cluster_selection_epsilon` (float, default=0.0): HDBSCAN cluster selection epsilon
- `cache_embeddings` (bool, default=True): Enable embedding caching

### **🚀 Simplified Usage Examples**

```python
# Quick clustering with defaults
config = ClusterConfig(min_cluster_size=15, use_llm_summaries=True)
result = hdbscan_cluster_categories(df, "text", config=config)

# Advanced hierarchical clustering
config = ClusterConfig(
    min_cluster_size=20,
    hierarchical=True,
    use_llm_summaries=True,
    max_coarse_clusters=10,
    context="customer feedback about AI responses"
)
result = hdbscan_cluster_categories(df, "feedback", config=config)
```

## ⚙️ Advanced Features

### 🤖 **LLM-Powered Cluster Naming**
Generate human-readable cluster names using LLM analysis:

#### **🆕 Configuration Object Approach**
```python
config = ClusterConfig(
    use_llm_summaries=True,        # Enable LLM naming
    context="customer feedback about AI responses",  # Provide context
    embedding_model="openai"       # Higher quality embeddings
)
df_clustered = hdbscan_cluster_categories(df, "feedback_text", config=config)
```

**Example output:**
- Instead of: "cluster_0", "cluster_1"
- You get: "enthusiastic and positive tone", "lacks specific technical details"

### 🧠 **Enhanced LLM Hierarchical Clustering**
**NEW**: Advanced LLM-powered concept center generation for superior coarse clustering:

#### **🆕 Configuration Object Approach**
```python
config = ClusterConfig(
    hierarchical=True,               # Enable enhanced LLM hierarchical clustering
    max_coarse_clusters=15,          # Maximum high-level categories
    use_llm_summaries=True,          # Generate readable fine cluster names
    context="properties of AI model responses"
)
df_clustered = hdbscan_cluster_categories(df, "properties", config=config)
```

**What makes this enhanced?**
- Uses advanced systems prompt to consolidate similar properties intelligently
- LLM generates concept centers by merging redundant properties (e.g., "step-by-step guidance on health" + "step-by-step guidance on math" → "provides detailed information")
- Embedding-based assignment ensures fine clusters map to most similar concept centers
- Improved error handling with retry logic for robust API calls
- Uses `text-embedding-3-large` for higher quality embeddings

### 🔗 **Legacy LLM Coarse Clustering**
For backward compatibility, the original approach is still available:

```python
config = ClusterConfig(
    use_llm_coarse_clustering=True,  # Use legacy LLM coarse clustering
    max_coarse_clusters=15,          # Maximum high-level categories
    context="properties of AI model responses"
)
df_clustered = hdbscan_cluster_categories(df, "properties", config=config)
```

### 💾 **Embedding Caching**
Automatically cache embeddings to speed up repeated runs:

```python
config = ClusterConfig(
    cache_embeddings=True,         # Enable caching (default)
    input_model_name="gpt-4",      # Differentiate cache by model
    embedding_model="openai"
)
df_clustered = hdbscan_cluster_categories(df, "text_column", config=config)
```

### 📊 **Precomputed Embeddings**
Use existing embeddings to skip computation:

```python
# Load precomputed embeddings
embeddings = load_precomputed_embeddings("embeddings.pkl")

config = ClusterConfig(
    precomputed_embeddings=embeddings,
    hierarchical=True
)
df_clustered = hdbscan_cluster_categories(df, "text_column", config=config)
```

## 🚀 Command Line Usage

```bash
# Basic HDBSCAN clustering
python hierarchical_clustering.py \
    --file data.jsonl \
    --method hdbscan \
    --hierarchical \
    --assign-outliers \
    --enable-dim-reduction \
    --embedding-model all-MiniLM-L6-v2 \
    --min-cluster-size 15

# Enhanced hierarchical HDBSCAN with LLM concept centers
python hierarchical_clustering.py \
    --file data.jsonl \
    --method hdbscan \
    --min-cluster-size 10 \
    --hierarchical \
    --assign-outliers \
    --enable-dim-reduction \
    --embedding-model all-MiniLM-L6-v2 \
    --max-coarse-clusters 20 \
    --context "properties seen in AI responses"

# Legacy LLM coarse clustering (still supported)
python hierarchical_clustering.py \
    --file data.jsonl \
    --method hdbscan \
    --min-cluster-size 10 \
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
- `--hierarchical`: Enable enhanced LLM-powered hierarchical clustering with concept centers
- `--assign-outliers`: Assign outliers to nearest clusters
- `--enable-dim-reduction`: Enable UMAP dimensionality reduction
- `--use-llm-coarse-clustering`: Use legacy LLM coarse clustering approach
- `--max-coarse-clusters`: Maximum coarse clusters for hierarchical/LLM clustering
- `--precomputed-embeddings`: Path to precomputed embeddings file
- `--input-model-name`: Name of input model for cache differentiation

## 📈 Performance Guide

| Dataset Size | Recommended Method | Embedding Model | Features | Expected Time |
|-------------|-------------------|------------------|----------|---------------|
| <1k values | `bertopic_hierarchical_cluster_categories` | `"all-MiniLM-L6-v2"` | LLM summaries | <1 minute |
| 1k-10k values | `bertopic_hierarchical_cluster_categories` | `"all-MiniLM-L6-v2"` | LLM summaries | 2-5 minutes |
| 10k-50k values | `hdbscan_cluster_categories` | `"all-MiniLM-L6-v2"` | Enhanced hierarchical | 5-15 minutes |
| 50k+ values | `hdbscan_cluster_categories` | `"all-MiniLM-L6-v2"` | Dim reduction, enhanced hierarchical | 15-45 minutes |

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

## 🚀 **Optimized Deduplication-First Clustering**

**NEW**: For datasets with duplicate values, use the optimized deduplication approach for better performance and quality:

```bash
# Optimized clustering for duplicate-heavy datasets
python dedupe_cluster_categories.py \
    --file your_data.jsonl \
    --column category \
    --min-cluster-size 10 \
    --hierarchical \
    --max-coarse-clusters 12 \
    --embedding-model openai \
    --output optimized_results
```

### **Why Use Deduplication-First?**

**📊 Efficiency Gains:**
- **3.2x faster**: Processes only unique values (e.g., 12K unique vs 41K total)
- **Memory efficient**: No redundant embedding computations
- **Cost effective**: Fewer API calls for embeddings

**🎯 Quality Improvements:**
- **Equal representation**: Each unique value influences clustering equally
- **Semantic clustering**: Based on meaning, not frequency
- **Cleaner centroids**: Cluster centers represent true semantic differences
- **Better results**: No frequency bias in cluster formation

### **When to Use Each Approach:**

| Scenario | Approach | Reason |
|----------|----------|--------|
| **Many duplicates** (>2x reduction) | `dedupe_cluster_categories.py` | ✅ Better quality + performance |
| **Mostly unique values** (<1.5x reduction) | `hierarchical_clustering.py` | ✅ Direct processing sufficient |
| **Frequency weighting desired** | `hierarchical_clustering.py` | ✅ Preserves value frequency |
| **Large datasets** (>10K duplicates) | `dedupe_cluster_categories.py` | ✅ Essential for performance |

### **Deduplication Workflow:**

1. **🔍 Analyze**: Check duplicate ratio in your data
2. **📋 Deduplicate**: Extract unique values for clustering
3. **🎯 Cluster**: Run enhanced LLM clustering on unique values
4. **🗺️ Map**: Apply cluster results back to full dataset
5. **💾 Save**: Output both unique clusters and full mapped dataset

```python
# Check if deduplication will help
import pandas as pd
df = pd.read_json("your_data.jsonl", lines=True)
reduction_ratio = len(df) / df["your_column"].nunique()
print(f"Reduction ratio: {reduction_ratio:.1f}x")

# If ratio > 2.0, use deduplication approach
if reduction_ratio > 2.0:
    print("✅ Use dedupe_cluster_categories.py for better results")
else:
    print("ℹ️ Direct clustering with hierarchical_clustering.py is fine")
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

## 🔧 Troubleshooting

### **Import Errors**

If you encounter `ModuleNotFoundError: No module named 'clustering'`, try one of these solutions:

#### **Solution 1: Run from Parent Directory (Recommended)**
```bash
# Make sure you're in the LMM-Vibes directory, not the clustering subdirectory
cd /path/to/LMM-Vibes
python clustering/hierarchical_clustering.py --file your_data.jsonl --method hdbscan
```

#### **Solution 2: Add Parent Directory to Python Path**
```bash
# From the clustering directory
cd clustering
PYTHONPATH=.. python hierarchical_clustering.py --file ../your_data.jsonl --method hdbscan
```

#### **Solution 3: Use Module Syntax**
```bash
# From the parent directory
python -m clustering.hierarchical_clustering --file your_data.jsonl --method hdbscan
```

### **Common Issues**

- **"No module named 'clustering'"**: You're running from the wrong directory. Use Solution 1 above.
- **"No module named 'clustering_utils'"**: The script can't find its dependencies. Use Solution 1 or 2 above.
- **Permission errors**: Make sure the script is executable: `chmod +x clustering/hierarchical_clustering.py`

### **Verifying Your Setup**

Test that everything is working:
```bash
# From the LMM-Vibes directory
python -c "from clustering.hierarchical_clustering import ClusterConfig; print('✅ Setup is working!')"
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
- **🆕 Simplified Configuration**: New `ClusterConfig` class consolidates all 16+ parameters into one object
- **📦 Backward Compatibility**: All existing code continues to work without changes
- **🚀 Deduplication-first optimization**: Automatically handles duplicate values for better performance and quality
- **🧠 Enhanced LLM hierarchical clustering**: Uses concept centers and intelligent property consolidation
- **🏷️ LLM-powered naming**: Clusters get human-readable names like "technical accuracy issues" instead of "cluster_3"
- **🌳 Automatic hierarchy**: Fine topics are discovered within coarse topics
- **🎯 Context awareness**: Provide context to get domain-specific cluster names
- **🔧 Intelligent property merging**: LLM consolidates redundant properties while preserving informative distinctions
- **🛡️ Robust error handling**: Retry logic and improved embedding API calls for reliability
- **💾 Caching system**: Speeds up repeated experiments with same data
- **📊 Multiple output formats**: Results saved in parquet, CSV, and JSON formats
- **⚖️ Equal representation**: Each unique value influences clustering equally, not by frequency

## 🆕 What's New in the Latest Version

### **Simplified Configuration System**
- **Before**: 16 individual parameters to manage
- **After**: Single `ClusterConfig` object with all parameters
- **Benefit**: Easier parameter management, better maintainability

```python
# OLD: Many parameters to remember and manage
df_clustered = hdbscan_cluster_categories(
    df, "text", min_cluster_size=30, embedding_model="all-MiniLM-L6-v2",
    verbose=True, include_embeddings=True, use_llm_summaries=False, 
    context=None, precomputed_embeddings=None, enable_dim_reduction=False,
    assign_outliers=False, hierarchical=False, min_grandparent_size=5,
    use_llm_coarse_clustering=False, max_coarse_clusters=15, 
    input_model_name=None, min_samples=None, cluster_selection_epsilon=0.0
)

# NEW: Clean, organized configuration
config = ClusterConfig(
    min_cluster_size=30,
    embedding_model="all-MiniLM-L6-v2",
    use_llm_summaries=True,
    hierarchical=True
)
df_clustered = hdbscan_cluster_categories(df, "text", config=config)
```

### **Improved Function Architecture**
- **Reduced function length by 68%**: Main clustering function is now ~110 lines instead of 350+
- **Eliminated code duplication**: Common logic extracted into reusable helper functions
- **Better maintainability**: Easier to add new features and fix issues 
# Step 0: Cluster Prompts
Add category labels (broad_category_id	broad_category	narrower_category_id	narrower_category) to each prompt in the dataframe.

# Step 1:Get Differences

This tool analyzes and identifies key differences between two language models' responses to the same prompt. It processes data from various Arena datasets to extract meaningful comparisons.

## Flow
1. Loads and filters Arena dataset battles (supports multiple datasets)
2. For each battle:
   - Extracts user prompts and model responses
   - Uses an LLM to analyze differences between responses
   - Outputs structured JSON comparing model behaviors

## Key Features
- Supports multiple datasets (arena, webdev)
- Supports both OpenAI and vLLM models
- Batch processing with intermediate saves
- Wandb integration for tracking
- JSON-structured difference analysis
- OpenAI Batch API support for large-scale processing

## Usage

### Basic Usage

Process arena dataset with OpenAI model:
```bash
python generate_differences.py \
  --dataset arena \
  --num_samples 100 \
  --model_name gpt-4o-mini \
  --output_file differences/arena_results.csv
```

Process webdev dataset with OpenAI model:
```bash
python generate_differences.py \
  --dataset webdev \
  --num_samples 100 \
  --model_name gpt-4o-mini \
  --exclude_ties \
  --output_file differences/webdev_results.csv
```

### Advanced Usage

Run with vLLM model:
```bash
python generate_differences.py \
  --dataset arena \
  --num_samples 1000 \
  --model_name llama-3-70b-instruct \
  --tensor_parallel_size 4 \
  --max_model_len 8192 \
  --temperature 0.7 \
  --output_file differences/large_run.csv
```

### Batch Mode (OpenAI Batch API)

Generate batch requests for large-scale processing:
```bash
python generate_differences.py \
  --dataset arena \
  --num_samples 1000 \
  --model_name gpt-4o-mini \
  --batches \
  --output_file batches/arena_batch_requests.jsonl
```

This creates:
- `batches/arena_batch_requests.jsonl` - OpenAI Batch API requests
- `batches/arena_batch_requests_metadata.json` - Metadata for each request

## Parameters

### Required
- `--dataset`: Dataset to process (`arena` or `webdev`)

### Optional
- `--num_samples`: Number of battles to process
- `--model_name`: Model to use for analysis (default: gpt-4o-mini)
- `--output_file`: Where to save results (defaults to `differences/[dataset]_differences.csv`)
- `--tensor_parallel_size`: For vLLM models (default: 1)
- `--temperature`: Model temperature (default: 0.6)
- `--top_p`: Top-p sampling (default: 0.95)
- `--max_tokens`: Max output tokens (default: 16000)
- `--max_model_len`: Max input length (default: 32000)
- `--filter_english`: Only process English conversations
- `--exclude_ties`: Skip tied battles (webdev dataset only)
- `--max_workers`: Max threads for OpenAI API calls (default: 16)

### Batch Mode
- `--batches`: Enable batch mode (generates OpenAI Batch API files)
- `--metadata_file`: Metadata file path for batch mode

## Supported Datasets

### Arena Dataset
- **Source**: `lmarena-ai/arena-human-preference-100k`
- **Split**: train
- **Features**: General conversation battles between models
- **Filters**: English conversations, specific model subset

### WebDev Dataset
- **Source**: `lmarena-ai/webdev-arena-preference-10k`
- **Split**: test
- **Features**: Web development task battles
- **Filters**: English conversations, exclude ties option

## Step 2: Post Processing

With the output file, run post_processing.py to get it in the right format for clustering:

```bash
python post_processing.py --input_file differences/arena_results.jsonl
```

This should save it to `differences/arena_results_processed.jsonl`

## Step 3: Clustering

The `hierarchical_clustering.py` script uses `hdbscan` to find thematic clusters in the generated difference properties.

### **⚠️ IMPORTANT: Running the Clustering Script**

The clustering script can be run from different directories. Here are the recommended approaches:

#### **Option 1: Run from Parent Directory (Recommended)**
```bash
# Navigate to the LMM-Vibes directory
cd /path/to/LMM-Vibes

# Run the script with full path
python clustering/hierarchical_clustering.py \
    --file differences/arena_results_processed.jsonl \
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
    --file ../differences/arena_results_processed.jsonl \
    --method hdbscan \
    --min-cluster-size 15 \
    --hierarchical
```

#### **Option 3: Use Python Module Syntax**
```bash
# From parent directory, use module syntax
python -m clustering.hierarchical_clustering \
    --file differences/arena_results_processed.jsonl \
    --method hdbscan \
    --min-cluster-size 15 \
    --hierarchical
```

### **🔧 Troubleshooting Import Errors**

If you encounter `ModuleNotFoundError: No module named 'clustering'`, try one of these solutions:

- **Run from parent directory**: Make sure you're in the LMM-Vibes directory, not the clustering subdirectory
- **Add parent to Python path**: `PYTHONPATH=.. python clustering/hierarchical_clustering.py`
- **Use module syntax**: `python -m clustering.hierarchical_clustering`

### **🆕 NEW: Simplified Configuration Approach**

The clustering module now supports a streamlined configuration system for easier parameter management:

#### **Python API (Recommended)**
```python
from hierarchical_clustering import ClusterConfig, hdbscan_cluster_categories
import pandas as pd

# Load your processed data
df = pd.read_json("differences/arena_results_processed.jsonl", lines=True)

# Create a simple configuration
config = ClusterConfig(
    min_cluster_size=15,
    embedding_model="all-MiniLM-L6-v2",
    hierarchical=True,
    assign_outliers=True,
    use_llm_summaries=True,
    context="properties seen in AI responses"
)

# Run clustering with simplified interface
df_clustered = hdbscan_cluster_categories(df, "property_description", config=config)
```

#### **Command Line (Legacy, Still Supported)**
```bash
# From the LMM-Vibes directory (recommended)
python clustering/hierarchical_clustering.py \
    --file differences/arena_results_processed.jsonl \
    --method hdbscan \
    --hierarchical \
    --assign-outliers \
    --embedding-model all-MiniLM-L6-v2 \
    --min-cluster-size 15 \
    --use-llm-summaries
```

**Key improvements:**
- **📦 Simplified interface**: One config object instead of 16+ parameters
- **🔄 Backward compatibility**: All existing commands continue to work
- **🛠️ Better maintainability**: Easier to add new features and modify behavior

### 🚀 Optimized Clustering for Duplicate-Heavy Data

**NEW**: For datasets with many duplicate values (like categories), use the optimized deduplication approach for 3x faster processing and better quality:

```bash
# From the LMM-Vibes directory
python clustering/dedupe_cluster_categories.py \
    --file differences/arena_results_processed.jsonl \
    --column category \
    --min-cluster-size 10 \
    --hierarchical \
    --max-coarse-clusters 12 \
    --embedding-model openai \
    --output optimized_categories
```

**Benefits:**
- **3x faster**: Only processes unique values (e.g., 12K unique vs 41K total)
- **Better quality**: Equal weight per unique value, not frequency-based
- **Cost effective**: Fewer embedding API calls
- **Auto-mapping**: Results mapped back to full dataset

**When to use:** If your reduction ratio is >2x (check with `df.groupby('column').size().max() / df.groupby('column').size().min()`)

### Standard Clustering Tuning

If the initial clustering gives you too many or too few clusters, you can tweak the following HDBSCAN parameters:

#### If you have too many clusters:

- **Increase `--min-cluster-size`**: This is the most direct way to reduce the number of clusters. A higher value means HDBSCAN will require more points to form a stable cluster, leading to fewer, larger clusters. For example, try `--min-cluster-size 30`.
- **Set `--cluster-selection-epsilon`**: This parameter merges clusters that are close to each other. By setting it to a small positive value, you can reduce the number of fine-grained clusters. For example, `--cluster-selection-epsilon 0.5`. Start with small values and increase to merge more aggressively.

Example command to get fewer clusters:
```bash
# From the LMM-Vibes directory
python clustering/hierarchical_clustering.py \
    --file differences/arena_results_processed.jsonl \
    --method hdbscan \
    --min-cluster-size 30 \
    --cluster-selection-epsilon 0.5
```

#### If you have too few clusters or too many outliers:

- **Decrease `--min-cluster-size`**: A smaller value will allow HDBSCAN to identify smaller, more numerous clusters. For example, try `--min-cluster-size 5`.
- **Decrease `--min-samples`**: This parameter controls how conservative the clustering is. Lowering it makes the algorithm more willing to form clusters from sparser data, reducing the number of points classified as outliers. If not set, it defaults to `--min-cluster-size`. Try setting it explicitly, e.g., `--min-samples 5`.
- **Use `--assign-outliers`**: If you want to eliminate outliers and assign every point to a cluster, use this flag.

Example command to get more clusters:
```bash
# From the LMM-Vibes directory
python clustering/hierarchical_clustering.py \
    --file oai_batch_data/arena_full_vibe_results_parsed_processed.jsonl \
    --method hdbscan \
    --min-cluster-size 5 \
    --hierarchical \
    --assign-outliers
```

## Step 4: Visualize Clusters

### 🆕 **NEW: Sunburst Chart Visualization (Recommended)**

Create a beautiful, interactive sunburst chart that mimics the style of existing cluster visualizations:

```bash
python sunburst_cluster_visualization.py \
    --file cluster_results/arena_results_processed_hdbscan_clustered/arena_results_processed_hdbscan_clustered_lightweight.parquet \
    --column property_description \
    --output-dir sunburst_visualization \
    --title "My Cluster Analysis"
```

**Key Features:**
- **Automatic structure detection**: Works with both hierarchical (fine+coarse) and flat (fine-only) clustering
- **Interactive exploration**: Click to drill down, hover for details, see examples
- **Mobile responsive**: Works on both desktop and mobile devices
- **Beautiful design**: Uses the same styling as existing sunburst charts

**Parameters:**
- `--file`: Path to your clustered parquet file
- `--column`: Base name of the clustered column (e.g., 'property_description')
- `--output-dir`: Directory to save visualization files (optional, defaults to 'sunburst_visualization')
- `--title`: Custom title for the visualization (optional)

### 📊 **Interactive Bar Chart Visualization**

For a simpler bar chart interface:

```bash
python interactive_cluster_visualization.py \
    --file cluster_results/arena_results_processed_hdbscan_clustered/arena_results_processed_hdbscan_clustered_lightweight.parquet \
    --column property_description \
    --fine-only  # Optional: show only fine clusters without hierarchy
```

### 🌐 **How to View Your Visualizations**

After running either visualization script, you'll get HTML files that you can view in your browser:

#### **Method 1: Direct Browser Opening**
Simply double-click the generated HTML file or drag it into your browser window.

#### **Method 2: Local HTTP Server (Recommended)**
For better performance and to avoid potential file loading issues:

```bash
# Navigate to the visualization directory
cd sunburst_visualization  # or wherever your HTML files are

# Start a local HTTP server
python -m http.server 8000

# Then open your browser and go to:
# http://localhost:8000/sunburst_property-description.html
```

If port 8000 is busy, use a different port:
```bash
python -m http.server 8080
# Then go to: http://localhost:8080/sunburst_property-description.html
```

#### **Method 3: VS Code Live Server**
If you're using VS Code, you can install the "Live Server" extension and right-click on the HTML file to open it with Live Server.

### 🔧 **Troubleshooting Visualizations**

- **Chart not displaying**: Make sure all files (HTML, CSS, JS, JSON) are in the same directory
- **Data not loading**: Use the HTTP server method instead of opening files directly
- **Performance issues**: For very large datasets, consider filtering your data first or using the `--fine-only` flag

## 🌐 Interactive Cluster Viewer (Web App)

For an easy-to-use interactive web interface to explore your clustering results:

```bash
# Install requirements
pip install flask plotly

# Launch the web app
python simple_cluster_viewer.py

# Open your browser to: http://127.0.0.1:5001
```

**Features:**
- 📁 **File selection**: Dropdown to choose any summary table file
- 📊 **Coarse clusters**: Interactive bar chart with cluster counts
- 🔍 **Drill-down**: Click on bars to explore fine-grained clusters
- 📝 **Examples**: View property descriptions and evidence for each cluster
- 🚀 **Fast loading**: Optimized for large datasets

**Requirements:** The web app automatically discovers `*summary_table.jsonl` files in your `cluster_results/` directory.
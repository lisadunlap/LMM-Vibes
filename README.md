# Step 0: Cluster Prompts
Add category labels (broad_category_id	broad_category	narrower_category_id	narrower_category) to each prompt in the dataframe.

```bash
python cluster_prompts.py \
  --save-path './test' \
  --min-cluster-size 50 
```

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

Process coding dataset:
```bash
python generate_differences.py \
  --dataset coding \
  --num_samples 100 \
  --model_name gpt-4o-mini \
  --output_file differences/coding_results_no_example.csv
```

```bash
python generate_differences.py \
  --dataset coding \
  --num_samples 100 \
  --model_name gpt-4o-mini \
  --output_file differences/coding_results.csv
```

```bash
python generate_differences.py \
  --dataset fictional \
  --num_samples 100 \
  --model_name gpt-4o-mini \
  --output_file differences/fictional_results.csv
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

```bash
python post_processing.py --input_file differences/coding_results_no_example.jsonl
```

```bash
python post_processing.py --input_file differences/coding_results.jsonl
```

```bash
python post_processing.py --input_file differences/fictional_results.jsonl
```

This should save it to f`{input_file}_processed.jsonl`

## Step 3: Clustering

```bash
python hierarchical_clustering.py \
    --file differences/arena_results_processed.jsonl \
    --method hdbscan \
    --hierarchical \
    --assign-outliers \
    --enable-dim-reduction \
    --embedding-model all-MiniLM-L6-v2 \
    --min-cluster-size 15
```

```bash
python hierarchical_clustering.py \
    --file differences/coding_results_processed.jsonl \
    --method hdbscan \
    --hierarchical \
    --assign-outliers \
    --enable-dim-reduction \
    --embedding-model all-MiniLM-L6-v2 \
    --min-cluster-size 2
```

## Step 4: Visualize Clusters

To get an interactive visualization, run:

```bash
python interactive_cluster_visualization.py \
    --file cluster_results/arena_results_processed_hdbscan_clustered/arena_results_processed_hdbscan_clustered_lightweight.parquet
```

```bash
python interactive_cluster_visualization.py \
    --file cluster_results/coding_results_processed_hdbscan_clustered/coding_results_processed_hdbscan_clustered_lightweight.parquet
```
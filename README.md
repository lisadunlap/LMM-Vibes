# Step 1:Get Differences

This tool analyzes and identifies key differences between two language models' responses to the same prompt. It processes data from the Arena dataset to extract meaningful comparisons.

## Flow
1. Loads and filters Arena dataset battles
2. For each battle:
   - Extracts user prompts and model responses
   - Truncates long responses if needed
   - Uses an LLM to analyze differences between responses
   - Outputs structured JSON comparing model behaviors

## Key Features
- Supports both OpenAI and vLLM models
- Automatic response truncation for long outputs
- Batch processing with intermediate saves
- Wandb integration for tracking
- JSON-structured difference analysis

## Usage

Basic run with OpenAI model:
```bash
python get_differences_arena_data.py \
  --num_samples 100 \
  --model_name gpt-4-turbo \
  --output_file differences/results.csv
```

Run with custom parameters:
```bash
python get_differences_arena_data.py \
  --num_samples 1000 \
  --model_name llama-3-70b-instruct \
  --tensor_parallel_size 4 \
  --max_model_len 8192 \
  --temperature 0.7 \
  --output_file differences/large_run.csv
```

## Parameters
- `--num_samples`: Number of battles to process
- `--model_name`: Model to use for analysis
- `--output_file`: Where to save results
- `--tensor_parallel_size`: For vLLM models
- `--temperature`: Model temperature (default: 0.6)
- `--top_p`: Top-p sampling (default: 0.95)
- `--max_tokens`: Max output tokens (default: 2048)
- `--max_model_len`: Max input length (default: 16384)
- `--filter_english`: Only process English conversations
- `--exclude_ties`: Skip tied battles
- `--auto_truncate`: Enable response truncation

## Step 2: post processing

with the output file, run post_processing.py to get it in the right format for clustering

```
python post_processing.py --input_file differences/results.jsonl
```

which should save it to `differences/results_processed.jsonl`

## Step 3: Clustering

```
python hierarchical_clustering.py \
    --file differences/results_processed.jsonl \
    --method hdbscan \
    --hierarchical \
    --assign-outliers \
    --enable-dim-reduction \
    --embedding-model all-MiniLM-L6-v2 \
    --min-cluster-size 15
```

## Step 4: visualize clusters

To get an interactive visualization that is probably not broken, run
<!-- cluster_results/test_processed_hdbscan_clustered/test_processed_hdbscan_clustered_lightweight.parquet -->
```
python interactive_cluster_visualization.py --file cluster_results/results_processed_hdbscan_clustered/results_processed_hdbscan_clustered_lightweight.parquet
```
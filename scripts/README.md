# LMM-Vibes Pipeline Scripts

This directory contains scripts for running the complete LMM-Vibes pipeline on full datasets.

## Recent Updates

**🔧 Unified Wandb Logging**: The pipeline now uses a single wandb run for all stages, providing consolidated logging instead of separate runs for each stage.

**🛠️ OpenAI Extractor Fix**: Fixed the "too many values to unpack" error that occurred with single-model data formats.

**⚡ Error Handling**: Added proper error handling to fail fast when no properties are extracted.

## Scripts Overview

### `run_full_pipeline.py`
The main script that provides a flexible command-line interface for running the pipeline on any dataset.

### Dataset-specific convenience scripts:
- `run_arena_pipeline.py` - For the Arena dataset
- `run_webdev_pipeline.py` - For the WebDev dataset  
- `run_wildbench_pipeline.py` - For the WildBench dataset

### Test script:
- `test_openai_extractor.py` - Test the OpenAI extractor with single-model data

## Usage

### Basic Usage

Run the pipeline on a specific dataset:
```bash
python scripts/run_full_pipeline.py \
    --data_path data/arena_single.jsonl \
    --output_dir results/arena_full_results
```

### Convenience Scripts

For common datasets, you can use the convenience scripts:

```bash
# Arena dataset
python scripts/run_arena_pipeline.py --output_dir results/arena_full

# WebDev dataset  
python scripts/run_webdev_pipeline.py --output_dir results/webdev_full

# WildBench dataset
python scripts/run_wildbench_pipeline.py --output_dir results/wildbench_full
```

### Advanced Options

#### Sample a subset of data
```bash
python scripts/run_arena_pipeline.py \
    --sample_size 1000 \
    --output_dir results/arena_sample_1k
```

#### Adjust clustering parameters
```bash
python scripts/run_full_pipeline.py \
    --data_path data/arena_single.jsonl \
    --output_dir results/arena_custom \
    --min_cluster_size 15 \
    --max_coarse_clusters 20 \
    --max_workers 12
```

#### Disable hierarchical clustering
```bash
python scripts/run_arena_pipeline.py \
    --no_hierarchical \
    --output_dir results/arena_no_hierarchical
```

#### Enable wandb logging
```bash
python scripts/run_arena_pipeline.py \
    --use_wandb \
    --output_dir results/arena_with_wandb
```

## Wandb Integration

The pipeline now provides **unified wandb logging** with a single run that tracks:

- **Pipeline configuration** (model, parameters, etc.)
- **Stage-by-stage metrics** (conversations, properties, clusters processed) - logged as summary statistics
- **Extraction results** (API calls, success rates, response lengths) - logged as summary statistics  
- **Clustering results** (number of clusters, outlier rates) - logged as summary statistics
- **Final dataset summary** (total properties, models analyzed) - logged as summary statistics
- **Sample results table** (preview of final output) - logged as regular table

### Summary Statistics vs Regular Metrics

**Numeric metrics are logged as summary statistics only once at the end of the pipeline:**
- All stage execution times
- Extraction success rates and error counts
- Parsing success rates and error counts  
- Clustering metrics (cluster counts, outlier rates)
- Final dataset statistics
- Model performance metrics

**Tables and artifacts are logged immediately:**
- Sample extraction inputs/outputs
- Sample parsed properties
- Model statistics tables
- Clustering results tables

This ensures clean, organized wandb runs without duplicate or noisy metric logging.

### Wandb Configuration

```bash
# Enable wandb logging
python scripts/run_arena_pipeline.py \
    --use_wandb \
    --output_dir results/arena_with_wandb

# Custom wandb project
python scripts/run_full_pipeline.py \
    --data_path data/arena_single.jsonl \
    --output_dir results/arena_custom \
    --use_wandb \
    --wandb_project my-custom-project
```

### Viewing Results

After running with `--use_wandb`, you can view:
- **Metrics**: Track extraction success rates, clustering quality, etc.
- **Logs**: See detailed stage-by-stage progress
- **Tables**: Browse sample results and extraction outputs
- **Config**: Review all pipeline parameters used

## Command Line Options

### Common Options (all scripts)

- `--output_dir`: Directory to save results (required for main script)
- `--sample_size`: Number of samples to use (default: full dataset)
- `--min_cluster_size`: Minimum cluster size for HDBSCAN (default varies by dataset)
- `--max_coarse_clusters`: Maximum number of coarse clusters (default varies by dataset)
- `--max_workers`: Number of parallel workers (default: 16)
- `--no_hierarchical`: Disable hierarchical clustering
- `--use_wandb`: Enable wandb logging
- `--quiet`: Disable verbose output

### Full Pipeline Script Only

- `--data_path`: Path to input dataset (required)
- `--method`: Analysis method (single_model, multi_model)
- `--system_prompt`: System prompt to use
- `--clusterer`: Clustering algorithm (hdbscan, kmeans)
- `--embedding_model`: Embedding model to use

## Dataset Requirements

Input datasets must be JSONL files with the following required columns:
- `prompt`: The input prompt
- `model`: The model that generated the response
- `model_response`: The model's response

## Output Files

Each run creates the following files in the output directory:

- `clustered_results.parquet`: Main results with clustering information
- `full_dataset.json`: Complete dataset in PropertyDataset format
- `full_dataset.parquet`: Complete dataset in Parquet format
- `model_stats.json`: Model statistics and cluster analysis
- `summary.txt`: Human-readable summary of results

## Performance Considerations

### Memory Usage
- Large datasets (>1GB) may require significant RAM
- Consider using `--sample_size` for initial exploration
- Monitor memory usage during processing

### Processing Time
- Full datasets can take hours to process
- Use `--max_workers` to control parallelization
- Enable `--use_wandb` for progress tracking on long runs

### Recommended Parameters by Dataset

#### Arena Dataset
```bash
python scripts/run_arena_pipeline.py \
    --min_cluster_size 15 \
    --max_coarse_clusters 30 \
    --max_workers 16
```

#### WebDev Dataset (1.8GB)
```bash
python scripts/run_webdev_pipeline.py \
    --min_cluster_size 8 \
    --max_coarse_clusters 12 \
    --max_workers 8
```

#### WildBench Dataset (764MB)
```bash
python scripts/run_wildbench_pipeline.py \
    --min_cluster_size 5 \
    --max_coarse_clusters 10 \
    --max_workers 8
```

## Error Handling

### New Error Handling Features

- **Zero Properties Error**: The pipeline now fails fast with a clear error message if no properties are extracted
- **OpenAI Extractor Fixes**: Resolved unpacking errors with single-model data formats
- **Wandb Coordination**: All stages now use the same wandb run

### Troubleshooting

### Common Issues

1. **Memory Error**: Reduce `--sample_size` or increase system RAM
2. **Slow Processing**: Increase `--max_workers` (up to CPU core count)
3. **OpenAI API Errors**: Check API key and rate limits
4. **Missing Dataset**: Verify file path and dataset format
5. **No Properties Extracted**: Check API connectivity and data format

### Example Error Solutions

```bash
# If running out of memory
python scripts/run_arena_pipeline.py --sample_size 5000

# If processing too slowly
python scripts/run_arena_pipeline.py --max_workers 16

# If need to debug
python scripts/run_arena_pipeline.py --quiet --sample_size 100

# Test OpenAI extractor
python scripts/test_openai_extractor.py
```

## Example Workflow

1. **Test with the extractor** to verify data format:
   ```bash
   python scripts/test_openai_extractor.py
   ```

2. **Start with a sample** to test parameters:
   ```bash
   python scripts/run_arena_pipeline.py \
       --sample_size 1000 \
       --output_dir results/arena_test \
       --use_wandb
   ```

3. **Review results** in wandb dashboard and adjust parameters if needed

4. **Run full pipeline**:
   ```bash
   python scripts/run_arena_pipeline.py \
       --output_dir results/arena_full_final \
       --use_wandb
   ```

5. **Analyze results** using the generated files and wandb dashboard 
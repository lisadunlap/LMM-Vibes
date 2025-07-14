#!/usr/bin/env python3
"""
Run the complete LMM-Vibes pipeline on full datasets.

This script runs the full pipeline using the explain() function on complete datasets
with configurable parameters.
"""

import argparse
import os
import pandas as pd
import json
from pathlib import Path
import time
from datetime import datetime

from lmmvibes import explain
from lmmvibes.core.data_objects import PropertyDataset


def load_dataset(data_path, method="single_model"):
    """Load dataset from jsonl file."""
    print(f"Loading dataset from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    # Load the dataset
    df = pd.read_json(data_path, lines=True)
    
    # Verify required columns
    if method == "single_model":
        required_cols = {"prompt", "model", "model_response"}
    elif method == "side_by_side":
        required_cols = {"prompt", "model_a", "model_a_response", "model_b", "model_b_response"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset missing required columns: {required_cols - set(df.columns)}")
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    return df


def run_pipeline(
    data_path,
    output_dir,
    method="single_model",
    system_prompt="single_model_system_prompt",
    clusterer="hdbscan",
    min_cluster_size=15,
    max_coarse_clusters=30,
    embedding_model="openai",
    hierarchical=True,
    max_workers=4,
    use_wandb=False,
    verbose=True,
    sample_size=None,
    extraction_cache_dir=None,
    clustering_cache_dir=None,
    metrics_cache_dir=None
):
    """Run the complete pipeline on a dataset."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    df = load_dataset(data_path, method)
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} rows from {len(df)} total rows")
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Starting pipeline with {len(df)} conversations")
    
    # Record start time
    start_time = time.time()
    
    # Run the full pipeline
    print("Running full pipeline with explain()...")
    clustered_df, model_stats = explain(
        df,
        method=method,
        system_prompt=system_prompt,
        clusterer=clusterer,
        min_cluster_size=min_cluster_size,
        max_coarse_clusters=max_coarse_clusters,
        embedding_model=embedding_model,
        hierarchical=hierarchical,
        max_workers=max_workers,
        use_wandb=use_wandb,
        verbose=verbose,
        output_dir=str(output_path),
        extraction_cache_dir=extraction_cache_dir,
        clustering_cache_dir=clustering_cache_dir,
        metrics_cache_dir=metrics_cache_dir
    )
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # Verify outputs
    print("\nVerifying pipeline outputs...")
    
    # Check basic structure
    assert len(clustered_df) > 0, "Should return clustered DataFrame"
    assert len(model_stats) > 0, "Should return model statistics"
    
    # Check required columns
    required_columns = ['fine_cluster_id', 'fine_cluster_label']
    for col in required_columns:
        if col not in clustered_df.columns:
            print(f"Warning: Missing column: {col}")
    
    # Check for properties - FAIL if no properties were extracted
    if 'property_description' in clustered_df.columns:
        properties_with_desc = clustered_df['property_description'].notna().sum()
        print(f"Properties extracted: {properties_with_desc}")
        
        if properties_with_desc == 0:
            raise RuntimeError(
                "ERROR: No properties were successfully extracted from the conversations. "
                "This could be due to:\n"
                "1. OpenAI API errors (check your API key and quotas)\n"
                "2. Data format issues (verify your input data structure)\n"
                "3. Model or prompt configuration problems\n"
                "Check the logs above for specific error messages."
            )
    else:
        raise RuntimeError(
            "ERROR: 'property_description' column not found in results. "
            "This indicates a fundamental issue with the pipeline execution."
        )
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Dataset: {data_path}")
    print(f"Output directory: {output_path}")
    print(f"Runtime: {runtime:.2f} seconds ({runtime/60:.1f} minutes)")
    print(f"Input conversations: {len(df)}")
    print(f"Clustered properties: {len(clustered_df)}")
    print(f"Models analyzed: {len(model_stats)}")
    print(f"Fine clusters: {len(clustered_df['fine_cluster_id'].unique())}")
    
    if 'coarse_cluster_id' in clustered_df.columns:
        coarse_clusters = len(clustered_df['coarse_cluster_id'].unique())
        print(f"Coarse clusters: {coarse_clusters}")
    
    # Show sample clusters
    print(f"\nSample cluster labels:")
    unique_labels = clustered_df['fine_cluster_label'].dropna().unique()
    for i, label in enumerate(unique_labels[:5]):
        cluster_size = (clustered_df['fine_cluster_label'] == label).sum()
        print(f"  {i+1}. {label} (size: {cluster_size})")
    
    # Show sample model stats
    print(f"\nSample model stats:")
    for model_name, stats in list(model_stats.items())[:3]:
        print(f"  {model_name}:")
        for stat in stats["fine"][:2]:
            print(f"    • {stat.property_description[:50]}... (score: {stat.score:.2f})")
    
    print("="*60)
    print("✅ Full pipeline completed successfully!")
    print("="*60)
    
    return clustered_df, model_stats


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Run LMM-Vibes pipeline on full datasets")
    
    # Dataset and output
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the input dataset (jsonl file)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    
    # Pipeline parameters
    parser.add_argument("--method", type=str, default="single_model",
                        choices=["single_model", "multi_model"],
                        help="Analysis method (default: single_model)")
    parser.add_argument("--system_prompt", type=str, default="single_model_system_prompt",
                        help="System prompt to use")
    parser.add_argument("--clusterer", type=str, default="hdbscan",
                        choices=["hdbscan", "kmeans"],
                        help="Clustering algorithm (default: hdbscan)")
    parser.add_argument("--min_cluster_size", type=int, default=15,
                        help="Minimum cluster size (default: 15)")
    parser.add_argument("--max_coarse_clusters", type=int, default=30,
                        help="Maximum number of coarse clusters (default: 30)")
    parser.add_argument("--embedding_model", type=str, default="openai",
                        help="Embedding model to use (default: openai)")
    parser.add_argument("--max_workers", type=int, default=16,
                        help="Maximum number of workers (default: 4)")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Sample size to use (default: use full dataset)")
    
    # Flags
    parser.add_argument("--no_hierarchical", action="store_true",
                        help="Disable hierarchical clustering")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable verbose output")
    
    args = parser.parse_args()
    
    # Run pipeline
    clustered_df, model_stats = run_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        method=args.method,
        system_prompt=args.system_prompt,
        clusterer=args.clusterer,
        min_cluster_size=args.min_cluster_size,
        max_coarse_clusters=args.max_coarse_clusters,
        embedding_model=args.embedding_model,
        hierarchical=not args.no_hierarchical,
        max_workers=args.max_workers,
        use_wandb=args.use_wandb,
        verbose=not args.quiet,
        sample_size=args.sample_size
    )
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 
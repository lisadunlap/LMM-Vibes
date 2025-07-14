#!/usr/bin/env python3
"""
Run the LMM-Vibes pipeline on the WebDev dataset.

This is a convenience script for running the full pipeline on the webdev dataset
with optimized parameters.
"""

import argparse
import os
from run_full_pipeline import run_pipeline
import pandas as pd

def main():
    """Main function for webdev dataset processing."""
    parser = argparse.ArgumentParser(description="Run LMM-Vibes pipeline on WebDev dataset")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, 
                        default="results/bigcodebench_full_pipeline",
                        help="Output directory for results (default: results/bigcodebench_full_pipeline)")
    
    # Optional overrides
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Sample size to use (default: use full dataset)")
    parser.add_argument("--min_cluster_size", type=int, default=15,
                        help="Minimum cluster size (default: 15)")
    parser.add_argument("--max_coarse_clusters", type=int, default=30,
                        help="Maximum number of coarse clusters (default: 30)")
    parser.add_argument("--max_workers", type=int, default=16,
                        help="Maximum number of workers (default: 16)")
    
    # Flags
    parser.add_argument("--no_hierarchical", action="store_true",
                        help="Disable hierarchical clustering")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable verbose output")
    parser.add_argument("--clear_cache", action="store_true",
                        help="Clear the cache before running")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable caching entirely")
    
    args = parser.parse_args()
    
    # # Handle cache management
    # if args.clear_cache:
    #     import shutil
    #     import os
    #     cache_dir = os.path.expanduser("~/.cache/litellm")
    #     if os.path.exists(cache_dir):
    #         print(f"Clearing cache directory: {cache_dir}")
    #         shutil.rmtree(cache_dir)
    #         print("Cache cleared successfully")
    
    # if args.no_cache:
    #     import os
    #     os.environ["LITELLM_CACHE_ENABLED"] = "false"
    #     print("Caching disabled")
    
    # Set the data path
    data_path = "data/helm/helm_bigcodebench_results_processed.jsonl"
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Error: HELM dataset not found at {data_path}")
        print("Please make sure the dataset is available.")
        return
    
    print("="*60)
    print("HELM DATASET PIPELINE")
    print("="*60)
    print(f"Dataset: {data_path}")
    print(f"Output: {args.output_dir}")
    if args.sample_size:
        print(f"Sample size: {args.sample_size}")
    else:
        print("Using full dataset")
    print("="*60)
    
    # Run pipeline with webdev-optimized parameters
    pipeline_kwargs = {
        'data_path': data_path,
        'output_dir': args.output_dir,
        'method': "single_model",
        'system_prompt': "single_model_system_prompt",
        'clusterer': "hdbscan",
        'min_cluster_size': args.min_cluster_size,
        'max_coarse_clusters': args.max_coarse_clusters,
        'embedding_model': "openai",
        'hierarchical': not args.no_hierarchical,
        'max_workers': args.max_workers,
        'use_wandb': args.use_wandb,
        'verbose': not args.quiet,
        'sample_size': args.sample_size
    }
    
    clustered_df, model_stats = run_pipeline(**pipeline_kwargs)
    
    print(f"\n🎉 HELM pipeline completed! Results saved to: {args.output_dir}")
    print(f"📄 Parsing failures saved to: {args.output_dir}/parsing_failures.json")


if __name__ == "__main__":
    main() 
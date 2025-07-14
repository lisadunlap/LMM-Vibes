#!/usr/bin/env python3
"""
Run the LMM-Vibes pipeline on the Arena dataset.

This is a convenience script for running the full pipeline on the arena dataset
with optimized parameters.
"""

import argparse
import os
from run_full_pipeline import run_pipeline


def main():
    """Main function for arena dataset processing."""
    parser = argparse.ArgumentParser(description="Run LMM-Vibes pipeline on Arena dataset")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, 
                        default="results/arena_sbs_full_pipeline",
                        help="Output directory for results (default: results/arena_sbs_full_pipeline)")
    
    # Optional overrides
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Sample size to use (default: use full dataset)")
    parser.add_argument("--min_cluster_size", type=int, default=15,
                        help="Minimum cluster size (default: 15)")
    parser.add_argument("--max_coarse_clusters", type=int, default=30,
                        help="Maximum number of coarse clusters (default: 30)")
    parser.add_argument("--max_workers", type=int, default=16,
                        help="Maximum number of workers (default: 8)")
    
    # Flags
    parser.add_argument("--no_hierarchical", action="store_true",
                        help="Disable hierarchical clustering")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable verbose output")
    
    args = parser.parse_args()
    
    # Set the data path
    data_path = "data/arena_sbs.jsonl"
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Error: Arena dataset not found at {data_path}")
        print("Please make sure the dataset is available.")
        return
    
    print("="*60)
    print("ARENA DATASET PIPELINE")
    print("="*60)
    print(f"Dataset: {data_path}")
    print(f"Output: {args.output_dir}")
    if args.sample_size:
        print(f"Sample size: {args.sample_size}")
    else:
        print("Using full dataset")
    print("="*60)
    
    # Run pipeline with arena-optimized parameters
    clustered_df, model_stats = run_pipeline(
        data_path=data_path,
        output_dir=args.output_dir,
        method="side_by_side",
        system_prompt="sbs_w_metrics_system_prompt",
        clusterer="hdbscan",
        min_cluster_size=args.min_cluster_size,
        max_coarse_clusters=args.max_coarse_clusters,
        embedding_model="openai",
        hierarchical=not args.no_hierarchical,
        max_workers=args.max_workers,
        use_wandb=args.use_wandb,
        verbose=not args.quiet,
        sample_size=args.sample_size
    )
    
    print(f"\n🎉 Arena pipeline completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 
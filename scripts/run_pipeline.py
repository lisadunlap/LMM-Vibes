#!/usr/bin/env python3
"""
Run the LMM-Vibes pipeline on the WebDev dataset.

This is a convenience script for running the full pipeline on the webdev dataset
with optimized parameters.
"""

import argparse
import os
from run_full_pipeline import run_pipeline
from lmmvibes import compute_metrics_only
import pandas as pd
import json

def main():
    """Main function for webdev dataset processing."""
    parser = argparse.ArgumentParser(description="Run LMM-Vibes pipeline on WebDev dataset")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, 
                        help="Output directory for results (default: results/webdev_full_pipeline)")
    parser.add_argument("--input_file", type=str,
                        help="Input file for results (default: data/arena_webdev_sbs.jsonl)")
    parser.add_argument("--system_prompt", type=str,
                        default="single_model_system_prompt",
                        help="System prompt for the pipeline (default: webdev_system_prompt_no_examples)")
    parser.add_argument("--method", type=str,
                        default="single_model",
                        help="Method for the pipeline (default: side_by_side)")
    
    # Optional overrides
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Sample size to use (default: use full dataset)")
    parser.add_argument("--min_cluster_size", type=int, default=8,
                        help="Minimum cluster size (default: 8)")
    parser.add_argument("--max_coarse_clusters", type=int, default=12,
                        help="Maximum number of coarse clusters (default: 12)")
    parser.add_argument("--max_workers", type=int, default=64,
                        help="Maximum number of workers (default: 64)")
    
    # Flags
    parser.add_argument("--hierarchical", action="store_true",
                        help="Enable hierarchical clustering")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable verbose output")
    
    # run specific components (only metrics)
    parser.add_argument("--run_metrics", action="store_true",
                        help="Run only the metrics component")
    
    args = parser.parse_args()
    
    # Set the data path
    data_path = args.input_file
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Error: WebDev dataset not found at {data_path}")
        print("Please make sure the dataset is available.")
        return
    
    print("="*60)
    print("LMM-VIBES PIPELINE")
    print("="*60)
    print(f"Dataset: {data_path}")
    print(f"Output: {args.output_dir}")
    if args.sample_size:
        print(f"Sample size: {args.sample_size}")
    else:
        print("Using full dataset")
    print("="*60)
    
    # Handle metrics-only mode
    if args.run_metrics:
        print("\n🔧 Running metrics-only mode...")
        print("This will load existing pipeline results and compute metrics only.")
        data_path = args.output_dir
        
        # Check if the input path exists and looks like pipeline results
        if not os.path.exists(data_path):
            print(f"Error: Input path not found: {data_path}")
            print("For metrics-only mode, provide a path to existing pipeline results.")
            print("This can be:")
            print("  - A file: results/previous_run/full_dataset.json")
            print("  - A directory: results/previous_run/")
            return
        
        # Run metrics-only computation
        clustered_df, model_stats = compute_metrics_only(
            input_path=data_path,
            method=args.method,
            output_dir=args.output_dir,
            use_wandb=args.use_wandb,
            verbose=not args.quiet,
            metrics_kwargs={"compute_confidence_intervals": True}
        )
        
        print(f"\n🎉 Metrics computation completed! Results saved to: {args.output_dir}")
        return
    
    # Run pipeline with webdev-optimized parameters
    clustered_df, model_stats = run_pipeline(
        data_path=data_path,
        output_dir=args.output_dir,
        method=args.method,
        system_prompt=args.system_prompt,
        clusterer="hdbscan",
        min_cluster_size=args.min_cluster_size,
        max_coarse_clusters=args.max_coarse_clusters,
        embedding_model="openai",
        hierarchical=False,
        max_workers=args.max_workers,
        use_wandb=args.use_wandb,
        verbose=not args.quiet,
        sample_size=args.sample_size,
        metrics_kwargs={"compute_confidence_intervals": True}
    )
    
    print(f"\n🎉 WebDev pipeline completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 
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
    parser.add_argument("--metrics_only", action="store_true",
                        help="Only run metrics calculation on existing clustered data")
    
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
    
    if args.metrics_only:
        # Run only metrics calculation
        print("="*60)
        print("METRICS ONLY MODE")
        print("="*60)
        print(f"Output: {args.output_dir}")
        print("="*60)
        
        # Try to load existing clustered data - check multiple formats
        clustered_data_path = None
        clustered_df = None
        
        # First try JSON format for clustered DataFrame (preserves nested data)
        clustered_json_path = os.path.join(args.output_dir, "clustered_results.jsonl")
        full_dataset_json_path = os.path.join(args.output_dir, "full_dataset.json")
        parquet_path = os.path.join(args.output_dir, "clustered_results.parquet")
        
        # Option 1: Load clustered DataFrame directly from JSON
        if os.path.exists(clustered_json_path):
            print(f"Loading clustered data from: {clustered_json_path}")
            try:
                clustered_df = pd.read_json(clustered_json_path, lines=True, orient='records')
                print("✅ Successfully loaded clustered DataFrame from JSONL")
            except Exception as e:
                print(f"❌ Failed to load clustered DataFrame JSONL: {e}")
        
        # Option 2: Load from full PropertyDataset JSON (fallback)
        if clustered_df is None and os.path.exists(full_dataset_json_path):
            print(f"Loading from full PropertyDataset JSON: {full_dataset_json_path}")
            try:
                from stringsight import PropertyDataset
                dataset = PropertyDataset.from_json(full_dataset_json_path)
                clustered_df = dataset.to_dataframe(type="clusters")
                print("✅ Successfully loaded from PropertyDataset JSON")
            except Exception as e:
                print(f"❌ Failed to load from PropertyDataset JSON: {e}")
        
        # Option 3: Fallback to parquet with error handling (limited data)
        if clustered_df is None and os.path.exists(parquet_path):
            print(f"Loading clustered data from parquet: {parquet_path}")
            print("⚠️  WARNING: Parquet format excludes nested data columns (like 'score')")
            print("⚠️  Metrics may not work properly without complete data")
            try:
                # Try with different parquet options to handle nested data
                clustered_df = pd.read_parquet(parquet_path, engine='pyarrow')
                print("✅ Successfully loaded from parquet format")
                print("📋 Checking for required columns...")
                
                # Check if essential columns for metrics are present
                required_cols = ['score']  # Add other essential columns as needed
                missing_cols = [col for col in required_cols if col not in clustered_df.columns]
                if missing_cols:
                    print(f"❌ Missing essential columns for metrics: {missing_cols}")
                    print("💡 Consider using JSON format which preserves all data")
                    clustered_df = None  # Don't use incomplete data
                else:
                    print("✅ All required columns present")
                    
            except Exception as e1:
                print(f"❌ Failed with pyarrow engine: {e1}")
                try:
                    # Try with fastparquet engine as fallback
                    clustered_df = pd.read_parquet(parquet_path, engine='fastparquet')
                    print("✅ Successfully loaded with fastparquet engine")
                    print("⚠️  Note: Data may be incomplete due to parquet limitations")
                except Exception as e2:
                    print(f"❌ Failed with fastparquet engine: {e2}")
        
        if clustered_df is None:
            print(f"Error: Could not load clustered data from any of:")
            print(f"  - {clustered_json_path} (preferred - complete data)")
            print(f"  - {full_dataset_json_path} (fallback - complete data)")
            print(f"  - {parquet_path} (limited - excludes nested data)")
            print("\nRecommendations:")
            print("  1. Run the full pipeline first to generate data files")
            print("  2. Ensure JSON files are available for complete data")
            print("  3. Check that the output directory path is correct")
            return
        
        # Run metrics calculation only
        from stringsight import PropertyDataset
        from stringsight.metrics.single_model import SingleModelMetrics
        
        # Create PropertyDataset from clustered DataFrame
        dataset = PropertyDataset.from_dataframe(clustered_df, method="single_model")
        
        # Run metrics stage
        metrics_stage = SingleModelMetrics(output_dir=args.output_dir)
        dataset = metrics_stage.run(dataset)
        
        # Save updated results
        model_stats = dataset.model_stats
        import json
        stats_path = os.path.join(args.output_dir, "model_stats.json")
        with open(stats_path, 'w') as f:
            # Convert ModelStats objects to dictionaries for JSON serialization
            serializable_stats = {}
            for model, levels in model_stats.items():
                serializable_stats[model] = {}
                for level, stats_list in levels.items():
                    serializable_stats[model][level] = [stat.to_dict() for stat in stats_list]
            json.dump(serializable_stats, f, indent=2)
        
        print(f"\n🎉 Metrics calculation completed! Results saved to: {args.output_dir}")
        print(f"📊 Model stats saved to: {stats_path}")
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
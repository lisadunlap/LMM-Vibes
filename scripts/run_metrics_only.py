#!/usr/bin/env python3
"""
Run just the metrics computation part of the LMM-Vibes pipeline.

This script loads existing pipeline results (from extraction and clustering stages)
and runs only the metrics computation stage. Useful for:
- Recomputing metrics with different parameters
- Running metrics on results from previous pipeline runs
- Debugging metrics computation without re-running the full pipeline
"""

import argparse
import os
import pandas as pd
import json
from pathlib import Path
import time
from datetime import datetime

from lmmvibes.core.data_objects import PropertyDataset
from lmmvibes.metrics import get_metrics
from lmmvibes.pipeline import Pipeline


def load_existing_results(input_path: str, method: str = "single_model") -> PropertyDataset:
    """
    Load existing pipeline results from various file formats.
    
    Args:
        input_path: Path to the results file (can be .json, .parquet, .pkl, or directory)
        method: "single_model" or "side_by_side"
        
    Returns:
        PropertyDataset with existing results
    """
    input_path = Path(input_path)
    
    if input_path.is_dir():
        # Try to load from a directory containing pipeline outputs
        possible_files = [
            input_path / "full_dataset.json",
            input_path / "full_dataset.parquet", 
            input_path / "clustered_results.parquet",
            input_path / "dataset.json",
            input_path / "dataset.parquet"
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                print(f"Loading from: {file_path}")
                return PropertyDataset.load(str(file_path))
        
        raise FileNotFoundError(f"No recognizable dataset file found in {input_path}")
    
    elif input_path.is_file():
        # Load from a specific file
        print(f"Loading from: {input_path}")
        return PropertyDataset.load(str(input_path))
    
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


def run_metrics_only(
    input_path: str,
    output_dir: str,
    method: str = "single_model",
    metrics_kwargs: dict = None,
    use_wandb: bool = False,
    verbose: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Run only the metrics computation stage on existing pipeline results.
    
    Args:
        input_path: Path to existing pipeline results
        output_dir: Directory to save metrics results
        method: "single_model" or "side_by_side"
        metrics_kwargs: Additional arguments for metrics computation
        use_wandb: Whether to enable wandb logging
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (clustered_df, model_stats)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading existing results from: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Method: {method}")
    
    # Load existing dataset
    dataset = load_existing_results(input_path, method)
    
    # Verify we have the required data for metrics
    if not dataset.clusters:
        raise ValueError("No clusters found in the dataset. Metrics computation requires clustered data.")
    
    if not dataset.properties:
        raise ValueError("No properties found in the dataset. Metrics computation requires extracted properties.")
    
    print(f"Loaded dataset with:")
    print(f"  - {len(dataset.conversations)} conversations")
    print(f"  - {len(dataset.properties)} properties")
    print(f"  - {len(dataset.clusters)} clusters")
    print(f"  - Models: {dataset.all_models}")
    
    # Create metrics stage
    metrics_config = {
        'method': method,
        'output_dir': str(output_path),
        'use_wandb': use_wandb,
        'verbose': verbose,
        **(metrics_kwargs or {})
    }
    
    metrics_stage = get_metrics(**metrics_config)
    
    # Create a minimal pipeline with just the metrics stage
    pipeline = Pipeline("Metrics-Only", [metrics_stage])
    
    # Run metrics computation
    print("\n" + "="*60)
    print("COMPUTING METRICS")
    print("="*60)
    
    start_time = time.time()
    result_dataset = pipeline.run(dataset)
    runtime = time.time() - start_time
    
    print(f"\n✅ Metrics computation completed in {runtime:.2f}s")
    
    # Convert back to DataFrame format
    clustered_df = result_dataset.to_dataframe()
    model_stats = result_dataset.model_stats
    
    # Save results
    print(f"\nSaving results to: {output_path}")
    
    # 1. Save clustered DataFrame as parquet
    clustered_parquet_path = output_path / "metrics_results.parquet"
    clustered_df.to_parquet(clustered_parquet_path, index=False)
    print(f"  ✓ Saved metrics DataFrame (parquet): {clustered_parquet_path}")
    
    # 2. Save complete PropertyDataset as JSON
    dataset_json_path = output_path / "metrics_dataset.json"
    result_dataset.save(str(dataset_json_path), format="json")
    print(f"  ✓ Saved metrics PropertyDataset (JSON): {dataset_json_path}")
    
    # 3. Save metrics artifacts
    # Functional metrics attach a compatibility structure under data.model_stats["functional_metrics"]
    fm = None
    if isinstance(model_stats, dict) and "functional_metrics" in model_stats:
        fm = model_stats["functional_metrics"]
    
    if fm:
        # Save the three functional JSONs if not already written by the metrics stage
        mc_path = output_path / "model_cluster_scores.json"
        cl_path = output_path / "cluster_scores.json"
        ms_path = output_path / "model_scores.json"
        with open(mc_path, 'w') as f:
            json.dump(fm.get("model_cluster_scores", {}), f, indent=2)
        with open(cl_path, 'w') as f:
            json.dump(fm.get("cluster_scores", {}), f, indent=2)
        with open(ms_path, 'w') as f:
            json.dump(fm.get("model_scores", {}), f, indent=2)
        print(f"  ✓ Saved functional metrics JSONs: {mc_path}, {cl_path}, {ms_path}")
    else:
        # Legacy fallback: serialize ModelStats objects
        stats_path = output_path / "metrics_stats.json"
        stats_for_json = {}
        for model_name, stats in model_stats.items():
            stats_for_json[str(model_name)] = {
                "fine": [stat.to_dict() for stat in stats.get("fine", [])]
            }
            if "coarse" in stats:
                stats_for_json[str(model_name)]["coarse"] = [stat.to_dict() for stat in stats["coarse"]]
        with open(stats_path, 'w') as f:
            json.dump(stats_for_json, f, indent=2)
        print(f"  ✓ Saved legacy model statistics (JSON): {stats_path}")
 
    # Print summary
    print(f"\n📊 Metrics Summary:")
    print(f"  - Models analyzed: {len(model_stats)}")
    if fm:
        # Brief: count model-cluster entries
        print(f"  - Model-cluster combinations: {sum(len(v) for v in fm.get('model_cluster_scores', {}).values())}")
    else:
        for model_name, stats in model_stats.items():
            print(f"  - {model_name}: {len(stats.get('fine', []))} fine clusters")
            if 'coarse' in stats:
                print(f"    {len(stats['coarse'])} coarse clusters")
    
    return clustered_df, model_stats


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Run only the metrics computation part of the LMM-Vibes pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run metrics on existing pipeline results
  python scripts/run_metrics_only.py \\
      --input results/previous_run/full_dataset.json \\
      --output results/metrics_only \\
      --method single_model

  # Run metrics on a directory containing pipeline outputs
  python scripts/run_metrics_only.py \\
      --input results/previous_run/ \\
      --output results/metrics_only \\
      --method side_by_side

  # Run metrics with custom output directory
  python scripts/run_metrics_only.py \\
      --input results/previous_run/full_dataset.parquet \\
      --output results/metrics_custom \\
      --method single_model \\
      --metrics-output-dir custom_metrics_output
        """
    )
    
    # Required arguments
    parser.add_argument("--input", type=str, required=True,
                        help="Path to existing pipeline results (file or directory)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for metrics results")
    
    # Pipeline parameters
    parser.add_argument("--method", type=str, default="single_model",
                        choices=["single_model", "side_by_side"],
                        help="Analysis method (default: single_model)")
    
    # Metrics-specific options
    parser.add_argument("--metrics-output-dir", type=str, default=None,
                        help="Custom output directory for metrics files (overrides --output)")
    
    # Flags
    parser.add_argument("--use-wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable verbose output")
    
    args = parser.parse_args()
    
    # Prepare metrics kwargs
    metrics_kwargs = {}
    if args.metrics_output_dir:
        metrics_kwargs['output_dir'] = args.metrics_output_dir
    
    # Run metrics-only pipeline
    try:
        clustered_df, model_stats = run_metrics_only(
            input_path=args.input,
            output_dir=args.output,
            method=args.method,
            metrics_kwargs=metrics_kwargs,
            use_wandb=args.use_wandb,
            verbose=not args.quiet
        )
        
        print(f"\n🎉 Metrics computation completed successfully!")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error during metrics computation: {e}")
        raise


if __name__ == "__main__":
    main() 
"""
Public API for LMM-Vibes.

This module provides the main explain() function that users will interact with.
"""

from typing import Dict, List, Any, Callable, Optional, Union, Tuple
import pandas as pd
from .core.data_objects import PropertyDataset
from .pipeline import Pipeline, PipelineBuilder
import time


def explain(
    df: pd.DataFrame,
    method: str = "side_by_side",
    system_prompt: str = "one_sided_system_prompt_no_examples",
    prompt_builder: Optional[Callable[[pd.Series], str]] = None,
    *,
    # Extraction parameters
    model_name: str = "gpt-4.1",
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 16000,
    max_workers: int = 16,
    # Clustering parameters  
    clusterer: Union[str, "PipelineStage"] = "hdbscan",
    min_cluster_size: int = 30,
    embedding_model: str = "openai",
    hierarchical: bool = False,
    assign_outliers: bool = False,
    max_coarse_clusters: int = 25,
    # Metrics parameters
    metrics_kwargs: Optional[Dict[str, Any]] = None,
    # Caching & logging
    use_wandb: bool = True,
    wandb_project: Optional[str] = None,
    include_embeddings: bool = True,
    verbose: bool = True,
    # Output parameters
    output_dir: Optional[str] = None,
    # Pipeline configuration
    custom_pipeline: Optional[Pipeline] = None,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Explain model behavior patterns from conversation data.
    
    This is the main entry point for LMM-Vibes. It takes a DataFrame of conversations
    and returns the same data with extracted properties and clusters.
    
    Args:
        df: DataFrame with conversation data
        method: "side_by_side" or "single_model" 
        system_prompt: System prompt for property extraction
        prompt_builder: Optional custom prompt builder function
        
        # Extraction parameters
        model_name: LLM model for property extraction
        temperature: Temperature for LLM
        top_p: Top-p for LLM
        max_tokens: Max tokens for LLM
        max_workers: Max parallel workers for API calls
        
        # Clustering parameters
        clusterer: Clustering method ("hdbscan", "hdbscan_native", "hierarchical") or PipelineStage
        min_cluster_size: Minimum cluster size
        embedding_model: Embedding model ("openai" or sentence-transformer model)
        hierarchical: Whether to create hierarchical clusters
        assign_outliers: Whether to assign outliers to nearest clusters
        
        # Metrics parameters
        metrics_kwargs: Additional metrics configuration
        
        # Caching & logging
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name
        include_embeddings: Whether to include embeddings in output
        verbose: Whether to print progress
        
        # Output parameters
        output_dir: Directory to save results (optional). If provided, saves:
                   - clustered_results.parquet: DataFrame with all results
                   - full_dataset.json: Complete PropertyDataset (JSON format)
                   - full_dataset.parquet: Complete PropertyDataset (parquet format)
                   - model_stats.json: Model statistics and rankings
                   - summary.txt: Human-readable summary
        
        # Pipeline configuration
        custom_pipeline: Custom pipeline to use instead of default
        **kwargs: Additional configuration options
        
    Returns:
        Tuple of (clustered_df, model_stats)
        - clustered_df: Original DataFrame with added property and cluster columns
        - model_stats: Dictionary of model statistics and rankings
        
    Example:
        >>> import pandas as pd
        >>> from lmmvibes import explain
        >>> 
        >>> # Load your conversation data
        >>> df = pd.read_csv("conversations.csv")
        >>> 
        >>> # Explain model behavior and save results
        >>> clustered_df, model_stats = explain(
        ...     df,
        ...     method="side_by_side",
        ...     min_cluster_size=20,
        ...     hierarchical=True,
        ...     output_dir="results/"  # Automatically saves results
        ... )
        >>> 
        >>> # Explore the results
        >>> print(clustered_df.columns)
        >>> print(model_stats.keys())
    """
    
    # Create PropertyDataset from input DataFrame
    dataset = PropertyDataset.from_dataframe(df, method=method)
    
    # 2️⃣  Initialize wandb if enabled
    wandb_run_name = f"explain_{method}_{int(time.time())}"
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project or "lmm-vibes",
            name=wandb_run_name,
            config={
                "method": method,
                "system_prompt": system_prompt,
                "model_name": model_name,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "max_workers": max_workers,
                "clusterer": clusterer,
                "min_cluster_size": min_cluster_size,
                "embedding_model": embedding_model,
                "hierarchical": hierarchical,
                "assign_outliers": assign_outliers,
                "max_coarse_clusters": max_coarse_clusters,
                "include_embeddings": include_embeddings,
                "output_dir": output_dir,
            },
            reinit=False  # Don't reinitialize if already exists
        )
    
    # Use custom pipeline if provided, otherwise build default pipeline
    if custom_pipeline is not None:
        pipeline = custom_pipeline
        # Ensure the custom pipeline uses the same wandb configuration
        if hasattr(pipeline, 'use_wandb'):
            pipeline.use_wandb = use_wandb
            pipeline.wandb_project = wandb_project or "lmm-vibes"
            if use_wandb:
                pipeline._wandb_ok = True  # Mark that wandb is already initialized
    else:
        pipeline = _build_default_pipeline(
            method=method,
            system_prompt=system_prompt,
            prompt_builder=prompt_builder,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_workers=max_workers,
            clusterer=clusterer,
            min_cluster_size=min_cluster_size,
            embedding_model=embedding_model,
            hierarchical=hierarchical,
            assign_outliers=assign_outliers,
            metrics_kwargs=metrics_kwargs,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            include_embeddings=include_embeddings,
            verbose=verbose,
            **kwargs
        )
    
    # 4️⃣  Execute pipeline
    result_dataset = pipeline.run(dataset)
    
    # Convert back to DataFrame format
    clustered_df = result_dataset.to_dataframe()
    model_stats = result_dataset.model_stats
    
    # 5️⃣  Save results if output_dir is provided
    if output_dir is not None:
        _save_results_to_dir(result_dataset, clustered_df, model_stats, output_dir, verbose)
    
    # Log accumulated summary metrics from pipeline stages
    if use_wandb and hasattr(pipeline, 'log_final_summary'):
        pipeline.log_final_summary()
    
    # Log final results to wandb if enabled
    if use_wandb:
        _log_final_results_to_wandb(clustered_df, model_stats)
    
    return clustered_df, model_stats


def _build_default_pipeline(
    method: str,
    system_prompt: str,
    prompt_builder: Optional[Callable],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_workers: int,
    clusterer: Union[str, "PipelineStage"],
    min_cluster_size: int,
    embedding_model: str,
    hierarchical: bool,
    assign_outliers: bool,
    metrics_kwargs: Optional[Dict[str, Any]],
    use_wandb: bool,
    wandb_project: Optional[str],
    include_embeddings: bool,
    verbose: bool,
    **kwargs
) -> Pipeline:
    """
    Build the default pipeline based on configuration.
    
    This function constructs the standard pipeline stages based on the user's
    configuration. It handles the complexity of importing and configuring
    the appropriate stages.
    """
    
    # Import stages (lazy imports to avoid circular dependencies)
    from .extractors import get_extractor
    from .postprocess import LLMJsonParser, PropertyValidator
    from .clusterers import get_clusterer
    from .metrics import get_metrics
    
    # Build pipeline using PipelineBuilder
    builder = PipelineBuilder(name=f"LMM-Vibes-{method}")
    
    # Configure common options
    common_config = {
        'verbose': verbose,
        'use_wandb': use_wandb,
        'wandb_project': wandb_project or "lmm-vibes"
    }
    
    # 1. Property extraction stage
    extractor = get_extractor(
        model_name=model_name,
        system_prompt=system_prompt,
        prompt_builder=prompt_builder,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_workers=max_workers,
        **common_config
    )
    builder.extract_properties(extractor)
    
    # 2. JSON parsing stage
    parser = LLMJsonParser(**common_config)
    builder.parse_properties(parser)
    
    # 3. Property validation stage
    validator = PropertyValidator(**common_config)
    builder.add_stage(validator)
    
    # 4. Clustering stage
    if isinstance(clusterer, str):
        clusterer_stage = get_clusterer(
            clusterer,
            min_cluster_size=min_cluster_size,
            embedding_model=embedding_model,
            hierarchical=hierarchical,
            assign_outliers=assign_outliers,
            include_embeddings=include_embeddings,
            **common_config
        )
    else:
        clusterer_stage = clusterer
    
    builder.cluster_properties(clusterer_stage)
    
    # 5. Metrics computation stage
    metrics_stage = get_metrics(
        method,
        **(metrics_kwargs or {}),
        **common_config
    )
    builder.compute_metrics(metrics_stage)
    
    # Build and return the pipeline
    pipeline = builder.configure(**common_config).build()
    
    # If wandb is already initialized globally, mark the pipeline as having wandb available
    if use_wandb:
        import wandb
        if wandb.run is not None and hasattr(pipeline, '_wandb_ok'):
            pipeline._wandb_ok = True
    
    return pipeline


def _log_final_results_to_wandb(df: pd.DataFrame, model_stats: Dict[str, Any]):
    """Log final results to wandb."""
    try:
        import wandb
        
        # Log dataset summary as summary metrics (not regular metrics)
        if wandb.run is not None:
            wandb.run.summary["final_dataset_shape"] = str(df.shape)
            wandb.run.summary["final_total_conversations"] = len(df['question_id'].unique()) if 'question_id' in df.columns else len(df)
            wandb.run.summary["final_total_properties"] = len(df)
            wandb.run.summary["final_unique_models"] = len(df['model'].unique()) if 'model' in df.columns else 0
        
        # Log clustering results if present
        cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
        if cluster_cols:
            for col in cluster_cols:
                if col.endswith('_id'):
                    cluster_ids = df[col].unique()
                    n_clusters = len([c for c in cluster_ids if c != -1])
                    n_outliers = sum(1 for c in cluster_ids if c == -1)
                    
                    level = "fine" if "fine" in col else "coarse" if "coarse" in col else "main"
                    # Log these as summary metrics
                    if wandb.run is not None:
                        wandb.run.summary[f"final_{level}_clusters"] = n_clusters
                        wandb.run.summary[f"final_{level}_outliers"] = n_outliers
                        wandb.run.summary[f"final_{level}_outlier_rate"] = n_outliers / len(df) if len(df) > 0 else 0

        # Log model statistics as tables - one table per model
        if model_stats:
            if wandb.run is not None:
                wandb.run.summary["final_models_analyzed"] = len(model_stats)
            
            # Create detailed tables for each model
            for model_name, stats in model_stats.items():
                # Log fine-grained clusters table
                if "fine" in stats and len(stats["fine"]) > 0:
                    fine_table_data = []
                    for stat in stats["fine"]:
                        row = stat.to_dict()
                        # Add examples_count for convenience
                        row["examples_count"] = len(stat.examples) if stat.examples else 0
                        # Rename property_description to cluster_label for consistency
                        row["cluster_label"] = row.pop("property_description")
                        fine_table_data.append(row)
                    
                    # Create and log the fine-grained table
                    fine_table = wandb.Table(columns=list(fine_table_data[0].keys()), data=[list(row.values()) for row in fine_table_data])
                    wandb.log({f"Metrics/model_stats_{model_name}_fine_clusters": fine_table})
                
                # Log coarse-grained clusters table if available
                if "coarse" in stats and len(stats["coarse"]) > 0:
                    coarse_table_data = []
                    for stat in stats["coarse"]:
                        row = stat.to_dict()
                        # Add examples_count for convenience
                        row["examples_count"] = len(stat.examples) if stat.examples else 0
                        # Rename property_description to cluster_label for consistency
                        row["cluster_label"] = row.pop("property_description")
                        coarse_table_data.append(row)
                    
                    # Create and log the coarse-grained table
                    coarse_table = wandb.Table(columns=list(coarse_table_data[0].keys()), data=[list(row.values()) for row in coarse_table_data])
                    wandb.log({f"Metrics/model_stats_{model_name}_coarse_clusters": coarse_table})
                
                # Log model summary statistics as summary metrics
                if "fine" in stats and wandb.run is not None:
                    fine_stats = stats["fine"]
                    avg_score = sum(stat.score for stat in fine_stats) / len(fine_stats)
                    avg_quality_score = sum(stat.quality_score for stat in fine_stats) / len(fine_stats)
                    total_size = sum(stat.size for stat in fine_stats)
                    
                    # Log summary statistics for this model as summary metrics
                    wandb.run.summary[f"model_{model_name}_fine_clusters_count"] = len(fine_stats)
                    wandb.run.summary[f"model_{model_name}_avg_score"] = avg_score
                    wandb.run.summary[f"model_{model_name}_avg_quality_score"] = avg_quality_score
                    wandb.run.summary[f"model_{model_name}_total_size"] = total_size
                    wandb.run.summary[f"model_{model_name}_max_score"] = max(stat.score for stat in fine_stats)
                    wandb.run.summary[f"model_{model_name}_min_score"] = min(stat.score for stat in fine_stats)
                    wandb.run.summary[f"model_{model_name}_max_quality_score"] = max(stat.quality_score for stat in fine_stats)
                    wandb.run.summary[f"model_{model_name}_min_quality_score"] = min(stat.quality_score for stat in fine_stats)
                    
                    if "coarse" in stats:
                        coarse_stats = stats["coarse"]
                        coarse_avg_quality_score = sum(stat.quality_score for stat in coarse_stats) / len(coarse_stats)
                        wandb.run.summary[f"model_{model_name}_coarse_clusters_count"] = len(coarse_stats)
                        wandb.run.summary[f"model_{model_name}_coarse_avg_score"] = sum(stat.score for stat in coarse_stats) / len(coarse_stats)
                        wandb.run.summary[f"model_{model_name}_coarse_avg_quality_score"] = coarse_avg_quality_score
        
        print("✅ Successfully logged detailed model statistics to wandb")
        print(f"   • Dataset summary metrics")
        print(f"   • Clustering results")
        print(f"   • Model statistics tables: {len(model_stats)} models")
        print(f"   • Summary metrics logged to run summary")
        
    except Exception as e:
        print(f"Failed to log final results to wandb: {e}")
        import traceback
        traceback.print_exc()


def _save_results_to_dir(
    result_dataset: PropertyDataset,
    clustered_df: pd.DataFrame,
    model_stats: Dict[str, Any],
    output_dir: str,
    verbose: bool = True
):
    """Save results to the specified output directory."""
    import pathlib
    import json
    
    # Create output directory if it doesn't exist
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Saving results to: {output_path}")
    
    # 1. Save clustered DataFrame as parquet
    clustered_parquet_path = output_path / "clustered_results.parquet"
    clustered_df.to_parquet(clustered_parquet_path, index=False)
    if verbose:
        print(f"  ✓ Saved clustered DataFrame (parquet): {clustered_parquet_path}")
    
    # 2. Save complete PropertyDataset as JSON
    dataset_json_path = output_path / "full_dataset.json"
    result_dataset.save(str(dataset_json_path), format="json")
    if verbose:
        print(f"  ✓ Saved full PropertyDataset (JSON): {dataset_json_path}")
    
    # 3. Save complete PropertyDataset as parquet
    dataset_parquet_path = output_path / "full_dataset.parquet"
    result_dataset.save(str(dataset_parquet_path), format="parquet")
    if verbose:
        print(f"  ✓ Saved full PropertyDataset (parquet): {dataset_parquet_path}")
    
    # 4. Save model statistics as JSON
    stats_path = output_path / "model_stats.json"
    
    # Convert ModelStats objects to dictionaries for JSON serialization
    stats_for_json = {}
    for model_name, stats in model_stats.items():
        stats_for_json[str(model_name)] = {
            "fine": [stat.to_dict() for stat in stats["fine"]]
        }
        if "coarse" in stats:
            stats_for_json[str(model_name)]["coarse"] = [stat.to_dict() for stat in stats["coarse"]]
    
    with open(stats_path, 'w') as f:
        json.dump(stats_for_json, f, indent=2)
    if verbose:
        print(f"  ✓ Saved model statistics (JSON): {stats_path}")
    
    # 5. Save summary statistics
    summary_path = output_path / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("LMM-Vibes Results Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total conversations: {len(clustered_df['question_id'].unique()) if 'question_id' in clustered_df.columns else len(clustered_df)}\n")
        f.write(f"Total properties: {len(clustered_df)}\n")
        f.write(f"Models analyzed: {len(model_stats)}\n")
        
        # Clustering info
        if 'property_description_fine_cluster_id' in clustered_df.columns:
            n_fine_clusters = len(clustered_df['property_description_fine_cluster_id'].unique())
            f.write(f"Fine clusters: {n_fine_clusters}\n")
        
        if 'property_description_coarse_cluster_id' in clustered_df.columns:
            n_coarse_clusters = len(clustered_df['property_description_coarse_cluster_id'].unique())
            f.write(f"Coarse clusters: {n_coarse_clusters}\n")
        
        f.write(f"\nFiles saved:\n")
        f.write(f"  - clustered_results.parquet: Complete DataFrame with clusters\n")
        f.write(f"  - full_dataset.json: Complete PropertyDataset object (JSON format)\n")
        f.write(f"  - full_dataset.parquet: Complete PropertyDataset object (parquet format)\n")
        f.write(f"  - model_stats.json: Model statistics and rankings\n")
        f.write(f"  - summary.txt: This summary file\n")
        
        # Model rankings
        f.write(f"\nModel Rankings (by average score):\n")
        model_avg_scores = {}
        for model_name, stats in model_stats.items():
            if "fine" in stats and len(stats["fine"]) > 0:
                avg_score = sum(stat.score for stat in stats["fine"]) / len(stats["fine"])
                model_avg_scores[model_name] = avg_score
        
        for i, (model_name, avg_score) in enumerate(sorted(model_avg_scores.items(), key=lambda x: x[1], reverse=True)):
            f.write(f"  {i+1}. {model_name}: {avg_score:.3f}\n")
    
    if verbose:
        print(f"  ✓ Saved summary: {summary_path}")
        print(f"🎉 All results saved to: {output_path}")
        print(f"    • DataFrame: clustered_results.parquet")
        print(f"    • PropertyDataset: full_dataset.json + full_dataset.parquet")
        print(f"    • Model stats: model_stats.json")
        print(f"    • Summary: summary.txt")


# Convenience functions for common use cases
def explain_side_by_side(
    df: pd.DataFrame,
    system_prompt: str = "one_sided_system_prompt",
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function for side-by-side model comparison.
    
    Args:
        df: DataFrame with columns: model_a, model_b, model_a_response, model_b_response, winner
        system_prompt: System prompt for extraction
        **kwargs: Additional arguments passed to explain()
        
    Returns:
        Tuple of (clustered_df, model_stats)
    """
    return explain(df, method="side_by_side", system_prompt=system_prompt, **kwargs)


def explain_single_model(
    df: pd.DataFrame,
    system_prompt: str = "single_model_system_prompt",
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function for single model analysis.
    
    Args:
        df: DataFrame with columns: model, model_response, score
        system_prompt: System prompt for extraction
        **kwargs: Additional arguments passed to explain()
        
    Returns:
        Tuple of (clustered_df, model_stats)
    """
    return explain(df, method="single_model", system_prompt=system_prompt, **kwargs)


def explain_with_custom_pipeline(
    df: pd.DataFrame,
    pipeline: Pipeline,
    method: str = "side_by_side"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run explanation with a custom pipeline.
    
    Args:
        df: Input DataFrame
        pipeline: Custom Pipeline instance
        method: Data format method
        
    Returns:
        Tuple of (clustered_df, model_stats)
    """
    return explain(df, method=method, custom_pipeline=pipeline) 
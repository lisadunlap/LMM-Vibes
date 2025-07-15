"""
Public API for LMM-Vibes.

This module provides the main explain() function that users will interact with.
"""

from typing import Dict, List, Any, Callable, Optional, Union, Tuple
import pandas as pd
from .core.data_objects import PropertyDataset
from .pipeline import Pipeline, PipelineBuilder
from .prompts import get_default_system_prompt
import time


def explain(
    df: pd.DataFrame,
    method: str = "single_model",
    system_prompt: str = None,
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
    # Cache configuration
    extraction_cache_dir: Optional[str] = None,
    clustering_cache_dir: Optional[str] = None,
    metrics_cache_dir: Optional[str] = None,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Explain model behavior patterns from conversation data.
    
    This is the main entry point for LMM-Vibes. It takes a DataFrame of conversations
    and returns the same data with extracted properties and clusters.
    
    Args:
        df: DataFrame with conversation data
        method: "side_by_side" or "single_model"
        system_prompt: System prompt for property extraction (if None, will be auto-determined)
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
    
    # Auto-determine system prompt if not provided
    if system_prompt is None:
        # Check if data contains score/preference information
        contains_score = _check_contains_score(df, method)
        system_prompt = get_default_system_prompt(method, contains_score)
        if verbose:
            print(f"Auto-selected system prompt for method '{method}' (contains_score={contains_score})")

    # Print the system prompt for verification
    if verbose:
        print("\n" + "="*80)
        print("SYSTEM PROMPT")
        print("="*80)
        print(system_prompt)
        print("="*80 + "\n")
    
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
            extraction_cache_dir=extraction_cache_dir,
            clustering_cache_dir=clustering_cache_dir,
            metrics_cache_dir=metrics_cache_dir,
            **kwargs
        )
    
    # 4️⃣  Execute pipeline
    result_dataset = pipeline.run(dataset)

       # Check for 0 properties before attempting to save
    if len([p for p in result_dataset.properties if p.property_description is not None]) == 0:
        raise RuntimeError(
            "\n" + "="*60 + "\n"
            "ERROR: Pipeline completed with 0 valid properties!\n"
            "="*60 + "\n"
            "This indicates that all property extraction attempts failed.\n"
            "Common causes:\n\n"
            "1. JSON PARSING FAILURES:\n"
            "   - LLM returning natural language instead of JSON\n"
            "   - Check logs above for 'Failed to parse JSON' errors\n\n"
            "2. SYSTEM PROMPT MISMATCH:\n"
            "   - Current system_prompt may not suit your data format\n"
            "   - Try a different system_prompt parameter\n\n"
            "3. API/MODEL ISSUES:\n"
            "   - OpenAI API key invalid or quota exceeded\n"
            "   - Model configuration problems\n\n"
            "Cannot save results with 0 properties.\n"
            "="*60
        )
    
    # Convert back to DataFrame format
    clustered_df = result_dataset.to_dataframe()
    model_stats = result_dataset.model_stats
    
    # 5️⃣  Save results if output_dir is provided
    if output_dir is not None:
        _save_results_to_dir(result_dataset, clustered_df, model_stats, output_dir, verbose, pipeline)
    
    # Log accumulated summary metrics from pipeline stages
    if use_wandb and hasattr(pipeline, 'log_final_summary'):
        pipeline.log_final_summary()
    
    # Log final results to wandb if enabled
    if use_wandb:
        _log_final_results_to_wandb(clustered_df, model_stats)
    
    return clustered_df, model_stats


def _check_contains_score(df: pd.DataFrame, method: str) -> bool:
    """
    Check if the DataFrame contains score/preference information.
    
    Args:
        df: Input DataFrame
        method: Analysis method
        
    Returns:
        True if the data contains scores, False otherwise
    """
    if method == "side_by_side":
        if "score" in df.columns:
            # Check if score column has any non-empty, non-None values
            return df["score"].notna().any() and (df["score"] != {}).any()
        return False
    
    elif method == "single_model":
        # Check for score column
        if "score" in df.columns:
            # Check if score column has any non-empty, non-None values
            return df["score"].notna().any() and (df["score"] != {}).any()
        return False
    
    else:
        # Default to False for unknown methods
        return False


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
    extraction_cache_dir: Optional[str] = None,
    clustering_cache_dir: Optional[str] = None,
    metrics_cache_dir: Optional[str] = None,
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
    extractor_kwargs = {
        'model_name': model_name,
        'system_prompt': system_prompt,
        'prompt_builder': prompt_builder,
        'temperature': temperature,
        'top_p': top_p,
        'max_tokens': max_tokens,
        'max_workers': max_workers,
        **common_config
    }
    
    # Add cache directory for extraction if provided
    if extraction_cache_dir:
        extractor_kwargs['cache_dir'] = extraction_cache_dir
    
    extractor = get_extractor(**extractor_kwargs)
    builder.extract_properties(extractor)
    
    # 2. JSON parsing stage
    parser = LLMJsonParser(**common_config)
    builder.parse_properties(parser)
    
    # 3. Property validation stage
    validator = PropertyValidator(**common_config)
    builder.add_stage(validator)
    
    # 4. Clustering stage
    clusterer_kwargs = {
        'min_cluster_size': min_cluster_size,
        'embedding_model': embedding_model,
        'hierarchical': hierarchical,
        'assign_outliers': assign_outliers,
        'include_embeddings': include_embeddings,
        **common_config
    }
    
    # Add cache directory for clustering if provided
    if clustering_cache_dir:
        clusterer_kwargs['cache_dir'] = clustering_cache_dir
    
    if isinstance(clusterer, str):
        clusterer_stage = get_clusterer(clusterer, **clusterer_kwargs)
    else:
        clusterer_stage = clusterer
    
    builder.cluster_properties(clusterer_stage)
    
    # 5. Metrics computation stage
    metrics_kwargs = {
        'method': method,
        **(metrics_kwargs or {}),
        **common_config
    }
    
    # Add cache directory for metrics if provided
    if metrics_cache_dir:
        metrics_kwargs['cache_dir'] = metrics_cache_dir
    
    metrics_stage = get_metrics(**metrics_kwargs)
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
                    # avg_quality_score = sum(stat.quality_score for stat in fine_stats) / len(fine_stats)
                    total_size = sum(stat.size for stat in fine_stats)
                    
                    # Log summary statistics for this model as summary metrics
                    wandb.run.summary[f"model_{model_name}_fine_clusters_count"] = len(fine_stats)
                    wandb.run.summary[f"model_{model_name}_avg_score"] = avg_score
                    # wandb.run.summary[f"model_{model_name}_avg_quality_score"] = avg_quality_score
                    wandb.run.summary[f"model_{model_name}_total_size"] = total_size
                    wandb.run.summary[f"model_{model_name}_max_score"] = max(stat.score for stat in fine_stats)
                    wandb.run.summary[f"model_{model_name}_min_score"] = min(stat.score for stat in fine_stats)
                    for key in fine_stats[0].quality_score.keys():
                        # Check if quality_score[key] is a scalar or dict
                        first_quality_value = fine_stats[0].quality_score[key]
                        if isinstance(first_quality_value, (int, float)):
                            # Handle scalar quality scores
                            wandb.run.summary[f"model_{model_name}_max_quality_score_{key}"] = max(stat.quality_score[key] for stat in fine_stats)
                            wandb.run.summary[f"model_{model_name}_min_quality_score_{key}"] = min(stat.quality_score[key] for stat in fine_stats)
                        elif isinstance(first_quality_value, dict):
                            # Handle nested dictionary quality scores
                            for sub_key in first_quality_value.keys():
                                try:
                                    sub_values = [stat.quality_score[key][sub_key] for stat in fine_stats if sub_key in stat.quality_score[key]]
                                    if sub_values and all(isinstance(v, (int, float)) for v in sub_values):
                                        wandb.run.summary[f"model_{model_name}_max_quality_score_{key}_{sub_key}"] = max(sub_values)
                                        wandb.run.summary[f"model_{model_name}_min_quality_score_{key}_{sub_key}"] = min(sub_values)
                                except (KeyError, TypeError):
                                    continue  # Skip if there are issues with this sub_key
                    
                    if "coarse" in stats:
                        coarse_stats = stats["coarse"]
                        # coarse_avg_quality_score = sum(stat.quality_score for stat in coarse_stats) / len(coarse_stats)
                        wandb.run.summary[f"model_{model_name}_coarse_clusters_count"] = len(coarse_stats)
                        wandb.run.summary[f"model_{model_name}_coarse_avg_score"] = sum(stat.score for stat in coarse_stats) / len(coarse_stats)
                        # wandb.run.summary[f"model_{model_name}_coarse_avg_quality_score"] = coarse_avg_quality_score
                        for key in coarse_stats[0].quality_score.keys():
                            # Check if quality_score[key] is a scalar or dict
                            first_quality_value = coarse_stats[0].quality_score[key]
                            if isinstance(first_quality_value, (int, float)):
                                # Handle scalar quality scores
                                wandb.run.summary[f"model_{model_name}_coarse_max_quality_score_{key}"] = max(stat.quality_score[key] for stat in coarse_stats)
                                wandb.run.summary[f"model_{model_name}_coarse_min_quality_score_{key}"] = min(stat.quality_score[key] for stat in coarse_stats)
                            elif isinstance(first_quality_value, dict):
                                # Handle nested dictionary quality scores
                                for sub_key in first_quality_value.keys():
                                    try:
                                        sub_values = [stat.quality_score[key][sub_key] for stat in coarse_stats if sub_key in stat.quality_score[key]]
                                        if sub_values and all(isinstance(v, (int, float)) for v in sub_values):
                                            wandb.run.summary[f"model_{model_name}_coarse_max_quality_score_{key}_{sub_key}"] = max(sub_values)
                                            wandb.run.summary[f"model_{model_name}_coarse_min_quality_score_{key}_{sub_key}"] = min(sub_values)
                                    except (KeyError, TypeError):
                                        continue  # Skip if there are issues with this sub_key
        
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
    verbose: bool = True,
    pipeline: Optional[Pipeline] = None
):
    """Save results to the specified output directory."""
    import pathlib
    import json
    
    # Create output directory if it doesn't exist
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Saving results to: {output_path}")
    
    # 1. Save clustered DataFrame as JSON (primary format - preserves data structures)
    clustered_json_path = output_path / "clustered_results.json"
    
    try:
        # Save as JSON with proper handling of nested structures
        clustered_df.to_json(clustered_json_path, orient='records', lines=True, force_ascii=False)
        if verbose:
            print(f"  ✓ Saved clustered DataFrame (JSON): {clustered_json_path}")
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Failed to save JSON: {e}")
    
    # 2. Save clustered DataFrame as parquet (secondary format - for efficiency)
    clustered_parquet_path = output_path / "clustered_results.parquet"
    
    # For parquet, exclude columns with complex nested data that can't be reliably serialized
    df_for_parquet = clustered_df.copy()
    
    # Identify problematic columns (those with nested dicts/lists)
    problematic_cols = []
    for col in df_for_parquet.columns:
        if df_for_parquet[col].dtype == 'object':
            # Check if column contains nested structures
            sample_non_null = df_for_parquet[col].dropna()
            if len(sample_non_null) > 0:
                sample_val = sample_non_null.iloc[0]
                if isinstance(sample_val, (dict, list)):
                    problematic_cols.append(col)
                    continue
                # Also check if it's a string representation of nested data
                if isinstance(sample_val, str) and (sample_val.startswith('{') or sample_val.startswith('[')):
                    try:
                        import ast
                        ast.literal_eval(sample_val)  # If this works, it's probably nested data as string
                        problematic_cols.append(col)
                        continue
                    except:
                        pass  # Not nested data, keep the column
    
    # Remove problematic columns and save a note about what was excluded
    if problematic_cols:
        df_for_parquet = df_for_parquet.drop(columns=problematic_cols)
        if verbose:
            print(f"  ⚠️  Excluding columns with nested data from parquet: {problematic_cols}")
            print(f"  💡 Use clustered_results.json for complete data with nested structures")
    
    try:
        df_for_parquet.to_parquet(clustered_parquet_path, index=False, engine='pyarrow')
        if verbose:
            print(f"  ✓ Saved clustered DataFrame (parquet): {clustered_parquet_path}")
            if problematic_cols:
                print(f"    Note: {len(problematic_cols)} columns with nested data excluded")
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Failed to save parquet: {e}")
            print(f"  📄 Saving as CSV instead...")
        # Fallback to CSV if parquet still fails
        csv_path = output_path / "clustered_results.csv"
        try:
            df_for_parquet.to_csv(csv_path, index=False)
            if verbose:
                print(f"  ✓ Saved clustered DataFrame (CSV): {csv_path}")
        except Exception as csv_e:
            if verbose:
                print(f"  ❌ Failed to save CSV: {csv_e}")
                print(f"  💡 Data available in JSON format: {clustered_json_path}")
    
    # 3. Save complete PropertyDataset as JSON
    dataset_json_path = output_path / "full_dataset.json"
    result_dataset.save(str(dataset_json_path), format="json")
    if verbose:
        print(f"  ✓ Saved full PropertyDataset (JSON): {dataset_json_path}")
    
    # 5. Save model statistics as JSON
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
    
    # 6. Save parsing failures if pipeline is provided
    if pipeline:
        # Find the LLMJsonParser stage
        from .postprocess import LLMJsonParser
        parser_stage = None
        for stage in pipeline.stages:
            if isinstance(stage, LLMJsonParser):
                parser_stage = stage
                break
        
        if parser_stage and hasattr(parser_stage, 'get_parsing_failures'):
            failures = parser_stage.get_parsing_failures()
            if failures:
                # Convert failures to serializable format
                serializable_failures = []
                for failure in failures:
                    # Convert any non-serializable objects to strings
                    serializable_failure = {
                        'property_id': failure['property_id'],
                        'question_id': failure['question_id'],
                        'model': str(failure['model']),
                        'raw_response': failure['raw_response'],
                        'error_type': failure['error_type'],
                        'error_message': failure['error_message'],
                        'consecutive_errors': failure['consecutive_errors'],
                        'index': failure['index']
                    }
                    
                    # Add optional fields if they exist
                    if 'parsed_json' in failure:
                        serializable_failure['parsed_json'] = failure['parsed_json']
                    if 'property_dict' in failure:
                        serializable_failure['property_dict'] = failure['property_dict']
                    
                    serializable_failures.append(serializable_failure)
                
                # Save parsing failures
                failures_path = output_path / "parsing_failures.json"
                with open(failures_path, 'w') as f:
                    json.dump(serializable_failures, f, indent=2)
                
                if verbose:
                    print(f"  ✓ Saved parsing failures: {failures_path}")
                    print(f"    • Total parsing failures: {len(serializable_failures)}")
                    
                    # Show summary of error types
                    error_types = {}
                    for failure in serializable_failures:
                        error_type = failure['error_type']
                        error_types[error_type] = error_types.get(error_type, 0) + 1
                    
                    if error_types:
                        print(f"    • Error types:")
                        for error_type, count in error_types.items():
                            print(f"      - {error_type}: {count}")
    
    # 7. Save summary statistics
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
        f.write(f"  - clustered_results.json: Complete DataFrame with clusters (JSON)\n")
        f.write(f"  - clustered_results.parquet: Complete DataFrame with clusters (parquet)\n")
        f.write(f"  - full_dataset.json: Complete PropertyDataset object (JSON format)\n")
        f.write(f"  - full_dataset.parquet: Complete PropertyDataset object (parquet format)\n")
        f.write(f"  - model_stats.json: Model statistics and rankings\n")
        f.write(f"  - parsing_failures.json: Detailed parsing failure information\n")
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
        print(f"    • DataFrame: clustered_results.json + clustered_results.parquet")
        print(f"    • PropertyDataset: full_dataset.json + full_dataset.parquet")
        print(f"    • Model stats: model_stats.json")
        print(f"    • Parsing failures: parsing_failures.json")
        print(f"    • Summary: summary.txt")


# Convenience functions for common use cases
def explain_side_by_side(
    df: pd.DataFrame,
    system_prompt: str = None,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function for side-by-side model comparison.
    
    Args:
        df: DataFrame with columns: model_a, model_b, model_a_response, model_b_response, winner
        system_prompt: System prompt for extraction (if None, will be auto-determined)
        **kwargs: Additional arguments passed to explain()
        
    Returns:
        Tuple of (clustered_df, model_stats)
    """
    return explain(df, method="side_by_side", system_prompt=system_prompt, **kwargs)


def explain_single_model(
    df: pd.DataFrame,
    system_prompt: str = None,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function for single model analysis.
    
    Args:
        df: DataFrame with columns: model, model_response, score
        system_prompt: System prompt for extraction (if None, will be auto-determined)
        **kwargs: Additional arguments passed to explain()
        
    Returns:
        Tuple of (clustered_df, model_stats)
    """
    return explain(df, method="single_model", system_prompt=system_prompt, **kwargs)


def explain_with_custom_pipeline(
    df: pd.DataFrame,
    pipeline: Pipeline,
    method: str = "single_model"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Explain model behavior using a custom pipeline.
    
    Args:
        df: DataFrame with conversation data
        pipeline: Custom pipeline to use
        method: "side_by_side" or "single_model"
        
    Returns:
        Tuple of (clustered_df, model_stats)
    """
    dataset = PropertyDataset.from_dataframe(df, method=method)
    result_dataset = pipeline.run(dataset)
    return result_dataset.to_dataframe(), result_dataset.model_stats


def compute_metrics_only(
    input_path: str,
    method: str = "single_model",
    output_dir: Optional[str] = None,
    metrics_kwargs: Optional[Dict[str, Any]] = None,
    use_wandb: bool = False,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run only the metrics computation stage on existing pipeline results.
    
    This function loads existing pipeline results (from extraction and clustering stages)
    and runs only the metrics computation stage. Useful for:
    - Recomputing metrics with different parameters
    - Running metrics on results from previous pipeline runs
    - Debugging metrics computation without re-running the full pipeline
    
    Args:
        input_path: Path to existing pipeline results (file or directory)
        method: "single_model" or "side_by_side"
        output_dir: Directory to save metrics results (optional)
        metrics_kwargs: Additional arguments for metrics computation
        use_wandb: Whether to enable wandb logging
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (clustered_df, model_stats)
        
    Example:
        >>> from lmmvibes import compute_metrics_only
        >>> 
        >>> # Run metrics on existing pipeline results
        >>> clustered_df, model_stats = compute_metrics_only(
        ...     input_path="results/previous_run/full_dataset.json",
        ...     method="single_model",
        ...     output_dir="results/metrics_only"
        ... )
        >>> 
        >>> # Or run on a directory containing pipeline outputs
        >>> clustered_df, model_stats = compute_metrics_only(
        ...     input_path="results/previous_run/",
        ...     method="side_by_side"
        ... )
    """
    from pathlib import Path
    from .metrics import get_metrics
    from .pipeline import Pipeline
    import json
    
    input_path = Path(input_path)
    
    # Load existing dataset
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
                if verbose:
                    print(f"Loading from: {file_path}")
                dataset = PropertyDataset.load(str(file_path))
                break
        else:
            raise FileNotFoundError(f"No recognizable dataset file found in {input_path}")
    
    elif input_path.is_file():
        # Load from a specific file
        if verbose:
            print(f"Loading from: {input_path}")
        dataset = PropertyDataset.load(str(input_path))
    
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    # Verify we have the required data for metrics
    if not dataset.clusters:
        raise ValueError("No clusters found in the dataset. Metrics computation requires clustered data.")
    
    if not dataset.properties:
        raise ValueError("No properties found in the dataset. Metrics computation requires extracted properties.")
    
    if verbose:
        print(f"Loaded dataset with:")
        print(f"  - {len(dataset.conversations)} conversations")
        print(f"  - {len(dataset.properties)} properties")
        print(f"  - {len(dataset.clusters)} clusters")
        print(f"  - Models: {dataset.all_models}")
    
    # Create metrics stage
    metrics_config = {
        'method': method,
        'use_wandb': use_wandb,
        'verbose': verbose,
        **(metrics_kwargs or {})
    }
    
    # Add output directory if provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        metrics_config['output_dir'] = str(output_path)
    
    metrics_stage = get_metrics(**metrics_config)
    
    # Create a minimal pipeline with just the metrics stage
    pipeline = Pipeline("Metrics-Only", [metrics_stage])
    
    # Run metrics computation
    if verbose:
        print("\n" + "="*60)
        print("COMPUTING METRICS")
        print("="*60)
    
    result_dataset = pipeline.run(dataset)
    
    # Convert back to DataFrame format
    clustered_df = result_dataset.to_dataframe()
    model_stats = result_dataset.model_stats
    
    # Save results if output_dir is provided
    if output_dir:
        if verbose:
            print(f"\nSaving results to: {output_dir}")
        
        # 1. Save clustered DataFrame as parquet
        clustered_parquet_path = output_path / "metrics_results.parquet"
        clustered_df.to_parquet(clustered_parquet_path, index=False)
        if verbose:
            print(f"  ✓ Saved metrics DataFrame (parquet): {clustered_parquet_path}")
        
        # 2. Save complete PropertyDataset as JSON
        dataset_json_path = output_path / "metrics_dataset.json"
        result_dataset.save(str(dataset_json_path), format="json")
        if verbose:
            print(f"  ✓ Saved metrics PropertyDataset (JSON): {dataset_json_path}")
        
        # 3. Save model statistics as JSON
        stats_path = output_path / "metrics_stats.json"
        
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
        
        # Print summary
        if verbose:
            print(f"\n📊 Metrics Summary:")
            print(f"  - Models analyzed: {len(model_stats)}")
            for model_name, stats in model_stats.items():
                print(f"  - {model_name}: {len(stats['fine'])} fine clusters")
                if 'coarse' in stats:
                    print(f"    {len(stats['coarse'])} coarse clusters")
    
    return clustered_df, model_stats 
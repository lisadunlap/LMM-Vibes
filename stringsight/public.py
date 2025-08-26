"""
Public API for LMM-Vibes.

This module provides the main explain() function that users will interact with.
"""

from typing import Dict, List, Any, Callable, Optional, Union, Tuple
import pandas as pd
from .core.data_objects import PropertyDataset
from .pipeline import Pipeline, PipelineBuilder
from .prompts import get_default_system_prompt, single_model_system_prompt_custom
import time


def explain(
    df: pd.DataFrame,
    method: str = "single_model",
    system_prompt: str = None,
    prompt_builder: Optional[Callable[[pd.Series], str]] = None,
    task_description: Optional[str] = None,
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
    prettify_labels: bool = False,
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
        task_description: Optional description of the task; when provided with
            method="single_model" and no explicit system_prompt, a task-aware
            system prompt is constructed from single_model_system_prompt_custom.
        
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
        >>> from stringsight import explain
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
        # If single_model and user provided task_description, use the custom template
        if method == "single_model" and task_description:
            system_prompt = single_model_system_prompt_custom.format(task_description=task_description)
            if verbose:
                print(f"Using custom single-model system prompt with task_description (contains_score={contains_score})")
        else:
            system_prompt = get_default_system_prompt(method, contains_score)
            if verbose:
                print(f"Auto-selected system prompt for method '{method}' (contains_score={contains_score})")
    else:
        # If prompt is less than 50 characters, assume it's a prompt name and resolve it
        if len(system_prompt) < 50:
            try:
                from . import prompts
                system_prompt = getattr(prompts, system_prompt)
            except AttributeError:
                # Build a clear list of available prompt names (string-valued attrs containing 'prompt')
                from inspect import getmembers
                available_prompts = sorted([
                    name for name, value in getmembers(prompts)
                    if isinstance(value, str) and 'prompt' in name
                ])
                raise ValueError(f"Unknown prompt name: '{system_prompt}'. Available prompts: {available_prompts}")
        # If the resolved/explicit prompt contains a task_description placeholder, format it when provided
        if "{task_description}" in system_prompt and task_description is not None:
            system_prompt = system_prompt.format(task_description=task_description)
    
    # Print the system prompt for verification
    if verbose:
        print("\n" + "="*80)
        print("SYSTEM PROMPT")
        print("="*80)
        print(system_prompt)
        print("="*80 + "\n")
    if len(system_prompt) < 50:
        raise ValueError("System prompt is too short. Please provide a longer system prompt.")
    
    # Create PropertyDataset from input DataFrame
    dataset = PropertyDataset.from_dataframe(df, method=method)
    
    # Print initial dataset information
    if verbose:
        print(f"\n📋 Initial dataset summary:")
        print(f"   • Conversations: {len(dataset.conversations)}")
        print(f"   • Models: {len(dataset.all_models)}")
        if len(dataset.all_models) <= 20:
            print(f"   • Model names: {', '.join(sorted(dataset.all_models))}")
        print()
    
    # 2️⃣  Initialize wandb if enabled (and explicitly disable via env when off)
    # Ensure environment flag aligns with the provided setting to prevent
    # accidental logging by submodules that import wandb directly.
    import os as _os
    if not use_wandb:
        _os.environ["WANDB_DISABLED"] = "true"
    else:
        _os.environ.pop("WANDB_DISABLED", None)

    # 2️⃣  Initialize wandb if enabled
    # Create run name based on input filename if available
    if use_wandb:
        try:
            import wandb
            import os
            
            # Try to get input filename from the DataFrame or use a default
            input_filename = "unknown_dataset"
            if hasattr(df, 'name') and df.name:
                input_filename = df.name
            elif hasattr(df, '_metadata') and df._metadata and 'filename' in df._metadata:
                input_filename = df._metadata['filename']
            else:
                # Try to infer from the DataFrame source if it has a path attribute
                # This is a fallback for when we can't determine the filename
                input_filename = f"dataset_{len(df)}_rows"
            
            # Clean the filename for wandb (remove extension, replace spaces/special chars)
            if isinstance(input_filename, str):
                # Remove file extension and clean up the name
                input_filename = os.path.splitext(os.path.basename(input_filename))[0]
                # Replace spaces and special characters with underscores
                input_filename = input_filename.replace(' ', '_').replace('-', '_')
                # Remove any remaining special characters
                import re
                input_filename = re.sub(r'[^a-zA-Z0-9_]', '', input_filename)
            
            wandb_run_name = f"{input_filename}_{method}"
            
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
        except ImportError:
            # wandb not installed or not available
            use_wandb = False
    
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
            prettify_labels=prettify_labels,
            metrics_kwargs=metrics_kwargs,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            include_embeddings=include_embeddings,
            verbose=verbose,
            extraction_cache_dir=extraction_cache_dir,
            clustering_cache_dir=clustering_cache_dir,
            metrics_cache_dir=metrics_cache_dir,
            output_dir=output_dir,
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
    clustered_df = result_dataset.to_dataframe(type="all", method=method)
    model_stats = result_dataset.model_stats
    
    # Save final summary if output_dir is provided
    if output_dir is not None:
        _save_final_summary(result_dataset, clustered_df, model_stats, output_dir, verbose)
        
        # Also save the full dataset for backward compatibility with compute_metrics_only and other tools
        import pathlib
        import json
        
        output_path = pathlib.Path(output_dir)
        
        # Save full dataset as JSON
        full_dataset_json_path = output_path / "full_dataset.json"
        result_dataset.save(str(full_dataset_json_path))
        if verbose:
            print(f"  ✓ Saved full dataset: {full_dataset_json_path}")
    
    # Log accumulated summary metrics from pipeline stages
    if use_wandb and hasattr(pipeline, 'log_final_summary'):
        pipeline.log_final_summary()
    
    # Log final results to wandb if enabled
    if use_wandb:
        try:
            import wandb
            _log_final_results_to_wandb(clustered_df, model_stats)
        except ImportError:
            # wandb not installed or not available
            use_wandb = False
    
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
    prettify_labels: bool,
    metrics_kwargs: Optional[Dict[str, Any]],
    use_wandb: bool,
    wandb_project: Optional[str],
    include_embeddings: bool,
    verbose: bool,
    extraction_cache_dir: Optional[str] = None,
    clustering_cache_dir: Optional[str] = None,
    metrics_cache_dir: Optional[str] = None,
    output_dir: Optional[str] = "./results",
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
    
    # Create stage-specific output directories if output_dir is provided
    if output_dir:
        extraction_output = output_dir
        parsing_output = output_dir
        validation_output = output_dir
        clustering_output = output_dir
        metrics_output = output_dir
    else:
        extraction_output = parsing_output = validation_output = clustering_output = metrics_output = None
    
    # 1. Property extraction stage
    extractor_kwargs = {
        'model_name': model_name,
        'system_prompt': system_prompt,
        'prompt_builder': prompt_builder,
        'temperature': temperature,
        'top_p': top_p,
        'max_tokens': max_tokens,
        'max_workers': max_workers,
        'output_dir': extraction_output,
        **common_config
    }
    
    # Add cache directory for extraction if provided
    if extraction_cache_dir:
        extractor_kwargs['cache_dir'] = extraction_cache_dir
    
    extractor = get_extractor(**extractor_kwargs)
    builder.extract_properties(extractor)
    
    # 2. JSON parsing stage
    parser_kwargs = {
        'output_dir': parsing_output,
        **common_config
    }
    parser = LLMJsonParser(**parser_kwargs)
    builder.parse_properties(parser)
    
    # 3. Property validation stage
    validator_kwargs = {
        'output_dir': validation_output,
        **common_config
    }
    validator = PropertyValidator(**validator_kwargs)
    builder.add_stage(validator)
    
    # 4. Clustering stage
    clusterer_kwargs = {
        'min_cluster_size': min_cluster_size,
        'embedding_model': embedding_model,
        'hierarchical': hierarchical,
        'assign_outliers': assign_outliers,
        'include_embeddings': include_embeddings,
        'prettify_labels': prettify_labels,
        'output_dir': clustering_output,
        **common_config
    }
    # Forward any additional clusterer-specific kwargs (e.g., groupby_column)
    if kwargs:
        clusterer_kwargs.update(kwargs)
    
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
        'output_dir': metrics_output,
        'compute_bootstrap': metrics_kwargs.get('compute_confidence_intervals', True) if metrics_kwargs else True,
        'bootstrap_samples': metrics_kwargs.get('bootstrap_samples', 100) if metrics_kwargs else 100,
        'log_to_wandb': use_wandb,
        'generate_plots': True,
        **(metrics_kwargs or {}),
        **common_config
    }
    
    # Add cache directory for metrics if provided
    if metrics_cache_dir:
        metrics_kwargs['cache_dir'] = metrics_cache_dir
    
    metrics_stage = get_metrics(**metrics_kwargs)
    builder.compute_metrics(metrics_stage)
    
    # Build and return the pipeline
    pipeline = builder.configure(output_dir=output_dir, **common_config).build()
    
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
                    n_clusters = len([c for c in cluster_ids if c is not None and c >= 0])
                    n_outliers = sum(1 for c in cluster_ids if c is not None and c < 0)
                    
                    level = "fine" if "fine" in col else "coarse" if "coarse" in col else "main"
                    # Log these as summary metrics
                    if wandb.run is not None:
                        wandb.run.summary[f"final_{level}_clusters"] = n_clusters
                        wandb.run.summary[f"final_{level}_outliers"] = n_outliers
                        wandb.run.summary[f"final_{level}_outlier_rate"] = n_outliers / len(df) if len(df) > 0 else 0

        # Handle new functional metrics format
        if model_stats and "functional_metrics" in model_stats:
            functional_metrics = model_stats["functional_metrics"]
            
            # Log summary statistics for functional metrics
            if wandb.run is not None:
                model_cluster_scores = functional_metrics.get("model_cluster_scores", {})
                cluster_scores = functional_metrics.get("cluster_scores", {})
                model_scores = functional_metrics.get("model_scores", {})
                
                wandb.run.summary["final_models_analyzed"] = len(model_scores)
                wandb.run.summary["final_clusters_analyzed"] = len(cluster_scores)
                
                # Log model-level summary statistics
                for model_name, model_data in model_scores.items():
                    if isinstance(model_data, dict):
                        size = model_data.get("size", 0)
                        quality = model_data.get("quality", {})
                        
                        wandb.run.summary[f"model_{model_name}_total_size"] = size
                        
                        # Log quality metrics
                        for metric_name, metric_value in quality.items():
                            if isinstance(metric_value, (int, float)):
                                wandb.run.summary[f"model_{model_name}_avg_{metric_name}"] = metric_value
                
                # Log cluster-level summary statistics
                for cluster_name, cluster_data in cluster_scores.items():
                    if isinstance(cluster_data, dict):
                        size = cluster_data.get("size", 0)
                        quality = cluster_data.get("quality", {})
                        
                        wandb.run.summary[f"cluster_{cluster_name}_total_size"] = size
                        
                        # Log quality metrics
                        for metric_name, metric_value in quality.items():
                            if isinstance(metric_value, (int, float)):
                                wandb.run.summary[f"cluster_{cluster_name}_avg_{metric_name}"] = metric_value
            
            print("✅ Successfully logged functional metrics to wandb")
            print(f"   • Dataset summary metrics")
            print(f"   • Clustering results")
            print(f"   • Functional metrics: {len(model_scores) if 'model_scores' in functional_metrics else 0} models, {len(cluster_scores) if 'cluster_scores' in functional_metrics else 0} clusters")
            print(f"   • Summary metrics logged to run summary")
    except ImportError:
        # wandb not installed or not available
        return
    
    # Handle legacy format for backward compatibility (commented out but kept for reference)
    # if model_stats:
    #     if wandb.run is not None:
    #         wandb.run.summary["final_models_analyzed"] = len(model_stats)
    #     # ... rest of legacy logging code ...


def _save_final_summary(
    result_dataset: PropertyDataset,
    clustered_df: pd.DataFrame,
    model_stats: Dict[str, Any],
    output_dir: str,
    verbose: bool = True
):
    """Save a final summary of the explain run to a text file."""
    import pathlib
    import json
    
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\nSaving final summary to: {output_path / 'summary.txt'}")
    
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
        
        f.write(f"\nOutput files:\n")
        f.write(f"  - raw_properties.jsonl: Raw LLM responses\n")
        f.write(f"  - extraction_stats.json: Extraction statistics\n")
        f.write(f"  - extraction_samples.jsonl: Sample inputs/outputs\n")
        f.write(f"  - parsed_properties.jsonl: Parsed property objects\n")
        f.write(f"  - parsing_stats.json: Parsing statistics\n")
        f.write(f"  - parsing_failures.jsonl: Failed parsing attempts\n")
        f.write(f"  - validated_properties.jsonl: Validated properties\n")
        f.write(f"  - validation_stats.json: Validation statistics\n")
        f.write(f"  - clustered_results.jsonl: Complete clustered data\n")
        f.write(f"  - embeddings.parquet: Embeddings data\n")
        f.write(f"  - clustered_results_lightweight.jsonl: Data without embeddings\n")
        f.write(f"  - summary_table.jsonl: Clustering summary\n")
        f.write(f"  - model_cluster_scores.json: Per model-cluster combination metrics\n")
        f.write(f"  - cluster_scores.json: Per cluster metrics (aggregated across models)\n")
        f.write(f"  - model_scores.json: Per model metrics (aggregated across clusters)\n")
        f.write(f"  - full_dataset.json: Complete PropertyDataset (JSON format)\n")
        f.write(f"  - full_dataset.parquet: Complete PropertyDataset (parquet format, or .jsonl if mixed data types)\n")
        
        # Model rankings - handle both new functional metrics and legacy format
        f.write(f"\nModel Rankings (by average quality score):\n")
        model_avg_scores = {}
        
        # Check for new functional metrics format
        if model_stats and "functional_metrics" in model_stats:
            functional_metrics = model_stats["functional_metrics"]
            model_scores = functional_metrics.get("model_scores", {})
            
            for model_name, model_data in model_scores.items():
                if isinstance(model_data, dict):
                    quality = model_data.get("quality", {})
                    if quality:
                        # Use the first quality metric as the ranking score
                        first_metric_value = next(iter(quality.values()))
                        if isinstance(first_metric_value, (int, float)):
                            model_avg_scores[model_name] = first_metric_value
        
        # Fallback to legacy format if functional metrics not available
        else:
            for model_name, stats in model_stats.items():
                if "fine" in stats and len(stats["fine"]) > 0:
                    avg_score = sum(stat.score for stat in stats["fine"]) / len(stats["fine"])
                    model_avg_scores[model_name] = avg_score
        
        for i, (model_name, avg_score) in enumerate(sorted(model_avg_scores.items(), key=lambda x: x[1], reverse=True)):
            f.write(f"  {i+1}. {model_name}: {avg_score:.3f}\n")
    
    if verbose:
        print(f"  ✓ Saved final summary: {summary_path}")


# ------------------------------------------------------------------
# 🆕  Fixed-taxonomy "label" entry point
# ------------------------------------------------------------------

def _build_fixed_axes_pipeline(
    *,
    extractor: "FixedAxesLabeler",
    taxonomy: Dict[str, str],
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_workers: int,
    metrics_kwargs: Optional[Dict[str, Any]],
    use_wandb: bool,
    wandb_project: Optional[str],
    include_embeddings: bool,
    verbose: bool,
    output_dir: Optional[str],
    extraction_cache_dir: Optional[str] = None,
    metrics_cache_dir: Optional[str] = None,
    **kwargs,
):
    """Internal helper that constructs a pipeline for *label()* calls."""

    from .postprocess import LLMJsonParser, PropertyValidator
    from .clusterers.dummy_clusterer import DummyClusterer
    from .metrics import get_metrics

    builder = PipelineBuilder(name="LMM-Vibes-fixed-axes")

    common_cfg = {"verbose": verbose, "use_wandb": use_wandb, "wandb_project": wandb_project or "lmm-vibes"}

    # 1️⃣  Extraction / labeling (use pre-created extractor)
    builder.extract_properties(extractor)

    # 2️⃣  JSON parsing
    parser = LLMJsonParser(output_dir=output_dir, fail_fast=True, **common_cfg)
    builder.parse_properties(parser)

    # 3️⃣  Validation
    validator = PropertyValidator(output_dir=output_dir, **common_cfg)
    builder.add_stage(validator)

    # 4️⃣  Dummy clustering
    dummy_clusterer = DummyClusterer(allowed_labels=list(taxonomy.keys()), output_dir=output_dir, **common_cfg)
    builder.cluster_properties(dummy_clusterer)

    # 5️⃣  Metrics (single-model only)
    metrics_stage = get_metrics(method="single_model", output_dir=output_dir, **(metrics_kwargs or {}), **({"cache_dir": metrics_cache_dir} if metrics_cache_dir else {}), **common_cfg)
    builder.compute_metrics(metrics_stage)

    return builder.configure(output_dir=output_dir, **common_cfg).build()


def label(
    df: pd.DataFrame,
    *,
    taxonomy: Dict[str, str],
    model_name: str = "gpt-4.1",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 2048,
    max_workers: int = 8,
    metrics_kwargs: Optional[Dict[str, Any]] = None,
    use_wandb: bool = True,
    wandb_project: Optional[str] = None,
    include_embeddings: bool = True,
    verbose: bool = True,
    output_dir: Optional[str] = None,
    extraction_cache_dir: Optional[str] = None,
    metrics_cache_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the *fixed-taxonomy* analysis pipeline. This is just you're run of the mill LLM-judge with a given rubric. 

    The user provides a dataframe with a model and its responses alone with a taxonomy.

    Unlike :pyfunc:`explain`, this entry point does **not** perform clustering;
    each taxonomy label simply becomes its own cluster.  The input `df` **must**
    be in *single-model* format (columns `question_id`, `prompt`, `model`, `model_response`, …).
    """

    method = "single_model"  # hard-coded, we only support single-model here

    # Align environment with wandb toggle early to avoid accidental logging on import
    import os as _os
    if not use_wandb:
        _os.environ["WANDB_DISABLED"] = "true"
    else:
        _os.environ.pop("WANDB_DISABLED", None)
    if "model_b" in df.columns:
        raise ValueError("label() currently supports only single-model data.  Use explain() for side-by-side analyses.")

    # ------------------------------------------------------------------
    # Create extractor first to get the system prompt
    # ------------------------------------------------------------------
    from .extractors.fixed_axes_labeler import FixedAxesLabeler
    
    # Create the extractor to generate the system prompt from taxonomy
    extractor = FixedAxesLabeler(
        taxonomy=taxonomy,
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_workers=max_workers,
        cache_dir=extraction_cache_dir or ".cache/stringsight",
        output_dir=output_dir,
        verbose=verbose,
        use_wandb=use_wandb,
        wandb_project=wandb_project or "lmm-vibes"
    )
    
    # Print the system prompt for verification
    if verbose:
        print("\n" + "="*80)
        print("SYSTEM PROMPT")
        print("="*80)
        print(extractor.system_prompt)
        print("="*80 + "\n")
    
    # ------------------------------------------------------------------
    # Build dataset & pipeline
    # ------------------------------------------------------------------
    dataset = PropertyDataset.from_dataframe(df, method=method)

    # Initialize wandb if enabled
    if use_wandb:
        try:
            import wandb
            import os
            
            # Try to get input filename from the DataFrame or use a default
            input_filename = "unknown_dataset"
            if hasattr(df, 'name') and df.name:
                input_filename = df.name
            elif hasattr(df, '_metadata') and df._metadata and 'filename' in df._metadata:
                input_filename = df._metadata['filename']
            else:
                # Try to infer from the DataFrame source if it has a path attribute
                # This is a fallback for when we can't determine the filename
                input_filename = f"dataset_{len(df)}_rows"
            
            # Clean the filename for wandb (remove extension, replace spaces/special chars)
            if isinstance(input_filename, str):
                # Remove file extension and clean up the name
                input_filename = os.path.splitext(os.path.basename(input_filename))[0]
                # Replace spaces and special characters with underscores
                input_filename = input_filename.replace(' ', '_').replace('-', '_')
                # Remove any remaining special characters
                import re
                input_filename = re.sub(r'[^a-zA-Z0-9_]', '', input_filename)
            
            wandb_run_name = f"{input_filename}_label"
            
            wandb.init(
                project=wandb_project or "lmm-vibes",
                name=wandb_run_name,
                config={
                    "method": method,
                    "model_name": model_name,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "max_workers": max_workers,
                    "taxonomy_size": len(taxonomy),
                    "include_embeddings": include_embeddings,
                    "output_dir": output_dir,
                },
                reinit=False  # Don't reinitialize if already exists
            )
        except ImportError:
            # wandb not installed or not available
            use_wandb = False

    pipeline = _build_fixed_axes_pipeline(
        extractor=extractor,
        taxonomy=taxonomy,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_workers=max_workers,
        metrics_kwargs=metrics_kwargs,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        include_embeddings=include_embeddings,
        verbose=verbose,
        output_dir=output_dir,
        extraction_cache_dir=extraction_cache_dir,
        metrics_cache_dir=metrics_cache_dir,
        **kwargs,
    )

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    result_dataset = pipeline.run(dataset)
    clustered_df = result_dataset.to_dataframe(type="clusters", method=method)

    # Save final summary and full dataset if output_dir is provided (same as explain() function)
    if output_dir is not None:
        _save_final_summary(result_dataset, clustered_df, result_dataset.model_stats, output_dir, verbose)
        
        # Also save the full dataset for backward compatibility with compute_metrics_only and other tools
        import pathlib
        import json
        
        output_path = pathlib.Path(output_dir)
        
        # Save full dataset as JSON
        full_dataset_json_path = output_path / "full_dataset.json"
        result_dataset.save(str(full_dataset_json_path))
        if verbose:
            print(f"  ✓ Saved full dataset: {full_dataset_json_path}")

    return clustered_df, result_dataset.model_stats


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
    dataset = PropertyDataset.from_dataframe(df)
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
        >>> from stringsight import compute_metrics_only
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

    # Align environment with wandb toggle early to avoid accidental logging on import
    import os as _os
    if not use_wandb:
        _os.environ["WANDB_DISABLED"] = "true"
    else:
        _os.environ.pop("WANDB_DISABLED", None)
    
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
        
        # Count unique models from conversations for verification
        unique_models = set()
        for conv in dataset.conversations:
            if isinstance(conv.model, list):
                unique_models.update(conv.model)
            else:
                unique_models.add(conv.model)
        
        print(f"  - Total unique models: {len(unique_models)}")
        if len(unique_models) <= 20:
            model_list = sorted(list(unique_models))
            print(f"  - Model names: {', '.join(model_list)}")
        print()
    
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
    
    # Initialize wandb if enabled
    if use_wandb:
        try:
            import wandb
            import os
            
            # Try to get input filename from the input path
            input_filename = "unknown_dataset"
            if input_path.is_file():
                input_filename = input_path.name
            elif input_path.is_dir():
                # Try to find a recognizable dataset file in the directory
                possible_files = [
                    input_path / "full_dataset.json",
                    input_path / "full_dataset.parquet", 
                    input_path / "clustered_results.parquet",
                    input_path / "dataset.json",
                    input_path / "dataset.parquet"
                ]
                
                for file_path in possible_files:
                    if file_path.exists():
                        input_filename = file_path.name
                        break
                else:
                    # If no recognizable file found, use the directory name
                    input_filename = input_path.name
            
            # Clean the filename for wandb (remove extension, replace spaces/special chars)
            if isinstance(input_filename, str):
                # Remove file extension and clean up the name
                input_filename = os.path.splitext(os.path.basename(input_filename))[0]
                # Replace spaces and special characters with underscores
                input_filename = input_filename.replace(' ', '_').replace('-', '_')
                # Remove any remaining special characters
                import re
                input_filename = re.sub(r'[^a-zA-Z0-9_]', '', input_filename)
            
            wandb_run_name = f"{input_filename}_metrics_only"
            
            wandb.init(
                project="lmm-vibes",
                name=wandb_run_name,
                config={
                    "method": method,
                    "input_path": str(input_path),
                    "output_dir": output_dir,
                    "metrics_kwargs": metrics_kwargs,
                },
                reinit=False  # Don't reinitialize if already exists
            )
        except ImportError:
            # wandb not installed or not available
            use_wandb = False
    
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
        
        # Use the same saving mechanism as the full pipeline
        _save_final_summary(
            result_dataset=result_dataset,
            clustered_df=clustered_df,
            model_stats=model_stats,
            output_dir=output_dir,
            verbose=verbose
        )
        
        # Print summary
        if verbose:
            print(f"\n📊 Metrics Summary:")
            print(f"  - Models analyzed: {len(model_stats)}")
            
            # Handle new functional metrics format
            if model_stats and "functional_metrics" in model_stats:
                functional_metrics = model_stats["functional_metrics"]
                model_scores = functional_metrics.get("model_scores", {})
                cluster_scores = functional_metrics.get("cluster_scores", {})
                
                print(f"  - Functional metrics computed:")
                print(f"    • Model scores: {len(model_scores)} models")
                print(f"    • Cluster scores: {len(cluster_scores)} clusters")
                
                # Print model-level summary
                for model_name, model_data in model_scores.items():
                    if isinstance(model_data, dict):
                        size = model_data.get("size", 0)
                        quality = model_data.get("quality", {})
                        print(f"    • {model_name}: {size} conversations")
                        if quality:
                            for metric_name, metric_value in quality.items():
                                if isinstance(metric_value, (int, float)):
                                    print(f"      - {metric_name}: {metric_value:.3f}")
            
            # Handle legacy format for backward compatibility
            else:
                for model_name, stats in model_stats.items():
                    if "fine" in stats:
                        print(f"  - {model_name}: {len(stats['fine'])} fine clusters")
                    if "coarse" in stats:
                        print(f"    {len(stats['coarse'])} coarse clusters")
    
    return clustered_df, model_stats 
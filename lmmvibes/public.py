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
    # Metrics parameters
    metrics_kwargs: Optional[Dict[str, Any]] = None,
    # Caching & logging
    use_wandb: bool = True,
    wandb_project: Optional[str] = None,
    include_embeddings: bool = True,
    verbose: bool = True,
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
        >>> # Explain model behavior
        >>> clustered_df, model_stats = explain(
        ...     df,
        ...     method="side_by_side",
        ...     min_cluster_size=20,
        ...     hierarchical=True
        ... )
        >>> 
        >>> # Explore the results
        >>> print(clustered_df.columns)
        >>> print(model_stats.keys())
    """
    
    # Create PropertyDataset from input DataFrame
    dataset = PropertyDataset.from_dataframe(df, method=method)
    
    # 2️⃣  Initialize wandb if enabled
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project or "lmm-vibes",
            name=f"explain_{method}_{int(time.time())}",
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
                "include_embeddings": include_embeddings,
            }
        )
    
    # Use custom pipeline if provided, otherwise build default pipeline
    if custom_pipeline is not None:
        pipeline = custom_pipeline
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
    
    return pipeline


def _log_final_results_to_wandb(df: pd.DataFrame, model_stats: Dict[str, Any]):
    """Log final results to wandb."""
    try:
        import wandb
        
        # Log dataset summary
        wandb.log({
            "final_dataset_shape": str(df.shape),
            "final_total_conversations": len(df['question_id'].unique()) if 'question_id' in df.columns else len(df),
            "final_total_properties": len(df),
            "final_unique_models": len(df['model'].unique()) if 'model' in df.columns else 0,
        })
        
        # Log clustering results if present
        cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
        if cluster_cols:
            for col in cluster_cols:
                if col.endswith('_id'):
                    cluster_ids = df[col].unique()
                    n_clusters = len([c for c in cluster_ids if c != -1])
                    n_outliers = sum(1 for c in cluster_ids if c == -1)
                    
                    level = "fine" if "fine" in col else "coarse" if "coarse" in col else "main"
                    wandb.log({
                        f"final_{level}_clusters": n_clusters,
                        f"final_{level}_outliers": n_outliers,
                        f"final_{level}_outlier_rate": n_outliers / len(df) if len(df) > 0 else 0,
                    })
        
        # Log model statistics
        if model_stats:
            wandb.log({
                "final_models_analyzed": len(model_stats),
                "final_model_stats": model_stats,
            })
        
        # Log a sample of the final results
        if len(df) > 0:
            sample_size = min(20, len(df))
            sample_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
            
            # Select key columns for the sample
            key_cols = ['question_id', 'model', 'property_description', 'category', 'impact', 'type']
            if 'property_description_fine_cluster_label' in df.columns:
                key_cols.append('property_description_fine_cluster_label')
            if 'property_description_coarse_cluster_label' in df.columns:
                key_cols.append('property_description_coarse_cluster_label')
            
            available_cols = [col for col in key_cols if col in sample_df.columns]
            sample_for_table = sample_df[available_cols].astype(str)
            
            wandb.log({
                "final_results_sample": wandb.Table(dataframe=sample_for_table),
            })
        
    except Exception as e:
        print(f"Failed to log final results to wandb: {e}")


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
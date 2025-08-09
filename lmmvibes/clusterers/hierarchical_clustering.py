#!/usr/bin/env python3
"""
Hierarchical Text Clustering Module

Provides scalable hierarchical clustering for text data using semantic embeddings.
Supports multiple clustering algorithms including HDBSCAN, and traditional methods.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import time
from collections import defaultdict
import os
import pickle
import argparse
import concurrent.futures
from dataclasses import dataclass
from typing import Optional, Dict, Union, List, Any

# Core ML libraries
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer

# Try relative import first, fall back to absolute import
try:
    from .clustering_utils import _get_llm_cluster_summary, llm_coarse_cluster_with_centers, _get_embeddings, _setup_embeddings, save_clustered_results, initialize_wandb, load_precomputed_embeddings
except ImportError:
    from clustering_utils import _get_llm_cluster_summary, llm_coarse_cluster_with_centers, _get_embeddings, _setup_embeddings, save_clustered_results, initialize_wandb, load_precomputed_embeddings

# Import the new modular functions
try:
    from .clustering_utils import generate_coarse_labels, assign_fine_to_coarse
except ImportError:
    from clustering_utils import generate_coarse_labels, assign_fine_to_coarse

# Prompts for LLM clustering
try:
    from .clustering_prompts import coarse_clustering_systems_prompt, deduplication_clustering_systems_prompt, outlier_clustering_systems_prompt
except ImportError:
    from clustering_prompts import coarse_clustering_systems_prompt, deduplication_clustering_systems_prompt, outlier_clustering_systems_prompt

# Optional imports (will be checked when needed)
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
import litellm
import wandb
from bertopic import BERTopic
from bertopic.backend import OpenAIBackend

# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class ClusterConfig:
    """Configuration for clustering operations."""
    min_cluster_size: int = 30
    verbose: bool = True
    include_embeddings: bool = True
    context: Optional[str] = None
    precomputed_embeddings: Optional[Union[np.ndarray, Dict, str]] = None
    disable_dim_reduction: bool = False
    assign_outliers: bool = False
    hierarchical: bool = False
    min_grandparent_size: int = 3
    max_coarse_clusters: int = 15
    input_model_name: Optional[str] = None
    min_samples: Optional[int] = None
    cluster_selection_epsilon: float = 0.0
    cache_embeddings: bool = True
    # model settings
    embedding_model: str = "openai"
    summary_model: str = "gpt-4.1"
    cluster_assignment_model: str = "gpt-4.1-mini"
    # Dimension reduction settings
    dim_reduction_method: str = "adaptive"  # "adaptive", "umap", "pca", "none"
    umap_n_components: int = 100  # More conservative default
    umap_n_neighbors: int = 30    # Higher for better global structure
    umap_min_dist: float = 0.1    # Non-zero to preserve structure
    umap_metric: str = "cosine"   # Better for semantic similarity
    # wandb configuration
    use_wandb: bool = True
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    def __post_init__(self):
        """Set derived parameters after initialization."""
        if self.min_samples is None:
            self.min_samples = min(self.min_cluster_size, max(5, self.min_cluster_size // 2))
    
    @classmethod
    def from_args(cls, args):
        """Create config from argparse args."""
        # Handle wandb flag logic: default True, but --no-wandb overrides it
        use_wandb = not args.no_wandb if hasattr(args, 'no_wandb') else True
        
        return cls(
            min_cluster_size=args.min_cluster_size,
            embedding_model=args.embedding_model,
            verbose=not hasattr(args, 'quiet') or not args.quiet,
            include_embeddings=not args.no_embeddings,
            context=args.context,
            precomputed_embeddings=args.precomputed_embeddings,
            disable_dim_reduction=args.disable_dim_reduction,
            assign_outliers=args.assign_outliers,
            hierarchical=args.hierarchical,
            min_grandparent_size=args.min_grandparent_size,
            max_coarse_clusters=args.max_coarse_clusters,
            input_model_name=args.input_model_name,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            # Dimension reduction settings
            dim_reduction_method=getattr(args, 'dim_reduction_method', 'adaptive'),
            umap_n_components=getattr(args, 'umap_n_components', 100),
            umap_n_neighbors=getattr(args, 'umap_n_neighbors', 30),
            umap_min_dist=getattr(args, 'umap_min_dist', 0.1),
            umap_metric=getattr(args, 'umap_metric', 'cosine'),
            use_wandb=use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prepare_embeddings(unique_values: List[Any], config: ClusterConfig) -> np.ndarray:
    """
    Prepare embeddings for clustering with caching and optional dimensionality reduction.
    
    Args:
        unique_values: List of unique values to embed
        config: ClusterConfig containing embedding parameters
        
    Returns:
        np.ndarray: Processed embeddings ready for clustering
    """
    unique_strings = [str(value) for value in unique_values]
    
    if config.verbose:
        print(f"Preparing embeddings for {len(unique_values)} unique values...")
    
    # Get embeddings (either precomputed or compute fresh)
    if config.precomputed_embeddings is not None:
        if config.verbose:
            print("Using precomputed embeddings...")
        embeddings = config.precomputed_embeddings
        if isinstance(embeddings, dict):
            if config.verbose:
                print(f"Mapping {len(unique_values)} values to embeddings from dict with {len(embeddings)} entries...")
            try:
                embeddings = np.array([embeddings[str(val)] for val in unique_values])
                if config.verbose:
                    print(f"✅ Successfully mapped to {len(embeddings)} embeddings")
            except KeyError as e:
                print(f"❌ Error: Some values not found in precomputed embeddings: {e}")
                print(f"Available keys (first 5): {list(embeddings.keys())[:5]}")
                print(f"Missing values (first 5): {[str(val) for val in unique_values if str(val) not in embeddings][:5]}")
                raise
        else:
            if config.verbose:
                print(f"Using precomputed embeddings array with {len(embeddings)} entries...")
            embeddings = np.array(embeddings)
        
        if config.verbose:
            print(f"Embeddings shape: {embeddings.shape}")
    else:
        # Use caching if enabled
        embeddings, _ = _setup_embeddings(unique_strings, config.embedding_model, config.verbose)
        embeddings = np.array(embeddings)
    
    # Normalize embeddings
    if config.verbose:
        print("Normalizing embeddings...")
    if len(embeddings) > 1:
        embeddings = (embeddings - embeddings.mean(axis=0)) / (embeddings.std(axis=0) + 1e-8)
    
    # Keep original embeddings for output (before any dimensionality reduction)
    original_embeddings = embeddings.copy()
    
    # Improved dimension reduction that preserves semantic coherence
    if not config.disable_dim_reduction:
        n_points, n_dims = embeddings.shape
        
        # Determine method (improved adaptive logic)
        if config.dim_reduction_method == "adaptive":
            # More conservative adaptive logic that considers dataset size
            if n_points < 100:
                # For very small datasets, skip dimension reduction entirely
                method = "none"
            elif n_points > 5000 or n_dims > 200:
                # For large datasets, use UMAP
                method = "umap"
            else:
                # For medium datasets, skip dimension reduction
                method = "none"
        else:
            method = config.dim_reduction_method
        
        if method == "umap":
            if config.verbose:
                print(f"Applying UMAP dimensionality reduction...")
            
            # Adaptive parameters with better safeguards
            n_components = min(config.umap_n_components, n_dims - 1)
            # Ensure n_neighbors is much smaller than n_points to avoid spectral embedding issues
            n_neighbors = min(config.umap_n_neighbors, max(5, n_points // 3))
            
            # Additional safety check
            if n_neighbors >= n_points:
                if config.verbose:
                    print(f"Dataset too small for UMAP (n_points={n_points}, n_neighbors={n_neighbors}), skipping dimension reduction")
                method = "none"
            else:
                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=config.umap_min_dist,
                    metric=config.umap_metric,
                    random_state=42,
                    verbose=config.verbose
                )
                embeddings = reducer.fit_transform(embeddings)
                
                if config.verbose:
                    print(f"Reduced to shape: {embeddings.shape}")
                
        elif method == "pca":
            if config.verbose:
                print(f"Applying PCA dimensionality reduction...")
            
            from sklearn.decomposition import PCA
            n_components = min(100, n_dims - 1, n_points - 1)
            reducer = PCA(n_components=n_components, random_state=42)
            embeddings = reducer.fit_transform(embeddings)
            
            if config.verbose:
                print(f"Reduced to shape: {embeddings.shape}")
                
        elif method == "none" and config.verbose:
            print("Skipping dimension reduction")
    
    return embeddings, original_embeddings


def generate_cluster_summaries(cluster_values: Dict[int, List], config: ClusterConfig, 
                             column_name: str, cluster_type: str = "cluster") -> Dict[int, str]:
    """
    Generate cluster summaries using LLM or generic labels.
    
    Args:
        cluster_values: Dict mapping cluster IDs to lists of values
        config: ClusterConfig containing summary parameters
        column_name: Name of the column being clustered
        
    Returns:
        Dict mapping cluster IDs to summary labels
    """
    if config.verbose:
        print(f"Generating LLM-based cluster summaries for {cluster_type} clusters...")
    
    cluster_label_map = {}
    for cluster_id, values in cluster_values.items():
        if cluster_id == -1:
            cluster_label_map[cluster_id] = "Outliers"
            continue
            
        summary = _get_llm_cluster_summary(values, config.summary_model, column_name, cluster_type, 50)
        cluster_label_map[cluster_id] = summary
        if config.verbose:
            print(f"    Cluster {cluster_id}: {summary} ({len(values)} items)")
    
    return cluster_label_map


def format_clustering_results(df: pd.DataFrame, column_name: str, 
                            unique_values: List, original_embeddings: np.ndarray,
                            cluster_labels: np.ndarray, cluster_label_map: Dict[int, str],
                            config: ClusterConfig, 
                            coarse_cluster_data: Optional[tuple] = None) -> pd.DataFrame:
    """
    Format clustering results into output DataFrame.
    
    Args:
        df: Original DataFrame
        column_name: Name of the column that was clustered
        unique_values: List of unique values that were clustered
        original_embeddings: Original embeddings before dimensionality reduction
        cluster_labels: Cluster assignment for each unique value
        cluster_label_map: Mapping from cluster ID to label
        config: ClusterConfig containing formatting parameters
        coarse_cluster_data: Optional tuple of (coarse_labels, coarse_label_map) for hierarchical
        
    Returns:
        pd.DataFrame: Formatted results with cluster assignments
    """
    df_copy = df.copy()
    
    # Create basic mappings
    value_to_cluster = dict(zip(unique_values, cluster_labels))
    value_to_label = {v: cluster_label_map[c] for v, c in value_to_cluster.items()}
    
    # Add fine cluster columns
    df_copy[f'{column_name}_fine_cluster_label'] = df_copy[column_name].map(value_to_label)
    df_copy[f'{column_name}_fine_cluster_id'] = df_copy[column_name].map(value_to_cluster)
    
    # Add coarse cluster columns if hierarchical
    if coarse_cluster_data is not None:
        coarse_labels, coarse_label_map = coarse_cluster_data
        value_to_coarse = dict(zip(unique_values, coarse_labels))
        value_to_coarse_label = {v: coarse_label_map[c] for v, c in value_to_coarse.items()}
        
        df_copy[f'{column_name}_coarse_cluster_label'] = df_copy[column_name].map(value_to_coarse_label)
        df_copy[f'{column_name}_coarse_cluster_id'] = df_copy[column_name].map(value_to_coarse)
    
    # Add embeddings if requested
    if config.include_embeddings:
        value_to_embedding = dict(zip(unique_values, original_embeddings.tolist()))
        df_copy[f'{column_name}_embedding'] = df_copy[column_name].map(value_to_embedding)
        
        # Get embeddings for cluster names
        unique_cluster_names = list(set(value_to_label.values()))
        cluster_name_embeddings = _get_embeddings(unique_cluster_names, config.embedding_model, config.verbose)
        cluster_name_to_embedding = dict(zip(unique_cluster_names, cluster_name_embeddings))
        df_copy[f'{column_name}_fine_cluster_label_embedding'] = df_copy[f'{column_name}_fine_cluster_label'].map(cluster_name_to_embedding)
    
    return df_copy


# =============================================================================
# MAIN CLUSTERING FUNCTIONS
# =============================================================================

def hdbscan_cluster_categories(df, column_name, config=None, **kwargs):
    """
    Fast HDBSCAN clustering for medium to large datasets.
    Now supports BERTopic-based outlier reduction if config.assign_outliers is True.
    """
    # Handle backward compatibility by creating config from kwargs
    if config is None:
        config_kwargs = {}
        for k, v in kwargs.items():
            if hasattr(ClusterConfig, k):
                config_kwargs[k] = v
        config = ClusterConfig(**config_kwargs)

    start_time = time.time()
    unique_values = df[column_name].unique()
    unique_strings = [str(value) for value in unique_values]

    if config.verbose:
        print(f"HDBSCAN clustering for {len(unique_values)} unique values...")
        if config.hierarchical:
            print("  Hierarchical mode enabled - will create grandparent clusters")

    # Step 1: Prepare embeddings
    embeddings, original_embeddings = prepare_embeddings(unique_values, config)

    # Step 2: Run HDBSCAN clustering
    if config.verbose:
        print("Starting HDBSCAN clustering...")
        print(f"Parameters: min_cluster_size={config.min_cluster_size}, data_shape={embeddings.shape}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
        algorithm='best',
        core_dist_n_jobs=-1,
        cluster_selection_epsilon=config.cluster_selection_epsilon
    )
    cluster_labels = clusterer.fit_predict(embeddings)

    if config.verbose:
        n_initial_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        print(f"HDBSCAN clustering completed! Found {n_initial_clusters} clusters and {n_noise} outliers")

    # Step 3: Handle outlier assignment if requested, using BERTopic
    if config.assign_outliers and -1 in cluster_labels:
        if config.verbose:
            print("Assigning outliers using BERTopic.reduce_outliers...")
        # Use OpenAIBackend if openai model, else None
        if config.embedding_model == "openai":
            from bertopic.backend import OpenAIBackend
            import openai
            client = openai.OpenAI()
            bertopic_embedding_model = OpenAIBackend(client, "text-embedding-3-large")
        else:
            bertopic_embedding_model = None  # Let BERTopic use default
        # Minimal BERTopic setup for outlier reduction
        topic_model = BERTopic(
            embedding_model=bertopic_embedding_model,
            hdbscan_model=clusterer,
            calculate_probabilities=False,
            verbose=config.verbose
        )
        # Fit-transform to get topics (simulate)
        topics, _ = topic_model.fit_transform(unique_strings, embeddings=embeddings)
        # Reduce outliers
        new_topics = topic_model.reduce_outliers(unique_strings, topics, strategy="c-tf-idf", threshold=0.1)
        new_topics = topic_model.reduce_outliers(unique_strings, new_topics, strategy="distributions")
        topic_model.update_topics(unique_strings, topics=new_topics)
        cluster_labels = np.array(new_topics)
        if config.verbose:
            n_final_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_final_noise = list(cluster_labels).count(-1)
            print(f"After BERTopic outlier reduction: {n_final_clusters} clusters, {n_final_noise} outliers")

    # Step 4: Generate cluster summaries
    from collections import defaultdict
    cluster_values = defaultdict(list)
    for value, cluster_id in zip(unique_values, cluster_labels):
        cluster_values[cluster_id].append(value)

    cluster_label_map = generate_cluster_summaries(cluster_values, config, column_name)

    # -------------------------------------------------------------
    # Step 4b (moved up): Deduplicate fine-cluster labels via LLM
    # -------------------------------------------------------------
    if config.verbose:
        print("Deduplicating fine-cluster labels…")

    fine_cluster_names = [cluster_label_map[cid] for cid in cluster_values.keys() if cid != -1]

    # Generate deduplicated labels
    deduped_names = generate_coarse_labels(
        fine_cluster_names,
        max_coarse_clusters=None,
        systems_prompt=deduplication_clustering_systems_prompt,
        model=config.summary_model,
        verbose=config.verbose,
    )

    # Assign fine labels to deduplicated labels
    fine_to_dedupe = assign_fine_to_coarse(
        fine_cluster_names,
        deduped_names,
        model=config.cluster_assignment_model,
        strategy="llm",
        verbose=config.verbose,
    )

    # Merge fine clusters that were deduplicated
    # 1. Update the label map so every original cluster id points to its deduped label
    for cid in list(cluster_values.keys()):
        if cid == -1:
            continue
        original_label = cluster_label_map[cid]
        deduped_label = fine_to_dedupe.get(original_label, original_label)
        cluster_label_map[cid] = deduped_label

    # 2. Build mapping from deduped label → new sequential id
    unique_dedup_labels = [lbl for lbl in sorted(set(cluster_label_map.values())) if lbl != "Outliers"]
    label_to_new_id = {lbl: idx for idx, lbl in enumerate(unique_dedup_labels)}

    # 3. Re-assign cluster_labels array so duplicates share the same id
    new_cluster_labels = []
    for original_cid in cluster_labels:
        if original_cid == -1:
            new_cluster_labels.append(-1)
        else:
            deduped = cluster_label_map[original_cid]
            new_cluster_labels.append(label_to_new_id.get(deduped, -1))

    cluster_labels = np.array(new_cluster_labels)

    # 4. Rebuild cluster_values & cluster_label_map using new ids
    cluster_values = defaultdict(list)
    for val, cid in zip(unique_values, cluster_labels):
        cluster_values[cid].append(val)

    cluster_label_map = {new_id: lbl for lbl, new_id in label_to_new_id.items()}
    cluster_label_map[-1] = "Outliers"  # Ensure outliers are properly mapped

    # -------------------------------------------------------------
    # Step 4a (moved down): Cluster outliers using LLM
    # Now assign outliers against BOTH deduped fine labels and outlier cluster labels
    # -------------------------------------------------------------
    if config.verbose:
        print("Clustering outliers using LLM...")

    # Get outlier items after dedup
    outlier_items = [unique_values[i] for i, label in enumerate(cluster_labels) if label == -1]

    if len(outlier_items) > 0:
        if config.verbose:
            print("Original number of outliers:", len(outlier_items))

        # 1) Generate candidate outlier-only cluster names
        outlier_cluster_names = generate_coarse_labels(
            outlier_items,
            max_coarse_clusters=len(outlier_items) // (config.min_cluster_size // 2),
            systems_prompt=outlier_clustering_systems_prompt,
            model=config.summary_model,
            verbose=config.verbose,
        )

        # 2) Build combined candidate label set: deduped fine labels + outlier labels
        existing_dedup_fine_labels = [cluster_label_map[cid] for cid in cluster_values.keys() if cid != -1]
        # remove "Outliers" if present in suggestions to avoid confusion
        outlier_cluster_names_clean = [n for n in outlier_cluster_names if n != "Outliers"]
        combined_candidate_labels = list(dict.fromkeys(existing_dedup_fine_labels + outlier_cluster_names_clean))

        # Map from fine label -> current id
        fine_label_to_id = {label: cid for cid, label in cluster_label_map.items() if cid != -1}

        # 3) Assign each outlier to the best label among the combined set
        outlier_assignments = assign_fine_to_coarse(
            outlier_items,
            combined_candidate_labels,
            model=config.cluster_assignment_model,
            strategy="llm",
            verbose=config.verbose,
        )

        # 4) Allocate new IDs for truly new outlier clusters (not matching any fine label)
        next_cluster_id = (max([cid for cid in cluster_values.keys() if cid != -1]) + 1) if any(cid != -1 for cid in cluster_values.keys()) else 0
        outlier_name_to_new_id = {}
        for name in outlier_cluster_names_clean:
            if name not in fine_label_to_id and name not in outlier_name_to_new_id:
                outlier_name_to_new_id[name] = next_cluster_id
                next_cluster_id += 1

        # 5) Update cluster_labels for outliers
        new_cluster_labels = cluster_labels.copy()
        for i, label in enumerate(cluster_labels):
            if label == -1:
                item = unique_values[i]
                assigned_label = outlier_assignments.get(item, "Outliers")
                if assigned_label in fine_label_to_id:
                    new_cluster_labels[i] = fine_label_to_id[assigned_label]
                elif assigned_label in outlier_name_to_new_id:
                    new_cluster_labels[i] = outlier_name_to_new_id[assigned_label]
                # else keep -1

        cluster_labels = new_cluster_labels

        # 6) Rebuild cluster_values and extend cluster_label_map to include new outlier clusters
        cluster_values = defaultdict(list)
        for value, cid in zip(unique_values, cluster_labels):
            cluster_values[cid].append(value)

        for name, cid in outlier_name_to_new_id.items():
            cluster_label_map[cid] = name

        if config.verbose:
            n_outlier_clusters = len(outlier_name_to_new_id)
            n_remaining_outliers = list(cluster_labels).count(-1)
            print(f"Assigned outliers to {n_outlier_clusters} clusters, {n_remaining_outliers} remain as outliers")
    else:
        if config.verbose:
            print("No outliers to cluster")

    # # Step 5: Handle hierarchical clustering if requested
    coarse_cluster_data = None
    if config.hierarchical:
        if config.verbose:
            print("Generating hierarchical coarse clusters…")

        fine_cluster_names = [cluster_label_map[c] for c in cluster_values.keys() if c != -1]
        max_coarse_clusters = min(config.max_coarse_clusters, max(1, len(fine_cluster_names) // 2))

        # Generate coarse cluster labels
        coarse_names = generate_coarse_labels(
            fine_cluster_names,
            max_coarse_clusters,
            systems_prompt=coarse_clustering_systems_prompt,
            model=config.summary_model,
            verbose=config.verbose,
        )

        # Assign fine labels to coarse labels
        coarse_assignments = assign_fine_to_coarse(
            fine_cluster_names,
            coarse_names,
            model=config.cluster_assignment_model,
            strategy="llm",
            verbose=config.verbose,
        )
        if config.verbose:
            print("coarse_assignments", coarse_assignments)
            print("coarse_names", coarse_names)
        # Map back to original values
        coarse_labels = []
        coarse_label_map = {}
        
        # Build coarse_labels in the same order as unique_values
        for value, fine_cluster_id in zip(unique_values, cluster_labels):
            if fine_cluster_id == -1:
                coarse_labels.append(-1)
                coarse_label_map[-1] = "Outliers"
            else:
                fine_name = cluster_label_map[fine_cluster_id]
                if fine_name in fine_cluster_names:
                    coarse_name = coarse_assignments[fine_name]
                    if coarse_name == "Outliers":
                        coarse_idx = -1
                    else:
                        coarse_idx = coarse_names.index(coarse_name)
                    coarse_labels.append(coarse_idx)
                    if coarse_idx != -1:
                        coarse_label_map[coarse_idx] = coarse_names[coarse_idx]
                else:
                    # Handle case where fine cluster name is not in fine_cluster_names
                    coarse_labels.append(-1)
                    coarse_label_map[-1] = "Outliers"
        coarse_cluster_data = (coarse_labels, coarse_label_map)

    # Step 6: Format results
    df_result = format_clustering_results(
        df, column_name, unique_values, original_embeddings,
        cluster_labels, cluster_label_map, config, coarse_cluster_data
    )

    if config.verbose:
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        total_time = time.time() - start_time
        print(f"Found {n_clusters} clusters and {n_noise} outliers in {total_time:.1f} seconds")

    return df_result

def hierarchical_cluster_categories(df, column_name, config=None, **kwargs):
    """
    Basic hierarchical clustering using sklearn's AgglomerativeClustering.
    
    Best for: Small to medium datasets with known cluster counts.
    
    Args:
        df: DataFrame containing the data
        column_name: Name of the column to cluster on
        config: ClusterConfig object with all parameters (preferred)
        **kwargs: Individual parameters for backward compatibility
    """
    # Handle backward compatibility by creating config from kwargs
    if config is None:
        config_kwargs = {}
        for k, v in kwargs.items():
            if hasattr(ClusterConfig, k):
                config_kwargs[k] = v
        config = ClusterConfig(**config_kwargs)
    
    # Set default values for hierarchical clustering parameters
    n_coarse_clusters = getattr(config, 'max_coarse_clusters', 10)
    n_fine_clusters = getattr(config, 'min_cluster_size', 50) * 2  # Default to 2x min_cluster_size
    
    start_time = time.time()
    unique_values = df[column_name].unique()
    unique_strings = [str(value) for value in unique_values]
    
    if config.verbose:
        print(f"Hierarchical clustering for {len(unique_values)} unique values...")
    
    # Get embeddings - note: this doesn't use caching yet, could be enhanced
    embeddings = _get_embeddings(unique_strings, config.embedding_model, config.verbose)
    embeddings = np.array(embeddings)
    
    # Clustering
    coarse_clustering = AgglomerativeClustering(n_clusters=n_coarse_clusters, linkage='ward')
    fine_clustering = AgglomerativeClustering(n_clusters=n_fine_clusters, linkage='ward')
    
    coarse_clusters = coarse_clustering.fit_predict(embeddings)
    fine_clusters = fine_clustering.fit_predict(embeddings)
    
    # Create basic mappings
    value_to_coarse_id = dict(zip(unique_values, coarse_clusters))
    value_to_fine_id = dict(zip(unique_values, fine_clusters))
    
    # Generate cluster labels (either LLM or generic)
    if config.verbose:
        print("Generating LLM-based cluster summaries...")
    
    # Group values by coarse clusters
    coarse_cluster_values = defaultdict(list)
    for value, cluster_id in zip(unique_values, coarse_clusters):
        coarse_cluster_values[cluster_id].append(value)
    
    # Group values by fine clusters  
    fine_cluster_values = defaultdict(list)
    for value, cluster_id in zip(unique_values, fine_clusters):
        fine_cluster_values[cluster_id].append(value)
    
    # Generate coarse cluster summaries
    coarse_label_map = {}
    for cluster_id, values in coarse_cluster_values.items():
        if len(values) < 5:  # Skip very small clusters
            coarse_label_map[cluster_id] = f"coarse_cluster_{cluster_id}"
            continue
            
        summary = _get_llm_cluster_summary(values, config.summary_model, column_name, "broad", 50)
        coarse_label_map[cluster_id] = summary
        
        if config.verbose:
            print(f"    Coarse cluster {cluster_id}: {summary} ({len(values)} items)")
    
    # Generate fine cluster summaries
    fine_label_map = {}
    for cluster_id, values in fine_cluster_values.items():
        if len(values) < 3:  # Skip very small clusters
            fine_label_map[cluster_id] = f"fine_cluster_{cluster_id}"
            continue
            
        summary = _get_llm_cluster_summary(values, config.summary_model, column_name, "specific", 30)
        fine_label_map[cluster_id] = summary
        
        if config.verbose and cluster_id % 5 == 0:  # Print progress for every 5th cluster
            print(f"    Fine cluster {cluster_id}: {summary} ({len(values)} items)")
    
    value_to_coarse_label = {v: coarse_label_map[c] for v, c in value_to_coarse_id.items()}
    value_to_fine_label = {v: fine_label_map[c] for v, c in value_to_fine_id.items()}
    
    # Create output
    df_copy = df.copy()
    df_copy[f'{column_name}_coarse_cluster_id'] = df_copy[column_name].map(value_to_coarse_id)
    df_copy[f'{column_name}_fine_cluster_id'] = df_copy[column_name].map(value_to_fine_id)
    df_copy[f'{column_name}_coarse_cluster_label'] = df_copy[column_name].map(value_to_coarse_label)
    df_copy[f'{column_name}_fine_cluster_label'] = df_copy[column_name].map(value_to_fine_label)
    
    if config.include_embeddings:
        value_to_embedding = dict(zip(unique_values, embeddings.tolist()))
        df_copy[f'{column_name}_embedding'] = df_copy[column_name].map(value_to_embedding)
    
    if config.verbose:
        print(f"Hierarchical clustering completed in {time.time() - start_time:.1f} seconds")
    
    return df_copy

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function with command-line argument support."""
    parser = argparse.ArgumentParser(description='Hierarchical Text Clustering')
    parser.add_argument('--file', '-f', required=True, 
                       help='Path to input JSONL file')
    parser.add_argument('--column', '-c', default='property_description',
                       help='Column name to cluster on (default: property_description)')
    parser.add_argument('--method', '-m', choices=['hdbscan', 'hierarchical', 'hdbscan_native'], 
                       default='hdbscan',
                       help='Clustering method (default: hdbscan)')
    parser.add_argument('--min-cluster-size', type=int, default=10,
                       help='Minimum cluster size (default: 15)')
    parser.add_argument('--min-samples', type=int,
                       help='min_samples for HDBSCAN. Lower values reduce outliers. (default: based on min_cluster_size)')
    parser.add_argument('--cluster-selection-epsilon', type=float, default=0.0,
                       help='Epsilon value for HDBSCAN cluster selection to merge clusters (default: 0.0, disabled, higher values merge more clusters)')
    parser.add_argument('--embedding-model', default='openai',
                       help='Embedding model: openai, all-MiniLM-L6-v2, etc. (default: openai)')
    parser.add_argument('--output', '-o', 
                       help='Output filename prefix (default: auto-generated)')
    parser.add_argument('--no-embeddings', action='store_true',
                       help='Exclude embeddings from output')
    parser.add_argument('--context', default='properties seen in AI responses',
                       help='Context for LLM summaries (default: "properties seen in AI responses")')
    parser.add_argument('--precomputed-embeddings', 
                       help='Path to precomputed embeddings file (.pkl or .npy)')
    parser.add_argument('--disable-dim-reduction', action='store_true',
                       help='Disable UMAP dimensionality reduction (default: False)')
    parser.add_argument('--dim-reduction-method', choices=['adaptive', 'umap', 'pca', 'none'], default='adaptive',
                       help='Dimension reduction method: adaptive (auto-choose), umap, pca, or none (default: adaptive)')
    parser.add_argument('--umap-n-components', type=int, default=100,
                       help='Number of UMAP components (default: 100)')
    parser.add_argument('--umap-n-neighbors', type=int, default=30,
                       help='Number of UMAP neighbors (default: 30)')
    parser.add_argument('--umap-min-dist', type=float, default=0.1,
                       help='UMAP minimum distance (default: 0.1)')
    parser.add_argument('--umap-metric', default='cosine',
                       help='UMAP distance metric (default: cosine)')
    parser.add_argument('--assign-outliers', action='store_true',
                       help='Assign HDBSCAN outliers to their nearest clusters (default: False)')
    parser.add_argument('--hierarchical', action='store_true',
                       help='Enable hierarchical HDBSCAN clustering (cluster the clusters) (default: False)')
    parser.add_argument('--min-grandparent-size', type=int, default=3,
                       help='Minimum size for grandparent clusters in hierarchical mode (default: 5)')
    parser.add_argument('--max-coarse-clusters', type=int, default=30,
                       help='Maximum coarse clusters for hierarchical clustering (default: 15)')
    parser.add_argument('--input-model-name', 
                       help='Name of the input model being analyzed (for cache differentiation)')
    parser.add_argument('--type', choices=['all', 'context-specific', 'general'], default='all',
                       help='Type of data to cluster (default: context-specific)')
    parser.add_argument('--remove-low-impact', action='store_true',
                       help='Remove low impact properties (default: False)')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')
    parser.add_argument('--wandb-project', 
                       help='wandb project name')
    parser.add_argument('--wandb-entity', 
                       help='wandb entity name')
    parser.add_argument('--wandb-run-name', 
                       help='wandb run name')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.file}...")
    df = pd.read_json(args.file, lines=True)

    if args.type == 'context-specific':
        def is_context_specific(row):
            return 'context' in row['type'].lower()
        df = df[df.apply(is_context_specific, axis=1)]
    elif args.type == 'general':
        def is_general(row):
            return 'general' in row['type'].lower()
        df = df[df.apply(is_general, axis=1)]
    elif args.type == 'all':
        pass
    
    if args.remove_low_impact:
        df = df[(df.impact.str.lower() == 'high') | (df.impact.str.lower() == 'medium')]
    
    if args.column not in df.columns:
        print(f"Error: Column '{args.column}' not found in data. Available columns: {list(df.columns)}")
        return None
    
    print(f"Loaded {len(df)} rows with {len(df[args.column].unique())} unique values in '{args.column}'")
    
    # Set up parameters
    include_embeddings = not args.no_embeddings
    
    # Load precomputed embeddings if provided
    precomputed_embeddings = None
    if args.precomputed_embeddings:
        precomputed_embeddings = load_precomputed_embeddings(args.precomputed_embeddings, verbose=True)
    
    # Create config from args
    config = ClusterConfig.from_args(args)
    if precomputed_embeddings:
        config.precomputed_embeddings = precomputed_embeddings
    
    # Initialize wandb if enabled
    method_name = args.method
    initialize_wandb(config, method_name, args.file)
    
    # Run clustering based on method
    if args.method == 'hdbscan':
        print(f"Running HDBSCAN clustering...")
        df_clustered = hdbscan_cluster_categories(df, args.column, config=config)
        method_name = "hdbscan"

    elif args.method == 'hierarchical':
        print(f"Running traditional hierarchical clustering...")
        df_clustered = hierarchical_cluster_categories(df, args.column, config=config)
        method_name = "hierarchical"
    
    # Generate output filename
    if args.output:
        output_prefix = args.output
    else:
        input_basename = os.path.splitext(os.path.basename(args.file))[0]
        output_prefix = f"{input_basename}_{method_name}_clustered"
    
    # Save results
    save_clustered_results(df_clustered, output_prefix, include_embeddings=include_embeddings, config=config)
    
    print(f"\n✅ Clustering complete! Final dataset shape: {df_clustered.shape}")
    
    # Close wandb run if it was initialized
    if config.use_wandb:
        wandb.finish()
        print("🔄 Wandb run completed")
    
    return df_clustered


if __name__ == "__main__":
    df_result = main() 


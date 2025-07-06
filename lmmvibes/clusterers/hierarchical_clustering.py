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
    embedding_model: str = "openai"
    verbose: bool = True
    include_embeddings: bool = True
    context: Optional[str] = None
    precomputed_embeddings: Optional[Union[np.ndarray, Dict, str]] = None
    disable_dim_reduction: bool = False
    assign_outliers: bool = False
    hierarchical: bool = False
    min_grandparent_size: int = 5
    max_coarse_clusters: int = 15
    input_model_name: Optional[str] = None
    min_samples: Optional[int] = None
    cluster_selection_epsilon: float = 0.0
    cache_embeddings: bool = True
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
        
        # Determine method (simplified logic)
        if config.dim_reduction_method == "adaptive":
            # Simple adaptive logic
            if n_points > 5000 or n_dims > 200:
                method = "umap" if n_points > 10000 else "umap"
            else:
                method = "none"
        else:
            method = config.dim_reduction_method
        
        if method == "umap":
            if config.verbose:
                print(f"Applying UMAP dimensionality reduction...")
            
            # Adaptive parameters
            n_components = min(config.umap_n_components, n_dims - 1)
            n_neighbors = min(config.umap_n_neighbors, n_points - 1)
            
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
            
        summary = _get_llm_cluster_summary(values, column_name, cluster_type, config.context, 50)
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
    print("format clustering results df_copy ", df_copy.columns)
    
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
    
    print("df_copy ", df_copy.columns)
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
    print(cluster_label_map)

    # Step 5: Handle hierarchical clustering if requested
    coarse_cluster_data = None
    if config.hierarchical:
        print("Generating hierarchical clusters...")
        fine_cluster_names = [cluster_label_map[c] for c in cluster_values.keys() if c != -1]
        coarse_assignments, coarse_names = llm_coarse_cluster_with_centers(
            fine_cluster_names, config.max_coarse_clusters, config.context, config.verbose
        )
        print("coarse_assignments", coarse_assignments)
        print("coarse_names", coarse_names)
        # Map back to original values
        coarse_labels = []
        coarse_label_map = {}
        for i, (cluster_id, _) in enumerate([(c, vals) for c, vals in cluster_values.items()]):
            if cluster_id == -1:
                coarse_labels.extend([-1] * len(cluster_values[cluster_id]))
                coarse_label_map[-1] = "Outliers"
            else:
                fine_name = cluster_label_map[cluster_id]
                if fine_name in fine_cluster_names:
                    coarse_idx = coarse_assignments[fine_cluster_names.index(fine_name)]
                    coarse_labels.extend([coarse_idx] * len(cluster_values[cluster_id]))
                    if coarse_idx != -1:
                        coarse_label_map[coarse_idx] = coarse_names[coarse_idx]
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


def hdbscan_native_hierarchical_cluster(df, column_name, config=None, **kwargs):
    """
    Hierarchical clustering using HDBSCAN's native hierarchy.
    
    This method runs HDBSCAN once and then extracts multiple levels of clusters
    by traversing the algorithm's internal condensed tree.
    
    Best for: Efficiently exploring the natural, multi-level structure of data.
    
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

    start_time = time.time()
    unique_values = df[column_name].unique()
    unique_strings = [str(value) for value in unique_values]
    
    if config.verbose:
        print(f"Native HDBSCAN hierarchical clustering for {len(unique_values)} unique values...")

    # Step 1: Prepare embeddings (including dimension reduction)
    embeddings, original_embeddings = prepare_embeddings(unique_values, config)

    # Step 2: Run HDBSCAN once
    if config.verbose:
        print("Starting HDBSCAN clustering...")

    if config.min_samples is None:
        config.min_samples = config.min_cluster_size

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        algorithm='best',
        core_dist_n_jobs=-1,
        prediction_data=True,  # Required for all_points_membership_vectors
        cluster_selection_epsilon=config.cluster_selection_epsilon
    )
    clusterer.fit(embeddings)
    if config.verbose:
        n_flat_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
        print(f"HDBSCAN fit completed. Found {n_flat_clusters} flat clusters.")

    # Step 3: Extract hierarchy from the condensed tree
    if config.verbose:
        print("Extracting native hierarchy from condensed tree...")
    
    try:
        membership_vectors = hdbscan.all_points_membership_vectors(clusterer)
    except Exception as e:
        print(f"Error getting membership vectors: {e}")
        print("This may be due to an older hdbscan version. Please upgrade: pip install --upgrade hdbscan")
        # As a fallback, we could just use clusterer.labels_ as one level. For now, we raise.
        raise

    tree = clusterer.condensed_tree_.to_pandas()
    
    # Create child-to-parent mapping for tree traversal
    child_to_parent = dict(zip(tree['child'], tree['parent']))
    
    # Determine the most specific (leaf) cluster for each point from the hierarchy
    point_to_cluster = {}

    # The output of all_points_membership_vectors can be sparse or dense depending on the library version.
    is_sparse = not isinstance(membership_vectors, np.ndarray)

    for point_idx, point_memberships in enumerate(membership_vectors):
        if is_sparse:
            # Handle sparse matrix row object, which has .indices and .data attributes
            if len(point_memberships.data) > 0:
                # Find the cluster with the highest membership strength
                cluster_id = point_memberships.indices[np.argmax(point_memberships.data)]
                point_to_cluster[point_idx] = cluster_id
            else:
                point_to_cluster[point_idx] = -1  # Outlier
        else:
            # Handle dense numpy array row
            if np.sum(point_memberships) > 0:
                cluster_id = np.argmax(point_memberships)
                point_to_cluster[point_idx] = cluster_id
            else:
                point_to_cluster[point_idx] = -1 # Outlier
    
    # Step 4: Generate all levels of the hierarchy
    hierarchy_levels = {}
    level_0_map = {unique_values[i]: cid for i, cid in point_to_cluster.items()}
    hierarchy_levels[0] = level_0_map

    max_levels = 10  # Safety break
    for i in range(1, max_levels):
        prev_level_map = hierarchy_levels[i-1]
        
        # Check if we have reached the root or only outliers
        prev_level_clusters = set(prev_level_map.values())
        if all(cid == -1 or cid not in child_to_parent or child_to_parent[cid] == cid for cid in prev_level_clusters):
            if config.verbose:
                print(f"Reached top of hierarchy at level {i-1}. Stopping.")
            break
            
        next_level_map = {}
        for value, cluster_id in prev_level_map.items():
            if cluster_id != -1 and cluster_id in child_to_parent:
                parent_id = child_to_parent[cluster_id]
                # Avoid self-loops at the root
                next_level_map[value] = parent_id if parent_id != cluster_id else -1
            else:
                next_level_map[value] = -1
        
        # Stop if the new level is identical to the previous one
        if next_level_map == prev_level_map:
            break
            
        hierarchy_levels[i] = next_level_map
    
    if config.verbose:
        print(f"Generated {len(hierarchy_levels)} levels of clusters.")

    # Step 5: Generate labels and create output DataFrame
    df_copy = df.copy()
    level_names = {0: 'fine', 1: 'coarse'}

    for level_idx, value_to_id_map in hierarchy_levels.items():
        level_name = level_names.get(level_idx, f'coarse_{level_idx}')
        if config.verbose:
            num_clusters = len(set(value_to_id_map.values())) - (1 if -1 in set(value_to_id_map.values()) else 0)
            print(f"\nProcessing Level {level_idx} ('{level_name}') with {num_clusters} clusters...")

        # Generate LLM summaries if requested
        print(f"  Generating LLM summaries for level '{level_name}'...")
            
        cluster_values = defaultdict(list)
        for value, cluster_id in value_to_id_map.items():
            cluster_values[cluster_id].append(value)
        
        id_to_label_map = {}
        for cluster_id, values in cluster_values.items():
            if cluster_id == -1:
                id_to_label_map[cluster_id] = "Outliers"
                continue
            
            # Use a slightly different prompt context for different levels
            cluster_type = "specific" if level_idx == 0 else "broad"
            summary = _get_llm_cluster_summary(values, column_name, cluster_type, config.context, 50)
            id_to_label_map[cluster_id] = summary
            
            if config.verbose and (cluster_id % 5 == 0 or num_clusters < 10):
                print(f"    Cluster {cluster_id}: {summary} ({len(values)} items)")

        value_to_label_map = {v: id_to_label_map[cid] for v, cid in value_to_id_map.items()}
       
        df_copy[f'{column_name}_{level_name}_cluster_id'] = df_copy[column_name].map(value_to_id_map)
        df_copy[f'{column_name}_{level_name}_cluster_label'] = df_copy[column_name].map(value_to_label_map)

    if config.include_embeddings:
        value_to_embedding = dict(zip(unique_values, original_embeddings.tolist()))
        df_copy[f'{column_name}_embedding'] = df_copy[column_name].map(value_to_embedding)

    if config.verbose:
        total_time = time.time() - start_time
        print(f"\nNative HDBSCAN clustering completed in {total_time:.1f} seconds")

    return df_copy


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
            
        summary = _get_llm_cluster_summary(values, column_name, "broad", config.context, 50)
        coarse_label_map[cluster_id] = summary
        
        if config.verbose:
            print(f"    Coarse cluster {cluster_id}: {summary} ({len(values)} items)")
    
    # Generate fine cluster summaries
    fine_label_map = {}
    for cluster_id, values in fine_cluster_values.items():
        if len(values) < 3:  # Skip very small clusters
            fine_label_map[cluster_id] = f"fine_cluster_{cluster_id}"
            continue
            
        summary = _get_llm_cluster_summary(values, column_name, "specific", config.context, 30)
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
        
    elif args.method == 'hdbscan_native':
        print(f"Running Native HDBSCAN hierarchical clustering...")
        df_clustered = hdbscan_native_hierarchical_cluster(df, args.column, config=config)
        method_name = "hdbscan_native"
    
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


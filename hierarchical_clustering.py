#!/usr/bin/env python3
"""
Hierarchical Text Clustering Module

Provides scalable hierarchical clustering for text data using semantic embeddings.
Supports multiple clustering algorithms including BERTopic, HDBSCAN, and traditional methods.
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

# Core ML libraries
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer

# Optional imports (will be checked when needed)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from bertopic import BERTopic
    import hdbscan
    import umap
    HAS_BERTOPIC = True
except ImportError:
    HAS_BERTOPIC = False

try:
    import litellm
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False


# =============================================================================
# MAIN CLUSTERING FUNCTIONS
# =============================================================================

def bertopic_hierarchical_cluster_categories(df, column_name, min_cluster_size=10, min_topic_size=10,
                                           max_coarse_topics=25, max_fine_topics_per_coarse=50,
                                           verbose=True, embedding_model="openai", 
                                           include_embeddings=True, cache_embeddings=True,
                                           use_llm_summaries=False, context=None,
                                           use_llm_coarse_clustering=False, max_coarse_clusters=15,
                                           input_model_name=None):
    """
    Two-stage hierarchical clustering using BERTopic: coarse topics → fine subtopics.
    
    Best for: Well-defined categories with natural subcategories.
    
    Args:
        df: DataFrame containing the data
        column_name: Name of the column to cluster on
        min_cluster_size: Minimum cluster size for BERTopic (default: 30)
        min_topic_size: Minimum size for fine-level topics (default: 10)
        max_coarse_topics: Maximum number of coarse topics (default: 25)
        max_fine_topics_per_coarse: Maximum fine topics per coarse topic (default: 20)
        verbose: Print progress information (default: True)
        embedding_model: "openai", "all-MiniLM-L6-v2", or "all-mpnet-base-v2" (default: "openai")
        include_embeddings: Include embeddings in output (default: True)
        cache_embeddings: Save/load embeddings from disk (default: True)
        use_llm_summaries: Use LLM to generate human-readable cluster names (default: False)
        context: Optional context for LLM summaries (e.g., "properties seen in AI responses")
        use_llm_coarse_clustering: Use LLM-only approach to create coarse clusters from fine cluster names (default: False)
        max_coarse_clusters: Maximum coarse clusters when using LLM coarse clustering (default: 15)
        input_model_name: Optional name of the input model being analyzed (for cache differentiation)
    
    Returns:
        DataFrame with added columns:
        - {column_name}_coarse_topic_id: Coarse topic ID (-1 for outliers)
        - {column_name}_coarse_topic_label: Human-readable coarse topic name
        - {column_name}_fine_topic_id: Fine topic ID (-1 for outliers) 
        - {column_name}_fine_topic_label: Human-readable fine topic name
        - {column_name}_embedding: Vector embeddings (if include_embeddings=True)
    
    Example:
        >>> df_clustered = bertopic_hierarchical_cluster_categories(
        ...     df, 'response_text', verbose=True, use_llm_summaries=True
        ... )
    """
    if not HAS_BERTOPIC:
        raise ImportError("Please install: pip install bertopic sentence-transformers")
    
    start_time = time.time()
    unique_values = df[column_name].unique()
    unique_strings = [str(value) for value in unique_values]
    
    if verbose:
        print(f"BERTopic hierarchical clustering for {len(unique_values)} unique values...")
    
    # Get embeddings (with caching)
    embeddings, embedding_model_obj = _setup_embeddings_with_cache(
        unique_strings, embedding_model, column_name, cache_embeddings, verbose, input_model_name
    )
    
    # Step 1: Coarse-level clustering
    if verbose:
        print("Step 1: Coarse-level topic modeling...")
    
    coarse_topic_model = _create_bertopic_model(
        embedding_model_obj, unique_values, max_coarse_topics, min_cluster_size, verbose
    )
    
    if embeddings is not None:
        coarse_topics, _ = coarse_topic_model.fit_transform(unique_strings, embeddings)
    else:
        coarse_topics, _ = coarse_topic_model.fit_transform(unique_strings)
    
    coarse_topic_info = coarse_topic_model.get_topic_info()
    if verbose:
        print(f"Found {len(coarse_topic_info)} coarse topics")
        for _, row in coarse_topic_info.head(5).iterrows():
            print(f"  Topic {row['Topic']}: {row['Name']} ({row['Count']} items)")

    # Step 2: Fine-level clustering within each coarse topic
    if verbose:
        print("Step 2: Fine-level topic modeling...")
    
    fine_topics, fine_topic_labels = _create_fine_topics(
        unique_values, coarse_topics, embedding_model_obj, embeddings, 
        min_topic_size, max_fine_topics_per_coarse, verbose
    )
    
    # Step 4: Optional LLM coarse clustering from fine topics
    if use_llm_coarse_clustering:
        if verbose:
            print("Step 4: Using LLM to create coarse clusters from fine topic names...")
        
        # Get fine topic names
        fine_topic_names = list(fine_topic_labels.values())
        unique_fine_names = list(set(fine_topic_names))
        
        # Use LLM to create coarse clusters from fine topic names
        fine_to_coarse_assignments, coarse_cluster_names = llm_coarse_cluster_from_fine(
            unique_fine_names, max_coarse_clusters, context, verbose
        )
        
        # Create mapping from fine topic ID to coarse cluster
        fine_name_to_coarse_id = dict(zip(unique_fine_names, fine_to_coarse_assignments))
        fine_name_to_coarse_name = {}
        for fine_name, coarse_id in fine_name_to_coarse_id.items():
            if coarse_id == -1:
                fine_name_to_coarse_name[fine_name] = "Outliers"
            else:
                fine_name_to_coarse_name[fine_name] = coarse_cluster_names[coarse_id]
        
        # Override coarse topics with LLM-based clustering
        new_coarse_topics = []
        new_coarse_topic_info = pd.DataFrame()
        coarse_topic_labels = {}
        
        for i, fine_topic_id in enumerate(fine_topics):
            fine_topic_name = fine_topic_labels.get(fine_topic_id, f"topic_{fine_topic_id}")
            coarse_cluster_name = fine_name_to_coarse_name.get(fine_topic_name, "Outliers")
            coarse_cluster_id = fine_name_to_coarse_id.get(fine_topic_name, -1)
            new_coarse_topics.append(coarse_cluster_id)
            coarse_topic_labels[coarse_cluster_id] = coarse_cluster_name
        
        # Create new coarse topic info DataFrame
        coarse_counts = defaultdict(int)
        for topic_id in new_coarse_topics:
            coarse_counts[topic_id] += 1
        
        topic_data = []
        for topic_id, count in coarse_counts.items():
            topic_name = coarse_topic_labels.get(topic_id, f"Topic {topic_id}")
            topic_data.append({
                'Topic': topic_id,
                'Count': count,
                'Name': topic_name
            })
        
        coarse_topic_info = pd.DataFrame(topic_data)
        coarse_topics = new_coarse_topics
        
        if verbose:
            n_coarse = len(set(coarse_topics)) - (1 if -1 in coarse_topics else 0)
            print(f"✅ Created {n_coarse} LLM-based coarse clusters from fine topics")

    # Step 3: Optional LLM-based summarization
    elif use_llm_summaries:
        if verbose:
            print("Step 3: Generating LLM-based cluster summaries...")
        coarse_topic_info, fine_topic_labels = _generate_llm_summaries(
            unique_values, coarse_topics, fine_topics, coarse_topic_info, 
            fine_topic_labels, column_name, context, verbose
        )
    
    # Create final mappings and output
    result_df = _create_bertopic_output(
        df, column_name, unique_values, coarse_topics, fine_topics,
        coarse_topic_info, fine_topic_labels, embeddings, include_embeddings
    )
    
    if verbose:
        total_time = time.time() - start_time
        print(f"BERTopic clustering completed in {total_time:.1f} seconds")
        _print_topic_summary(coarse_topic_info, fine_topic_labels, column_name)
    
    return result_df


def hdbscan_cluster_categories(df, column_name, min_cluster_size=30, 
                              embedding_model="openai", verbose=True, 
                              include_embeddings=True, use_llm_summaries=False, context=None,
                              precomputed_embeddings=None, enable_dim_reduction=False,
                              assign_outliers=False, hierarchical=False, min_grandparent_size=5,
                              use_llm_coarse_clustering=False, max_coarse_clusters=15,
                              input_model_name=None):
    """
    Fast HDBSCAN clustering for medium to large datasets.
    
    Best for: Natural cluster discovery, noise handling, scalability.
    
    Args:
        df: DataFrame containing the data
        column_name: Name of the column to cluster on
        min_cluster_size: Minimum cluster size (default: 10)
        embedding_model: Embedding method (default: "openai")
        verbose: Print progress (default: True)
        include_embeddings: Include embeddings in output (default: True)
        use_llm_summaries: Use LLM to generate cluster summaries (default: False)
        context: Optional context for LLM summaries (e.g., "properties seen in AI responses")
        precomputed_embeddings: Optional precomputed embeddings array/dict (default: None)
        enable_dim_reduction: Enable UMAP dimensionality reduction (default: False)
        assign_outliers: Assign HDBSCAN outliers to their nearest clusters using distance-based assignment
        hierarchical: Enable hierarchical clustering (cluster the clusters) (default: False)
        min_grandparent_size: Minimum size for grandparent clusters (default: 5)
        use_llm_coarse_clustering: Use LLM-only approach to create coarse clusters from fine cluster names (default: False)
        max_coarse_clusters: Maximum coarse clusters when using LLM coarse clustering (default: 15)
        input_model_name: Optional name of the input model being analyzed (for cache differentiation)
    """
    if not HAS_BERTOPIC:  # hdbscan comes with bertopic
        raise ImportError("Please install: pip install hdbscan sentence-transformers")
    
    start_time = time.time()
    unique_values = df[column_name].unique()
    unique_strings = [str(value) for value in unique_values]
    
    if verbose:
        print(f"HDBSCAN clustering for {len(unique_values)} unique values...")
        if hierarchical:
            print("  Hierarchical mode enabled - will create grandparent clusters")
    
    # Get embeddings (either precomputed or compute fresh)
    if precomputed_embeddings is not None:
        if verbose:
            print("Using precomputed embeddings...")
        embeddings = precomputed_embeddings
        if isinstance(embeddings, dict):
            if verbose:
                print(f"Mapping {len(unique_values)} values to embeddings from dict with {len(embeddings)} entries...")
            # If embeddings is a dict mapping values to embeddings
            try:
                embeddings = np.array([embeddings[str(val)] for val in unique_values])
                if verbose:
                    print(f"✅ Successfully mapped to {len(embeddings)} embeddings")
            except KeyError as e:
                print(f"❌ Error: Some values not found in precomputed embeddings: {e}")
                print(f"Available keys (first 5): {list(embeddings.keys())[:5]}")
                print(f"Missing values (first 5): {[str(val) for val in unique_values if str(val) not in embeddings][:5]}")
                raise
        else:
            if verbose:
                print(f"Using precomputed embeddings array with {len(embeddings)} entries...")
            # If embeddings is already an array
            embeddings = np.array(embeddings)
        
        if verbose:
            print(f"Embeddings shape: {embeddings.shape}")
    else:
        # For non-precomputed embeddings, we'll need to implement caching here too
        # For now, just use the regular _get_embeddings function
        embeddings = _get_embeddings(unique_strings, embedding_model, verbose)
        embeddings = np.array(embeddings)
    
    # Normalize embeddings
    if verbose:
        print("Normalizing embeddings...")
    embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)
    
    # Keep original embeddings for output (before any dimensionality reduction)
    original_embeddings = embeddings.copy()
    
    # Optional: Reduce dimensionality for faster clustering on large datasets
    if enable_dim_reduction and len(unique_values) > 10000 and embeddings.shape[1] > 100:
        if verbose:
            print(f"Large dataset detected ({len(unique_values)} points, {embeddings.shape[1]} dims). Applying UMAP dimensionality reduction...")
        try:
            import umap
            reducer = umap.UMAP(
                n_components=50,  # Reduce to 50 dimensions
                n_neighbors=15,
                min_dist=0.0,
                metric='cosine',
                random_state=42,
                verbose=verbose
            )
            embeddings = reducer.fit_transform(embeddings)  # Only modify embeddings used for clustering
            if verbose:
                print(f"Reduced embeddings to shape: {embeddings.shape}")
        except ImportError:
            if verbose:
                print("UMAP not available, proceeding with full dimensionality (may be slower)")
    
    # HDBSCAN clustering with optimized parameters for large datasets
    if verbose:
        print("Starting initial HDBSCAN clustering...")
        print(f"Parameters: min_cluster_size={min_cluster_size}, data_shape={embeddings.shape}")
    
    # Adjust parameters based on dataset size
    min_samples = min(min_cluster_size, max(5, min_cluster_size // 2))
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',  # Excess of Mass (faster than leaf)
        algorithm='best',  # Let HDBSCAN choose the best algorithm
        core_dist_n_jobs=-1  # Use all CPU cores
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    
    if verbose:
        n_initial_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        print(f"Initial HDBSCAN clustering completed! Found {n_initial_clusters} clusters and {n_noise} outliers")
    
    # Optional: Assign outliers to nearest clusters
    if assign_outliers and -1 in cluster_labels:
        if verbose:
            n_outliers = list(cluster_labels).count(-1)
            print(f"Assigning {n_outliers} outliers to nearest clusters...")
        
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Get outlier indices and non-outlier indices
        outlier_mask = cluster_labels == -1
        outlier_indices = np.where(outlier_mask)[0]
        non_outlier_indices = np.where(~outlier_mask)[0]
        
        if len(non_outlier_indices) > 0:  # Only if we have non-outlier clusters
            # Calculate distances from outliers to all non-outlier points
            outlier_embeddings = embeddings[outlier_indices]
            non_outlier_embeddings = embeddings[non_outlier_indices]
            non_outlier_labels = cluster_labels[non_outlier_indices]
            
            # For each outlier, find the nearest non-outlier point and assign its cluster
            distances = euclidean_distances(outlier_embeddings, non_outlier_embeddings)
            nearest_indices = np.argmin(distances, axis=1)
            
            # Assign outliers to the cluster of their nearest neighbor
            for i, outlier_idx in enumerate(outlier_indices):
                nearest_non_outlier_idx = non_outlier_indices[nearest_indices[i]]
                cluster_labels[outlier_idx] = cluster_labels[nearest_non_outlier_idx]
            
            if verbose:
                print(f"✅ Successfully assigned all outliers to nearest clusters")
    
    # Store initial cluster results
    initial_cluster_labels = cluster_labels.copy()
    grandparent_cluster_labels = None
    
    # Hierarchical clustering: cluster the clusters
    if hierarchical:
        if verbose:
            print("Starting hierarchical clustering (clustering the clusters)...")
        
        # Get unique cluster IDs (excluding outliers)
        unique_clusters = [c for c in set(cluster_labels) if c != -1]
        
        if len(unique_clusters) >= min_grandparent_size:
            # Compute cluster centroids
            cluster_centroids = []
            cluster_sizes = []
            
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_embeddings = embeddings[cluster_mask]
                centroid = np.mean(cluster_embeddings, axis=0)
                cluster_centroids.append(centroid)
                cluster_sizes.append(np.sum(cluster_mask))
            
            cluster_centroids = np.array(cluster_centroids)
            
            if verbose:
                print(f"  Computing grandparent clusters from {len(cluster_centroids)} cluster centroids...")
            
            # Run HDBSCAN on cluster centroids
            grandparent_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_grandparent_size,
                min_samples=min(min_grandparent_size, max(2, min_grandparent_size // 2)),
                metric='euclidean',
                cluster_selection_method='eom'
            )
            grandparent_labels = grandparent_clusterer.fit_predict(cluster_centroids)
            
            # Optional: Assign grandparent-level outliers to nearest grandparent clusters
            if assign_outliers and -1 in grandparent_labels:
                if verbose:
                    n_grandparent_outliers = list(grandparent_labels).count(-1)
                    print(f"  Assigning {n_grandparent_outliers} grandparent outliers to nearest grandparent clusters...")
                
                # Get outlier and non-outlier centroid indices
                grandparent_outlier_mask = grandparent_labels == -1
                grandparent_outlier_indices = np.where(grandparent_outlier_mask)[0]
                grandparent_non_outlier_indices = np.where(~grandparent_outlier_mask)[0]
                
                if len(grandparent_non_outlier_indices) > 0:  # Only if we have non-outlier grandparent clusters
                    # Calculate distances from outlier centroids to non-outlier centroids
                    outlier_centroids = cluster_centroids[grandparent_outlier_indices]
                    non_outlier_centroids = cluster_centroids[grandparent_non_outlier_indices]
                    non_outlier_grandparent_labels = grandparent_labels[grandparent_non_outlier_indices]
                    
                    # For each outlier centroid, find the nearest non-outlier centroid and assign its grandparent cluster
                    centroid_distances = euclidean_distances(outlier_centroids, non_outlier_centroids)
                    nearest_centroid_indices = np.argmin(centroid_distances, axis=1)
                    
                    # Assign outlier centroids to the grandparent cluster of their nearest neighbor
                    for i, outlier_centroid_idx in enumerate(grandparent_outlier_indices):
                        nearest_non_outlier_centroid_idx = grandparent_non_outlier_indices[nearest_centroid_indices[i]]
                        grandparent_labels[outlier_centroid_idx] = grandparent_labels[nearest_non_outlier_centroid_idx]
                    
                    if verbose:
                        print(f"  ✅ Successfully assigned all grandparent outliers to nearest grandparent clusters")
            
            # Map back to original data points
            cluster_to_grandparent = dict(zip(unique_clusters, grandparent_labels))
            grandparent_cluster_labels = np.array([
                cluster_to_grandparent.get(label, -1) for label in cluster_labels
            ])
            
            n_grandparent_clusters = len(set(grandparent_labels)) - (1 if -1 in grandparent_labels else 0)
            if verbose:
                print(f"✅ Created {n_grandparent_clusters} grandparent clusters")
        else:
            if verbose:
                print(f"  Not enough clusters ({len(unique_clusters)}) for hierarchical clustering (min: {min_grandparent_size})")
            grandparent_cluster_labels = np.full_like(cluster_labels, 0)  # Single grandparent cluster
    
    # Create basic mappings
    value_to_cluster = dict(zip(unique_values, initial_cluster_labels))
    if hierarchical:
        value_to_grandparent = dict(zip(unique_values, grandparent_cluster_labels))
    
    # Generate cluster labels (either LLM or generic)
    if use_llm_summaries:
        if verbose:
            print("Generating LLM-based cluster summaries...")
        
        # Group values by cluster
        cluster_values = defaultdict(list)
        for value, cluster_id in zip(unique_values, initial_cluster_labels):
            cluster_values[cluster_id].append(value)
        
        # Generate summaries for each cluster
        cluster_label_map = {}
        for cluster_id, values in cluster_values.items():
            if cluster_id == -1:
                cluster_label_map[cluster_id] = "Outliers"
                continue
                
            if len(values) < 5:  # Skip very small clusters
                cluster_label_map[cluster_id] = f"cluster_{cluster_id}"
                continue
                
            summary = _get_llm_cluster_summary(values, column_name, "cluster", context, 50)
            cluster_label_map[cluster_id] = summary
            
            if verbose:
                print(f"    Cluster {cluster_id}: {summary} ({len(values)} items)")
        
        # Generate grandparent cluster summaries if hierarchical
        grandparent_label_map = {}
        if hierarchical and grandparent_cluster_labels is not None:
            if verbose:
                print("Generating LLM-based grandparent cluster summaries...")
            
            grandparent_values = defaultdict(list)
            for value, grandparent_id in zip(unique_values, grandparent_cluster_labels):
                grandparent_values[grandparent_id].append(value)
            
            for grandparent_id, values in grandparent_values.items():
                if grandparent_id == -1:
                    grandparent_label_map[grandparent_id] = "Outliers"
                    continue
                    
                if len(values) < 10:  # Skip very small grandparent clusters
                    grandparent_label_map[grandparent_id] = f"grandparent_{grandparent_id}"
                    continue
                    
                summary = _get_llm_cluster_summary(values, column_name, "grandparent cluster", context, 100)
                grandparent_label_map[grandparent_id] = summary
                
                if verbose:
                    print(f"    Grandparent {grandparent_id}: {summary} ({len(values)} items)")
    else:
        cluster_label_map = {c: f"cluster_{c}" if c != -1 else "outlier" for c in set(initial_cluster_labels)}
        if hierarchical:
            grandparent_label_map = {c: f"grandparent_{c}" if c != -1 else "outlier" for c in set(grandparent_cluster_labels)}
    
    value_to_label = {v: cluster_label_map[c] for v, c in value_to_cluster.items()}
    if hierarchical:
        value_to_grandparent_label = {v: grandparent_label_map[c] for v, c in value_to_grandparent.items()}
    
    # Optional: Use LLM to create coarse clusters from fine cluster names
    if use_llm_coarse_clustering and not hierarchical:  # Don't use both hierarchical and LLM coarse clustering
        if verbose:
            print("Using LLM to create coarse clusters from fine cluster names...")
        
        # Get unique fine cluster names (excluding outliers initially)
        fine_cluster_names = list(set(value_to_label.values()))
        
        # Use LLM to create coarse clusters from fine cluster names
        fine_to_coarse_assignments, coarse_cluster_names = llm_coarse_cluster_from_fine(
            fine_cluster_names, max_coarse_clusters, context, verbose
        )
        
        # Create mapping from fine cluster name to coarse cluster
        fine_name_to_coarse_id = dict(zip(fine_cluster_names, fine_to_coarse_assignments))
        fine_name_to_coarse_name = {}
        for fine_name, coarse_id in fine_name_to_coarse_id.items():
            if coarse_id == -1:
                fine_name_to_coarse_name[fine_name] = "Outliers"
            else:
                fine_name_to_coarse_name[fine_name] = coarse_cluster_names[coarse_id]
        
        # Map original values to coarse clusters through fine clusters
        value_to_coarse_label = {}
        value_to_coarse_id = {}
        for value, fine_label in value_to_label.items():
            coarse_label = fine_name_to_coarse_name.get(fine_label, "Outliers") 
            coarse_id = fine_name_to_coarse_id.get(fine_label, -1)
            value_to_coarse_label[value] = coarse_label
            value_to_coarse_id[value] = coarse_id
        
        # Set up coarse cluster variables for output
        hierarchical = True  # Enable coarse cluster output
        value_to_grandparent_label = value_to_coarse_label
        value_to_grandparent = value_to_coarse_id
        
        if verbose:
            n_coarse = len(set(value_to_coarse_id.values())) - (1 if -1 in value_to_coarse_id.values() else 0)
            print(f"✅ Created {n_coarse} LLM-based coarse clusters")
    
    # Create output DataFrame
    df_copy = df.copy()
    df_copy[f'{column_name}_fine_cluster_label'] = df_copy[column_name].map(value_to_label)
    df_copy[f'{column_name}_fine_cluster_id'] = df_copy[column_name].map(value_to_cluster)
    
    if hierarchical:
        df_copy[f'{column_name}_coarse_cluster_label'] = df_copy[column_name].map(value_to_grandparent_label)
        df_copy[f'{column_name}_coarse_cluster_id'] = df_copy[column_name].map(value_to_grandparent)

    if include_embeddings:
        # Use original_embeddings (full embeddings) instead of potentially reduced embeddings
        value_to_embedding = dict(zip(unique_values, original_embeddings.tolist()))
        df_copy[f'{column_name}_embedding'] = df_copy[column_name].map(value_to_embedding)
        
        # Get embeddings for cluster names
        unique_cluster_names = list(set(value_to_label.values()))
        cluster_name_embeddings = _get_embeddings(unique_cluster_names, embedding_model, verbose)
        cluster_name_to_embedding = dict(zip(unique_cluster_names, cluster_name_embeddings))
        df_copy[f'{column_name}_fine_cluster_label_embedding'] = df_copy[f'{column_name}_fine_cluster_label'].map(cluster_name_to_embedding)
    
    if verbose:
        n_clusters = len(set(initial_cluster_labels)) - (1 if -1 in initial_cluster_labels else 0)
        n_noise = list(initial_cluster_labels).count(-1)
        total_time = time.time() - start_time
        print(f"Found {n_clusters} clusters and {n_noise} outliers in {total_time:.1f} seconds")
        
        if hierarchical and grandparent_cluster_labels is not None:
            n_grandparents = len(set(grandparent_cluster_labels)) - (1 if -1 in grandparent_cluster_labels else 0)
            print(f"Created {n_grandparents} grandparent clusters")

    return df_copy


def hierarchical_cluster_categories(df, column_name, n_coarse_clusters=10, n_fine_clusters=50,
                                   embedding_model="openai", verbose=True, 
                                   include_embeddings=True, use_llm_summaries=False, 
                                   context='properties seen in AI responses',
                                   input_model_name=None):
    """
    Basic hierarchical clustering using sklearn's AgglomerativeClustering.
    
    Best for: Small to medium datasets with known cluster counts.
    
    Args:
        df: DataFrame containing the data
        column_name: Name of the column to cluster on
        n_coarse_clusters: Number of coarse clusters (default: 10)
        n_fine_clusters: Number of fine clusters (default: 50)
        embedding_model: Embedding method (default: "openai")
        verbose: Print progress (default: True)
        include_embeddings: Include embeddings in output (default: True)
        use_llm_summaries: Use LLM to generate cluster summaries (default: False)
        context: Optional context for LLM summaries (e.g., "properties seen in AI responses")
        input_model_name: Optional name of the input model being analyzed (for cache differentiation)
    """
    start_time = time.time()
    unique_values = df[column_name].unique()
    unique_strings = [str(value) for value in unique_values]
    
    if verbose:
        print(f"Hierarchical clustering for {len(unique_values)} unique values...")
    
    # Get embeddings - note: this doesn't use caching yet, could be enhanced
    embeddings = _get_embeddings(unique_strings, embedding_model, verbose)
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
    if use_llm_summaries:
        if verbose:
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
                
            summary = _get_llm_cluster_summary(values, column_name, "broad", context, 50)
            coarse_label_map[cluster_id] = summary
            
            if verbose:
                print(f"    Coarse cluster {cluster_id}: {summary} ({len(values)} items)")
        
        # Generate fine cluster summaries
        fine_label_map = {}
        for cluster_id, values in fine_cluster_values.items():
            if len(values) < 3:  # Skip very small clusters
                fine_label_map[cluster_id] = f"fine_cluster_{cluster_id}"
                continue
                
            summary = _get_llm_cluster_summary(values, column_name, "specific", context, 30)
            fine_label_map[cluster_id] = summary
            
            if verbose and cluster_id % 5 == 0:  # Print progress for every 5th cluster
                print(f"    Fine cluster {cluster_id}: {summary} ({len(values)} items)")
        
        value_to_coarse_label = {v: coarse_label_map[c] for v, c in value_to_coarse_id.items()}
        value_to_fine_label = {v: fine_label_map[c] for v, c in value_to_fine_id.items()}
    else:
        # Use generic cluster labels
        value_to_coarse_label = {v: f"coarse_cluster_{c}" for v, c in value_to_coarse_id.items()}
        value_to_fine_label = {v: f"fine_cluster_{c}" for v, c in value_to_fine_id.items()}
    
    # Create output
    df_copy = df.copy()
    df_copy[f'{column_name}_coarse_cluster_id'] = df_copy[column_name].map(value_to_coarse_id)
    df_copy[f'{column_name}_fine_cluster_id'] = df_copy[column_name].map(value_to_fine_id)
    df_copy[f'{column_name}_coarse_cluster_label'] = df_copy[column_name].map(value_to_coarse_label)
    df_copy[f'{column_name}_fine_cluster_label'] = df_copy[column_name].map(value_to_fine_label)
    
    if include_embeddings:
        value_to_embedding = dict(zip(unique_values, embeddings.tolist()))
        df_copy[f'{column_name}_embedding'] = df_copy[column_name].map(value_to_embedding)
    
    if verbose:
        print(f"Hierarchical clustering completed in {time.time() - start_time:.1f} seconds")
    
    return df_copy


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_llm_cluster_summary(values, column_name, cluster_type, context=None, sample_size=50):
    """Get LLM-based summary for a cluster using the original prompt format."""
    import random
    
    # Subsample values for the prompt
    if len(values) <= sample_size:
        sampled_vals = values
    else:
        sampled_vals = random.sample(values, sample_size)
    
    # Convert to strings and create clean prompt
    sampled_strings = [str(val) for val in sampled_vals]
    values_text = '\n'.join(sampled_strings)
    
    # Create prompt with optional context (using the original prompt format)
    # if context:
    #     if len(values) > sample_size:
    #         prompt = f"These are {sample_size} representative values from a {cluster_type} cluster of {len(values)} total values, all {context}: {values_text}. Please provide a short (up to 6 words) high-level label that best describes what these values have in common. Do NOT give labels that would apply to all clusters (e.g. 'Properties of AI responses', 'a variety of AI responses')."
    #     else:
    #         prompt = f"These values are grouped together in a {cluster_type} cluster, all {context}: {values_text}. Please provide a short (up to 6 words) high-level label that best describes what these values have in common. Do NOT give labels that would apply to all clusters (e.g. 'Properties of AI responses', 'a variety of AI responses')."
    # else:
    #     if len(values) > sample_size:
    #         prompt = f"These are {sample_size} representative values from a {cluster_type} cluster of {len(values)} total values: {values_text}. Please provide a short (up to 6 words) high-level label that best describes what these values have in common. Do NOT give labels that would apply to all clusters (e.g. 'Properties of AI responses', 'a variety of AI responses')."
    #     else:
    #         prompt = f"These values are grouped together in a {cluster_type} cluster: {values_text}. Please provide a short (up to 6 words) high-level label that best describes what these values have in common. Do NOT give labels that would apply to all clusters (e.g. 'Properties of AI responses', 'a variety of AI responses')."
    # prompt = f"These are a sample of properties seen in the responses of LLM A but not LLM B for the same prompt. I want to summarize the properties in a way that is easy to understand and use for a user. Please provide a short (up to 10 words) description that best describes most or all of the properties. Remeber that these should still be formatted as properties of model responses that could be seen in the response of model A but not model B for the same prompt. Do NOT give labels that would apply to all clusters (e.g. 'Properties of AI responses', 'a variety of AI responses') or talk about the variation within the cluster (e.g. 'variations of model tone' is not useful but 'enthusiastic tone' is if the majority of the properties mention enthusiasm). Please output your decription and nothing else. Here are the values: {values_text}"
    prompt = f"""Given a large list of properties seen in the responses of an LLM, I have clustered these properties and now want to come up with a summary of the property that each cluster represents. Below are a list of properties that all belong to the same cluster. Please come up with a clear description (up to 8 words) of a LLM output property that accurately describes most or all of the properties in the cluster. This should be a property of a model response (up to 6 words), not a category of properties.
    
For instance "Speaking Tone and Emoji Usage" is a category of properties, but "uses an enthusiastic tone" or "uses emojis" is a property of a model response. Similarily, "various types of reasoning" is a category of properties, but "uses deductive reasoning to solve problems" or "uses inductive reasoning to solve problems" is a property of a model response. Similarly, descriptions like  "Provides detailed math responses" is not informative because it could be applied to many different clusters, so it is better to describe the property in a way that is specific to the cluster and more informative, even if it does not apply to all properties in the cluster.

Think about whether a user could easily understand the models behavior at a detailed level by looking at the cluster name. 

Output the cluster property description and nothing else.

Here are the values:\n{values_text}"""
    try:
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            caching=True,
            max_tokens=100  # Keep summary responses short
        )
        # remove quotes only if they appear at start/end of string
        content = response.choices[0].message.content.strip()
        if content.startswith('"') or content.startswith("'"):
            content = content[1:]
        if content.endswith('"') or content.endswith("'"):
            content = content[:-1]
        return content
    except Exception as e:
        print(f"Warning: Failed to get LLM summary for {cluster_type} cluster: {e}")
        return f"{cluster_type}_cluster_auto"  # Fallback name

def llm_coarse_cluster_from_fine(fine_cluster_names, max_coarse_clusters=15, context=None, verbose=True, model="gpt-4.1"):
    """
    Create coarse clusters from fine cluster names using LLM-only approach.
    
    Args:
        fine_cluster_names: List of fine cluster names to group into coarse clusters
        max_coarse_clusters: Maximum number of coarse clusters to create (default: 15)
        context: Optional context for the clustering task
        verbose: Whether to print progress
        
    Returns:
        tuple: (cluster_assignments, coarse_cluster_names) where cluster_assignments maps
               each fine cluster to a coarse cluster index
    """
    if not HAS_LITELLM:
        raise ImportError("Please install: pip install litellm")
    
    if verbose:
        print(f"Creating coarse clusters from {len(fine_cluster_names)} fine cluster names using LLM...")
    
    # Filter out outlier/noise labels
    valid_fine_names = [name for name in fine_cluster_names if name not in ["Outliers", "outlier", "Noise"]]
    
    if len(valid_fine_names) <= max_coarse_clusters:
        if verbose:
            print(f"Only {len(valid_fine_names)} valid fine clusters, using 1:1 mapping")
        # If we have fewer fine clusters than desired coarse clusters, use 1:1 mapping
        assignments = list(range(len(fine_cluster_names)))
        coarse_names = fine_cluster_names
        return assignments, coarse_names
    
    fine_names_text = '\n'.join(valid_fine_names)
    
    # Create context-aware prompt
    
    if context:
        prompt = f"""Below is a list of fine-grained cluster names, each representing {context}. I want to group these into at most {max_coarse_clusters} broader, coarse-grained clusters. Please create coarse cluster names that capture the higher-level themes and group the fine clusters appropriately.

Each coarse cluster name should be a clear, descriptive label (up to 6 words) that encompasses multiple fine clusters. Think about what the most informative label would be for a user to understand the most of the items in each cluster and how they differ from other clusters. Avoid overly broad or vague labels like "includes detailed analysis" which could be applied to more than one cluster. 

Output your final coarse cluster descriptions as a list with each cluster name on a new line. Do not put numbers or bullet points in front of the cluster names. Do not include any other text in your response.

Here are the fine cluster names to group:
{fine_names_text}"""
    else:
        prompt = f"""Below is a list of fine-grained cluster names. I want to group these into at most {max_coarse_clusters} broader, coarse-grained clusters. Please create coarse cluster names that capture the higher-level themes and group the fine clusters appropriately.

    Each coarse cluster name should be a clear, descriptive label (up to 6 words) that encompasses multiple fine clusters. Think about what the most informative label would be for a user to understand the most of the items in each cluster and how they differ from other clusters. Avoid overly broad or vague labels like "includes detailed analysis" which could be applied to more than one cluster. 

Output your final coarse cluster descriptions as a list with each cluster name on a new line. Do not put numbers or bullet points in front of the cluster names. Do not include any other text in your response.

Here are the fine cluster names to group:
{fine_names_text}"""
    
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            caching=True,
            max_tokens=300
        )
        
        # Parse response into cluster names
        coarse_cluster_names = [name.strip() for name in response.choices[0].message.content.strip().split('\n') if name.strip()]
        
        if verbose:
            print(f"Generated {len(coarse_cluster_names)} coarse cluster names")
            for i, name in enumerate(coarse_cluster_names):
                print(f"  {i}: {name}")
        
        # Get embeddings for fine and coarse cluster names
        fine_embeddings = _get_embeddings(valid_fine_names, "openai", verbose=False)
        coarse_embeddings = _get_embeddings(coarse_cluster_names, "openai", verbose=False)
        
        # Calculate cosine similarities and assign fine clusters to coarse clusters
        fine_embeddings = np.array(fine_embeddings)
        coarse_embeddings = np.array(coarse_embeddings)
        
        # Normalize embeddings for cosine similarity
        fine_embeddings = fine_embeddings / np.linalg.norm(fine_embeddings, axis=1, keepdims=True)
        coarse_embeddings = coarse_embeddings / np.linalg.norm(coarse_embeddings, axis=1, keepdims=True)
        
        cosine_similarities = np.dot(fine_embeddings, coarse_embeddings.T)
        fine_to_coarse_assignments = np.argmax(cosine_similarities, axis=1)
        
        # Create full assignment list including outliers
        full_assignments = []
        valid_idx = 0
        for name in fine_cluster_names:
            if name in ["Outliers", "outlier", "Noise"]:
                full_assignments.append(-1)  # Keep outliers as outliers
            else:
                full_assignments.append(fine_to_coarse_assignments[valid_idx])
                valid_idx += 1
        
        # Add "Outliers" to coarse cluster names if needed
        if -1 in full_assignments:
            coarse_cluster_names_with_outliers = coarse_cluster_names + ["Outliers"]
        else:
            coarse_cluster_names_with_outliers = coarse_cluster_names
        
        if verbose:
            print("Fine cluster to coarse cluster mapping:")
            for fine_name, coarse_idx in zip(fine_cluster_names, full_assignments):
                if coarse_idx == -1:
                    coarse_name = "Outliers"
                else:
                    coarse_name = coarse_cluster_names[coarse_idx]
                print(f"  '{fine_name}' -> '{coarse_name}'")
        
        return full_assignments, coarse_cluster_names_with_outliers
        
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to generate LLM-based coarse clusters: {e}")
        # Fallback: create simple numeric coarse clusters
        assignments = [i % max_coarse_clusters for i in range(len(fine_cluster_names))]
        coarse_names = [f"coarse_cluster_{i}" for i in range(max_coarse_clusters)]
        return assignments, coarse_names

def _setup_embeddings_with_cache(texts, embedding_model, column_name, cache_embeddings, verbose=False, input_model_name=None):
    """Setup embeddings with caching support."""
    # Create cache filename - include input_model_name if provided to avoid collisions
    if input_model_name:
        # Clean the model name for filename usage
        clean_model_name = input_model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        cache_filename = f"embeddings_cache_{column_name}_{embedding_model}_{clean_model_name}_{len(texts)}.pkl"
    else:
        cache_filename = f"embeddings_cache_{column_name}_{embedding_model}_{len(texts)}.pkl"
    
    # Try to load from cache first
    if cache_embeddings and os.path.exists(cache_filename):
        if verbose:
            print(f"Loading cached embeddings from {cache_filename}...")
        try:
            with open(cache_filename, 'rb') as f:
                cached_data = pickle.load(f)
                embeddings = cached_data['embeddings']
                embedding_model_obj = cached_data.get('embedding_model_obj', None)
                if verbose:
                    print(f"✅ Loaded {len(embeddings)} cached embeddings!")
                return embeddings, embedding_model_obj
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to load cache: {e}. Computing fresh embeddings...")
    print(f"saving embeddings to cache: {cache_filename}")
    
    # Compute embeddings if not cached
    embeddings, embedding_model_obj = _setup_embeddings(texts, embedding_model, verbose)
    
    # Save to cache
    if cache_embeddings:
        if verbose:
            print(f"💾 Saving embeddings to cache: {cache_filename}...")
        try:
            cache_data = {
                'embeddings': embeddings,
                'embedding_model_obj': embedding_model_obj,
                'embedding_model': embedding_model,
                'input_model_name': input_model_name,
                'num_texts': len(texts)
            }
            with open(cache_filename, 'wb') as f:
                pickle.dump(cache_data, f)
            if verbose:
                cache_size = os.path.getsize(cache_filename) / (1024**2)
                print(f"✅ Cached embeddings saved ({cache_size:.1f} MB)")
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to save cache: {e}")
    
    return embeddings, embedding_model_obj


def _setup_embeddings(texts, embedding_model, verbose=False):
    """Setup embeddings based on model type."""
    if embedding_model == "openai":
        if verbose:
            print("Using OpenAI embeddings...")
        embeddings = _get_openai_embeddings(texts)
        embeddings = np.array(embeddings)
        # Normalize embeddings
        embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)
        return embeddings, None
    else:
        if verbose:
            print(f"Using sentence transformer: {embedding_model}")
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("Please install: pip install sentence-transformers")
        model = SentenceTransformer(embedding_model)
        return None, model


def _get_embeddings(texts, embedding_model, verbose=False):
    """Get embeddings for texts."""
    if embedding_model == "openai":
        return _get_openai_embeddings(texts)
    else:
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("Please install: pip install sentence-transformers")
        if verbose:
            print(f"Computing embeddings with {embedding_model}...")
        model = SentenceTransformer(embedding_model)
        return model.encode(texts, show_progress_bar=verbose).tolist()


def _get_openai_embeddings(texts, batch_size=100):
    """Get embeddings using OpenAI API."""
    if not HAS_LITELLM:
        raise ImportError("Please install: pip install litellm")
    
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = litellm.embedding(
            model="text-embedding-3-small",
            input=batch,
            caching=True
        )
        batch_embeddings = [item['embedding'] for item in response['data']]
        embeddings.extend(batch_embeddings)
    return embeddings


def _create_bertopic_model(embedding_model_obj, unique_values, max_topics, min_cluster_size, verbose):
    """Create configured BERTopic model."""
    umap_model = umap.UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, 
        metric='cosine', random_state=42
    )
    
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=max(min_cluster_size, len(unique_values) // max_topics),
        min_samples=min(30, min_cluster_size),
        metric='euclidean'
    )
    
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2), stop_words="english", 
        min_df=2, max_features=5000
    )
    
    return BERTopic(
        embedding_model=embedding_model_obj,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=max_topics,
        verbose=verbose
    )


def _create_fine_topics(unique_values, coarse_topics, embedding_model_obj, embeddings,
                       min_topic_size, max_fine_topics_per_coarse, verbose):
    """Create fine-grained topics within each coarse topic."""
    coarse_topic_groups = defaultdict(list)
    coarse_topic_indices = defaultdict(list)
    
    for i, (value, topic) in enumerate(zip(unique_values, coarse_topics)):
        coarse_topic_groups[topic].append(str(value))
        coarse_topic_indices[topic].append(i)
    
    fine_topics = [-1] * len(unique_values)
    fine_topic_labels = {-1: "Outliers"}
    fine_topic_counter = 0
    
    for coarse_topic_id, documents in coarse_topic_groups.items():
        if coarse_topic_id == -1 or len(documents) < min_topic_size:
            continue
            
        if verbose:
            print(f"  Processing coarse topic {coarse_topic_id} ({len(documents)} documents)...")
        
        # Create fine-level model
        fine_model = _create_fine_bertopic_model(embedding_model_obj, documents, max_fine_topics_per_coarse)
        
        # Get fine topics
        if embeddings is not None:
            subset_indices = coarse_topic_indices[coarse_topic_id]
            subset_embeddings = np.array([embeddings[i] for i in subset_indices])  # Convert to numpy array
            fine_assignments, _ = fine_model.fit_transform(documents, subset_embeddings)
        else:
            fine_assignments, _ = fine_model.fit_transform(documents)
        
        # Map back to original indices
        for local_idx, fine_topic in enumerate(fine_assignments):
            original_idx = coarse_topic_indices[coarse_topic_id][local_idx]
            if fine_topic != -1:
                global_fine_topic_id = fine_topic_counter
                fine_topics[original_idx] = global_fine_topic_id
                
                # Get topic label
                try:
                    topic_words = fine_model.get_topic(fine_topic)
                    if topic_words:
                        top_words = [word for word, _ in topic_words[:3]]
                        fine_topic_labels[global_fine_topic_id] = f"Subtopic: {', '.join(top_words)}"
                except:
                    fine_topic_labels[global_fine_topic_id] = f"fine_topic_{global_fine_topic_id}"
                
                fine_topic_counter += 1
    
    return fine_topics, fine_topic_labels


def _create_fine_bertopic_model(embedding_model_obj, documents, max_topics):
    """Create BERTopic model for fine-level clustering."""
    n_docs = len(documents)
    
    fine_umap = umap.UMAP(
        n_neighbors=min(15, n_docs//2),
        n_components=min(5, n_docs//3),
        min_dist=0.0, metric='cosine', random_state=42
    )
    
    fine_hdbscan = hdbscan.HDBSCAN(
        min_cluster_size=max(5, n_docs // max_topics),
        min_samples=5, metric='euclidean'
    )
    
    fine_vectorizer = CountVectorizer(
        ngram_range=(1, 2), stop_words="english", 
        min_df=1, max_features=1000
    )
    
    return BERTopic(
        embedding_model=embedding_model_obj,
        umap_model=fine_umap,
        hdbscan_model=fine_hdbscan,
        vectorizer_model=fine_vectorizer,
        nr_topics=max_topics,
        verbose=False
    )


def _create_bertopic_output(df, column_name, unique_values, coarse_topics, fine_topics,
                           coarse_topic_info, fine_topic_labels, embeddings, include_embeddings):
    """Create final output DataFrame with BERTopic results."""
    # Get coarse topic labels
    coarse_topic_labels = {-1: "Outliers"}
    for _, row in coarse_topic_info.iterrows():
        coarse_topic_labels[row['Topic']] = row['Name']
    
    # Create mappings
    value_to_coarse_topic = dict(zip(unique_values, coarse_topics))
    value_to_fine_topic = dict(zip(unique_values, fine_topics))
    
    value_to_coarse_label = {
        value: coarse_topic_labels.get(topic, f"topic_{topic}")
        for value, topic in value_to_coarse_topic.items()
    }
    
    value_to_fine_label = {
        value: fine_topic_labels.get(topic, f"topic_{topic}")
        for value, topic in value_to_fine_topic.items()
    }
    
    # Create output DataFrame
    df_copy = df.copy()
    df_copy[f'{column_name}_coarse_topic_label'] = df_copy[column_name].map(value_to_coarse_label)
    df_copy[f'{column_name}_fine_topic_label'] = df_copy[column_name].map(value_to_fine_label)
    df_copy[f'{column_name}_coarse_topic_id'] = df_copy[column_name].map(value_to_coarse_topic)
    df_copy[f'{column_name}_fine_topic_id'] = df_copy[column_name].map(value_to_fine_topic)
    
    if include_embeddings and embeddings is not None:
        value_to_embedding = dict(zip(unique_values, embeddings))
        df_copy[f'{column_name}_embedding'] = df_copy[column_name].map(value_to_embedding)

    return df_copy


def _print_topic_summary(coarse_topic_info, fine_topic_labels, column_name):
    """Print summary of clustering results."""
    n_coarse = len([t for t in coarse_topic_info['Topic'] if t != -1])
    n_fine = len([t for t in fine_topic_labels.keys() if t != -1])
    
    print(f"\nClustering Summary for '{column_name}':")
    print(f"  Coarse topics: {n_coarse}")
    print(f"  Fine topics: {n_fine}")
    print(f"  Total unique values processed: {coarse_topic_info['Count'].sum()}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_clustered_results(parquet_path):
    """Load previously clustered results from parquet file."""
    df = pd.read_parquet(parquet_path)
    
    print(f"Loaded {len(df)} rows from {parquet_path}")
    cluster_cols = [col for col in df.columns if 'cluster' in col.lower() or 'topic' in col.lower()]
    embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
    
    if cluster_cols:
        print(f"Cluster columns: {cluster_cols}")
    if embedding_cols:
        print(f"Embedding columns: {embedding_cols}")
    
    return df


def save_clustered_results(df, base_filename, include_embeddings=True):
    """Save clustered results in multiple formats."""

    os.makedirs(f"cluster_results/{base_filename}", exist_ok=True)
    
    # Convert problematic columns to strings
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    
    # Save full results with embeddings
    if include_embeddings:
        full_path = f"cluster_results/{base_filename}/{base_filename}_with_embeddings.parquet"
        df.to_parquet(full_path, compression='snappy')
        print(f"Saved full results to: {full_path}")
    
    # Save lightweight version without embeddings
    embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
    df_light = df.drop(columns=embedding_cols) if embedding_cols else df
    
    light_parquet = f"cluster_results/{base_filename}/{base_filename}_lightweight.parquet"
    light_csv_gz = f"cluster_results/{base_filename}/{base_filename}.csv.gz"
    light_jsonl = f"cluster_results/{base_filename}/{base_filename}.jsonl"
    
    df_light.to_parquet(light_parquet, compression='snappy')
    df_light.to_csv(light_csv_gz, index=False, compression='gzip')
    df_light.to_json(light_jsonl, lines=True, orient="records")
    
    print(f"Saved lightweight results to: {light_parquet}, {light_csv_gz}, {light_jsonl}")
    
    # Print file sizes
    if include_embeddings:
        full_size = os.path.getsize(full_path) / (1024**2)
        print(f"  Full dataset: {full_size:.1f} MB")
    
    light_size = os.path.getsize(light_parquet) / (1024**2)
    csv_gz_size = os.path.getsize(light_csv_gz) / (1024**2)
    print(f"  Lightweight: {light_size:.1f} MB (parquet), {csv_gz_size:.1f} MB (csv.gz)")


def load_precomputed_embeddings(embeddings_path, verbose=True):
    """Load precomputed embeddings from various file formats."""
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    if verbose:
        print(f"Loading precomputed embeddings from {embeddings_path}...")
    
    if embeddings_path.endswith('.pkl'):
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                # Check if it's a cache file with 'embeddings' key
                if 'embeddings' in data:
                    embeddings = data['embeddings']
                    if verbose:
                        print(f"Loaded {len(embeddings)} embeddings from cache file")
                else:
                    # Assume it's a direct mapping of values to embeddings
                    embeddings = data
                    if verbose:
                        print(f"Loaded {len(embeddings)} embeddings from mapping file")
            else:
                # Assume it's a direct array/list of embeddings
                embeddings = data
                if verbose:
                    print(f"Loaded {len(embeddings)} embeddings from array file")
    
    elif embeddings_path.endswith('.npy'):
        embeddings = np.load(embeddings_path)
        if verbose:
            print(f"Loaded {len(embeddings)} embeddings from numpy file")
    
    elif embeddings_path.endswith('.parquet'):
        # Load from parquet file with embedding column
        if verbose:
            print("Loading parquet file...")
        df = pd.read_parquet(embeddings_path)
        
        embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
        if not embedding_cols:
            raise ValueError(f"No embedding columns found in {embeddings_path}")
        
        embedding_col = embedding_cols[0]  # Use first embedding column
        
        # Find the column that was clustered (should be the base name of embedding column)
        base_col = embedding_col.replace('_embedding', '')
        if base_col not in df.columns:
            # Try to find any text column that might be the source
            text_cols = [col for col in df.columns if col not in embedding_cols and 
                        df[col].dtype == 'object']
            if text_cols:
                base_col = text_cols[0]
                if verbose:
                    print(f"Using column '{base_col}' as source column")
            else:
                raise ValueError(f"Cannot find source text column in {embeddings_path}")
        
        if verbose:
            print(f"Creating value-to-embedding mapping from column '{base_col}'...")
        
        # Create mapping from values to embeddings
        embeddings = {}
        for _, row in df.iterrows():
            value = str(row[base_col])
            embedding = row[embedding_col]
            embeddings[value] = embedding
        
        if verbose:
            print(f"Loaded {len(embeddings)} embeddings from parquet file (column: {embedding_col})")
    
    else:
        raise ValueError(f"Unsupported file format: {embeddings_path}. Supported: .pkl, .npy, .parquet")
    
    return embeddings


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
    parser.add_argument('--method', '-m', choices=['bertopic', 'hdbscan', 'hierarchical'], 
                       default='hdbscan',
                       help='Clustering method (default: hdbscan)')
    parser.add_argument('--min-cluster-size', type=int, default=10,
                       help='Minimum cluster size (default: 15)')
    parser.add_argument('--embedding-model', default='openai',
                       help='Embedding model: openai, all-MiniLM-L6-v2, etc. (default: openai)')
    parser.add_argument('--output', '-o', 
                       help='Output filename prefix (default: auto-generated)')
    parser.add_argument('--no-embeddings', action='store_true',
                       help='Exclude embeddings from output')
    parser.add_argument('--no-llm-summaries', action='store_true',
                       help='Disable LLM-based cluster summaries')
    parser.add_argument('--context', default='properties seen in AI responses',
                       help='Context for LLM summaries (default: "properties seen in AI responses")')
    parser.add_argument('--max-coarse-topics', type=int, default=40,
                       help='Max coarse topics for BERTopic (default: 40)')
    parser.add_argument('--max-fine-topics', type=int, default=50,
                       help='Max fine topics per coarse topic for BERTopic (default: 20)')
    parser.add_argument('--precomputed-embeddings', 
                       help='Path to precomputed embeddings file (.pkl or .npy)')
    parser.add_argument('--enable-dim-reduction', action='store_true',
                       help='Enable UMAP dimensionality reduction (default: False)')
    parser.add_argument('--assign-outliers', action='store_true',
                       help='Assign HDBSCAN outliers to their nearest clusters (default: False)')
    parser.add_argument('--hierarchical', action='store_true',
                       help='Enable hierarchical HDBSCAN clustering (cluster the clusters) (default: False)')
    parser.add_argument('--min-grandparent-size', type=int, default=5,
                       help='Minimum size for grandparent clusters in hierarchical mode (default: 5)')
    parser.add_argument('--use-llm-coarse-clustering', action='store_true',
                       help='Use LLM-only approach to create coarse clusters from fine cluster names (default: False)')
    parser.add_argument('--max-coarse-clusters', type=int, default=15,
                       help='Maximum coarse clusters when using LLM coarse clustering (default: 15)')
    parser.add_argument('--input-model-name', 
                       help='Name of the input model being analyzed (for cache differentiation)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.file}...")
    df = pd.read_json(args.file, lines=True)
    
    if args.column not in df.columns:
        print(f"Error: Column '{args.column}' not found in data. Available columns: {list(df.columns)}")
        return None
    
    print(f"Loaded {len(df)} rows with {len(df[args.column].unique())} unique values in '{args.column}'")
    
    # Set up parameters
    include_embeddings = not args.no_embeddings
    use_llm_summaries = not args.no_llm_summaries
    
    # Load precomputed embeddings if provided
    precomputed_embeddings = None
    if args.precomputed_embeddings:
        precomputed_embeddings = load_precomputed_embeddings(args.precomputed_embeddings, verbose=True)
    
    # Run clustering based on method
    if args.method == 'bertopic':
        print(f"Running BERTopic hierarchical clustering...")
        df_clustered = bertopic_hierarchical_cluster_categories(
            df, args.column,
            min_cluster_size=args.min_cluster_size,
            max_coarse_topics=args.max_coarse_topics,
            max_fine_topics_per_coarse=args.max_fine_topics,
            embedding_model=args.embedding_model,
            verbose=True,
            include_embeddings=include_embeddings,
            use_llm_summaries=use_llm_summaries,
            context=args.context,
            use_llm_coarse_clustering=args.use_llm_coarse_clustering,
            max_coarse_clusters=args.max_coarse_clusters,
            input_model_name=args.input_model_name
        )
        method_name = "bertopic"
        
    elif args.method == 'hdbscan':
        print(f"Running HDBSCAN clustering...")
        df_clustered = hdbscan_cluster_categories(
            df, args.column,
            min_cluster_size=args.min_cluster_size,
            embedding_model=args.embedding_model,
            verbose=True,
            include_embeddings=include_embeddings,
            use_llm_summaries=use_llm_summaries,
            context=args.context,
            precomputed_embeddings=precomputed_embeddings,
            enable_dim_reduction=args.enable_dim_reduction,
            assign_outliers=args.assign_outliers,
            hierarchical=args.hierarchical,
            min_grandparent_size=args.min_grandparent_size,
            use_llm_coarse_clustering=args.use_llm_coarse_clustering,
            max_coarse_clusters=args.max_coarse_clusters,
            input_model_name=args.input_model_name
        )
        method_name = "hdbscan"
        
    elif args.method == 'hierarchical':
        print(f"Running traditional hierarchical clustering...")
        df_clustered = hierarchical_cluster_categories(
            df, args.column,
            embedding_model=args.embedding_model,
            verbose=True,
            include_embeddings=include_embeddings,
            use_llm_summaries=use_llm_summaries,
            context=args.context,
            input_model_name=args.input_model_name
        )
        method_name = "hierarchical"
    
    # Generate output filename
    if args.output:
        output_prefix = args.output
    else:
        input_basename = os.path.splitext(os.path.basename(args.file))[0]
        output_prefix = f"{input_basename}_{method_name}_clustered"
    
    # Save results
    save_clustered_results(df_clustered, output_prefix, include_embeddings=include_embeddings)
    
    print(f"\n✅ Clustering complete! Final dataset shape: {df_clustered.shape}")
    return df_clustered


if __name__ == "__main__":
    df_result = main() 

def _generate_llm_summaries(unique_values, coarse_topics, fine_topics, coarse_topic_info, 
                           fine_topic_labels, column_name, context, verbose, sample_size=50):
    """Generate LLM-based summaries for clusters using custom prompts."""
    if not HAS_LITELLM:
        raise ImportError("Please install: pip install litellm")
    
    import random
    
    # Group values by coarse topics
    coarse_cluster_values = defaultdict(list)
    for value, topic in zip(unique_values, coarse_topics):
        coarse_cluster_values[topic].append(value)
    
    # Group values by fine topics  
    fine_cluster_values = defaultdict(list)
    for value, topic in zip(unique_values, fine_topics):
        fine_cluster_values[topic].append(value)
    
    # Generate coarse topic summaries
    if verbose:
        print("  Generating coarse topic summaries...")
    
    new_coarse_names = {}
    for topic_id in coarse_cluster_values.keys():
        if topic_id == -1:
            new_coarse_names[topic_id] = "Outliers"
            continue
            
        values = coarse_cluster_values[topic_id]
        summary = _get_llm_cluster_summary(values, column_name, "broad", context, sample_size)
        new_coarse_names[topic_id] = summary
        
        if verbose:
            print(f"    Topic {topic_id}: {summary} ({len(values)} items)")
    
    # Update coarse topic info
    updated_coarse_info = coarse_topic_info.copy()
    for idx, row in updated_coarse_info.iterrows():
        topic_id = row['Topic']
        if topic_id in new_coarse_names:
            updated_coarse_info.at[idx, 'Name'] = new_coarse_names[topic_id]
    
    # Generate fine topic summaries
    if verbose:
        print("  Generating fine topic summaries...")
    
    updated_fine_labels = {}
    for topic_id in fine_cluster_values.keys():
        if topic_id == -1:
            updated_fine_labels[topic_id] = "Outliers"
            continue
            
        values = fine_cluster_values[topic_id]
        if len(values) < 5:  # Skip very small clusters
            updated_fine_labels[topic_id] = f"fine_topic_{topic_id}"
            continue
            
        summary = _get_llm_cluster_summary(values, column_name, "specific", context, sample_size)
        updated_fine_labels[topic_id] = summary
        
        if verbose and topic_id % 10 == 0:  # Print progress for every 10th topic
            print(f"    Fine topic {topic_id}: {summary} ({len(values)} items)")
    
    return updated_coarse_info, updated_fine_labels 

# def llm_coarse_cluster_from_fine(fine_cluster_names, max_coarse_clusters=15, context=None, verbose=True):
#     """
#     Create coarse clusters from fine cluster names using LLM-only approach.
    
#     Args:
#         fine_cluster_names: List of fine cluster names to group into coarse clusters
#         max_coarse_clusters: Maximum number of coarse clusters to create (default: 15)
#         context: Optional context for the clustering task
#         verbose: Whether to print progress
        
#     Returns:
#         tuple: (cluster_assignments, coarse_cluster_names) where cluster_assignments maps
#                each fine cluster to a coarse cluster index
#     """
#     if not HAS_LITELLM:
#         raise ImportError("Please install: pip install litellm")
    
#     if verbose:
#         print(f"Creating coarse clusters from {len(fine_cluster_names)} fine cluster names using LLM...")
    
#     # Filter out outlier/noise labels
#     valid_fine_names = [name for name in fine_cluster_names if name not in ["Outliers", "outlier", "Noise"]]
    
#     if len(valid_fine_names) <= max_coarse_clusters:
#         if verbose:
#             print(f"Only {len(valid_fine_names)} valid fine clusters, using 1:1 mapping")
#         # If we have fewer fine clusters than desired coarse clusters, use 1:1 mapping
#         assignments = list(range(len(fine_cluster_names)))
#         coarse_names = fine_cluster_names
#         return assignments, coarse_names
    
#     fine_names_text = '\n'.join(valid_fine_names)
    
#     # Create context-aware prompt
#     if context:
#         prompt = f"""Below is a list of fine-grained cluster names, each representing {context}. I want to group these into at most {max_coarse_clusters} broader, coarse-grained clusters. Please create coarse cluster names that capture the higher-level themes and group the fine clusters appropriately.

# Each coarse cluster name should be a clear, descriptive label (up to 8 words) that encompasses multiple fine clusters. Think about what the most informative label would be for a user to understand the most of the items in the cluster. Avoid overly broad or vague labels like "includes detailed analysis". 

# Output your final coarse cluster descriptions as a list with each cluster name on a new line. Do not put numbers or bullet points in front of the cluster names. Do not include any other text in your response.

# Here are the fine cluster names to group:
# {fine_names_text}"""
#     else:
#         prompt = f"""Below is a list of fine-grained cluster names. I want to group these into at most {max_coarse_clusters} broader, coarse-grained clusters. Please create coarse cluster names that capture the higher-level themes and group the fine clusters appropriately.

# Each coarse cluster name should be a clear, descriptive label (up to 8 words) that encompasses multiple fine clusters. Think about the broader categories or themes that emerge from the fine clusters.

# Output your final coarse cluster descriptions as a list with each cluster name on a new line. Do not put numbers or bullet points in front of the cluster names. Do not include any other text in your response.

# Here are the fine cluster names to group:
# {fine_names_text}"""
    
#     try:
#         response = litellm.completion(
#             model="gpt-4.1",
#             messages=[{"role": "user", "content": prompt}],
#             caching=True,
#             max_tokens=300
#         )
        
#         # Parse response into cluster names
#         coarse_cluster_names = [name.strip() for name in response.choices[0].message.content.strip().split('\n') if name.strip()]
        
#         if verbose:
#             print(f"Generated {len(coarse_cluster_names)} coarse cluster names")
#             for i, name in enumerate(coarse_cluster_names):
#                 print(f"  {i}: {name}")
        
#         # Get embeddings for fine and coarse cluster names
#         fine_embeddings = _get_embeddings(valid_fine_names, "openai", verbose=False)
#         coarse_embeddings = _get_embeddings(coarse_cluster_names, "openai", verbose=False)
        
#         # Calculate cosine similarities and assign fine clusters to coarse clusters
#         fine_embeddings = np.array(fine_embeddings)
#         coarse_embeddings = np.array(coarse_embeddings)
        
#         # Normalize embeddings for cosine similarity
#         fine_embeddings = fine_embeddings / np.linalg.norm(fine_embeddings, axis=1, keepdims=True)
#         coarse_embeddings = coarse_embeddings / np.linalg.norm(coarse_embeddings, axis=1, keepdims=True)
        
#         cosine_similarities = np.dot(fine_embeddings, coarse_embeddings.T)
#         fine_to_coarse_assignments = np.argmax(cosine_similarities, axis=1)
        
#         # Create full assignment list including outliers
#         full_assignments = []
#         valid_idx = 0
#         for name in fine_cluster_names:
#             if name in ["Outliers", "outlier", "Noise"]:
#                 full_assignments.append(-1)  # Keep outliers as outliers
#             else:
#                 full_assignments.append(fine_to_coarse_assignments[valid_idx])
#                 valid_idx += 1
        
#         # Add "Outliers" to coarse cluster names if needed
#         if -1 in full_assignments:
#             coarse_cluster_names_with_outliers = coarse_cluster_names + ["Outliers"]
#         else:
#             coarse_cluster_names_with_outliers = coarse_cluster_names
        
#         if verbose:
#             print("Fine cluster to coarse cluster mapping:")
#             for fine_name, coarse_idx in zip(fine_cluster_names, full_assignments):
#                 if coarse_idx == -1:
#                     coarse_name = "Outliers"
#                 else:
#                     coarse_name = coarse_cluster_names[coarse_idx]
#                 print(f"  '{fine_name}' -> '{coarse_name}'")
        
#         return full_assignments, coarse_cluster_names_with_outliers
        
#     except Exception as e:
#         if verbose:
#             print(f"Warning: Failed to generate LLM-based coarse clusters: {e}")
#         # Fallback: create simple numeric coarse clusters
#         assignments = [i % max_coarse_clusters for i in range(len(fine_cluster_names))]
#         coarse_names = [f"coarse_cluster_{i}" for i in range(max_coarse_clusters)]
#         return assignments, coarse_names


def llm_only_cluster_summaries(df, column_name, max_clusters=30, context=None, verbose=True):
    """
    Get LLM-based cluster summaries for a column - improved version.
    
    Args:
        df: DataFrame containing the data
        column_name: Name of the column to cluster on
        max_clusters: Maximum number of clusters to create (default: 30)
        context: Optional context describing what the values represent
        verbose: Whether to print progress
        
    Returns:
        tuple: (cluster_assignments, cluster_names) where cluster_assignments maps
               each unique value to a cluster index and cluster_names contains the cluster labels
    """
    if not HAS_LITELLM:
        raise ImportError("Please install: pip install litellm")
        
    unique_values = df[column_name].unique()
    unique_strings = [str(value) for value in unique_values]
    values_text = '\n'.join(unique_strings)
    
    if context:
        prompt = f"""Below is a list of {context}. I want to cluster these into at most {max_clusters} clusters. Please come up with cluster names, where each cluster name is a clear description (up to 6 words) that accurately describes most or all of the items in the cluster.

Think about the natural groupings and themes that emerge from the data. Each cluster should provide a unique insight that is not already provided by the other clusters. Avoid overly broad categories that could apply to most items.

Output your final cluster descriptions as a list with each cluster name on a new line. Do not put numbers or bullet points in front of the cluster names. Do not include any other text in your response.

Here are the values: {values_text}"""
    else:
        prompt = f"""Below is a list of values that I want to cluster into at most {max_clusters} clusters. Please come up with cluster names, where each cluster name is a clear description (up to 6 words) that accurately describes most or all of the items in the cluster.

Think about the natural groupings and themes that emerge from the data. Each cluster should provide a unique insight that is not already provided by the other clusters.

Output your final cluster descriptions as a list with each cluster name on a new line. Do not put numbers or bullet points in front of the cluster names. Do not include any other text in your response.

Here are the values: {values_text}"""
    
    try:
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            caching=True,
            max_tokens=400
        )
        
        # Parse the response into a list of cluster names
        cluster_names = [name.strip() for name in response.choices[0].message.content.strip().split('\n') if name.strip()]
        
        if verbose:
            print(f"Generated {len(cluster_names)} cluster names")
            for i, name in enumerate(cluster_names):
                print(f"  {i}: {name}")
        
        # Embed the strings and the cluster names and assign them based on cosine similarity
        embeddings = _get_embeddings(unique_strings, "openai", verbose)
        cluster_embeddings = _get_embeddings(cluster_names, "openai", verbose)
        
        embeddings = np.array(embeddings)
        cluster_embeddings = np.array(cluster_embeddings)
        
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        cluster_embeddings = cluster_embeddings / np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
        
        cosine_similarities = np.dot(embeddings, cluster_embeddings.T)
        cluster_assignments = np.argmax(cosine_similarities, axis=1)
        
        return cluster_assignments, cluster_names
        
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to generate LLM-based clusters: {e}")
        # Fallback to simple clustering
        cluster_assignments = [0] * len(unique_values)
        cluster_names = ["default_cluster"]
        return cluster_assignments, cluster_names 
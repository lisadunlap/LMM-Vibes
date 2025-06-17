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

def bertopic_hierarchical_cluster_categories(df, column_name, min_cluster_size=30, min_topic_size=10,
                                           max_coarse_topics=25, max_fine_topics_per_coarse=20,
                                           verbose=True, embedding_model="openai", 
                                           include_embeddings=True, cache_embeddings=True,
                                           use_llm_summaries=False, context=None):
    """
    RECOMMENDED: Hierarchical clustering using BERTopic with HDBSCAN.
    
    Creates natural topic hierarchies with automatic keyword-based labeling.
    
    Args:
        df: DataFrame containing the data
        column_name: Name of the column to cluster on
        min_cluster_size: Minimum cluster size for HDBSCAN (default: 30)
        min_topic_size: Minimum topic size for BERTopic (default: 10)
        max_coarse_topics: Maximum broad topics to extract (default: 25)
        max_fine_topics_per_coarse: Maximum subtopics per broad topic (default: 20)
        verbose: Whether to print progress (default: True)
        embedding_model: "openai", "all-MiniLM-L6-v2", or "all-mpnet-base-v2" (default: "openai")
        include_embeddings: Include embeddings in output (default: True)
        cache_embeddings: Save/load embeddings from disk (default: True)
        use_llm_summaries: Use LLM to generate cluster summaries instead of keywords (default: False)
        context: Optional context for LLM summaries (e.g., "properties seen in AI responses")
    
    Returns:
        DataFrame with new columns:
        - {column_name}_coarse_topic_label: Broad topic labels
        - {column_name}_fine_topic_label: Specific subtopic labels
        - {column_name}_coarse_topic_id: Numeric coarse topic IDs
        - {column_name}_fine_topic_id: Numeric fine topic IDs
        - {column_name}_embedding: Vector embeddings (if include_embeddings=True)
    """
    if not HAS_BERTOPIC:
        raise ImportError("Please install: pip install bertopic sentence-transformers hdbscan umap-learn")
    
    start_time = time.time()
    unique_values = df[column_name].unique()
    unique_strings = [str(value) for value in unique_values]
    
    if verbose:
        print(f"BERTopic hierarchical clustering for {len(unique_values)} unique values...")
    
    # Get embeddings (with caching)
    embeddings, embedding_model_obj = _setup_embeddings_with_cache(
        unique_strings, embedding_model, column_name, cache_embeddings, verbose
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
    
    # Step 3: Optional LLM-based summarization
    if use_llm_summaries:
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
                              assign_outliers=False, hierarchical=False, min_grandparent_size=5):
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
                                   include_embeddings=True):
    """
    Traditional agglomerative hierarchical clustering.
    
    Best for: Small datasets (<10k unique values), exact cluster counts.
    
    Args:
        df: DataFrame containing the data
        column_name: Name of the column to cluster on
        n_coarse_clusters: Number of broad clusters (default: 10)
        n_fine_clusters: Number of specific clusters (default: 50)
        embedding_model: Embedding method (default: "openai")
        verbose: Print progress (default: True)
        include_embeddings: Include embeddings in output (default: True)
    """
    start_time = time.time()
    unique_values = df[column_name].unique()
    unique_strings = [str(value) for value in unique_values]
    
    if verbose:
        print(f"Agglomerative clustering for {len(unique_values)} unique values...")
    
    # Get embeddings
    embeddings = _get_embeddings(unique_strings, embedding_model, verbose)
    embeddings = np.array(embeddings)
    
    # Clustering
    coarse_clustering = AgglomerativeClustering(n_clusters=n_coarse_clusters, linkage='ward')
    fine_clustering = AgglomerativeClustering(n_clusters=n_fine_clusters, linkage='ward')
    
    coarse_clusters = coarse_clustering.fit_predict(embeddings)
    fine_clusters = fine_clustering.fit_predict(embeddings)
    
    # Create output
    df_copy = df.copy()
    df_copy[f'{column_name}_coarse_cluster_id'] = df_copy[column_name].map(dict(zip(unique_values, coarse_clusters)))
    df_copy[f'{column_name}_fine_cluster_id'] = df_copy[column_name].map(dict(zip(unique_values, fine_clusters)))
    df_copy[f'{column_name}_coarse_cluster_label'] = df_copy[f'{column_name}_coarse_cluster_id'].apply(lambda x: f"coarse_cluster_{x}")
    df_copy[f'{column_name}_fine_cluster_label'] = df_copy[f'{column_name}_fine_cluster_id'].apply(lambda x: f"fine_cluster_{x}")
    
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
    prompt = f"""Given a large list of properties seen in the responses of an LLM, I have clustered these properties and now want to come up with a summary of the property that each cluster represents. Below are a list of properties that all belong to the same cluster. Please come up with a clear description (up to 8 words) of a LLM output property that accurately describes most or all of the properties in the cluster. This should be a property of a model response, not a category of properties. For instance "Speaking Tone and Emoji Usage" is a category of properties, but "uses an enthusiastic tone" or "uses emojis" is a property of a model response. Similarily, "various types of reasoning" is a category of properties, but "uses deductive reasoning to solve problems" or "uses inductive reasoning to solve problems" is a property of a model response. Think about whether a user could easily understand the models behavior at a detailed level by looking at the cluster name. 

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


def _setup_embeddings_with_cache(texts, embedding_model, column_name, cache_embeddings, verbose=False):
    """Setup embeddings with caching support."""
    # Create cache filename
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
    # Save full results with embeddings
    if include_embeddings:
        full_path = f"{base_filename}_with_embeddings.parquet"
        df.to_parquet(full_path, compression='snappy')
        print(f"Saved full results to: {full_path}")
    
    # Save lightweight version without embeddings
    embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
    df_light = df.drop(columns=embedding_cols) if embedding_cols else df
    
    light_parquet = f"{base_filename}_lightweight.parquet"
    light_csv_gz = f"{base_filename}.csv.gz"
    light_jsonl = f"{base_filename}.jsonl"
    
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
    parser.add_argument('--min-cluster-size', type=int, default=30,
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
    parser.add_argument('--max-fine-topics', type=int, default=20,
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
            context=args.context
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
            min_grandparent_size=args.min_grandparent_size
        )
        method_name = "hdbscan"
        
    elif args.method == 'hierarchical':
        print(f"Running traditional hierarchical clustering...")
        df_clustered = hierarchical_cluster_categories(
            df, args.column,
            embedding_model=args.embedding_model,
            verbose=True,
            include_embeddings=include_embeddings
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
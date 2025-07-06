#!/usr/bin/env python3
"""
Optimized category clustering with deduplication approach.
"""

import pandas as pd
import numpy as np
from clustering.hierarchical_clustering import hdbscan_cluster_categories, ClusterConfig

def dedupe_cluster_and_map(df, column_name, config=None, **clustering_kwargs):
    """
    Optimized clustering approach:
    1. Deduplicate categories to unique values
    2. Cluster the unique categories 
    3. Map cluster results back to original dataframe
    
    Args:
        df: DataFrame with potentially duplicated categories
        column_name: Name of the category column to cluster
        config: ClusterConfig object with all parameters (preferred)
        **clustering_kwargs: Arguments to pass to clustering function (for backward compatibility)
        
    Returns:
        tuple: (df_with_clusters, unique_categories_clustered)
    """
    print(f"🔍 Analyzing {column_name} column...")
    
    # Analyze the data
    total_rows = len(df)
    unique_values = df[column_name].nunique()
    reduction_ratio = total_rows / unique_values
    
    print(f"  Total rows: {total_rows:,}")
    print(f"  Unique {column_name}: {unique_values:,}")
    print(f"  Reduction ratio: {reduction_ratio:.1f}x")
    print()
    
    # Step 1: Create deduplicated dataset
    print("📋 Step 1: Creating deduplicated dataset...")
    unique_categories_df = df[[column_name]].drop_duplicates().reset_index(drop=True)
    print(f"  Created dataset with {len(unique_categories_df)} unique categories")
    
    # Step 2: Cluster the unique categories
    print("🎯 Step 2: Clustering unique categories...")
    
    # Create config if not provided
    if config is None:
        # Map kwargs to config fields
        config_kwargs = {}
        for k, v in clustering_kwargs.items():
            if hasattr(ClusterConfig, k):
                config_kwargs[k] = v
        config = ClusterConfig(**config_kwargs)
    
    print(f"  Using config: min_cluster_size={config.min_cluster_size}, hierarchical={config.hierarchical}")
    
    clustered_unique = hdbscan_cluster_categories(
        unique_categories_df, 
        column_name,
        config=config
    )
    
    # Step 3: Create mapping dictionary
    print("🔗 Step 3: Creating category-to-cluster mapping...")
    
    # Extract cluster information for mapping
    cluster_cols = [col for col in clustered_unique.columns if 'cluster' in col]
    mapping_dict = {}
    
    for _, row in clustered_unique.iterrows():
        category = row[column_name]
        mapping_dict[category] = {}
        for col in cluster_cols:
            mapping_dict[category][col] = row[col]
    
    print(f"  Created mapping for {len(mapping_dict)} categories")
    
    # Step 4: Map clusters back to original dataframe
    print("🗺️  Step 4: Mapping clusters back to original dataframe...")
    
    df_mapped = df.copy()
    for col in cluster_cols:
        df_mapped[col] = df_mapped[column_name].map(
            lambda x: str(mapping_dict.get(x, {}).get(col, 'unmapped'))
        )
    
    print("✅ Clustering and mapping complete!")
    print()
    
    return df_mapped, clustered_unique

def analyze_clustering_results(df_mapped, clustered_unique, column_name):
    """Analyze and display clustering results."""
    
    # Find cluster columns
    cluster_cols = [col for col in df_mapped.columns if 'cluster' in col]
    
    for col in cluster_cols:
        if 'label' in col and 'embedding' not in col:
            print(f"=== {col.upper()} RESULTS ===")
            
            # Convert cluster values to strings to handle arrays/complex types
            df_mapped[col] = df_mapped[col].astype(str)
            
            # Show cluster sizes in original dataset
            cluster_counts = df_mapped[col].value_counts()
            print("Cluster sizes in full dataset:")
            for cluster, count in cluster_counts.head(10).items():
                print(f"  {cluster}: {count:,} items")
            
            print()
            
            # Show sample categories per cluster
            print("Sample categories per cluster:")
            for cluster in cluster_counts.head(5).index:
                try:
                    sample_categories = df_mapped[df_mapped[col] == cluster][column_name].unique()[:3]
                    print(f"  🔹 {cluster}:")
                    for i, cat in enumerate(sample_categories, 1):
                        print(f"    {i}. {cat}")
                except Exception as e:
                    print(f"  🔹 {cluster}: Error getting samples - {e}")
            print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized category clustering with deduplication')
    parser.add_argument('--file', '-f', required=True, help='Path to input JSONL file')
    parser.add_argument('--column', '-c', default='category', help='Column name to cluster (default: category)')
    parser.add_argument('--min-cluster-size', type=int, default=15, help='Minimum cluster size')
    parser.add_argument('--hierarchical', action='store_true', help='Enable hierarchical clustering')
    parser.add_argument('--max-coarse-clusters', type=int, default=15, help='Max coarse clusters for hierarchical clustering')
    parser.add_argument('--embedding-model', default='openai', help='Embedding model')
    parser.add_argument('--output', '-o', help='Output filename prefix')
    parser.add_argument('--no-embeddings', action='store_true', help='Exclude embeddings from output')
    parser.add_argument('--no-llm-summaries', action='store_true', help='Disable LLM-based cluster summaries')
    parser.add_argument('--context', default='categories of properties seen in AI model responses', 
                       help='Context for LLM summaries')
    parser.add_argument('--assign-outliers', action='store_true', help='Assign outliers to nearest clusters')
    parser.add_argument('--enable-dim-reduction', action='store_true', 
                       help='Enable UMAP dimensionality reduction')
    parser.add_argument('--input-model-name', help='Name of input model for cache differentiation')
    parser.add_argument('--min-samples', type=int, help='min_samples for HDBSCAN')
    parser.add_argument('--cluster-selection-epsilon', type=float, default=0.0, 
                       help='Epsilon for HDBSCAN cluster selection')
    
    args = parser.parse_args()
    
    # Load data
    print(f"📂 Loading data from {args.file}...")
    df = pd.read_json(args.file, lines=True)
    print(f"Loaded {len(df)} rows")
    print()
    
    # Create config from args
    config = ClusterConfig(
        min_cluster_size=args.min_cluster_size,
        hierarchical=args.hierarchical,
        max_coarse_clusters=args.max_coarse_clusters,
        embedding_model=args.embedding_model,
        assign_outliers=args.assign_outliers,
        use_llm_summaries=not args.no_llm_summaries,
        context=args.context,
        include_embeddings=not args.no_embeddings,
        enable_dim_reduction=args.enable_dim_reduction,
        input_model_name=args.input_model_name,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon
    )
    
    # Run optimized clustering
    df_mapped, clustered_unique = dedupe_cluster_and_map(
        df, 
        args.column,
        config=config
    )
    
    # Analyze results
    analyze_clustering_results(df_mapped, clustered_unique, args.column)
    
    # Save results
    if args.output:
        output_prefix = args.output
    else:
        output_prefix = f"{args.column}_deduped_clustered"
    
    # Save the mapped full dataset
    from clustering.hierarchical_clustering import save_clustered_results
    save_clustered_results(df_mapped, output_prefix, include_embeddings=config.include_embeddings)
    
    # Save the unique categories with cluster info
    unique_output = f"{output_prefix}_unique_categories"
    save_clustered_results(clustered_unique, unique_output, include_embeddings=True)
    
    print(f"💾 Results saved:")
    print(f"  Full dataset (mapped): cluster_results/{output_prefix}/")
    print(f"  Unique categories: cluster_results/{unique_output}/")
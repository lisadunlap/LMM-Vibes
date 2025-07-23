#!/usr/bin/env python3
"""
Utility script to precompute embeddings for property descriptions.

This script can be run independently to generate embeddings for faster vector search.
Run with: python -m lmmvibes.viz.precompute_embeddings --results_dir path/to/results/
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import vector search functionality
from lmmvibes.viz.vector_search import PropertyVectorSearch


def main():
    parser = argparse.ArgumentParser(description="Precompute embeddings for property descriptions")
    parser.add_argument("--results_dir", required=True, help="Path to pipeline results directory")
    parser.add_argument("--embedding_model", default="openai", help="Embedding model to use")
    parser.add_argument("--force_recompute", action="store_true", help="Force recomputation of existing embeddings")
    
    args = parser.parse_args()
    
    results_path = Path(args.results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory {results_path} does not exist")
        sys.exit(1)
    
    clustered_path = results_path / "clustered_results.json"
    if not clustered_path.exists():
        print(f"Error: clustered_results.json not found in {results_path}")
        sys.exit(1)
    
    embeddings_path = results_path / "property_embeddings.npy"
    
    if embeddings_path.exists() and not args.force_recompute:
        print(f"Embeddings already exist at {embeddings_path}")
        print("Use --force_recompute to regenerate them")
        sys.exit(0)
    
    print(f"Loading data from {results_path}...")
    
    try:
        # Initialize vector search (this will compute embeddings)
        print("Initializing vector search engine...")
        vector_search = PropertyVectorSearch(results_path, embedding_model=args.embedding_model)
        
        # Get statistics
        stats = vector_search.get_statistics()
        print(f"✅ Successfully computed embeddings for {stats['total_properties']} properties")
        print(f"   - Total conversations: {stats['total_conversations']}")
        print(f"   - Unique models: {stats['unique_models']}")
        print(f"   - Unique clusters: {stats['unique_clusters']}")
        print(f"   - Embedding dimension: {stats['embedding_dimension']}")
        
        # Verify embeddings were saved
        if embeddings_path.exists():
            print(f"✅ Embeddings saved to {embeddings_path}")
        else:
            print("❌ Warning: Embeddings file not found after computation")
            
    except Exception as e:
        print(f"❌ Error computing embeddings: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
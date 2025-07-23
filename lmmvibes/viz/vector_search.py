"""Vector search functionality for LMM-Vibes property descriptions.

This module provides semantic search capabilities using approximate nearest neighbors
to find relevant behavioral properties based on user queries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import pickle
from dataclasses import dataclass

# Import existing embedding utilities
from lmmvibes.clusterers.clustering_utils import _get_openai_embeddings
from lmmvibes.core.caching import LMDBCache

@dataclass
class SearchResult:
    """A single search result with metadata."""
    property_description: str
    model: str
    cluster_id: str
    cluster_label: str
    similarity_score: float
    question_id: str
    evidence: Optional[str] = None
    category: Optional[str] = None
    impact: Optional[str] = None
    type: Optional[str] = None

class PropertyVectorSearch:
    """Vector search engine for behavioral properties."""
    
    def __init__(self, results_path: Path, embedding_model: str = "openai"):
        """
        Initialize the vector search engine.
        
        Args:
            results_path: Path to pipeline results directory
            embedding_model: Embedding model to use ("openai" or sentence-transformer model)
        """
        self.results_path = Path(results_path)
        self.embedding_model = embedding_model
        self.cache = LMDBCache()
        
        # Load data
        self.clustered_df = None
        self.property_embeddings = None
        self.property_descriptions = []
        self.property_metadata = []
        
        self._load_data()
    
    def _load_data(self):
        """Load clustered results and embeddings."""
        # Load clustered results
        clustered_path = self.results_path / "clustered_results.json"
        if clustered_path.exists():
            self.clustered_df = pd.read_json(clustered_path, lines=True)
        else:
            raise FileNotFoundError(f"clustered_results.json not found in {self.results_path}")
        
        # Load or compute embeddings
        embeddings_path = self.results_path / "property_embeddings.npy"
        if embeddings_path.exists():
            # Load precomputed embeddings
            self.property_embeddings = np.load(embeddings_path)
            print(f"Loaded {len(self.property_embeddings)} precomputed embeddings")
        else:
            # Compute embeddings on-the-fly
            print("Computing embeddings for property descriptions...")
            self._compute_embeddings()
        
        # Prepare property metadata
        self._prepare_metadata()
    
    def _compute_embeddings(self):
        """Compute embeddings for all unique property descriptions."""
        # Get unique property descriptions
        unique_descriptions = self.clustered_df['property_description'].unique().tolist()
        self.property_descriptions = unique_descriptions
        
        # Compute embeddings
        embeddings = _get_openai_embeddings(unique_descriptions)
        self.property_embeddings = np.array(embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        self.property_embeddings = self.property_embeddings / np.linalg.norm(
            self.property_embeddings, axis=1, keepdims=True
        )
        
        # Save embeddings for future use
        embeddings_path = self.results_path / "property_embeddings.npy"
        np.save(embeddings_path, self.property_embeddings)
        print(f"Saved {len(self.property_embeddings)} embeddings to {embeddings_path}")
    
    def _prepare_metadata(self):
        """Prepare metadata for each property description."""
        # Create a mapping from property description to metadata
        property_metadata = {}
        
        for _, row in self.clustered_df.iterrows():
            desc = row['property_description']
            if desc not in property_metadata:
                property_metadata[desc] = {
                    'models': set(),
                    'cluster_ids': set(),
                    'cluster_labels': set(),
                    'question_ids': set(),
                    'evidence': [],
                    'categories': set(),
                    'impacts': set(),
                    'types': set()
                }
            
            metadata = property_metadata[desc]
            metadata['models'].add(row.get('model', 'Unknown'))
            metadata['cluster_ids'].add(str(row.get('fine_cluster_id', '')))
            metadata['cluster_labels'].add(row.get('fine_cluster_label', ''))
            metadata['question_ids'].add(row.get('question_id', ''))
            
            if row.get('evidence'):
                metadata['evidence'].append(row['evidence'])
            if row.get('category'):
                metadata['categories'].add(row['category'])
            if row.get('impact'):
                metadata['impacts'].add(row['impact'])
            if row.get('type'):
                metadata['types'].add(row['type'])
        
        # Convert sets to lists for JSON serialization
        self.property_metadata = {}
        for desc, meta in property_metadata.items():
            self.property_metadata[desc] = {
                'models': list(meta['models']),
                'cluster_ids': list(meta['cluster_ids']),
                'cluster_labels': list(meta['cluster_labels']),
                'question_ids': list(meta['question_ids']),
                'evidence': meta['evidence'][:3],  # Keep first 3 pieces of evidence
                'categories': list(meta['categories']),
                'impacts': list(meta['impacts']),
                'types': list(meta['types'])
            }
    
    def search(self, query: str, top_k: int = 10, 
               min_similarity: float = 0.5) -> List[SearchResult]:
        """
        Search for properties similar to the query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of SearchResult objects sorted by similarity
        """
        if not query.strip():
            return []
        
        # Get query embedding
        query_embedding = _get_openai_embeddings([query])
        query_embedding = np.array(query_embedding[0], dtype=np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute similarities
        similarities = self.property_embeddings @ query_embedding
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity < min_similarity:
                continue
                
            desc = self.property_descriptions[idx]
            metadata = self.property_metadata[desc]
            
            # Create result object
            result = SearchResult(
                property_description=desc,
                model=metadata['models'][0] if metadata['models'] else 'Unknown',
                cluster_id=metadata['cluster_ids'][0] if metadata['cluster_ids'] else '',
                cluster_label=metadata['cluster_labels'][0] if metadata['cluster_labels'] else '',
                similarity_score=float(similarity),
                question_id=metadata['question_ids'][0] if metadata['question_ids'] else '',
                evidence=metadata['evidence'][0] if metadata['evidence'] else None,
                category=metadata['categories'][0] if metadata['categories'] else None,
                impact=metadata['impacts'][0] if metadata['impacts'] else None,
                type=metadata['types'][0] if metadata['types'] else None
            )
            results.append(result)
        
        return results
    
    def search_by_model(self, query: str, models: List[str], 
                       top_k: int = 10, min_similarity: float = 0.5) -> List[SearchResult]:
        """
        Search for properties within specific models.
        
        Args:
            query: Search query text
            models: List of model names to search within
            top_k: Number of results to return per model
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of SearchResult objects sorted by similarity
        """
        all_results = []
        
        for model in models:
            # Filter properties by model
            model_properties = [
                desc for desc in self.property_descriptions
                if model in self.property_metadata[desc]['models']
            ]
            
            if not model_properties:
                continue
            
            # Get indices for this model's properties
            model_indices = [
                i for i, desc in enumerate(self.property_descriptions)
                if desc in model_properties
            ]
            
            if not model_indices:
                continue
            
            # Get query embedding
            query_embedding = _get_openai_embeddings([query])
            query_embedding = np.array(query_embedding[0], dtype=np.float32)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Compute similarities for this model's properties
            model_embeddings = self.property_embeddings[model_indices]
            similarities = model_embeddings @ query_embedding
            
            # Get top-k results for this model
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            for top_idx in top_indices:
                similarity = similarities[top_idx]
                if similarity < min_similarity:
                    continue
                
                original_idx = model_indices[top_idx]
                desc = self.property_descriptions[original_idx]
                metadata = self.property_metadata[desc]
                
                result = SearchResult(
                    property_description=desc,
                    model=model,
                    cluster_id=metadata['cluster_ids'][0] if metadata['cluster_ids'] else '',
                    cluster_label=metadata['cluster_labels'][0] if metadata['cluster_labels'] else '',
                    similarity_score=float(similarity),
                    question_id=metadata['question_ids'][0] if metadata['question_ids'] else '',
                    evidence=metadata['evidence'][0] if metadata['evidence'] else None,
                    category=metadata['categories'][0] if metadata['categories'] else None,
                    impact=metadata['impacts'][0] if metadata['impacts'] else None,
                    type=metadata['types'][0] if metadata['types'] else None
                )
                all_results.append(result)
        
        # Sort all results by similarity
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return all_results[:top_k]
    
    def get_property_examples(self, property_description: str, 
                            max_examples: int = 5) -> List[Dict[str, Any]]:
        """
        Get example conversations for a specific property.
        
        Args:
            property_description: The property description to find examples for
            max_examples: Maximum number of examples to return
            
        Returns:
            List of example conversations
        """
        if property_description not in self.property_metadata:
            return []
        
        metadata = self.property_metadata[property_description]
        question_ids = metadata['question_ids'][:max_examples]
        
        examples = []
        for qid in question_ids:
            # Find the row in clustered_df with this question_id and property_description
            mask = (self.clustered_df['question_id'] == qid) & \
                   (self.clustered_df['property_description'] == property_description)
            
            matching_rows = self.clustered_df[mask]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                example = {
                    'question_id': qid,
                    'model': row.get('model', 'Unknown'),
                    'prompt': row.get('prompt', ''),
                    'response': row.get('model_response', ''),
                    'evidence': row.get('evidence', ''),
                    'score': row.get('score', 0),
                    'cluster_label': row.get('fine_cluster_label', '')
                }
                examples.append(example)
        
        return examples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        return {
            'total_properties': len(self.property_descriptions),
            'total_conversations': len(self.clustered_df),
            'unique_models': len(set().union(*[
                set(meta['models']) for meta in self.property_metadata.values()
            ])),
            'unique_clusters': len(set().union(*[
                set(meta['cluster_labels']) for meta in self.property_metadata.values()
            ])),
            'embedding_dimension': self.property_embeddings.shape[1] if self.property_embeddings is not None else 0
        } 
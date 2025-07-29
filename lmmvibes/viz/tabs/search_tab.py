"""
Search tab for pipeline results app.

This module contains the search tab functionality for semantic
search of behavioral properties using vector embeddings.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, List, Any
from pathlib import Path

# Import vector search functionality
try:
    from lmmvibes.viz.vector_search import PropertyVectorSearch, SearchResult
except ImportError:
    PropertyVectorSearch = None
    SearchResult = None


def create_search_tab(results_path: Path, all_models: List[str]):
    """Create the search tab with vector search functionality."""
    
    st.header("🔎 Vector Search")
    st.write("Search for behavioral properties using semantic similarity with vector embeddings.")
    
    # Initialize vector search engine
    @st.cache_resource
    def get_vector_search(results_path):
        if PropertyVectorSearch is None:
            st.error("Vector search is not available. Please install required dependencies.")
            return None
            
        try:
            return PropertyVectorSearch(results_path)
        except Exception as e:
            st.error(f"Failed to initialize vector search: {e}")
            return None
    
    vector_search = get_vector_search(results_path)
    
    if vector_search is None:
        st.error("Vector search is not available. Please ensure clustered_results.json exists.")
        st.stop()
    
    # Show search statistics
    stats = vector_search.get_statistics()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Properties", stats['total_properties'])
    with col2:
        st.metric("Total Conversations", stats['total_conversations'])
    with col3:
        st.metric("Unique Models", stats['unique_models'])
    with col4:
        st.metric("Unique Clusters", stats['unique_clusters'])
    
    st.divider()
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search for behavioral properties",
            placeholder="e.g., 'step by step reasoning', 'creative responses', 'formal tone'...",
            help="Enter a description of the behavioral property you're looking for"
        )
    
    with col2:
        top_k = st.number_input("Max results", min_value=5, max_value=50, value=10)
        min_similarity = st.slider("Min similarity", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    
    # Model filter
    model_filter = st.multiselect(
        "Filter by models (optional)",
        all_models,
        help="Leave empty to search across all models"
    )
    
    # Search button
    search_button = st.button("🔍 Search", type="primary")
    
    if search_button and search_query:
        with st.spinner("Searching..."):
            try:
                if model_filter:
                    # Search within specific models
                    results = vector_search.search_by_model(
                        search_query, model_filter, top_k=top_k, min_similarity=min_similarity
                    )
                else:
                    # Search across all models
                    results = vector_search.search(
                        search_query, top_k=top_k, min_similarity=min_similarity
                    )
                
                if results:
                    st.success(f"Found {len(results)} relevant properties")
                    
                    # Display results
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i}: {result.property_description[:80]}...", expanded=i<=3):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"**Property:** {result.property_description}")
                                st.markdown(f"**Model:** {result.model}")
                                st.markdown(f"**Cluster:** {result.cluster_label}")
                                st.markdown(f"**Similarity:** {result.similarity_score:.3f}")
                                
                                if result.category:
                                    st.markdown(f"**Category:** {result.category}")
                                if result.impact:
                                    st.markdown(f"**Impact:** {result.impact}")
                                if result.type:
                                    st.markdown(f"**Type:** {result.type}")
                            
                            with col2:
                                if result.evidence:
                                    st.markdown("**Evidence:**")
                                    st.text(result.evidence[:200] + "..." if len(result.evidence) > 200 else result.evidence)
                            
                            # Show examples directly if available
                            examples = vector_search.get_property_examples(result.property_description, max_examples=2)
                            if examples:
                                st.markdown("**Example Conversations:**")
                                for j, example in enumerate(examples, 1):
                                    with st.expander(f"Example {j}", expanded=False):
                                        st.markdown(f"**Question ID:** {example['question_id']}")
                                        st.markdown(f"**Model:** {example['model']}")
                                        st.markdown(f"**Score:** {example['score']}")
                                        
                                        if example['prompt']:
                                            st.markdown("**Prompt:**")
                                            st.text(example['prompt'][:300] + "..." if len(example['prompt']) > 300 else example['prompt'])
                                        
                                        if example['response']:
                                            st.markdown("**Response:**")
                                            st.text(example['response'][:300] + "..." if len(example['response']) > 300 else example['response'])
                                        
                                        if example['evidence']:
                                            st.markdown("**Evidence:**")
                                            st.text(example['evidence'])
                            else:
                                st.info("No examples available for this property.")
                    
                    # Show summary statistics
                    st.subheader("Search Summary")
                    similarity_scores = [r.similarity_score for r in results]
                    avg_similarity = np.mean(similarity_scores)
                    max_similarity = max(similarity_scores)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Similarity", f"{avg_similarity:.3f}")
                    with col2:
                        st.metric("Max Similarity", f"{max_similarity:.3f}")
                    with col3:
                        st.metric("Results Found", len(results))
                    
                    # Similarity distribution
                    fig = px.histogram(
                        x=similarity_scores,
                        nbins=10,
                        title="Similarity Score Distribution",
                        labels={'x': 'Similarity Score', 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.warning(f"No properties found matching '{search_query}' with similarity >= {min_similarity}")
                    st.info("Try:")
                    st.info("• Using different keywords")
                    st.info("• Lowering the similarity threshold")
                    st.info("• Checking spelling")
            
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.info("This might be due to missing embeddings. The system will compute them on first use.")
    
    elif search_query and not search_button:
        st.info("Click 'Search' to find relevant properties")
    
    # Show search tips
    with st.expander("💡 Search Tips", expanded=False):
        st.markdown("""
        **Effective search strategies:**
        
        • **Be specific:** Instead of "good responses", try "step-by-step explanations" or "creative problem solving"
        • **Use behavioral terms:** "formal tone", "technical accuracy", "user-friendly explanations"
        • **Combine concepts:** "detailed reasoning with examples", "concise but accurate responses"
        • **Try synonyms:** "thorough" instead of "detailed", "helpful" instead of "useful"
        
        **Understanding similarity scores:**
        • **0.9+**: Very similar properties
        • **0.7-0.9**: Related properties  
        • **0.5-0.7**: Somewhat related properties
        • **<0.5**: Weakly related (filtered out by default)
        """) 
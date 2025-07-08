"""Example viewer components for LMM-Vibes visualization.

Provides widgets for displaying and exploring actual model responses
that demonstrate specific behavioral properties.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import streamlit as st
import pandas as pd
import json


class ExampleViewerWidget:
    """Widget for viewing and exploring model response examples."""
    
    def __init__(self, model_stats: Dict[str, Any], results_path: Path):
        """Initialize with model statistics and results path.
        
        Args:
            model_stats: Dictionary of model performance statistics
            results_path: Path to results directory for loading examples
        """
        self.model_stats = model_stats
        self.results_path = results_path
    
    @st.cache_data
    def _load_property_examples(_self, property_ids: List[str]) -> pd.DataFrame:
        """Load specific property examples on-demand.
        
        Args:
            property_ids: List of property IDs to load
            
        Returns:
            DataFrame containing the requested examples
        """
        if not property_ids:
            return pd.DataFrame()
        
        # Load full dataset to get prompt/response details
        clustered_path = _self.results_path / "clustered_results.parquet"
        try:
            full_df = pd.read_parquet(clustered_path)
            return full_df[full_df['id'].isin(property_ids)]
        except Exception as e:
            st.error(f"Error loading examples: {e}")
            return pd.DataFrame()
    
    def render_cluster_examples(self, cluster_label: str, model_name: str, 
                              level: str = 'fine', show_metadata: bool = True) -> None:
        """Render examples for a specific cluster and model.
        
        Args:
            cluster_label: Name of the cluster to show examples for
            model_name: Model to show examples from
            level: Cluster level ('fine' or 'coarse')
            show_metadata: Whether to show additional metadata
        """
        # Get the stored example property IDs from model_stats
        model_data = self.model_stats.get(model_name, {})
        clusters = model_data.get(level, [])
        
        target_cluster = None
        for cluster in clusters:
            if cluster['property_description'] == cluster_label:
                target_cluster = cluster
                break
        
        if not target_cluster or not target_cluster.get('examples'):
            st.warning("No examples available for this cluster")
            return
        
        st.subheader(f"📝 Examples: {model_name}")
        st.caption(f"Cluster: {cluster_label}")
        
        # Display cluster stats
        if show_metadata:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Cluster Score", f"{target_cluster['score']:.3f}")
            with col2:
                st.metric("Cluster Size", target_cluster['size'])
            with col3:
                st.metric("Proportion", f"{target_cluster['proportion']:.3f}")
            with col4:
                quality_score = target_cluster.get('quality_score', 0)
                st.metric("Quality Score", f"{quality_score:.3f}")
        
        # Load examples
        example_ids = target_cluster['examples']
        examples_df = self._load_property_examples(example_ids)
        
        if examples_df.empty:
            st.warning("Could not load example data")
            return
        
        st.write(f"**Showing {len(examples_df)} example(s):**")
        
        # Display examples
        for i, (_, row) in enumerate(examples_df.iterrows(), 1):
            self._render_single_example(row, i, show_metadata)
    
    def render_interactive_example_browser(self, selected_models: List[str], 
                                         level: str = 'fine') -> None:
        """Render interactive browser for exploring examples across models.
        
        Args:
            selected_models: Models to include in browser
            level: Cluster level ('fine' or 'coarse')
        """
        st.subheader("🔍 Interactive Example Browser")
        
        if not selected_models:
            st.warning("Please select models to browse examples")
            return
        
        # Model selection
        selected_model = st.selectbox("Select model", selected_models)
        
        # Get clusters for selected model
        model_data = self.model_stats.get(selected_model, {})
        clusters = model_data.get(level, [])
        
        if not clusters:
            st.warning(f"No {level} clusters found for {selected_model}")
            return
        
        # Cluster selection with search
        cluster_names = [c['property_description'] for c in clusters]
        
        # Search filter
        search_term = st.text_input("🔎 Search clusters", placeholder="Enter search term...")
        
        if search_term:
            filtered_clusters = [c for c in clusters 
                               if search_term.lower() in c['property_description'].lower()]
        else:
            filtered_clusters = clusters
        
        if not filtered_clusters:
            st.warning("No clusters match your search")
            return
        
        # Sort options
        sort_by = st.selectbox(
            "Sort by",
            ['score', 'size', 'proportion', 'alphabetical'],
            help="How to order the clusters"
        )
        
        if sort_by == 'alphabetical':
            filtered_clusters.sort(key=lambda x: x['property_description'])
        else:
            filtered_clusters.sort(key=lambda x: x[sort_by], reverse=True)
        
        # Cluster selection
        cluster_options = [f"{c['property_description'][:80]}... (Score: {c['score']:.3f})" 
                          for c in filtered_clusters]
        selected_cluster_idx = st.selectbox(
            "Select cluster", 
            range(len(cluster_options)),
            format_func=lambda x: cluster_options[x]
        )
        
        if selected_cluster_idx is not None:
            selected_cluster = filtered_clusters[selected_cluster_idx]
            
            # Show examples for selected cluster
            self.render_cluster_examples(
                selected_cluster['property_description'], 
                selected_model, 
                level
            )
    
    def render_comparison_examples(self, cluster_label: str, model1: str, model2: str,
                                 level: str = 'fine') -> None:
        """Render side-by-side comparison of examples from two models.
        
        Args:
            cluster_label: Cluster to compare
            model1: First model name
            model2: Second model name
            level: Cluster level ('fine' or 'coarse')
        """
        st.subheader(f"⚔️ Example Comparison: {model1} vs {model2}")
        st.caption(f"Cluster: {cluster_label}")
        
        # Get examples for both models
        examples1 = self._get_cluster_examples(cluster_label, model1, level)
        examples2 = self._get_cluster_examples(cluster_label, model2, level)
        
        if examples1.empty and examples2.empty:
            st.warning("No examples available for either model in this cluster")
            return
        
        # Display side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**{model1} Examples:**")
            if not examples1.empty:
                for i, (_, row) in enumerate(examples1.iterrows(), 1):
                    with st.expander(f"Example {i}"):
                        self._render_example_content(row)
            else:
                st.info("No examples for this model")
        
        with col2:
            st.write(f"**{model2} Examples:**")
            if not examples2.empty:
                for i, (_, row) in enumerate(examples2.iterrows(), 1):
                    with st.expander(f"Example {i}"):
                        self._render_example_content(row)
            else:
                st.info("No examples for this model")
        
        # Show comparison insights
        self._render_comparison_insights(examples1, examples2, model1, model2)
    
    def render_property_search(self, selected_models: List[str]) -> None:
        """Render search interface for finding specific properties.
        
        Args:
            selected_models: Models to search within
        """
        st.subheader("🔎 Property Search")
        
        # Search input
        search_query = st.text_input(
            "Search for behavioral properties",
            placeholder="e.g. 'step by step', 'creative', 'formal tone'..."
        )
        
        if not search_query:
            st.info("Enter search terms to find relevant behavioral properties")
            return
        
        # Search across all model clusters
        matching_clusters = []
        for model in selected_models:
            model_data = self.model_stats.get(model, {})
            for level in ['fine', 'coarse']:
                clusters = model_data.get(level, [])
                for cluster in clusters:
                    if search_query.lower() in cluster['property_description'].lower():
                        matching_clusters.append({
                            'model': model,
                            'level': level,
                            'cluster': cluster,
                            'description': cluster['property_description'],
                            'score': cluster['score'],
                            'size': cluster['size']
                        })
        
        if not matching_clusters:
            st.warning("No matching properties found")
            return
        
        # Sort by relevance (score)
        matching_clusters.sort(key=lambda x: x['score'], reverse=True)
        
        st.write(f"**Found {len(matching_clusters)} matching properties:**")
        
        # Display results
        for i, match in enumerate(matching_clusters[:20]):  # Limit to top 20
            with st.expander(f"{match['model']} - {match['description'][:60]}..."):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Full Description:** {match['description']}")
                    st.write(f"**Model:** {match['model']}")
                    st.write(f"**Level:** {match['level'].title()}")
                
                with col2:
                    st.metric("Score", f"{match['score']:.3f}")
                    st.metric("Size", match['size'])
                
                # Show examples button
                if st.button(f"View Examples", key=f"search_examples_{i}"):
                    self.render_cluster_examples(
                        match['description'], 
                        match['model'], 
                        match['level']
                    )
    
    def _render_single_example(self, row: pd.Series, example_num: int, 
                             show_metadata: bool = True) -> None:
        """Render a single example with prompt, response, and metadata."""
        with st.expander(f"Example {example_num}: {row.get('id', 'Unknown')[:12]}..."):
            # Main content
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**💬 Prompt:**")
                prompt = row.get('prompt', row.get('user_prompt', 'N/A'))
                st.write(prompt)
            
            with col2:
                st.write("**🤖 Model Response:**")
                response = self._extract_response(row)
                st.write(response)
            
            # Extracted property
            st.write("**🎯 Extracted Behavioral Property:**")
            property_desc = row.get('property_description', 'N/A')
            st.success(property_desc)
            
            # Additional metadata if requested
            if show_metadata:
                st.write("**📊 Metadata:**")
                metadata_col1, metadata_col2, metadata_col3 = st.columns(3)
                
                with metadata_col1:
                    st.json({
                        'Question ID': row.get('question_id', 'N/A'),
                        'Property ID': row.get('id', 'N/A')
                    })
                
                with metadata_col2:
                    st.json({
                        'Cluster ID': row.get('fine_cluster_id', 'N/A'),
                        'Cluster Label': row.get('fine_cluster_label', 'N/A')
                    })
                
                with metadata_col3:
                    scores = row.get('score', {})
                    if isinstance(scores, dict):
                        st.json(scores)
                    else:
                        st.json({'score': scores})
    
    def _render_example_content(self, row: pd.Series) -> None:
        """Render just the content of an example (for comparison views)."""
        st.write("**Prompt:**")
        prompt = row.get('prompt', row.get('user_prompt', 'N/A'))
        st.caption(prompt)
        
        st.write("**Response:**")
        response = self._extract_response(row)
        st.write(response)
        
        st.write("**Property:**")
        property_desc = row.get('property_description', 'N/A')
        st.info(property_desc)
    
    def _extract_response(self, row: pd.Series) -> str:
        """Extract model response from row, handling different column formats."""
        return (row.get('model_response') or 
                row.get('model_a_response') or 
                row.get('model_b_response') or 
                row.get('responses', 'N/A'))
    
    def _get_cluster_examples(self, cluster_label: str, model_name: str, 
                            level: str) -> pd.DataFrame:
        """Get examples for a specific cluster and model."""
        model_data = self.model_stats.get(model_name, {})
        clusters = model_data.get(level, [])
        
        for cluster in clusters:
            if cluster['property_description'] == cluster_label:
                example_ids = cluster.get('examples', [])
                return self._load_property_examples(example_ids)
        
        return pd.DataFrame()
    
    def _render_comparison_insights(self, examples1: pd.DataFrame, examples2: pd.DataFrame,
                                  model1: str, model2: str) -> None:
        """Render insights from comparing examples between two models."""
        st.subheader("🔍 Comparison Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"{model1} Examples", len(examples1))
        
        with col2:
            st.metric(f"{model2} Examples", len(examples2))
        
        with col3:
            # Compare response lengths if data available
            if not examples1.empty and not examples2.empty:
                avg_len1 = examples1.apply(lambda x: len(self._extract_response(x)), axis=1).mean()
                avg_len2 = examples2.apply(lambda x: len(self._extract_response(x)), axis=1).mean()
                
                longer_model = model1 if avg_len1 > avg_len2 else model2
                st.metric("Longer Responses", longer_model)
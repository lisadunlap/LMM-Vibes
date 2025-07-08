"""Cluster heatmap components for LMM-Vibes visualization.

Provides interactive heatmap widgets for visualizing model performance
across behavioral clusters.
"""

from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ClusterHeatmapWidget:
    """Widget for creating interactive cluster performance heatmaps."""
    
    def __init__(self, model_stats: Dict[str, Any]):
        """Initialize with model statistics data.
        
        Args:
            model_stats: Dictionary of model performance statistics
        """
        self.model_stats = model_stats
        self.all_models = list(model_stats.keys())
    
    def render_performance_heatmap(self, selected_models: List[str], level: str = 'fine',
                                 top_n_clusters: int = 20, cluster_filter: Optional[str] = None) -> None:
        """Render interactive performance heatmap.
        
        Args:
            selected_models: Models to include in heatmap
            level: Cluster level ('fine' or 'coarse') 
            top_n_clusters: Maximum number of clusters to show
            cluster_filter: Optional text filter for cluster names
        """
        if not selected_models:
            st.warning("Please select models to display in heatmap")
            return
        
        st.subheader("🔥 Model × Cluster Performance Heatmap")
        
        # Controls
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_by = st.selectbox(
                "Sort clusters by",
                ['avg_score', 'max_score', 'score_variance', 'alphabetical'],
                help="How to order clusters in the heatmap"
            )
        with col2:
            color_scale = st.selectbox(
                "Color scale",
                ['RdYlGn', 'Viridis', 'RdBu', 'Spectral'],
                help="Color scheme for the heatmap"
            )
        with col3:
            center_score = st.number_input(
                "Center score", 
                value=1.0, 
                step=0.1,
                help="Score value to center color scale around"
            )
        
        # Get all clusters across selected models
        all_clusters = self._get_common_clusters(selected_models, level, top_n_clusters, cluster_filter)
        
        if not all_clusters:
            st.warning("No clusters found for selected models")
            return
        
        # Build score matrix
        score_matrix, cluster_labels = self._build_score_matrix(
            selected_models, all_clusters, level, sort_by
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=score_matrix,
            x=selected_models,
            y=cluster_labels,
            colorscale=color_scale,
            zmid=center_score,
            colorbar=dict(
                title="Score vs Median",
                titleside="right"
            ),
            hoverongaps=False,
            hovertemplate='<b>%{x}</b><br>%{y}<br><b>Score:</b> %{z:.3f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Model Performance Heatmap - {level.title()} Level Clusters",
            xaxis_title="Models",
            yaxis_title="Behavioral Clusters",
            height=max(500, len(cluster_labels) * 25),
            yaxis={'side': 'left'},
            font=dict(size=10)
        )
        
        # Make cluster labels more readable
        fig.update_yaxes(tickfont=dict(size=9))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        self._render_heatmap_insights(score_matrix, selected_models, cluster_labels)
    
    def render_cluster_similarity_heatmap(self, selected_models: List[str], level: str = 'fine') -> None:
        """Render heatmap showing cluster similarity across models.
        
        Args:
            selected_models: Models to analyze
            level: Cluster level ('fine' or 'coarse')
        """
        if len(selected_models) < 2:
            st.warning("Please select at least 2 models for similarity analysis")
            return
        
        st.subheader("🔗 Model Similarity Heatmap")
        st.caption("Cosine similarity based on cluster performance patterns")
        
        # Build model vectors based on cluster performance
        model_vectors = self._build_model_vectors(selected_models, level)
        
        if not model_vectors:
            st.warning("Insufficient data for similarity analysis")
            return
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(model_vectors, selected_models)
        
        # Create similarity heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=selected_models,
            y=selected_models,
            colorscale='Blues',
            zmin=0,
            zmax=1,
            colorbar=dict(title="Similarity"),
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br><b>Similarity:</b> %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Model Behavioral Similarity",
            xaxis_title="Models",
            yaxis_title="Models",
            height=400,
            width=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show most/least similar pairs
        self._render_similarity_insights(similarity_matrix, selected_models)
    
    def render_cluster_coverage_heatmap(self, selected_models: List[str], level: str = 'fine') -> None:
        """Render heatmap showing which clusters each model appears in.
        
        Args:
            selected_models: Models to analyze
            level: Cluster level ('fine' or 'coarse')
        """
        st.subheader("📊 Cluster Coverage Heatmap")
        st.caption("Which models appear in which behavioral clusters")
        
        # Get all unique clusters
        all_clusters = set()
        for model in selected_models:
            model_data = self.model_stats.get(model, {})
            clusters = model_data.get(level, [])
            for cluster in clusters:
                all_clusters.add(cluster['property_description'])
        
        all_clusters = sorted(list(all_clusters))[:30]  # Limit for readability
        
        # Build binary coverage matrix
        coverage_matrix = []
        for cluster in all_clusters:
            row = []
            for model in selected_models:
                model_data = self.model_stats.get(model, {})
                clusters = model_data.get(level, [])
                has_cluster = any(c['property_description'] == cluster for c in clusters)
                row.append(1 if has_cluster else 0)
            coverage_matrix.append(row)
        
        # Create coverage heatmap
        cluster_labels = [label[:50] + '...' if len(label) > 50 else label for label in all_clusters]
        
        fig = go.Figure(data=go.Heatmap(
            z=coverage_matrix,
            x=selected_models,
            y=cluster_labels,
            colorscale=[[0, 'white'], [1, 'darkblue']],
            showscale=False,
            hovertemplate='<b>%{x}</b><br>%{y}<br><b>Present:</b> %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Model Presence in Behavioral Clusters",
            xaxis_title="Models",
            yaxis_title="Behavioral Clusters",
            height=max(400, len(all_clusters) * 20),
            yaxis={'side': 'left'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Coverage statistics
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Clusters per Model:**")
            for model in selected_models:
                model_data = self.model_stats.get(model, {})
                n_clusters = len(model_data.get(level, []))
                st.write(f"• {model}: {n_clusters} clusters")
        
        with col2:
            st.write("**Models per Cluster (avg):**")
            models_per_cluster = [sum(row) for row in coverage_matrix]
            avg_models = np.mean(models_per_cluster) if models_per_cluster else 0
            st.metric("Average models per cluster", f"{avg_models:.1f}")
    
    def _get_common_clusters(self, selected_models: List[str], level: str, 
                           top_n: int, cluster_filter: Optional[str]) -> List[str]:
        """Get clusters that appear across the selected models."""
        all_clusters = set()
        for model in selected_models:
            model_data = self.model_stats.get(model, {})
            clusters = model_data.get(level, [])
            for cluster in clusters:
                cluster_name = cluster['property_description']
                if cluster_filter is None or cluster_filter.lower() in cluster_name.lower():
                    all_clusters.add(cluster_name)
        
        return sorted(list(all_clusters))[:top_n]
    
    def _build_score_matrix(self, selected_models: List[str], all_clusters: List[str],
                          level: str, sort_by: str) -> Tuple[List[List[float]], List[str]]:
        """Build the score matrix for heatmap visualization."""
        # Build score matrix
        score_matrix = []
        cluster_stats = []
        
        for cluster in all_clusters:
            row = []
            cluster_scores = []
            
            for model in selected_models:
                model_data = self.model_stats.get(model, {})
                clusters = model_data.get(level, [])
                # Find score for this cluster
                score = 0
                for c in clusters:
                    if c['property_description'] == cluster:
                        score = c['score']
                        break
                row.append(score)
                if score > 0:  # Only include non-zero scores in stats
                    cluster_scores.append(score)
            
            score_matrix.append(row)
            cluster_stats.append({
                'cluster': cluster,
                'avg_score': np.mean(cluster_scores) if cluster_scores else 0,
                'max_score': max(cluster_scores) if cluster_scores else 0,
                'score_variance': np.var(cluster_scores) if cluster_scores else 0
            })
        
        # Sort clusters based on selected criteria
        if sort_by == 'avg_score':
            sorted_indices = sorted(range(len(cluster_stats)), 
                                  key=lambda i: cluster_stats[i]['avg_score'], reverse=True)
        elif sort_by == 'max_score':
            sorted_indices = sorted(range(len(cluster_stats)), 
                                  key=lambda i: cluster_stats[i]['max_score'], reverse=True)
        elif sort_by == 'score_variance':
            sorted_indices = sorted(range(len(cluster_stats)), 
                                  key=lambda i: cluster_stats[i]['score_variance'], reverse=True)
        else:  # alphabetical
            sorted_indices = sorted(range(len(cluster_stats)), 
                                  key=lambda i: cluster_stats[i]['cluster'])
        
        # Reorder matrix and labels
        sorted_matrix = [score_matrix[i] for i in sorted_indices]
        sorted_labels = [cluster_stats[i]['cluster'] for i in sorted_indices]
        
        # Truncate long labels
        display_labels = []
        for label in sorted_labels:
            if len(label) > 60:
                display_labels.append(label[:57] + '...')
            else:
                display_labels.append(label)
        
        return sorted_matrix, display_labels
    
    def _render_heatmap_insights(self, score_matrix: List[List[float]], 
                               selected_models: List[str], cluster_labels: List[str]) -> None:
        """Render insights from the heatmap data."""
        st.subheader("🔍 Heatmap Insights")
        
        # Convert to numpy array for easier analysis
        matrix = np.array(score_matrix)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Performing Model-Cluster Pairs:**")
            # Find top 5 scores
            flat_indices = np.argsort(matrix.flatten())[-5:][::-1]
            for idx in flat_indices:
                row_idx, col_idx = np.unravel_index(idx, matrix.shape)
                score = matrix[row_idx, col_idx]
                if score > 0:  # Only show non-zero scores
                    st.write(f"• {selected_models[col_idx]} × {cluster_labels[row_idx][:40]}...")
                    st.caption(f"  Score: {score:.3f}")
        
        with col2:
            st.write("**Performance Summary:**")
            # Model averages
            model_avgs = np.mean(matrix, axis=0)
            best_model_idx = np.argmax(model_avgs)
            st.write(f"**Best Overall:** {selected_models[best_model_idx]} ({model_avgs[best_model_idx]:.3f})")
            
            # Cluster averages
            cluster_avgs = np.mean(matrix, axis=1)
            best_cluster_idx = np.argmax(cluster_avgs)
            st.write(f"**Most Challenging Cluster:** {cluster_labels[best_cluster_idx][:30]}...")
    
    def _build_model_vectors(self, selected_models: List[str], level: str) -> Dict[str, List[float]]:
        """Build performance vectors for each model."""
        # Get all unique clusters
        all_clusters = set()
        for model in selected_models:
            model_data = self.model_stats.get(model, {})
            clusters = model_data.get(level, [])
            for cluster in clusters:
                all_clusters.add(cluster['property_description'])
        
        all_clusters = sorted(list(all_clusters))
        
        # Build vectors
        model_vectors = {}
        for model in selected_models:
            vector = []
            model_data = self.model_stats.get(model, {})
            clusters = model_data.get(level, [])
            cluster_dict = {c['property_description']: c['score'] for c in clusters}
            
            for cluster in all_clusters:
                vector.append(cluster_dict.get(cluster, 0))
            
            model_vectors[model] = vector
        
        return model_vectors
    
    def _compute_similarity_matrix(self, model_vectors: Dict[str, List[float]], 
                                 selected_models: List[str]) -> List[List[float]]:
        """Compute cosine similarity matrix between models."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Convert to matrix
        vectors = [model_vectors[model] for model in selected_models]
        
        # Compute cosine similarity
        similarity = cosine_similarity(vectors)
        
        return similarity.tolist()
    
    def _render_similarity_insights(self, similarity_matrix: List[List[float]], 
                                  selected_models: List[str]) -> None:
        """Render insights from similarity analysis."""
        col1, col2 = st.columns(2)
        
        # Convert to numpy for easier analysis
        matrix = np.array(similarity_matrix)
        
        with col1:
            st.write("**Most Similar Pairs:**")
            # Find highest similarities (excluding diagonal)
            mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
            masked_matrix = np.where(mask, matrix, -1)
            
            flat_indices = np.argsort(masked_matrix.flatten())[-3:][::-1]
            for idx in flat_indices:
                row_idx, col_idx = np.unravel_index(idx, matrix.shape)
                similarity = matrix[row_idx, col_idx]
                if similarity > 0:
                    st.write(f"• {selected_models[row_idx]} ↔ {selected_models[col_idx]}")
                    st.caption(f"  Similarity: {similarity:.3f}")
        
        with col2:
            st.write("**Least Similar Pairs:**")
            flat_indices = np.argsort(masked_matrix.flatten())[:3]
            for idx in flat_indices:
                row_idx, col_idx = np.unravel_index(idx, matrix.shape)
                similarity = matrix[row_idx, col_idx]
                if similarity >= 0:
                    st.write(f"• {selected_models[row_idx]} ↔ {selected_models[col_idx]}")
                    st.caption(f"  Similarity: {similarity:.3f}")
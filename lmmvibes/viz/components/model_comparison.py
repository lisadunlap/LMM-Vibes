"""Model comparison components for LMM-Vibes visualization.

Provides reusable widgets for comparing model performance across
behavioral clusters and metrics.
"""

from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


class ModelComparisonWidget:
    """Widget for comparing models across various metrics and clusters."""
    
    def __init__(self, model_stats: Dict[str, Any]):
        """Initialize with model statistics data.
        
        Args:
            model_stats: Dictionary of model performance statistics
        """
        self.model_stats = model_stats
        self.all_models = list(model_stats.keys())
    
    def render_model_selector(self, key_suffix: str = "", default_models: Optional[List[str]] = None) -> List[str]:
        """Render model selection widget.
        
        Args:
            key_suffix: Unique suffix for widget key
            default_models: Default models to select
            
        Returns:
            List of selected model names
        """
        if default_models is None:
            default_models = self.all_models[:min(3, len(self.all_models))]
        
        return st.multiselect(
            "Select models to compare",
            self.all_models,
            default=default_models,
            key=f"model_selector_{key_suffix}",
            help="Choose models for comparison analysis"
        )
    
    def render_leaderboard_table(self, sort_by: str = 'avg_score') -> pd.DataFrame:
        """Render model leaderboard table.
        
        Args:
            sort_by: Metric to sort by ('avg_score', 'median_score', 'top_score')
            
        Returns:
            DataFrame with leaderboard data
        """
        st.subheader("📊 Model Leaderboard")
        
        # Compute rankings
        model_rankings = self._compute_model_rankings()
        
        # Sort by requested metric
        if sort_by != 'avg_score':
            model_rankings.sort(key=lambda x: x[1][sort_by], reverse=True)
        
        # Create leaderboard DataFrame
        leaderboard_data = []
        for rank, (model, stats) in enumerate(model_rankings, 1):
            leaderboard_data.append({
                'Rank': rank,
                'Model': model,
                'Avg Score': f"{stats['avg_score']:.3f}",
                'Median Score': f"{stats['median_score']:.3f}", 
                'Top Score': f"{stats['top_score']:.3f}",
                'Std Dev': f"{stats['std_score']:.3f}",
                '# Clusters': stats['num_clusters'],
                'Above Median': stats['above_median'],
                'Below Median': stats['below_median']
            })
        
        df_leaderboard = pd.DataFrame(leaderboard_data)
        
        # Render with custom styling
        st.dataframe(
            df_leaderboard,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Rank': st.column_config.NumberColumn('🏆 Rank', width='small'),
                'Model': st.column_config.TextColumn('🤖 Model', width='medium'),
                'Avg Score': st.column_config.NumberColumn('📊 Avg Score', width='small'),
                'Median Score': st.column_config.NumberColumn('📈 Median', width='small'),
                'Top Score': st.column_config.NumberColumn('🔥 Best', width='small'),
                'Std Dev': st.column_config.NumberColumn('📏 Std Dev', width='small'),
                '# Clusters': st.column_config.NumberColumn('🏷️ Clusters', width='small'),
                'Above Median': st.column_config.NumberColumn('⬆️ Above', width='small'),
                'Below Median': st.column_config.NumberColumn('⬇️ Below', width='small')
            }
        )
        
        return df_leaderboard
    
    def render_head_to_head_comparison(self, model1: str, model2: str, level: str = 'fine', 
                                     top_n: int = 15) -> None:
        """Render detailed head-to-head comparison between two models.
        
        Args:
            model1: First model name
            model2: Second model name  
            level: Cluster level ('fine' or 'coarse')
            top_n: Number of top clusters to compare
        """
        st.subheader(f"⚔️ Head-to-Head: {model1} vs {model2}")
        
        # Get cluster data for both models
        clusters1 = self._get_top_clusters_for_model(model1, level, top_n * 2)  # Get more to find overlaps
        clusters2 = self._get_top_clusters_for_model(model2, level, top_n * 2)
        
        # Find common clusters
        clusters1_dict = {c['property_description']: c for c in clusters1}
        clusters2_dict = {c['property_description']: c for c in clusters2}
        
        common_clusters = set(clusters1_dict.keys()) & set(clusters2_dict.keys())
        
        if not common_clusters:
            st.warning("No common clusters found between these models")
            return
        
        # Create comparison data
        comparison_data = []
        for cluster_name in list(common_clusters)[:top_n]:  # Limit to top_n
            c1 = clusters1_dict[cluster_name]
            c2 = clusters2_dict[cluster_name]
            
            comparison_data.append({
                'Cluster': cluster_name[:60] + ('...' if len(cluster_name) > 60 else ''),
                f'{model1} Score': c1['score'],
                f'{model2} Score': c2['score'],
                'Score Diff': c1['score'] - c2['score'],
                f'{model1} Size': c1['size'],
                f'{model2} Size': c2['size'],
                'Winner': model1 if c1['score'] > c2['score'] else model2 if c2['score'] > c1['score'] else 'Tie'
            })
        
        # Sort by absolute score difference
        comparison_data.sort(key=lambda x: abs(x['Score Diff']), reverse=True)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            model1_wins = sum(1 for row in comparison_data if row['Winner'] == model1)
            st.metric(f"{model1} Wins", model1_wins)
        with col2:
            model2_wins = sum(1 for row in comparison_data if row['Winner'] == model2) 
            st.metric(f"{model2} Wins", model2_wins)
        with col3:
            ties = sum(1 for row in comparison_data if row['Winner'] == 'Tie')
            st.metric("Ties", ties)
        
        # Display comparison table
        st.dataframe(
            df_comparison,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Score Diff': st.column_config.NumberColumn(
                    'Score Diff', 
                    format="%.3f",
                    help="Positive = Model 1 better, Negative = Model 2 better"
                )
            }
        )
        
        # Visualization
        fig = px.scatter(
            df_comparison, 
            x=f'{model1} Score', 
            y=f'{model2} Score',
            hover_data=['Cluster', 'Score Diff'],
            title=f"Cluster Performance: {model1} vs {model2}",
            labels={
                f'{model1} Score': f'{model1} Score',
                f'{model2} Score': f'{model2} Score'
            }
        )
        
        # Add diagonal line for equal performance
        max_score = max(df_comparison[f'{model1} Score'].max(), df_comparison[f'{model2} Score'].max())
        min_score = min(df_comparison[f'{model1} Score'].min(), df_comparison[f'{model2} Score'].min())
        fig.add_shape(
            type="line",
            x0=min_score, y0=min_score, x1=max_score, y1=max_score,
            line=dict(color="red", dash="dash"),
            name="Equal Performance"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_radar_chart(self, selected_models: List[str], 
                                     cluster_categories: Optional[List[str]] = None) -> None:
        """Render radar chart comparing models across cluster categories.
        
        Args:
            selected_models: Models to include in radar chart
            cluster_categories: Specific cluster categories to include
        """
        if len(selected_models) < 2:
            st.warning("Please select at least 2 models for radar chart comparison")
            return
        
        st.subheader("🕸️ Performance Radar Chart")
        
        # If no specific categories provided, use top categories
        if cluster_categories is None:
            cluster_categories = self._get_top_cluster_categories(n=8)
        
        # Build radar chart data
        fig = go.Figure()
        
        for model in selected_models:
            scores = []
            for category in cluster_categories:
                # Find score for this category
                model_data = self.model_stats.get(model, {})
                clusters = model_data.get('fine', [])
                
                category_scores = [c['score'] for c in clusters 
                                 if category.lower() in c['property_description'].lower()]
                avg_score = np.mean(category_scores) if category_scores else 0
                scores.append(avg_score)
            
            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=cluster_categories,
                fill='toself',
                name=model,
                hovertemplate='Model: %{fullData.name}<br>Category: %{theta}<br>Score: %{r:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(2.0, max([max(trace.r) for trace in fig.data if hasattr(trace, 'r')]))]
                )
            ),
            showlegend=True,
            title="Model Performance Across Behavioral Categories",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _compute_model_rankings(self) -> List[Tuple[str, Dict[str, float]]]:
        """Compute model rankings with various metrics."""
        model_scores = {}
        for model, stats in self.model_stats.items():
            fine_scores = [stat['score'] for stat in stats.get('fine', [])]
            if fine_scores:
                model_scores[model] = {
                    'avg_score': np.mean(fine_scores),
                    'median_score': np.median(fine_scores),
                    'num_clusters': len(fine_scores),
                    'top_score': max(fine_scores),
                    'std_score': np.std(fine_scores),
                    'above_median': sum(1 for s in fine_scores if s > 1.0),
                    'below_median': sum(1 for s in fine_scores if s < 1.0)
                }
            else:
                model_scores[model] = {
                    'avg_score': 0, 'median_score': 0, 'num_clusters': 0, 
                    'top_score': 0, 'std_score': 0, 'above_median': 0, 'below_median': 0
                }
        
        return sorted(model_scores.items(), key=lambda x: x[1]['avg_score'], reverse=True)
    
    def _get_top_clusters_for_model(self, model_name: str, level: str = 'fine', 
                                   top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top N clusters for a specific model."""
        model_data = self.model_stats.get(model_name, {})
        clusters = model_data.get(level, [])
        return sorted(clusters, key=lambda x: x['score'], reverse=True)[:top_n]
    
    def _get_top_cluster_categories(self, n: int = 8) -> List[str]:
        """Get the most common cluster categories across all models."""
        all_descriptions = []
        for model_data in self.model_stats.values():
            clusters = model_data.get('fine', [])
            all_descriptions.extend([c['property_description'] for c in clusters])
        
        # Extract key terms (simplified approach)
        # This could be enhanced with proper NLP
        terms = []
        for desc in all_descriptions:
            words = desc.lower().split()
            # Look for key behavioral terms
            key_terms = ['reasoning', 'creative', 'formal', 'detailed', 'concise', 
                        'helpful', 'accurate', 'structured', 'emotional', 'technical']
            for term in key_terms:
                if any(term in word for word in words):
                    terms.append(term.title())
        
        # Get most common terms
        from collections import Counter
        common_terms = Counter(terms).most_common(n)
        return [term for term, count in common_terms]
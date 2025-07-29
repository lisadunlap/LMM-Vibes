"""
Clusters tab for pipeline results app.

This module contains the clusters tab functionality for exploring
hierarchical behavioral clusters with detailed property information.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path


def create_clusters_tab(clustered_df: pd.DataFrame):
    """Create the clusters tab with hierarchical cluster viewer."""
    
    st.header("Cluster Viewer")
    st.write("Explore hierarchical behavioral clusters with detailed property information")
    
    # Build hierarchical structure from clustered_df
    coarse_clusters = []
    fine_clusters_by_parent = {}
    
    # Get cluster information from clustered_df
    if ('property_description_fine_cluster_id' in clustered_df.columns and 
        'property_description_coarse_cluster_id' in clustered_df.columns):
        
        # Group by fine and coarse cluster IDs to build hierarchy
        cluster_groups = clustered_df.groupby([
            'property_description_fine_cluster_id', 
            'property_description_fine_cluster_label', 
            'property_description_coarse_cluster_id', 
            'property_description_coarse_cluster_label'
        ]).agg({
            'property_description': 'count',
            'id': 'count'  # Count of properties in this cluster
        }).reset_index()
        
        # Build coarse clusters
        coarse_cluster_data = {}
        for _, row in cluster_groups.iterrows():
            coarse_id = row['property_description_coarse_cluster_id']
            coarse_label = row['property_description_coarse_cluster_label']
            
            if coarse_id not in coarse_cluster_data:
                coarse_cluster_data[coarse_id] = {
                    'id': coarse_id,
                    'label': coarse_label,
                    'size': 0,
                    'fine_clusters': []
                }
            
            # Add fine cluster info
            fine_cluster = {
                'id': row['property_description_fine_cluster_id'],
                'label': row['property_description_fine_cluster_label'],
                'size': row['property_description'],
                'parent_id': coarse_id,
                'parent_label': coarse_label,
                'property_descriptions': []
            }
            
            coarse_cluster_data[coarse_id]['fine_clusters'].append(fine_cluster)
            coarse_cluster_data[coarse_id]['size'] += 1
        
        # Convert to lists and sort
        coarse_clusters = list(coarse_cluster_data.values())
        coarse_clusters.sort(key=lambda x: x['size'], reverse=True)
        
        # Build fine clusters by parent
        for coarse_cluster in coarse_clusters:
            fine_clusters_by_parent[coarse_cluster['id']] = coarse_cluster['fine_clusters']
            
            # Get property descriptions for each fine cluster
            for fine_cluster in coarse_cluster['fine_clusters']:
                fine_cluster_data = clustered_df[
                    (clustered_df['property_description_fine_cluster_id'] == fine_cluster['id']) & 
                    (clustered_df['property_description_fine_cluster_label'] == fine_cluster['label'])
                ]
                fine_cluster['property_descriptions'] = fine_cluster_data['property_description'].unique().tolist()
    
    # Create two-column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Cluster Viewer")
        
        # Track selected cluster
        if 'selected_cluster' not in st.session_state:
            st.session_state.selected_cluster = None
        
        # Get all fine clusters directly from clustered_df
        if 'property_description_fine_cluster_id' in clustered_df.columns:
            # Get unique fine clusters with their sizes
            fine_clusters_data = clustered_df.groupby([
                'property_description_fine_cluster_id', 
                'property_description_fine_cluster_label'
            ]).agg({
                'property_description': 'count',
                'id': 'count'
            }).reset_index()
            
            # Sort by size (largest first)
            fine_clusters_data = fine_clusters_data.sort_values('property_description', ascending=False)
            
            # Display fine clusters
            for _, row in fine_clusters_data.iterrows():
                cluster_id = row['property_description_fine_cluster_id']
                cluster_label = row['property_description_fine_cluster_label']
                cluster_size = row['property_description']
                
                # Check if this cluster is selected
                is_selected = (st.session_state.selected_cluster and 
                             st.session_state.selected_cluster['id'] == cluster_id)
                
                # Create clickable button using the cluster name with size
                button_text = f"{cluster_label} (Size: {cluster_size})"
                
                if st.button(
                    button_text,
                    key=f"fine_{cluster_id}",
                    help="Click to view details",
                    use_container_width=True
                ):
                    # Get property descriptions for this cluster
                    cluster_data = clustered_df[
                        clustered_df['property_description_fine_cluster_id'] == cluster_id
                    ]
                    property_descriptions = cluster_data['property_description'].unique().tolist()
                    
                    # Create cluster object for selection
                    selected_cluster = {
                        'id': cluster_id,
                        'label': cluster_label,
                        'size': cluster_size,
                        'property_descriptions': property_descriptions
                    }
                    st.session_state.selected_cluster = selected_cluster
            
        else:
            st.error("No fine cluster data found. Please ensure the pipeline generated clustering results.")
    
    with col2:
        st.subheader("Cluster Details")
        
        # Add close button (X) in the header
        if st.button("✕", key="close_details", help="Close details"):
            st.session_state.selected_cluster = None
        
        if st.session_state.selected_cluster:
            cluster = st.session_state.selected_cluster
            
            # Main description with better styling
            st.markdown(f"""
            <div style="
                padding: 12px;
                margin: 8px 0;
                background-color: #f8f9fa;
                border-radius: 6px;
                border-left: 4px solid #3182ce;
            ">
                <div style="font-weight: 600; font-size: 16px; margin-bottom: 8px;">
                    {cluster['label']}
                </div>
                <div style="font-size: 14px; color: #666;">
                    <strong>Size:</strong> {cluster['size']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Parent information
            if cluster.get('parent_label'):
                st.markdown(f"""
                <div style="
                    padding: 8px 12px;
                    margin: 8px 0;
                    background-color: #f0f8ff;
                    border-radius: 4px;
                    font-size: 14px;
                ">
                    <strong>Parent:</strong> {cluster['parent_label']}
                </div>
                """, unsafe_allow_html=True)
            
            # Property descriptions section
            st.subheader(f"Property Descriptions ({len(cluster['property_descriptions'])})")
            
            for i, desc in enumerate(cluster['property_descriptions']):
                st.markdown(f"""
                <div style="
                    padding: 10px 12px;
                    margin: 6px 0;
                    background-color: #f8f9fa;
                    border-radius: 6px;
                    border-left: 3px solid #3182ce;
                    font-size: 14px;
                    line-height: 1.4;
                ">
                    {desc}
                </div>
                """, unsafe_allow_html=True)
            
            # Show some statistics with better styling
            st.subheader("Cluster Statistics")
            
            total_clusters = len(clustered_df['property_description_fine_cluster_id'].unique()) if 'property_description_fine_cluster_id' in clustered_df.columns else 0
            total_properties = len(clustered_df)
            
            # Calculate min and max properties per cluster
            if 'property_description_fine_cluster_id' in clustered_df.columns:
                cluster_sizes = clustered_df.groupby('property_description_fine_cluster_id').size()
                min_properties = cluster_sizes.min() if not cluster_sizes.empty else 0
                max_properties = cluster_sizes.max() if not cluster_sizes.empty else 0
                global_cluster_count = cluster_sizes.sum() if not cluster_sizes.empty else 0
            else:
                min_properties = 0
                max_properties = 0
                global_cluster_count = 0
            
            col1_stat, col2_stat, col3_stat = st.columns(3)
            with col1_stat:
                st.metric("Total Fine Clusters", total_clusters)
                st.metric("Total Properties", total_properties)
            with col2_stat:
                st.metric("Min Properties/Cluster", min_properties)
                st.metric("Max Properties/Cluster", max_properties)
            with col3_stat:
                st.metric("Global Cluster Count", global_cluster_count)
            
            # Show largest clusters
            st.subheader("Largest Clusters")
            if 'property_description_fine_cluster_id' in clustered_df.columns:
                largest_clusters = clustered_df.groupby(['property_description_fine_cluster_id', 'property_description_fine_cluster_label']).size().nlargest(5)
                for (cluster_id, cluster_label), size in largest_clusters.items():
                    st.markdown(f"""
                    <div style="
                        padding: 8px 12px;
                        margin: 4px 0;
                        background-color: #f8f9fa;
                        border-radius: 4px;
                        font-size: 14px;
                    ">
                        <strong>{cluster_label}</strong> (Size: {size})
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Select a cluster from the left panel to view details") 
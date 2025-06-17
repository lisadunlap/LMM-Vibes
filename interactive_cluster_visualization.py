"""
Interactive Hierarchical Cluster Visualization

This module provides interactive visualizations for hierarchical clustering results,
including sunburst charts and drill-down pie charts that can be saved as HTML files.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import defaultdict
import json


def load_clustering_results(parquet_path):
    """Load clustering results from parquet file."""
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows with {df.shape[1]} columns")
    
    # Identify clustering columns
    cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
    print(f"Found clustering columns: {cluster_cols}")
    
    return df


def create_sunburst_visualization(df, column_name, title=None, save_path=None):
    """
    Create an interactive sunburst chart showing hierarchical clusters.
    
    Args:
        df: DataFrame with clustering results
        column_name: Base name of the clustered column (e.g., 'property_description')
        title: Title for the chart
        save_path: Path to save HTML file (optional)
    
    Returns:
        Plotly figure object
    """
    # Identify relevant columns
    coarse_label_col = f"{column_name}_coarse_cluster_label"
    fine_label_col = f"{column_name}_fine_cluster_label"
    
    if coarse_label_col not in df.columns or fine_label_col not in df.columns:
        raise ValueError(f"Required columns not found. Need {coarse_label_col} and {fine_label_col}")
    
    # Create hierarchy data
    hierarchy_data = []
    
    # Group by coarse and fine clusters
    grouped = df.groupby([coarse_label_col, fine_label_col]).size().reset_index(name='count')
    
    # Add root level
    total_count = len(df)
    hierarchy_data.append({
        'ids': 'root',
        'labels': f'All Data ({total_count:,} items)',
        'parents': '',
        'values': total_count
    })
    
    # Add coarse level
    coarse_counts = df[coarse_label_col].value_counts()
    for coarse_label, count in coarse_counts.items():
        hierarchy_data.append({
            'ids': f'coarse_{coarse_label}',
            'labels': f'{coarse_label} ({count:,} items)',
            'parents': 'root',
            'values': count
        })
    
    # Add fine level
    for _, row in grouped.iterrows():
        coarse_label = row[coarse_label_col]
        fine_label = row[fine_label_col]
        count = row['count']
        
        hierarchy_data.append({
            'ids': f'fine_{coarse_label}_{fine_label}',
            'labels': f'{fine_label} ({count:,} items)',
            'parents': f'coarse_{coarse_label}',
            'values': count
        })
    
    # Convert to DataFrame for easier handling
    hierarchy_df = pd.DataFrame(hierarchy_data)
    
    # Create sunburst chart
    fig = go.Figure(go.Sunburst(
        ids=hierarchy_df['ids'],
        labels=hierarchy_df['labels'],
        parents=hierarchy_df['parents'],
        values=hierarchy_df['values'],
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent}<extra></extra>',
        maxdepth=3,
    ))
    
    fig.update_layout(
        title=title or f"Hierarchical Clusters for {column_name}",
        title_x=0.5,
        font_size=12,
        width=800,
        height=800
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Sunburst chart saved to: {save_path}")
    
    return fig


def create_drill_down_visualization(df, column_name, title=None, save_path=None):
    """
    Create drill-down pie charts with navigation controls.
    
    Args:
        df: DataFrame with clustering results
        column_name: Base name of the clustered column
        title: Title for the chart
        save_path: Path to save HTML file
    
    Returns:
        HTML string with interactive visualization
    """
    coarse_label_col = f"{column_name}_coarse_cluster_label"
    fine_label_col = f"{column_name}_fine_cluster_label"
    
    if coarse_label_col not in df.columns or fine_label_col not in df.columns:
        raise ValueError(f"Required columns not found. Need {coarse_label_col} and {fine_label_col}")
    
    # Prepare data structures
    coarse_counts = df[coarse_label_col].value_counts().to_dict()
    
    # Group fine clusters by coarse clusters
    fine_by_coarse = {}
    for coarse_label in coarse_counts.keys():
        coarse_subset = df[df[coarse_label_col] == coarse_label]
        fine_counts = coarse_subset[fine_label_col].value_counts().to_dict()
        fine_by_coarse[coarse_label] = fine_counts
    
    # Sample values for each fine cluster (for inspection)
    cluster_samples = {}
    for coarse_label in coarse_counts.keys():
        cluster_samples[coarse_label] = {}
        coarse_subset = df[df[coarse_label_col] == coarse_label]
        
        for fine_label in fine_by_coarse[coarse_label].keys():
            fine_subset = coarse_subset[coarse_subset[fine_label_col] == fine_label]
            # Get sample values (up to 20 for display)
            sample_values = fine_subset[column_name].unique()[:20].tolist()
            cluster_samples[coarse_label][fine_label] = [str(v) for v in sample_values]
    
    # Create HTML with embedded JavaScript
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title or f'Interactive Cluster Explorer - {column_name}'}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .chart-container {{ display: inline-block; width: 48%; vertical-align: top; }}
            .info-panel {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .navigation {{ margin: 10px 0; }}
            .btn {{ padding: 10px 15px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer; }}
            .btn:hover {{ background: #0056b3; }}
            .samples {{ max-height: 200px; overflow-y: auto; font-size: 12px; }}
            h1, h2 {{ color: #333; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title or f'Interactive Cluster Explorer - {column_name}'}</h1>
            
            <div class="navigation">
                <button class="btn" onclick="showCoarseLevel()" id="coarseBtn">Coarse Clusters</button>
                <button class="btn" onclick="goBack()" id="backBtn" style="display:none;">← Back to Coarse</button>
            </div>
            
            <div class="chart-container">
                <div id="chartDiv" style="width:100%; height:500px;"></div>
            </div>
            
            <div class="chart-container">
                <div class="info-panel">
                    <h2 id="infoTitle">Cluster Information</h2>
                    <div id="infoContent">Click on a cluster to see details and sample values.</div>
                </div>
            </div>
        </div>

        <script>
            // Data structures
            const coarseCounts = {json.dumps(coarse_counts)};
            const fineByCourse = {json.dumps(fine_by_coarse)};
            const clusterSamples = {json.dumps(cluster_samples)};
            
            let currentLevel = 'coarse';
            let currentCoarseCluster = null;
            
            function showCoarseLevel() {{
                currentLevel = 'coarse';
                currentCoarseCluster = null;
                
                const labels = Object.keys(coarseCounts);
                const values = Object.values(coarseCounts);
                
                const data = [{{
                    labels: labels,
                    values: values,
                    type: 'pie',
                    textinfo: 'label+percent',
                    textposition: 'auto',
                    hovertemplate: '<b>%{{label}}</b><br>Count: %{{value}}<br>Percentage: %{{percent}}<extra></extra>'
                }}];
                
                const layout = {{
                    title: 'Coarse Clusters - Click to drill down',
                    showlegend: true,
                    height: 500
                }};
                
                Plotly.newPlot('chartDiv', data, layout);
                
                // Add click event
                document.getElementById('chartDiv').on('plotly_click', function(data) {{
                    const pointIndex = data.points[0].pointIndex;
                    const clickedLabel = labels[pointIndex];
                    showFineLevel(clickedLabel);
                }});
                
                // Update navigation
                document.getElementById('coarseBtn').style.display = 'none';
                document.getElementById('backBtn').style.display = 'none';
                
                // Update info panel
                document.getElementById('infoTitle').textContent = 'Coarse Clusters Overview';
                document.getElementById('infoContent').innerHTML = 
                    'Total clusters: ' + labels.length + '<br>' +
                    'Total items: ' + values.reduce((a, b) => a + b, 0) + '<br>' +
                    'Click on any cluster to see fine-grained subclusters.';
            }}
            
            function showFineLevel(coarseLabel) {{
                currentLevel = 'fine';
                currentCoarseCluster = coarseLabel;
                
                const fineData = fineByCourse[coarseLabel];
                const labels = Object.keys(fineData);
                const values = Object.values(fineData);
                
                const data = [{{
                    labels: labels,
                    values: values,
                    type: 'pie',
                    textinfo: 'label+percent',
                    textposition: 'auto',
                    hovertemplate: '<b>%{{label}}</b><br>Count: %{{value}}<br>Percentage: %{{percent}}<extra></extra>'
                }}];
                
                const layout = {{
                    title: `Fine Clusters within "${{coarseLabel}}" - Click to see samples`,
                    showlegend: true,
                    height: 500
                }};
                
                Plotly.newPlot('chartDiv', data, layout);
                
                // Add click event for samples
                document.getElementById('chartDiv').on('plotly_click', function(data) {{
                    const pointIndex = data.points[0].pointIndex;
                    const clickedLabel = labels[pointIndex];
                    showSamples(coarseLabel, clickedLabel, values[pointIndex]);
                }});
                
                // Update navigation
                document.getElementById('backBtn').style.display = 'inline-block';
                
                // Update info panel
                document.getElementById('infoTitle').textContent = `Fine Clusters in "${{coarseLabel}}"`;
                document.getElementById('infoContent').innerHTML = 
                    'Subclusters: ' + labels.length + '<br>' +
                    'Total items in this coarse cluster: ' + values.reduce((a, b) => a + b, 0) + '<br>' +
                    'Click on any subcluster to see sample values.';
            }}
            
            function showSamples(coarseLabel, fineLabel, count) {{
                const samples = clusterSamples[coarseLabel][fineLabel];
                
                document.getElementById('infoTitle').textContent = `Samples from "${{fineLabel}}"`;
                
                let sampleHTML = `
                    <strong>Cluster:</strong> ${{fineLabel}}<br>
                    <strong>Parent:</strong> ${{coarseLabel}}<br>
                    <strong>Count:</strong> ${{count}} items<br><br>
                    <strong>Sample values (up to 20):</strong><br>
                    <div class="samples">
                `;
                
                samples.forEach((sample, index) => {{
                    sampleHTML += `${{index + 1}}. ${{sample}}<br>`;
                }});
                
                if (samples.length === 20) {{
                    sampleHTML += '<em>... and more</em>';
                }}
                
                sampleHTML += '</div>';
                
                document.getElementById('infoContent').innerHTML = sampleHTML;
            }}
            
            function goBack() {{
                showCoarseLevel();
            }}
            
            // Initialize with coarse level
            showCoarseLevel();
        </script>
    </body>
    </html>
    """
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"Interactive drill-down visualization saved to: {save_path}")
    
    return html_template


def create_cluster_treemap(df, column_name, title=None, save_path=None):
    """
    Create a treemap visualization showing hierarchical clusters.
    
    Args:
        df: DataFrame with clustering results
        column_name: Base name of the clustered column
        title: Title for the chart
        save_path: Path to save HTML file
    
    Returns:
        Plotly figure object
    """
    coarse_label_col = f"{column_name}_coarse_cluster_label"
    fine_label_col = f"{column_name}_fine_cluster_label"
    
    if coarse_label_col not in df.columns or fine_label_col not in df.columns:
        raise ValueError(f"Required columns not found.")
    
    # Create treemap data
    treemap_data = []
    
    # Group by coarse and fine clusters
    grouped = df.groupby([coarse_label_col, fine_label_col]).size().reset_index(name='count')
    
    for _, row in grouped.iterrows():
        coarse_label = row[coarse_label_col]
        fine_label = row[fine_label_col]
        count = row['count']
        
        treemap_data.append({
            'ids': f"{coarse_label}/{fine_label}",
            'labels': fine_label,
            'parents': coarse_label,
            'values': count
        })
    
    # Add coarse level parents
    coarse_counts = df[coarse_label_col].value_counts()
    for coarse_label, count in coarse_counts.items():
        treemap_data.append({
            'ids': coarse_label,
            'labels': coarse_label,
            'parents': "",
            'values': 0  # Will be calculated automatically
        })
    
    treemap_df = pd.DataFrame(treemap_data)
    
    fig = go.Figure(go.Treemap(
        ids=treemap_df['ids'],
        labels=treemap_df['labels'],
        parents=treemap_df['parents'],
        values=treemap_df['values'],
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent}<extra></extra>',
        maxdepth=2,
    ))
    
    fig.update_layout(
        title=title or f"Cluster Treemap for {column_name}",
        title_x=0.5,
        font_size=12,
        width=1000,
        height=600
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Treemap saved to: {save_path}")
    
    return fig


def create_cluster_statistics_dashboard(df, column_name, save_path=None):
    """
    Create a comprehensive dashboard with multiple views of the clustering results.
    
    Args:
        df: DataFrame with clustering results
        column_name: Base name of the clustered column
        save_path: Path to save HTML file
    """
    coarse_label_col = f"{column_name}_coarse_cluster_label"
    fine_label_col = f"{column_name}_fine_cluster_label"
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Coarse Cluster Distribution', 'Fine Cluster Distribution (Top 20)', 
                       'Cluster Size Distribution', 'Hierarchical Relationship'),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "bar"}]]
    )
    
    # 1. Coarse cluster pie chart
    coarse_counts = df[coarse_label_col].value_counts()
    fig.add_trace(
        go.Pie(labels=coarse_counts.index, values=coarse_counts.values, name="Coarse"),
        row=1, col=1
    )
    
    # 2. Fine cluster bar chart (top 20)
    fine_counts = df[fine_label_col].value_counts().head(20)
    fig.add_trace(
        go.Bar(x=fine_counts.values, y=fine_counts.index, orientation='h', name="Fine Top 20"),
        row=1, col=2
    )
    
    # 3. Cluster size histogram
    fig.add_trace(
        go.Histogram(x=fine_counts.values, nbinsx=20, name="Size Distribution"),
        row=2, col=1
    )
    
    # 4. Hierarchical relationship (samples from each coarse cluster)
    hierarchy_data = []
    hierarchy_labels = []
    
    for coarse_label in coarse_counts.head(10).index:  # Top 10 coarse clusters
        coarse_subset = df[df[coarse_label_col] == coarse_label]
        fine_in_coarse = coarse_subset[fine_label_col].value_counts()
        
        hierarchy_data.extend(fine_in_coarse.values)
        hierarchy_labels.extend([f"{coarse_label[:20]}... -> {fine_label[:15]}..." 
                                for fine_label in fine_in_coarse.index])
    
    fig.add_trace(
        go.Bar(x=hierarchy_data[:20], y=hierarchy_labels[:20], orientation='h', 
               name="Hierarchy Sample"),
        row=2, col=2
    )
    
    fig.update_layout(
        title=f"Clustering Statistics Dashboard - {column_name}",
        height=800,
        showlegend=False
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Statistics dashboard saved to: {save_path}")
    
    return fig


def visualize_clustering_results(parquet_path, column_name, output_dir="cluster_visualizations"):
    """
    Create all visualization types for clustering results.
    
    Args:
        parquet_path: Path to the clustered results parquet file
        column_name: Base name of the clustered column
        output_dir: Directory to save HTML files
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_clustering_results(parquet_path)
    
    base_name = column_name.replace('_', '-')
    
    print("Creating visualizations...")
    
    # 1. Sunburst chart
    print("1. Creating sunburst chart...")
    create_sunburst_visualization(
        df, column_name, 
        title=f"Hierarchical Cluster Sunburst - {column_name}",
        save_path=f"{output_dir}/{base_name}_sunburst.html"
    )
    
    # 2. Interactive drill-down
    print("2. Creating interactive drill-down...")
    create_drill_down_visualization(
        df, column_name,
        title=f"Interactive Cluster Explorer - {column_name}",
        save_path=f"{output_dir}/{base_name}_interactive.html"
    )
    
    # 3. Treemap
    print("3. Creating treemap...")
    create_cluster_treemap(
        df, column_name,
        title=f"Cluster Treemap - {column_name}",
        save_path=f"{output_dir}/{base_name}_treemap.html"
    )
    
    # 4. Statistics dashboard
    print("4. Creating statistics dashboard...")
    create_cluster_statistics_dashboard(
        df, column_name,
        save_path=f"{output_dir}/{base_name}_dashboard.html"
    )
    
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("Available files:")
    print(f"  - {base_name}_sunburst.html (Hierarchical sunburst chart)")
    print(f"  - {base_name}_interactive.html (Click-through drill-down)")
    print(f"  - {base_name}_treemap.html (Treemap visualization)")
    print(f"  - {base_name}_dashboard.html (Statistics dashboard)")
    
    return output_dir


# Example usage
if __name__ == "__main__":
    # Example: visualize property_description clustering results
    
    # Load your clustered results
    parquet_file = "all_one_sided_comparisons_clustered_lightweight.parquet"
    column_name = "property_description"
    
    # Create all visualizations
    visualize_clustering_results(parquet_file, column_name)
    
    # Or create individual visualizations
    # df = load_clustering_results(parquet_file)
    
    # # Just the interactive drill-down (most useful for exploration)
    # create_drill_down_visualization(
    #     df, column_name,
    #     title="Property Description Cluster Explorer",
    #     save_path="property_clusters_interactive.html"
    # )
    
    # # Just the sunburst (good overview)
    # fig = create_sunburst_visualization(
    #     df, column_name,
    #     title="Property Description Hierarchical Clusters",
    #     save_path="property_clusters_sunburst.html"
    # )
    # fig.show()  # Display in browser if running interactively 
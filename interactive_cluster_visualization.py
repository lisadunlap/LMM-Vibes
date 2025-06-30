"""
Simple Interactive Bar Chart for Cluster Visualization

This module provides a clean, interactive bar chart for exploring hierarchical clustering results.
"""

import pandas as pd
import plotly.graph_objects as go
import json
import argparse

def load_clustering_results(parquet_path):
    """Load clustering results from parquet file."""
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows with {df.shape[1]} columns")
    
    # Identify clustering columns
    cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
    print(f"Found clustering columns: {cluster_cols}")
    
    return df


def create_interactive_bar_chart(df, column_name, title=None, save_path=None):
    """
    Create a simple interactive bar chart for exploring clusters.
    
    Args:
        df: DataFrame with clustering results
        column_name: Base name of the clustered column (e.g., 'property_description')
        title: Title for the chart
        save_path: Path to save HTML file
    
    Returns:
        HTML string with interactive visualization
    """
    coarse_label_col = f"{column_name}_coarse_cluster_label"
    fine_label_col = f"{column_name}_fine_cluster_label"
    
    if coarse_label_col not in df.columns or fine_label_col not in df.columns:
        raise ValueError(f"Required columns not found. Need {coarse_label_col} and {fine_label_col}")
    
    # Prepare data - get all coarse clusters
    all_coarse_counts = df[coarse_label_col].value_counts().sort_values(ascending=False).to_dict()
    # Only filter out "outliers" if there are other meaningful clusters; if it's the only cluster, keep it
    coarse_counts = all_coarse_counts
    if len(all_coarse_counts) > 1:
        coarse_counts = {k: v for k, v in all_coarse_counts.items() 
                        if str(k).lower().strip() != 'outliers'}
    
    # Get fine clusters for each coarse cluster
    fine_by_coarse = {}
    cluster_samples = {}
    
    for coarse_label in coarse_counts.keys():
        coarse_subset = df[df[coarse_label_col] == coarse_label]
        all_fine_counts = coarse_subset[fine_label_col].value_counts().sort_values(ascending=False).to_dict()
        # Only filter out "outliers" if there are other meaningful clusters; if it's the only cluster, keep it
        fine_counts = all_fine_counts
        if len(all_fine_counts) > 1:
            fine_counts = {k: v for k, v in all_fine_counts.items() 
                          if str(k).lower().strip() != 'outliers'}
        fine_by_coarse[coarse_label] = fine_counts
        
        # Get sample values for each fine cluster
        cluster_samples[coarse_label] = {}
        for fine_label in fine_counts.keys():
            fine_subset = coarse_subset[coarse_subset[fine_label_col] == fine_label]
            sample_values = fine_subset[column_name].unique()[:10].tolist()
            cluster_samples[coarse_label][fine_label] = [str(v) for v in sample_values]
    
    # Create HTML
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title or f'Interactive Cluster Bar Chart - {column_name}'}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: #f8f9fa;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 30px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #e9ecef;
                padding-bottom: 20px;
            }}
            h1 {{ 
                color: #2c3e50; 
                margin: 0;
                font-size: 28px;
                font-weight: 600;
            }}
            .navigation {{ 
                margin: 20px 0; 
                text-align: center;
            }}
            .btn {{ 
                padding: 12px 24px; 
                margin: 0 10px; 
                background: #3498db; 
                color: white; 
                border: none; 
                border-radius: 6px; 
                cursor: pointer; 
                font-size: 14px;
                font-weight: 500;
                transition: all 0.2s;
            }}
            .btn:hover {{ 
                background: #2980b9; 
                transform: translateY(-1px);
            }}
            .btn:disabled {{
                background: #bdc3c7;
                cursor: not-allowed;
                transform: none;
            }}
            .chart-container {{ 
                margin: 20px 0;
                border: 1px solid #e9ecef;
                border-radius: 6px;
                overflow: hidden;
            }}
            .info-panel {{ 
                background: #f8f9fa; 
                padding: 20px; 
                margin: 20px 0; 
                border-radius: 6px; 
                border-left: 4px solid #3498db;
            }}
            .samples {{ 
                max-height: 300px; 
                overflow-y: auto; 
                font-size: 13px; 
                line-height: 1.5;
                background: white;
                padding: 15px;
                border-radius: 4px;
                margin-top: 10px;
            }}
            .sample-item {{
                padding: 8px 0;
                border-bottom: 1px solid #f1f3f4;
            }}
            .sample-item:last-child {{
                border-bottom: none;
            }}
            .breadcrumb {{
                background: #e9ecef;
                padding: 10px 20px;
                border-radius: 4px;
                margin-bottom: 20px;
                font-size: 14px;
                color: #6c757d;
            }}
            .cluster-list {{
                margin-top: 20px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 6px;
            }}
            .cluster-grid {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }}
            .cluster-item {{
                cursor: pointer;
                padding: 8px 16px;
                background: white;
                border: 1px solid #e9ecef;
                border-radius: 4px;
                transition: all 0.2s;
            }}
            .cluster-item:hover {{
                background: #e9ecef;
            }}
            .cluster-item strong {{
                font-weight: 600;
            }}
            .cluster-item .cluster-count {{
                font-size: 0.8em;
                color: #6c757d;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{title or f'Interactive Cluster Explorer - {column_name}'}</h1>
            </div>
            
            <div class="breadcrumb" id="breadcrumb">
                All Clusters
            </div>
            
            <div class="navigation">
                <button class="btn" onclick="goBack()" id="backBtn" style="display:none;">← Back to Coarse Clusters</button>
            </div>
            
            <div class="chart-container">
                <div id="chartDiv" style="width:100%; height:500px;"></div>
            </div>
            
            <div class="info-panel">
                <h3 id="infoTitle">Cluster Information</h3>
                <div id="infoContent">Click on any bar to explore that cluster in detail.</div>
            </div>
            
            <div class="info-panel">
                <h3 id="clusterListTitle">All Cluster Names</h3>
                <div id="clusterListContent" class="cluster-list">
                    Loading cluster names...
                </div>
            </div>
        </div>

        <script>
            // Data
            const coarseCounts = {json.dumps(coarse_counts)};
            const fineByCourse = {json.dumps(fine_by_coarse)};
            const clusterSamples = {json.dumps(cluster_samples)};
            
            let currentLevel = 'coarse';
            let currentCoarseCluster = null;
            
            function updateClusterList(level, coarseLabel = null) {{
                const listContainer = document.getElementById('clusterListContent');
                const listTitle = document.getElementById('clusterListTitle');
                
                if (level === 'coarse') {{
                    listTitle.textContent = 'All Coarse Cluster Names';
                    const clusters = Object.keys(coarseCounts);
                    let listHTML = '<div class="cluster-grid">';
                    
                    clusters.forEach((cluster, index) => {{
                        const count = coarseCounts[cluster];
                        listHTML += `
                            <div class="cluster-item" onclick="showFineLevel('${{cluster}}')">
                                <strong>${{index + 1}}. ${{cluster}}</strong>
                                <span class="cluster-count">(${{count.toLocaleString()}} items)</span>
                            </div>
                        `;
                    }});
                    
                    listHTML += '</div>';
                    listContainer.innerHTML = listHTML;
                }} else {{
                    listTitle.textContent = `Fine Cluster Names in "${{coarseLabel}}"`;
                    const clusters = Object.keys(fineByCourse[coarseLabel]);
                    let listHTML = '<div class="cluster-grid">';
                    
                    clusters.forEach((cluster, index) => {{
                        const count = fineByCourse[coarseLabel][cluster];
                        listHTML += `
                            <div class="cluster-item" onclick="showSamples('${{coarseLabel}}', '${{cluster}}', ${{count}})">
                                <strong>${{index + 1}}. ${{cluster}}</strong>
                                <span class="cluster-count">(${{count.toLocaleString()}} items)</span>
                            </div>
                        `;
                    }});
                    
                    listHTML += '</div>';
                    listContainer.innerHTML = listHTML;
                }}
            }}
            
            function showCoarseLevel() {{
                currentLevel = 'coarse';
                currentCoarseCluster = null;
                
                const labels = Object.keys(coarseCounts);
                const values = Object.values(coarseCounts);
                
                const data = [{{
                    x: labels,
                    y: values,
                    type: 'bar',
                    marker: {{
                        color: '#3498db',
                        line: {{
                            color: '#2980b9',
                            width: 1
                        }}
                    }},
                    hovertemplate: '<b>%{{x}}</b><br>Count: %{{y:,}}<br>Click to drill down<extra></extra>'
                }}];
                
                const layout = {{
                    title: {{
                        text: 'Coarse Clusters (Click any bar to drill down)',
                        font: {{ size: 16 }}
                    }},
                    xaxis: {{ 
                        title: 'Cluster Labels',
                        tickangle: -45
                    }},
                    yaxis: {{ 
                        title: 'Number of Items'
                    }},
                    margin: {{ t: 60, b: 100, l: 60, r: 30 }},
                    plot_bgcolor: 'white',
                    paper_bgcolor: 'white'
                }};
                
                Plotly.newPlot('chartDiv', data, layout, {{displayModeBar: false}});
                
                // Add click event
                document.getElementById('chartDiv').on('plotly_click', function(data) {{
                    const pointIndex = data.points[0].pointIndex;
                    const clickedLabel = labels[pointIndex];
                    showFineLevel(clickedLabel);
                }});
                
                // Update UI
                document.getElementById('backBtn').style.display = 'none';
                document.getElementById('breadcrumb').textContent = 'All Clusters';
                
                // Update info panel
                document.getElementById('infoTitle').textContent = 'Overview';
                document.getElementById('infoContent').innerHTML = 
                    `<strong>Total coarse clusters:</strong> ${{labels.length}}<br>
                     <strong>Total items:</strong> ${{values.reduce((a, b) => a + b, 0).toLocaleString()}}<br>
                     <strong>Largest cluster:</strong> ${{labels[0]}} (${{Math.max(...values).toLocaleString()}} items)<br><br>
                     Click on any bar above to see the fine-grained subclusters within it.`;
                
                updateClusterList('coarse');
            }}
            
            function showFineLevel(coarseLabel) {{
                currentLevel = 'fine';
                currentCoarseCluster = coarseLabel;
                
                const fineData = fineByCourse[coarseLabel];
                const labels = Object.keys(fineData);
                const values = Object.values(fineData);
                
                const data = [{{
                    x: labels,
                    y: values,
                    type: 'bar',
                    marker: {{
                        color: '#e74c3c',
                        line: {{
                            color: '#c0392b',
                            width: 1
                        }}
                    }},
                    hovertemplate: '<b>%{{x}}</b><br>Count: %{{y:,}}<br>Click to see samples<extra></extra>'
                }}];
                
                const layout = {{
                    title: {{
                        text: `Fine Clusters within "${{coarseLabel}}" (Click bars to see samples)`,
                        font: {{ size: 16 }}
                    }},
                    xaxis: {{ 
                        title: 'Fine Cluster Labels',
                        tickangle: -45
                    }},
                    yaxis: {{ 
                        title: 'Number of Items'
                    }},
                    margin: {{ t: 60, b: 100, l: 60, r: 30 }},
                    plot_bgcolor: 'white',
                    paper_bgcolor: 'white'
                }};
                
                Plotly.newPlot('chartDiv', data, layout, {{displayModeBar: false}});
                
                // Add click event for samples
                document.getElementById('chartDiv').on('plotly_click', function(data) {{
                    const pointIndex = data.points[0].pointIndex;
                    const clickedLabel = labels[pointIndex];
                    showSamples(coarseLabel, clickedLabel, values[pointIndex]);
                }});
                
                // Update UI
                document.getElementById('backBtn').style.display = 'inline-block';
                document.getElementById('breadcrumb').textContent = `All Clusters > ${{coarseLabel}}`;
                
                // Update info panel
                document.getElementById('infoTitle').textContent = `Fine clusters in "${{coarseLabel}}"`;
                document.getElementById('infoContent').innerHTML = 
                    `<strong>Fine clusters:</strong> ${{labels.length}}<br>
                     <strong>Total items:</strong> ${{values.reduce((a, b) => a + b, 0).toLocaleString()}}<br>
                     <strong>Largest subcluster:</strong> ${{labels[0]}} (${{Math.max(...values).toLocaleString()}} items)<br><br>
                     Click on any bar to see sample values from that cluster.`;
                
                updateClusterList('fine', coarseLabel);
            }}
            
            function showSamples(coarseLabel, fineLabel, count) {{
                const samples = clusterSamples[coarseLabel][fineLabel];
                
                document.getElementById('infoTitle').textContent = `Sample values from "${{fineLabel}}"`;
                
                let sampleHTML = `
                    <div style="margin-bottom: 15px;">
                        <strong>Cluster:</strong> ${{fineLabel}}<br>
                        <strong>Parent cluster:</strong> ${{coarseLabel}}<br>
                        <strong>Total items:</strong> ${{count.toLocaleString()}}<br>
                    </div>
                    <strong>Sample values (showing up to 10):</strong>
                    <div class="samples">
                `;
                
                samples.forEach((sample, index) => {{
                    sampleHTML += `<div class="sample-item">${{index + 1}}. ${{sample}}</div>`;
                }});
                
                if (samples.length === 10) {{
                    sampleHTML += '<div class="sample-item" style="font-style: italic; color: #666;">... and more</div>';
                }}
                
                sampleHTML += '</div>';
                
                document.getElementById('infoContent').innerHTML = sampleHTML;
            }}
            
            function goBack() {{
                showCoarseLevel();
            }}
            
            // Initialize
            showCoarseLevel();
        </script>
    </body>
    </html>
    """
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"Interactive bar chart saved to: {save_path}")
    
    return html_template


def create_simple_cluster_visualization(parquet_path, column_name, output_file=None):
    """
    Create a simple interactive bar chart for exploring clusters.
    
    Args:
        parquet_path: Path to the clustered results parquet file
        column_name: Base name of the clustered column
        output_file: Name of output HTML file (optional)
    """
    # Load data
    df = load_clustering_results(parquet_path)
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = column_name.replace('_', '-')
        output_file = f"{base_name}_interactive_bars.html"
    
    print("Creating simple interactive bar chart...")
    
    create_interactive_bar_chart(
        df, column_name,
        title=f"Interactive Cluster Explorer - {column_name.replace('_', ' ').title()}",
        save_path=output_file
    )
    
    print(f"✓ Interactive bar chart saved to: {output_file}")
    print("Open this file in your browser to explore the clusters!")
    
    return output_file


# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create interactive cluster visualization')
    parser.add_argument('--file', '-f', required=True,
                       help='Path to clustered results parquet file')
    parser.add_argument('--column', '-c', default='property_description',
                       help='Base name of the clustered column (default: property_description)')
    args = parser.parse_args()

    parquet_file = args.file
    column_name = args.column
    
    # Create the simple interactive visualization
    create_simple_cluster_visualization(parquet_file, column_name, output_file=args.file.replace(".parquet", "_interactive_bars.html")) 
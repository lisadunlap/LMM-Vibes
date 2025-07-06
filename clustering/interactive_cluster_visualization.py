"""
Simple Interactive Cluster List for Cluster Visualization

This module provides a clean, interactive cluster list for exploring hierarchical clustering results.
"""

import pandas as pd
import json
import argparse

def load_clustering_results(parquet_path):
    """Load clustering results from parquet file."""
    df = pd.read_csv(parquet_path)
    print(f"Loaded {len(df)} rows with {df.shape[1]} columns")
    print(f"Columns: {df.columns}")
    
    # Identify clustering columns
    cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
    print(f"Found clustering columns: {cluster_cols}")
    
    return df


def detect_clustering_structure(df, column_name):
    """
    Detect whether the data has both fine and coarse clusters or just fine clusters.
    
    Returns:
        dict with structure info and column names
    """
    coarse_label_col = f"{column_name}_coarse_cluster_label"
    fine_label_col = f"{column_name}_fine_cluster_label"
    
    has_coarse = coarse_label_col in df.columns
    has_fine = fine_label_col in df.columns
    
    if not has_fine:
        raise ValueError(f"Fine cluster column '{fine_label_col}' not found in data")
    
    if has_coarse and has_fine:
        print("Detected hierarchical clustering: both fine and coarse clusters")
        return {
            'type': 'hierarchical',
            'coarse_col': coarse_label_col,
            'fine_col': fine_label_col,
            'has_coarse': True,
            'has_fine': True
        }
    elif has_fine:
        print("Detected flat clustering: only fine clusters")
        return {
            'type': 'flat',
            'fine_col': fine_label_col,
            'has_coarse': False,
            'has_fine': True
        }
    else:
        raise ValueError(f"No clustering columns found for column '{column_name}'")


def create_interactive_cluster_list(df, column_name, title=None, save_path=None, show_only_fine=False, max_clusters=None):
    """
    Create a simple interactive cluster list for exploring clusters.
    
    Args:
        df: DataFrame with clustering results
        column_name: Base name of the clustered column (e.g., 'property_description')
        title: Title for the chart
        save_path: Path to save HTML file
        show_only_fine: If True, shows only fine-grained clusters without hierarchy
        max_clusters: Maximum number of clusters to show (for large datasets)
    
    Returns:
        HTML string with interactive visualization
    """
    # Detect clustering structure
    structure_info = detect_clustering_structure(df, column_name)
    
    print(f"=== DEBUGGING CLUSTER STRUCTURE ===")
    print(f"Column name: {column_name}")
    print(f"Structure info: {structure_info}")
    print(f"DataFrame columns: {list(df.columns)}")
    print(f"DataFrame shape: {df.shape}")
    
    if structure_info['type'] == 'hierarchical':
        coarse_label_col = structure_info['coarse_col']
        fine_label_col = structure_info['fine_col']
        
        print(f"\n=== HIERARCHICAL CLUSTERING DEBUG ===")
        print(f"Coarse column: {coarse_label_col}")
        print(f"Fine column: {fine_label_col}")
        
        # Check for missing values
        print(f"Missing values in coarse column: {df[coarse_label_col].isna().sum()}")
        print(f"Missing values in fine column: {df[fine_label_col].isna().sum()}")
        
        # Show unique values in both columns
        print(f"Unique coarse labels: {df[coarse_label_col].unique()}")
        print(f"Unique fine labels: {df[fine_label_col].unique()}")
        
        # First, filter out any rows with outlier fine clusters
        # Only filter out fine clusters that are actually outliers, not coarse clusters named "Outliers"
        df_filtered = df[
            ~df[fine_label_col].astype(str).str.lower().str.contains('outlier', na=False)
        ]
        
        print(f"Rows after filtering outliers: {len(df_filtered)} (from {len(df)})")
        
        # Also check if we should filter out coarse clusters named "Outliers"
        print(f"Coarse clusters before filtering: {df_filtered[coarse_label_col].value_counts()}")
        
        # Optionally filter out coarse clusters named "Outliers" if desired
        # Uncomment the next line if you want to exclude the "Outliers" coarse cluster
        # df_filtered = df_filtered[df_filtered[coarse_label_col] != "Outliers"]
        
        print(f"Rows after optional coarse outlier filtering: {len(df_filtered)}")
        
        # Get all fine cluster counts from the filtered data
        all_fine_counts = df_filtered[fine_label_col].value_counts().sort_values(ascending=False).to_dict()
        
        print(f"Fine cluster counts: {all_fine_counts}")
        
        # Group fine clusters by their coarse cluster
        fine_by_coarse = {}
        cluster_samples = {}
        
        # Calculate coarse cluster counts by summing their fine cluster counts
        coarse_counts = {}
        
        # Check for consistency between fine and coarse labels
        print(f"\n=== CHECKING FINE-COARSE CONSISTENCY ===")
        for fine_label in all_fine_counts.keys():
            fine_subset = df_filtered[df_filtered[fine_label_col] == fine_label]
            coarse_labels_for_fine = fine_subset[coarse_label_col].unique()
            print(f"Fine label '{fine_label}' appears in coarse labels: {coarse_labels_for_fine}")
            
            if len(coarse_labels_for_fine) > 1:
                print(f"  WARNING: Fine label '{fine_label}' appears in multiple coarse clusters!")
        
        for fine_label, fine_count in all_fine_counts.items():
            # Get the coarse cluster for this fine cluster
            fine_subset = df_filtered[df_filtered[fine_label_col] == fine_label]
            if len(fine_subset) > 0:
                coarse_label = fine_subset[coarse_label_col].iloc[0]
                
                # Initialize if not exists
                if coarse_label not in fine_by_coarse:
                    fine_by_coarse[coarse_label] = {}
                    cluster_samples[coarse_label] = {}
                    coarse_counts[coarse_label] = 0
                
                # Add fine cluster to its coarse cluster
                fine_by_coarse[coarse_label][fine_label] = fine_count
                coarse_counts[coarse_label] += fine_count
                
                # Get sample values for this fine cluster
                sample_values = fine_subset[column_name].unique()[:10].tolist()
                cluster_samples[coarse_label][fine_label] = [str(v) for v in sample_values]
        
        # Sort coarse counts by value
        coarse_counts = dict(sorted(coarse_counts.items(), key=lambda x: x[1], reverse=True))
        
        # Limit number of clusters if specified
        if max_clusters and len(coarse_counts) > max_clusters:
            print(f"Limiting to top {max_clusters} coarse clusters (from {len(coarse_counts)})")
            top_coarse_clusters = list(coarse_counts.keys())[:max_clusters]
            coarse_counts = {k: v for k, v in coarse_counts.items() if k in top_coarse_clusters}
            fine_by_coarse = {k: v for k, v in fine_by_coarse.items() if k in top_coarse_clusters}
            cluster_samples = {k: v for k, v in cluster_samples.items() if k in top_coarse_clusters}
        
        # Sort fine counts within each coarse cluster
        for coarse_label in fine_by_coarse:
            fine_by_coarse[coarse_label] = dict(sorted(fine_by_coarse[coarse_label].items(), key=lambda x: x[1], reverse=True))
        
        # Sort all_fine_counts by value for show_only_fine mode
        all_fine_counts = dict(sorted(all_fine_counts.items(), key=lambda x: x[1], reverse=True))
        
        # Limit fine clusters if specified and in show_only_fine mode
        if max_clusters and show_only_fine and len(all_fine_counts) > max_clusters:
            print(f"Limiting to top {max_clusters} fine clusters (from {len(all_fine_counts)})")
            top_fine_clusters = list(all_fine_counts.keys())[:max_clusters]
            all_fine_counts = {k: v for k, v in all_fine_counts.items() if k in top_fine_clusters}
        
        # Debug: Print cluster counts for verification
        print(f"\n=== FINAL CLUSTER COUNTS ===")
        print(f"Total rows after filtering outliers: {len(df_filtered)}")
        print(f"Coarse clusters: {len(coarse_counts)}")
        print(f"Fine clusters: {len(all_fine_counts)}")
        print(f"Coarse cluster counts: {coarse_counts}")
        print(f"Fine by coarse structure: {fine_by_coarse}")
        print(f"Total items in coarse clusters: {sum(coarse_counts.values())}")
        print(f"Total items in fine clusters: {sum(all_fine_counts.values())}")
        
    else:
        # Flat structure: only fine clusters
        fine_label_col = structure_info['fine_col']
        
        # Filter out outlier clusters
        df_filtered = df[
            ~df[fine_label_col].astype(str).str.lower().str.contains('outlier', na=False)
        ]
        
        fine_counts = df_filtered[fine_label_col].value_counts().sort_values(ascending=False)
        all_fine_counts = {k: v for k, v in fine_counts.items() 
                          if 'outliers' not in str(k).lower()}
        
        # Limit number of clusters if specified
        if max_clusters and len(all_fine_counts) > max_clusters:
            print(f"Limiting to top {max_clusters} fine clusters (from {len(all_fine_counts)})")
            top_fine_clusters = list(all_fine_counts.keys())[:max_clusters]
            all_fine_counts = {k: v for k, v in all_fine_counts.items() if k in top_fine_clusters}
        
        # For flat structure, we don't have coarse clusters
        coarse_counts = {}
        fine_by_coarse = {}
        cluster_samples = {}
        
        # Get sample values for each fine cluster
        for fine_label in all_fine_counts.keys():
            fine_subset = df_filtered[df_filtered[fine_label_col] == fine_label]
            sample_values = fine_subset[column_name].unique()[:10].tolist()
            cluster_samples[fine_label] = [str(v) for v in sample_values]
        
        # Debug: Print cluster counts for verification
        print(f"Total rows after filtering outliers: {len(df_filtered)}")
        print(f"Fine clusters: {len(all_fine_counts)}")
        print(f"Total items in fine clusters: {sum(all_fine_counts.values())}")
    
    # Create HTML
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title or f'Interactive Cluster Explorer - {column_name}'}</title>
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
            
            <div class="info-panel">
                <h3 id="infoTitle">Cluster Information</h3>
                <div id="infoContent">Click on any cluster to explore it in detail.</div>
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
            const allFineCounts = {json.dumps(all_fine_counts)};
            const showOnlyFine = {json.dumps(show_only_fine)};
            const isFlatStructure = {json.dumps(structure_info['type'] == 'flat')};
            
            let currentLevel = showOnlyFine || isFlatStructure ? 'fine' : 'coarse';
            let currentCoarseCluster = null;
            
            function updateClusterList(level, coarseLabel = null) {{
                const listContainer = document.getElementById('clusterListContent');
                const listTitle = document.getElementById('clusterListTitle');
                
                if (level === 'coarse' && !showOnlyFine && !isFlatStructure) {{
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
                }} else if (!showOnlyFine && !isFlatStructure) {{
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
                }} else {{
                    // Show only fine clusters mode (either show_only_fine or flat structure)
                    listTitle.textContent = 'All Fine Cluster Names';
                    const clusters = Object.keys(allFineCounts);
                    let listHTML = '<div class="cluster-grid">';
                    
                    clusters.forEach((cluster, index) => {{
                        const count = allFineCounts[cluster];
                        if (isFlatStructure) {{
                            listHTML += `
                                <div class="cluster-item" onclick="showSamplesFlat('${{cluster}}', ${{count}})">
                                    <strong>${{index + 1}}. ${{cluster}}</strong>
                                    <span class="cluster-count">(${{count.toLocaleString()}} items)</span>
                                </div>
                            `;
                        }} else {{
                            // Find the parent coarse cluster
                            let parentCluster = null;
                            for (const [coarse, fines] of Object.entries(fineByCourse)) {{
                                if (cluster in fines) {{
                                    parentCluster = coarse;
                                    break;
                                }}
                            }}
                            listHTML += `
                                <div class="cluster-item" onclick="showSamples('${{parentCluster}}', '${{cluster}}', ${{count}})">
                                    <strong>${{index + 1}}. ${{cluster}}</strong>
                                    <span class="cluster-count">(${{count.toLocaleString()}} items)</span>
                                </div>
                            `;
                        }}
                    }});
                    
                    listHTML += '</div>';
                    listContainer.innerHTML = listHTML;
                }}
            }}
            
            function showCoarseLevel() {{
                if (showOnlyFine || isFlatStructure) {{
                    showAllFineClusters();
                    return;
                }}
                
                currentLevel = 'coarse';
                currentCoarseCluster = null;
                
                const labels = Object.keys(coarseCounts);
                const values = Object.values(coarseCounts);
                
                // Update UI
                document.getElementById('backBtn').style.display = 'none';
                document.getElementById('breadcrumb').textContent = 'All Clusters';
                
                // Update info panel
                document.getElementById('infoTitle').textContent = 'Overview';
                document.getElementById('infoContent').innerHTML = 
                    `<strong>Total coarse clusters:</strong> ${{labels.length}}<br>
                     <strong>Total items:</strong> ${{values.reduce((a, b) => a + b, 0).toLocaleString()}}<br>
                     <strong>Largest cluster:</strong> ${{labels[0]}} (${{Math.max(...values).toLocaleString()}} items)<br><br>
                     Click on any cluster below to see the fine-grained subclusters within it.`;
                
                updateClusterList('coarse');
            }}
            
            function showAllFineClusters() {{
                const labels = Object.keys(allFineCounts);
                const values = Object.values(allFineCounts);
                
                // Update UI
                document.getElementById('backBtn').style.display = 'none';
                document.getElementById('breadcrumb').textContent = 'All Fine-Grained Clusters';
                
                // Update info panel
                document.getElementById('infoTitle').textContent = 'Fine-Grained Clusters Overview';
                document.getElementById('infoContent').innerHTML = 
                    `<strong>Total fine clusters:</strong> ${{labels.length}}<br>
                     <strong>Total items:</strong> ${{values.reduce((a, b) => a + b, 0).toLocaleString()}}<br>
                     <strong>Largest cluster:</strong> ${{labels[0]}} (${{Math.max(...values).toLocaleString()}} items)<br><br>
                     Click on any cluster to see sample values from that cluster.`;
                
                updateClusterList('fine');
            }}
            
            function showFineLevel(coarseLabel) {{
                currentLevel = 'fine';
                currentCoarseCluster = coarseLabel;
                
                const fineData = fineByCourse[coarseLabel];
                const labels = Object.keys(fineData);
                const values = Object.values(fineData);
                
                // Update UI
                document.getElementById('backBtn').style.display = 'inline-block';
                document.getElementById('breadcrumb').textContent = `All Clusters > ${{coarseLabel}}`;
                
                // Update info panel
                document.getElementById('infoTitle').textContent = `Fine clusters in "${{coarseLabel}}"`;
                document.getElementById('infoContent').innerHTML = 
                    `<strong>Fine clusters:</strong> ${{labels.length}}<br>
                     <strong>Total items:</strong> ${{values.reduce((a, b) => a + b, 0).toLocaleString()}}<br>
                     <strong>Largest subcluster:</strong> ${{labels[0]}} (${{Math.max(...values).toLocaleString()}} items)<br><br>
                     Click on any cluster to see sample values from that cluster.`;
                
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
            
            function showSamplesFlat(fineLabel, count) {{
                const samples = clusterSamples[fineLabel];
                
                document.getElementById('infoTitle').textContent = `Sample values from "${{fineLabel}}"`;
                
                let sampleHTML = `
                    <div style="margin-bottom: 15px;">
                        <strong>Cluster:</strong> ${{fineLabel}}<br>
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
        print(f"Interactive cluster list saved to: {save_path}")
    
    return html_template


def create_simple_cluster_visualization(parquet_path, column_name, output_file=None, show_only_fine=False, max_clusters=None):
    """
    Create a simple interactive cluster list for exploring clusters.
    
    Args:
        parquet_path: Path to the clustered results parquet file
        column_name: Base name of the clustered column
        output_file: Name of output HTML file (optional)
        show_only_fine: If True, shows only fine-grained clusters without hierarchy
        max_clusters: Maximum number of clusters to show (for large datasets)
    """
    # Load data
    df = load_clustering_results(parquet_path)
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = column_name.replace('_', '-')
        suffix = '_fine_clusters' if show_only_fine else '_interactive_list'
        output_file = f"{base_name}{suffix}.html"
    
    print("Creating simple interactive cluster list...")
    
    create_interactive_cluster_list(
        df, column_name,
        title=f"Interactive Cluster Explorer - {column_name.replace('_', ' ').title()}",
        save_path=output_file,
        show_only_fine=show_only_fine,
        max_clusters=max_clusters
    )
    
    print(f"✓ Interactive cluster list saved to: {output_file}")
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
    parser.add_argument('--fine-only', '-fo', action='store_true',
                       help='Show only fine-grained clusters without hierarchy')
    parser.add_argument('--max-clusters', '-m', type=int, default=None,
                       help='Maximum number of clusters to show (for large datasets)')
    args = parser.parse_args()

    parquet_file = args.file
    column_name = args.column
    
    # Create the simple interactive visualization
    output_file = args.file.replace(".csv.gz", "_fine_clusters.html" if args.fine_only else "_interactive_list.html")
    create_simple_cluster_visualization(parquet_file, column_name, output_file=output_file, show_only_fine=args.fine_only, max_clusters=args.max_clusters) 
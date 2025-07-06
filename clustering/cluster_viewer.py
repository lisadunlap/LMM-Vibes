#!/usr/bin/env python3
"""
Interactive Cluster Visualization Web App

A simple Flask web app for exploring hierarchical clustering results.
Features:
- Dropdown to select cluster summary files
- Interactive visualization of coarse clusters with counts
- Drill-down to fine-grained clusters
- Display property description examples
"""

import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import glob

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class ClusterViewer:
    def __init__(self, cluster_results_dir="cluster_results"):
        self.cluster_results_dir = cluster_results_dir
        self.current_data = None
        self.current_file = None
    
    def get_available_files(self):
        """Get all available summary table files."""
        pattern = os.path.join(self.cluster_results_dir, "**", "*summary_table.jsonl")
        files = glob.glob(pattern, recursive=True)
        return [os.path.relpath(f, self.cluster_results_dir) for f in files]
    
    def load_summary_file(self, file_path):
        """Load a summary table file."""
        full_path = os.path.join(self.cluster_results_dir, file_path)
        if not os.path.exists(full_path):
            return None
        
        try:
            data = pd.read_json(full_path, lines=True)
            self.current_data = data
            self.current_file = file_path
            return data
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            return None
    
    def get_coarse_clusters(self):
        """Get coarse cluster summary."""
        if self.current_data is None:
            return []
        
        coarse_summary = self.current_data.groupby('coarse_label').agg({
            'count': 'sum',
            'percent': 'sum',
            'fine_label': 'nunique'
        }).reset_index()
        
        coarse_summary.columns = ['coarse_label', 'total_count', 'total_percent', 'fine_clusters']
        coarse_summary = coarse_summary.sort_values('total_count', ascending=False)
        
        return coarse_summary.to_dict('records')
    
    def get_fine_clusters(self, coarse_label):
        """Get fine clusters for a specific coarse cluster."""
        if self.current_data is None:
            return []
        
        fine_data = self.current_data[self.current_data['coarse_label'] == coarse_label]
        fine_data = fine_data.sort_values('count', ascending=False)
        
        return fine_data.to_dict('records')
    
    def get_cluster_examples(self, fine_label):
        """Get examples for a specific fine cluster."""
        if self.current_data is None:
            return {}
        
        cluster_data = self.current_data[self.current_data['fine_label'] == fine_label]
        if len(cluster_data) == 0:
            return {}
        
        return cluster_data.iloc[0]['examples']

viewer = ClusterViewer()

@app.route('/')
def index():
    """Main page with file selection and visualization."""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error loading page: {str(e)}", 500

@app.route('/api/files')
def get_files():
    """API endpoint to get available files."""
    files = viewer.get_available_files()
    return jsonify(files)

@app.route('/api/load_file', methods=['POST'])
def load_file():
    """API endpoint to load a summary file."""
    file_path = request.json.get('file_path')
    data = viewer.load_summary_file(file_path)
    
    if data is None:
        return jsonify({'error': 'Failed to load file'}), 400
    
    coarse_clusters = viewer.get_coarse_clusters()
    return jsonify({
        'success': True,
        'coarse_clusters': coarse_clusters,
        'file_path': file_path
    })

@app.route('/api/coarse_clusters')
def get_coarse_clusters():
    """API endpoint to get coarse cluster visualization data."""
    coarse_clusters = viewer.get_coarse_clusters()
    
    if not coarse_clusters:
        return jsonify({'error': 'No data loaded'}), 400
    
    # Create bar chart data
    labels = [item['coarse_label'] for item in coarse_clusters]
    counts = [item['total_count'] for item in coarse_clusters]
    fine_counts = [item['fine_clusters'] for item in coarse_clusters]
    
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=labels,
        y=counts,
        name='Total Count',
        text=[f"{count}<br>({fine} fine)" for count, fine in zip(counts, fine_counts)],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Fine Clusters: %{customdata}<extra></extra>',
        customdata=fine_counts
    ))
    
    fig.update_layout(
        title='Coarse Clusters Overview',
        xaxis_title='Coarse Cluster',
        yaxis_title='Count',
        height=500,
        xaxis={'tickangle': -45}
    )
    
    return jsonify({
        'plot': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder)),
        'data': coarse_clusters
    })

@app.route('/api/fine_clusters/<coarse_label>')
def get_fine_clusters(coarse_label):
    """API endpoint to get fine clusters for a coarse cluster."""
    fine_clusters = viewer.get_fine_clusters(coarse_label)
    
    if not fine_clusters:
        return jsonify({'error': 'No data found for this coarse cluster'}), 400
    
    # Create bar chart for fine clusters
    labels = [item['fine_label'] for item in fine_clusters]
    counts = [item['count'] for item in fine_clusters]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=labels,
        y=counts,
        name='Count',
        text=counts,
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Fine Clusters in "{coarse_label}"',
        xaxis_title='Fine Cluster',
        yaxis_title='Count',
        height=400,
        xaxis={'tickangle': -45}
    )
    
    return jsonify({
        'plot': json.loads(json.dumps(fig, cls=PlotlyJSONEncoder)),
        'data': fine_clusters
    })

@app.route('/api/examples/<fine_label>')
def get_examples(fine_label):
    """API endpoint to get examples for a fine cluster."""
    examples = viewer.get_cluster_examples(fine_label)
    
    if not examples:
        return jsonify({'error': 'No examples found for this fine cluster'}), 400
    
    return jsonify({
        'fine_label': fine_label,
        'examples': examples
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
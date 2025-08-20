"""Dashboard visualization for StringSight pipeline results.

This module provides a Gradio interface for exploring model performance, 
cluster analysis, and detailed examples from pipeline output.

Usage:
    from stringsight.dashboard import launch_app
    launch_app(results_dir="path/to/results")
"""

from .app import launch_app, create_app

__all__ = ["launch_app", "create_app"] 
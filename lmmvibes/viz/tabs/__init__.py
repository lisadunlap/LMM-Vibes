"""
Tabs package for pipeline results app.

This package contains separate modules for each tab in the pipeline results app.
"""

from .overview_tab import create_overview_tab
from .examples_tab import create_examples_tab
from .clusters_tab import create_clusters_tab
from .frequencies_tab import create_frequencies_tab
from .search_tab import create_search_tab

__all__ = [
    'create_overview_tab',
    'create_examples_tab', 
    'create_clusters_tab',
    'create_frequencies_tab',
    'create_search_tab'
] 
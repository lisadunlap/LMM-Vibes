"""
Traditional hierarchical clustering stage.
"""

from ..core.stage import PipelineStage
from ..core.data_objects import PropertyDataset


class HierarchicalClusterer(PipelineStage):
    """Stub hierarchical clusterer."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def run(self, data: PropertyDataset) -> PropertyDataset:
        # TODO: Implement hierarchical clustering
        return data 
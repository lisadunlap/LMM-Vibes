"""
Single model metrics computation stage.
"""

from ..core.stage import PipelineStage
from ..core.data_objects import PropertyDataset


class SingleModelMetrics(PipelineStage):
    """Stub single model metrics."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def run(self, data: PropertyDataset) -> PropertyDataset:
        # TODO: Implement single model metrics
        return data 
"""
PyTorch Lightning modules for connectomics training tasks.

Includes:
- SemanticSegmentationModule: Cross-entropy based semantic segmentation
- InstanceSegmentationModule: Discriminative loss + semantic head
- AffinitySegmentationModule: Affinity-based boundary prediction
- Vista3DModule: 3D Vista/GAPE with semantic + instance + geometry heads
- Vista2DModule: 2D Vista/GAPE with semantic + instance + geometry heads
"""

from neurons.modules.semantic_seg import SemanticSegmentationModule
from neurons.modules.instance_seg import InstanceSegmentationModule
from neurons.modules.affinity_seg import AffinitySegmentationModule
from neurons.modules.vista3d_module import Vista3DModule
from neurons.modules.vista2d_module import Vista2DModule

__all__ = [
    "SemanticSegmentationModule",
    "InstanceSegmentationModule",
    "AffinitySegmentationModule",
    "Vista3DModule",
    "Vista2DModule",
]

"""
PyTorch Lightning modules for connectomics training tasks.

Includes:
- SemanticSegmentationModule: Cross-entropy based semantic segmentation
- InstanceSegmentationModule: Discriminative loss + semantic head
- AffinitySegmentationModule: Affinity-based boundary prediction
- Vista3DModule: Vista3D foundation model with auto/interactive modes
"""

from neurons.modules.semantic_seg import SemanticSegmentationModule
from neurons.modules.instance_seg import InstanceSegmentationModule
from neurons.modules.affinity_seg import AffinitySegmentationModule
from neurons.modules.vista3d_module import Vista3DModule

__all__ = [
    "SemanticSegmentationModule",
    "InstanceSegmentationModule",
    "AffinitySegmentationModule",
    "Vista3DModule",
]

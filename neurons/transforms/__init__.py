"""
Domain-specific MONAI dictionary transforms for connectomics / EM data.

Image augmentations:
- ``ElasticDeformationd``  — elastic warp simulating tissue deformation
- ``MissingSectiond``      — simulate missing Z-slices
- ``Defectsd``             — line artifacts and intensity shifts

Label transforms:
- ``RelabelAfterCropd``      — connected-component relabeling after crop
"""

from neurons.transforms.elastic_deformation import ElasticDeformationd
from neurons.transforms.missing_section import MissingSectiond
from neurons.transforms.defects import Defectsd
from neurons.transforms.relabel_after_crop import RelabelAfterCropd

__all__ = [
    "ElasticDeformationd",
    "MissingSectiond",
    "Defectsd",
    "RelabelAfterCropd",
]

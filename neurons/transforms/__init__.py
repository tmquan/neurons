"""
Domain-specific MONAI dictionary transforms for connectomics / EM data.

Image augmentations:
- ``ElasticDeformationd``  — elastic warp simulating tissue deformation
- ``MissingSectiond``      — simulate missing Z-slices
- ``Defectd``              — line artifacts and intensity shifts

Label transforms:
- ``FindBoundariesd``                 — set boundary voxels to 0 in labels
- ``Labeld`` / ``RelabelAfterCropd``  — connected-component relabeling after crop
- ``Directiond``                      — per-pixel direction toward instance center
- ``Covarianced``                     — per-pixel spatial covariance features
"""

from neurons.transforms.elastic_deformation import ElasticDeformationd
from neurons.transforms.find_boundaries import FindBoundariesd
from neurons.transforms.missing_section import MissingSectiond
from neurons.transforms.defect import Defectd
from neurons.transforms.label import Labeld, RelabelAfterCropd, InstanceWeightsd
from neurons.transforms.direction import Directiond
from neurons.transforms.covariance import Covarianced

__all__ = [
    "ElasticDeformationd",
    "FindBoundariesd",
    "MissingSectiond",
    "Defectd",
    "Labeld",
    "RelabelAfterCropd",
    "InstanceWeightsd",
    "Directiond",
    "Covarianced",
]

"""
Domain-specific MONAI dictionary transforms for connectomics / EM data.

Label transforms:
- ``FindBoundariesd``          — set boundary voxels to 0 in labels
- ``Labeld``                   — connected-component relabeling after crop
- ``Directiond``               — per-pixel direction toward instance center
- ``Covarianced``              — per-pixel spatial covariance features
- ``RandSpatialCropForegroundd`` — foreground-biased random spatial crop
- ``RandResolutionZoomd``      — random resolution zoom augmentation

Elastic deformation uses MONAI's ``Rand3DElasticd`` (configured in datamodules).
"""

from neurons.transforms.label import Labeld
from neurons.transforms.direction import Directiond
from neurons.transforms.covariance import Covarianced
from neurons.transforms.find_boundaries import FindBoundariesd
from neurons.transforms.rand_crop_foreground import RandSpatialCropForegroundd
from neurons.transforms.resolution_zoom import RandResolutionZoomd

__all__ = [
    "Labeld",
    "Directiond",
    "Covarianced",
    "FindBoundariesd",
    "RandSpatialCropForegroundd",
    "RandResolutionZoomd",
]

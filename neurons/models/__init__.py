"""
Model architectures for connectomics segmentation.

Includes:
- BaseModel: Abstract base class for all models
- Vista3DWrapper: 3D Vista/GAPE architecture (semantic + instance + geometry)
- Vista2DWrapper: 2D Vista/GAPE architecture (semantic + instance + geometry)
- SegResNetWrapper: MONAI SegResNet fallback with customizable heads
"""

from neurons.models.base import BaseModel
from neurons.models.segresnet import SegResNetWrapper
from neurons.models.vista3d_model import Vista3DWrapper
from neurons.models.vista2d_model import Vista2DWrapper

__all__ = [
    "BaseModel",
    "SegResNetWrapper",
    "Vista3DWrapper",
    "Vista2DWrapper",
]

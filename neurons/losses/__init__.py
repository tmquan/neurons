"""
Loss functions for connectomics segmentation.

Includes:
- DiscriminativeLoss: Instance embedding loss from De Brabandere et al. (2017)
- DiscriminativeLossVectorized: GPU-optimized version using einops scatter ops
- BoundaryLoss: Boundary-aware cross entropy loss
- BoundaryAwareCrossEntropy: Distance-based boundary weighting
- WeightedBoundaryLoss: Combined class-frequency and boundary weighting
- Vista3DLoss: Combined semantic + instance + geometry loss for 3D
- Vista2DLoss: Combined semantic + instance + geometry loss for 2D
"""

from neurons.losses.discriminative import DiscriminativeLoss, DiscriminativeLossVectorized
from neurons.losses.boundary import BoundaryLoss, BoundaryAwareCrossEntropy
from neurons.losses.weighted_boundary import WeightedBoundaryLoss
from neurons.losses.vista3d_losses import Vista3DLoss
from neurons.losses.vista2d_losses import Vista2DLoss

__all__ = [
    "DiscriminativeLoss",
    "DiscriminativeLossVectorized",
    "BoundaryLoss",
    "BoundaryAwareCrossEntropy",
    "WeightedBoundaryLoss",
    "Vista3DLoss",
    "Vista2DLoss",
]

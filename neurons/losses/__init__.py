"""
Loss functions for connectomics segmentation.

Includes:
- CentroidEmbeddingLoss: Classic discriminative loss (pull/push/reg on centroids)
- SkeletonEmbeddingLoss: Geometry-aware loss (pull to skeleton, DT sampling)
- DiscriminativeLoss: Alias for CentroidEmbeddingLoss (backward compat)
- DiscriminativeLossVectorized: Alias for CentroidEmbeddingLoss (backward compat)
- BoundaryLoss: Boundary-aware cross entropy loss
- BoundaryAwareCrossEntropy: Distance-based boundary weighting
- WeightedBoundaryLoss: Combined class-frequency and boundary weighting
- Vista3DLoss: Combined semantic + instance loss for 3D
- Vista2DLoss: Combined semantic + instance loss for 2D
"""

from neurons.losses.discriminative import (
    CentroidEmbeddingLoss,
    SkeletonEmbeddingLoss,
    DiscriminativeLoss,
    DiscriminativeLossVectorized,
)
from neurons.losses.boundary import BoundaryLoss, BoundaryAwareCrossEntropy
from neurons.losses.weighted_boundary import WeightedBoundaryLoss
from neurons.losses.vista3d_losses import Vista3DLoss
from neurons.losses.vista2d_losses import Vista2DLoss

__all__ = [
    "CentroidEmbeddingLoss",
    "SkeletonEmbeddingLoss",
    "DiscriminativeLoss",
    "DiscriminativeLossVectorized",
    "BoundaryLoss",
    "BoundaryAwareCrossEntropy",
    "WeightedBoundaryLoss",
    "Vista3DLoss",
    "Vista2DLoss",
]

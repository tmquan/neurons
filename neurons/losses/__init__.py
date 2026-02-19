"""
Loss functions for connectomics segmentation.

Includes:
- CentroidEmbeddingLoss: Classic discriminative loss (pull/push/reg on centroids)
- SkeletonEmbeddingLoss: Geometry-aware loss (pull to skeleton, DT sampling)
- DiscriminativeLoss: Alias for CentroidEmbeddingLoss (backward compat)
- DiscriminativeLossVectorized: Alias for CentroidEmbeddingLoss (backward compat)
- Vista3DLoss: Combined semantic + instance loss for 3D
- Vista2DLoss: Combined semantic + instance loss for 2D
"""

from neurons.losses.discriminative import (
    CentroidEmbeddingLoss,
    SkeletonEmbeddingLoss,
    DiscriminativeLoss,
    DiscriminativeLossVectorized,
)
from neurons.losses.vista3d_losses import Vista3DLoss
from neurons.losses.vista2d_losses import Vista2DLoss

__all__ = [
    "CentroidEmbeddingLoss",
    "SkeletonEmbeddingLoss",
    "DiscriminativeLoss",
    "DiscriminativeLossVectorized",
    "Vista3DLoss",
    "Vista2DLoss",
]

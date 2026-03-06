"""
Loss functions for connectomics segmentation.

Three standalone losses (each in its own module):
- SemanticLoss: CE + IoU + Dice (``losses.semantic``)
- InstanceLoss: pull / push / norm (``losses.instance``)
- GeometryLoss: dir / cov / raw  (``losses.geometry``)

Combined losses:
- Vista3DLoss: composes all three for volumetric training
- Vista2DLoss: composes all three for image training

Legacy embedding losses (``losses.discriminative``):
- CentroidEmbeddingLoss, SkeletonEmbeddingLoss
- DiscriminativeLoss, DiscriminativeLossVectorized (aliases)
"""

from neurons.losses.semantic import SemanticLoss
from neurons.losses.instance import InstanceLoss
from neurons.losses.geometry import GeometryLoss
from neurons.losses.vista3d_losses import Vista3DLoss
from neurons.losses.vista2d_losses import Vista2DLoss
from neurons.losses.discriminative import (
    CentroidEmbeddingLoss,
    SkeletonEmbeddingLoss,
    DiscriminativeLoss,
    DiscriminativeLossVectorized,
)

__all__ = [
    # Standalone losses
    "SemanticLoss",
    "InstanceLoss",
    "GeometryLoss",
    # Combined losses
    "Vista3DLoss",
    "Vista2DLoss",
    # Legacy embedding losses
    "CentroidEmbeddingLoss",
    "SkeletonEmbeddingLoss",
    "DiscriminativeLoss",
    "DiscriminativeLossVectorized",
]

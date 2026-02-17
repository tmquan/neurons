"""
Loss functions for connectomics segmentation.

Includes:
- DiscriminativeLoss: Instance embedding loss from De Brabandere et al. (2017)
- DiscriminativeLossVectorized: GPU-optimized version using einops scatter ops
- BoundaryLoss: Boundary-aware cross entropy loss
- BoundaryAwareCrossEntropy: Distance-based boundary weighting
- WeightedBoundaryLoss: Combined class-frequency and boundary weighting
"""

from neurons.losses.discriminative import DiscriminativeLoss, DiscriminativeLossVectorized
from neurons.losses.boundary import BoundaryLoss, BoundaryAwareCrossEntropy
from neurons.losses.weighted_boundary import WeightedBoundaryLoss

__all__ = [
    "DiscriminativeLoss",
    "DiscriminativeLossVectorized",
    "BoundaryLoss",
    "BoundaryAwareCrossEntropy",
    "WeightedBoundaryLoss",
]

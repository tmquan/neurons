"""
Loss functions for connectomics segmentation.

Includes:
- CentroidEmbeddingLoss: Classic discriminative loss (pull/push/reg on centroids)
- SkeletonEmbeddingLoss: Geometry-aware loss (pull to skeleton, DT sampling)
- GeometryLoss: dir/cov/raw regression (configurable: smooth_l1, mse, l1)
- SemanticLoss: CE + Dice on semantic logits (2D and 3D variants)
- InstanceLoss: pull/push/norm discriminative on instance embeddings (2D and 3D)
- Vista3DLoss: Combined semantic + instance + geometry loss for 3D
- Vista2DLoss: Combined semantic + instance + geometry loss for 2D
- DiscriminativeLoss: Alias for CentroidEmbeddingLoss (backward compat)
- DiscriminativeLossVectorized: Alias for CentroidEmbeddingLoss (backward compat)
"""

from neurons.losses.discriminative import (
    CentroidEmbeddingLoss,
    SkeletonEmbeddingLoss,
    GeometryLoss,
    DiscriminativeLoss,
    DiscriminativeLossVectorized,
)
from neurons.losses.vista3d_losses import (
    Vista3DLoss,
    SemanticLoss as SemanticLoss3D,
    InstanceLoss as InstanceLoss3D,
)
from neurons.losses.vista2d_losses import (
    Vista2DLoss,
    SemanticLoss as SemanticLoss2D,
    InstanceLoss as InstanceLoss2D,
)

__all__ = [
    "CentroidEmbeddingLoss",
    "SkeletonEmbeddingLoss",
    "GeometryLoss",
    "SemanticLoss2D",
    "SemanticLoss3D",
    "InstanceLoss2D",
    "InstanceLoss3D",
    "Vista3DLoss",
    "Vista2DLoss",
    "DiscriminativeLoss",
    "DiscriminativeLossVectorized",
]

"""
Vista2D combined loss for image-based segmentation.

Composes three standalone losses:
- SemanticLoss  (from ``neurons.losses.semantic``)
- InstanceLoss  (from ``neurons.losses.instance``)
- GeometryLoss  (from ``neurons.losses.geometry``)
"""

from neurons.losses.cosmos import BaseCombinedLoss


class Vista2DLoss(BaseCombinedLoss):
    """Compose SemanticLoss + InstanceLoss + GeometryLoss for Vista2D."""

    _SPATIAL_DIMS = 2

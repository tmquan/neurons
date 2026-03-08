"""
Vista3D combined loss for volumetric segmentation.

Composes three standalone losses:
- SemanticLoss  (from ``neurons.losses.semantic``)
- InstanceLoss  (from ``neurons.losses.instance``)
- GeometryLoss  (from ``neurons.losses.geometry``)
"""

from neurons.losses.loss import BaseCombinedLoss


class Vista3DLoss(BaseCombinedLoss):
    """Compose SemanticLoss + InstanceLoss + GeometryLoss for Vista3D."""

    _SPATIAL_DIMS = 3

"""
Loss functions for connectomics segmentation.

Standalone task losses:
- :class:`SemanticLoss` -- CE + IoU + Dice
- :class:`InstanceLoss` -- pull / push / norm
- :class:`GeometryLoss` -- dir / cov / raw

Combined multi-head loss used by every Lightning module:
- :class:`CombinedLoss` -- weighted sum of the three task losses
"""

from neurons.losses.semantic import SemanticLoss
from neurons.losses.instance import InstanceLoss
from neurons.losses.geometry import GeometryLoss
from neurons.losses.combined import CombinedLoss

__all__ = [
    "SemanticLoss",
    "InstanceLoss",
    "GeometryLoss",
    "CombinedLoss",
]

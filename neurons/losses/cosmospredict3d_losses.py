"""
Cosmos-Predict2.5 **3D** combined loss for volumetric segmentation.

Composes three standalone losses:

- :class:`SemanticLoss`  -- CE + IoU + Dice
- :class:`InstanceLoss`  -- pull / push / norm
- :class:`GeometryLoss`  -- direction / covariance / raw reconstruction

Optionally adds a **feature-consistency loss** that penalises the L2
distance between DiT features at two augmented views of the same input,
encouraging the backbone representations to be invariant to data
augmentation.  This is useful when fine-tuning the frozen backbone.
"""

from neurons.losses.cosmos import BaseCombinedLossWithConsistency


class CosmosPredict3DLoss(BaseCombinedLossWithConsistency):
    """Composite loss for Cosmos-Predict2.5 3D volumetric segmentation heads.

    Mirrors the :class:`Vista3DLoss` pattern but adds an optional
    ``weight_flow_consistency`` term for feature-level self-consistency.
    """

    _SPATIAL_DIMS = 3

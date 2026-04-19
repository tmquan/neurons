"""
Cosmos-Transfer2.5 **3D** Lightning module for volumetric segmentation.

Only the **automatic** training mode is supported.  See
:class:`BaseCosmosModule` for the full training / evaluation logic.
"""

from neurons.modules.cosmos import BaseCosmosModule
from neurons.models.cosmostransfer3d_model import CosmosTransfer3DWrapper
from neurons.losses import CombinedLoss


class CosmosTransfer3DModule(BaseCosmosModule):
    """Cosmos-Transfer2.5 3-D volumetric segmentation module.

    Three output heads: ``semantic`` ``[B, C, D, H, W]``,
    ``instance`` ``[B, E, D, H, W]``, ``geometry`` ``[B, G, D, H, W]``.
    """

    _model_cls = CosmosTransfer3DWrapper
    _loss_cls = CombinedLoss

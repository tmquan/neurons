"""
Cosmos-Transfer2.5 **3D** Lightning Module for volumetric segmentation training.

Supports two training modes (combinable in a single step):

- **automatic**: predict from the volume alone.
- **proofread**: additional context via fractionary labels or interactive
  point prompts.

Includes :meth:`compatibility_check` to verify that the selected 2B / 14B
variant can handle 3-D input within memory constraints.
"""

from neurons.modules.cosmos import BaseCosmosModule
from neurons.models.cosmostransfer3d_model import (
    CosmosTransfer3DWrapper,
    verify_fit,
)
from neurons.losses.cosmostransfer3d_losses import CosmosTransfer3DLoss


class CosmosTransfer3DModule(BaseCosmosModule):
    """PyTorch Lightning module for 3-D Cosmos-Transfer2.5 segmentation.

    Three-head architecture:

    - ``semantic``  [B, C, D, H, W]
    - ``instance``  [B, E, D, H, W]
    - ``geometry``  [B, G, D, H, W]

    Args:
        model_config: Forwarded to :class:`CosmosTransfer3DWrapper`.
        optimizer_config: Optimizer / scheduler settings.
        loss_config: Forwarded to :class:`CosmosTransfer3DLoss`.
        training_config: Training behaviour (modes, point sampling, ...).
    """

    _model_cls = CosmosTransfer3DWrapper
    _loss_cls = CosmosTransfer3DLoss
    _verify_fit_fn = staticmethod(verify_fit)

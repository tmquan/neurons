"""
Cosmos-Predict2.5 **3D** Lightning Module for volumetric segmentation training.

Supports two training modes (combinable in a single step):

- **automatic**: predict from the volume alone.
- **proofread**: additional context via fractionary labels or interactive
  point prompts.

Includes :meth:`compatibility_check` to verify that the selected 2B / 14B
variant can handle 3-D input within memory constraints.
"""

from neurons.modules.cosmos import BaseCosmosModule
from neurons.models.cosmospredict3d_model import (
    CosmosPredict3DWrapper,
    verify_fit,
)
from neurons.losses.cosmospredict3d_losses import CosmosPredict3DLoss


class CosmosPredict3DModule(BaseCosmosModule):
    """PyTorch Lightning module for 3-D Cosmos-Predict2.5 segmentation.

    Three-head architecture:

    - ``semantic``  [B, C, D, H, W]
    - ``instance``  [B, E, D, H, W]
    - ``geometry``  [B, G, D, H, W]

    Args:
        model_config: Forwarded to :class:`CosmosPredict3DWrapper`.
        optimizer_config: Optimizer / scheduler settings.
        loss_config: Forwarded to :class:`CosmosPredict3DLoss`.
        training_config: Training behaviour (modes, point sampling, ...).
    """

    model_cls = CosmosPredict3DWrapper
    loss_cls = CosmosPredict3DLoss
    verify_fit_fn = staticmethod(verify_fit)

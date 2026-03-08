"""
Vista2D Lightning Module for image-based segmentation training.

Supports two training modes that can be combined in a single step:

- **automatic**: predict everything from the image alone.
- **proofread**: additional context (fractionary labels or interactive
  point prompts) is provided.  Sub-modes:

  - *fractionary*: partial annotation exists — resolve labels and forward
    with ``semantic_ids``.
  - *interactive*: fully annotated — simulate point prompts sampled from GT.

When ``training_modes`` contains both, every batch runs both forward passes
and the losses are averaged.
"""

from neurons.modules.vista import BaseVistaModule
from neurons.models.vista2d_model import Vista2DWrapper
from neurons.losses.vista2d_losses import Vista2DLoss


class Vista2DModule(BaseVistaModule):
    """PyTorch Lightning module for Vista2D-based image segmentation.

    Three-head architecture:
    - semantic: per-pixel class logits  [B, C, H, W]
    - instance: per-pixel embeddings    [B, E, H, W]
    - geometry: per-pixel geometry      [B, G, H, W]

    Args:
        model_config: Model configuration dict.
        optimizer_config: Optimizer configuration dict.
        loss_config: Loss function configuration dict.
        training_config: Training configuration dict (contains
            ``training_modes``, ``num_pos_points``, etc.).
    """

    _SPATIAL_DIMS = 2
    _model_cls = Vista2DWrapper
    _loss_cls = Vista2DLoss

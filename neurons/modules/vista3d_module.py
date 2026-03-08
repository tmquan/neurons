"""
Vista3D Lightning Module for volumetric segmentation training.

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
from neurons.models.vista3d_model import Vista3DWrapper
from neurons.losses.vista3d_losses import Vista3DLoss


class Vista3DModule(BaseVistaModule):
    """PyTorch Lightning module for Vista3D-based volumetric segmentation.

    Three-head architecture:
    - semantic: per-voxel class logits  [B, C, D, H, W]
    - instance: per-voxel embeddings    [B, E, D, H, W]
    - geometry: per-voxel geometry      [B, G, D, H, W]

    Args:
        model_config: Model configuration dict.
        optimizer_config: Optimizer configuration dict.
        loss_config: Loss function configuration dict.
        training_config: Training configuration dict (contains
            ``training_modes``, ``num_pos_points``, etc.).
    """

    _SPATIAL_DIMS = 3
    _model_cls = Vista3DWrapper
    _loss_cls = Vista3DLoss

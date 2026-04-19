"""Vista2D Lightning module for image-based connectomics segmentation."""

from neurons.modules.vista import BaseVistaModule
from neurons.models.vista2d_model import Vista2DWrapper
from neurons.losses import CombinedLoss


class Vista2DModule(BaseVistaModule):
    """Vista2D image segmentation module.

    Three output heads: ``semantic`` ``[B, C, H, W]``,
    ``instance`` ``[B, E, H, W]``, ``geometry`` ``[B, G, H, W]``.
    """

    _SPATIAL_DIMS = 2
    _model_cls = Vista2DWrapper
    _loss_cls = CombinedLoss

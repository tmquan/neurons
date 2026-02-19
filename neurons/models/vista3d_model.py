"""
Vista3D model wrapper for volumetric connectomics segmentation.

3D version of the Vista/GAPE architecture with two parallel task heads:
- Semantic: per-voxel class logits (num_classes channels)
- Instance: per-voxel embedding vectors for discriminative clustering (emb_dim channels)
"""

from typing import Any, Dict

import torch
import torch.nn as nn

_SPATIAL_DIMS = 3
_CONV = nn.Conv3d
_NORM = nn.BatchNorm3d


class Vista3DWrapper(nn.Module):
    """
    3D version of the Vista architecture for volumetric segmentation.

    Args:
        in_channels: Number of input channels (default: 1 for EM).
        num_classes: Number of semantic classes (default: 16).
            Set higher than currently needed to leave headroom for
            future class additions without retraining the backbone.
        emb_dim: Instance embedding dimensionality (default: 16).
        feature_size: Base feature dimension from backbone (default: 48).
        encoder_name: Backbone encoder ('segresnet' or 'swin').

    Example:
        >>> model = Vista3DWrapper(in_channels=1, num_classes=16, emb_dim=16)
        >>> x = torch.randn(1, 1, 64, 64, 64)
        >>> out = model(x)
        >>> out['semantic'].shape   # [1, 16, 64, 64, 64]
        >>> out['instance'].shape   # [1, 16, 64, 64, 64]
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 16,
        emb_dim: int = 16,
        feature_size: int = 48,
        encoder_name: str = "segresnet",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.feature_size = feature_size

        self._build_backbone(encoder_name, **kwargs)

        self.head_semantic = nn.Sequential(
            _CONV(feature_size, 64, 3, padding=1), _NORM(64), nn.ReLU(inplace=True),
            _CONV(64, num_classes, 1),
        )
        self.head_instance = nn.Sequential(
            _CONV(feature_size, 64, 3, padding=1), _NORM(64), nn.ReLU(inplace=True),
            _CONV(64, emb_dim, 1),
        )

    def _build_backbone(self, encoder_name: str, **kwargs: Any) -> None:
        """Build backbone, falling back to SegResNet if Vista3D is unavailable."""
        try:
            from monai.networks.nets import vista3d
            self.vista3d = vista3d.Vista3D(
                in_channels=self.in_channels,
                encoder_name=encoder_name,
                feature_size=self.feature_size,
                **kwargs,
            )
            self._has_vista3d = True
        except (ImportError, AttributeError):
            from monai.networks.nets import SegResNet
            self.backbone = SegResNet(
                spatial_dims=_SPATIAL_DIMS,
                in_channels=self.in_channels,
                out_channels=self.feature_size,
                init_filters=self.feature_size,
            )
            self._has_vista3d = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through backbone + two parallel heads."""
        feat = self.vista3d(x) if self._has_vista3d else self.backbone(x)

        return {
            "semantic": self.head_semantic(feat),
            "instance": self.head_instance(feat),
        }

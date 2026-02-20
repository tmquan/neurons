"""
Vista3D model wrapper for volumetric connectomics segmentation.

3D version of the Vista architecture with three parallel task heads:
- Semantic: per-voxel class logits (num_classes channels)
- Instance: per-voxel embedding vectors for discriminative clustering (emb_dim channels)
- Geometry: per-voxel direction, structure tensor, and RGBA reconstruction
"""

from typing import Any, Dict, Optional

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
        feature_size: Base feature dimension from backbone (default: 64).
        encoder_name: Vista3D internal encoder ('segresnet' or 'swin').

    Example:
        >>> model = Vista3DWrapper(in_channels=1, num_classes=16, emb_dim=16)
        >>> x = torch.randn(1, 1, 64, 64, 64)
        >>> out = model(x)
        >>> out['semantic'].shape   # [1, 16, 64, 64, 64]
        >>> out['instance'].shape   # [1, 16, 64, 64, 64]
        >>> out['geometry'].shape   # [1, 16, 64, 64, 64]  (dir=3 + cov=9 + raw=4)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 16,
        emb_dim: int = 16,
        feature_size: int = 64,
        encoder_name: str = "vista3d",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.feature_size = feature_size
        self.spatial_dims = _SPATIAL_DIMS

        S = _SPATIAL_DIMS
        self.geom_channels = S + S * S + 4  # dir + cov + rgba

        self._build_backbone(encoder_name, **kwargs)

        self.head_semantic = nn.Sequential(
            _CONV(feature_size, 64, 3, padding=1), _NORM(64), nn.ReLU(inplace=True),
            _CONV(64, num_classes, 1),
        )
        self.head_instance = nn.Sequential(
            _CONV(feature_size, 64, 3, padding=1), _NORM(64), nn.ReLU(inplace=True),
            _CONV(64, emb_dim, 1),
        )
        self.head_geometry = nn.Sequential(
            _CONV(feature_size, 64, 3, padding=1), _NORM(64), nn.ReLU(inplace=True),
            _CONV(64, self.geom_channels, 1),
        )

    def _build_backbone(self, encoder_name: str, **kwargs: Any) -> None:
        """Build Vista3D backbone, falling back to SegResNet if unavailable."""
        try:
            from monai.networks.nets import vista3d
            self.backbone = vista3d.Vista3D(
                in_channels=self.in_channels,
                encoder_name=encoder_name,
                feature_size=self.feature_size,
                **kwargs,
            )
            self._use_vista3d = True
        except (ImportError, AttributeError):
            from monai.networks.nets import SegResNet
            self.backbone = SegResNet(
                spatial_dims=_SPATIAL_DIMS,
                in_channels=self.in_channels,
                out_channels=self.feature_size,
                init_filters=self.feature_size,
            )
            self._use_vista3d = False

    def forward(
        self,
        x: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through backbone + three parallel heads.

        Args:
            x: Input tensor [B, C, D, H, W].
            class_ids: Optional per-voxel semantic class labels [B, D, H, W].
                Passed through so the loss can compute per-class instance losses.
        """
        feat = self.backbone(x)

        out: Dict[str, torch.Tensor] = {
            "semantic": self.head_semantic(feat),
            "instance": self.head_instance(feat),
            "geometry": self.head_geometry(feat),
        }
        if class_ids is not None:
            out["class_ids"] = class_ids
        return out

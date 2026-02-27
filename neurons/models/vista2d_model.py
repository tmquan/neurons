"""
Vista2D model wrapper for image-based connectomics segmentation.

2D version of the Vista architecture with three parallel task heads:
- Semantic: per-pixel class logits (num_classes channels)
- Instance: per-pixel embedding vectors for discriminative clustering (emb_dim channels)
- Geometry: per-pixel direction, structure tensor, and RGBA reconstruction
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from neurons.models.point_prompt_encoder import PointPromptEncoder

_SPATIAL_DIMS = 2
_CONV = nn.Conv2d


def _NORM(ch: int) -> nn.GroupNorm:
    num_groups = max(g for g in (1, 2, 4, 8, 16, 32) if ch % g == 0)
    return nn.GroupNorm(num_groups, ch)


class Vista2DWrapper(nn.Module):
    """
    2D version of the Vista architecture for image segmentation.

    Args:
        in_channels: Number of input channels (default: 1 for EM).
        num_classes: Number of semantic classes (default: 16).
            Set higher than currently needed to leave headroom for
            future class additions without retraining the backbone.
        emb_dim: Instance embedding dimensionality (default: 16).
        feature_size: Base feature dimension from backbone (default: 64).
        encoder_name: Vista3D internal encoder ('segresnet' or 'swin').

    Example:
        >>> model = Vista2DWrapper(in_channels=1, num_classes=16, emb_dim=16)
        >>> x = torch.randn(4, 1, 256, 256)
        >>> out = model(x)
        >>> out['semantic'].shape   # [4, 16, 256, 256]
        >>> out['instance'].shape   # [4, 16, 256, 256]
        >>> out['geometry'].shape   # [4, 10, 256, 256]  (dir=2 + cov=4 + raw=4)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 16,
        emb_dim: int = 16,
        feature_size: int = 64,
        encoder_name: str = "vista3d",
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.feature_size = feature_size
        self.spatial_dims = _SPATIAL_DIMS
        self.dropout = dropout

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

        self.point_encoder = PointPromptEncoder(
            num_classes=num_classes,
            feature_size=feature_size,
            spatial_dims=_SPATIAL_DIMS,
        )

    def _build_backbone(self, encoder_name: str, **kwargs: Any) -> None:
        """Build backbone encoder: SegResNetDS2 or SegResNet fallback."""
        if encoder_name in ("vista3d", "segresnet_ds2"):
            try:
                from monai.networks.nets.segresnet_ds import SegResNetDS2
                self.backbone = SegResNetDS2(
                    spatial_dims=_SPATIAL_DIMS,
                    in_channels=self.in_channels,
                    out_channels=self.feature_size,
                    init_filters=self.feature_size,
                    blocks_down=(1, 2, 2, 4, 4),
                    norm="instance",
                    dsdepth=1,
                )
                return
            except ImportError:
                import warnings
                warnings.warn(
                    "SegResNetDS2 not available, falling back to SegResNet. "
                    "Install monai>=1.3 for Vista encoder support.",
                    stacklevel=2,
                )

        from monai.networks.nets import SegResNet
        self.backbone = SegResNet(
            spatial_dims=_SPATIAL_DIMS,
            in_channels=self.in_channels,
            out_channels=self.feature_size,
            init_filters=self.feature_size,
            dropout_prob=self.dropout,
        )
        self._use_vista3d = False

    def forward(
        self,
        x: torch.Tensor,
        semantic_ids: Optional[torch.Tensor] = None,
        point_prompts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through backbone + three parallel heads.

        Args:
            x: Input tensor [B, C, H, W].
            semantic_ids: Optional per-pixel semantic class labels [B, H, W].
                Passed through so the loss can compute per-class instance losses.
            point_prompts: Optional dict with keys ``pos_points``,
                ``neg_points``, ``target_semantic_ids``, ``target_instance_ids``
                as produced by :func:`sample_point_prompts`.
        """
        feat = self.backbone(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        if point_prompts is not None:
            feat = feat + self.point_encoder(
                pos_points=point_prompts["pos_points"],
                neg_points=point_prompts["neg_points"],
                target_semantic_ids=point_prompts["target_semantic_ids"],
                target_instance_ids=point_prompts["target_instance_ids"],
                spatial_shape=feat.shape[2:],
            )

        out: Dict[str, torch.Tensor] = {
            "semantic": self.head_semantic(feat),
            "instance": self.head_instance(feat),
            "geometry": self.head_geometry(feat),
        }
        if semantic_ids is not None:
            out["semantic_ids"] = semantic_ids
        return out

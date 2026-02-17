"""
Vista3D model wrapper for connectomics segmentation.

Vista3D is NVIDIA's 3D foundation model for medical image segmentation,
available in MONAI. This wrapper adapts it for connectomics applications.
"""

import warnings
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from einops import rearrange

from neurons.models.base import BaseModel


class Vista3DWrapper(BaseModel):
    """
    Wrapper for MONAI Vista3D foundation model adapted for connectomics.

    Vista3D supports:
    - Interactive segmentation with point/box prompts
    - Automatic segmentation for batch processing
    - Fine-tuning for domain-specific tasks

    Falls back to SegResNet if Vista3D is not available in the
    installed MONAI version.

    Args:
        in_channels: Number of input channels (default: 1 for EM).
        num_classes: Number of output classes (default: 2).
        pretrained: Load pretrained weights (default: True).
        freeze_encoder: Freeze encoder for fine-tuning (default: False).
        encoder_name: Encoder architecture ('segresnet' or 'swin').
        feature_size: Base feature size for encoder.
        use_point_prompts: Enable interactive point prompts.
        use_automatic_mode: Enable automatic segmentation mode.

    Example:
        >>> model = Vista3DWrapper(in_channels=1, num_classes=2)
        >>> x = torch.randn(1, 1, 64, 128, 128)
        >>> output = model(x)
        >>> logits = output["logits"]  # [1, 2, 64, 128, 128]
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        encoder_name: str = "segresnet",
        feature_size: int = 48,
        use_point_prompts: bool = False,
        use_automatic_mode: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=num_classes,
            spatial_dims=3,
        )

        self.pretrained = pretrained
        self.freeze_encoder_flag = freeze_encoder
        self.encoder_name = encoder_name
        self.feature_size = feature_size
        self.use_point_prompts = use_point_prompts
        self.use_automatic_mode = use_automatic_mode

        self._build_model(**kwargs)

        if freeze_encoder:
            self.freeze_encoder()

    def _build_model(self, **kwargs: Any) -> None:
        """
        Build the Vista3D model architecture.

        Falls back to SegResNet if Vista3D is not available.
        """
        try:
            from monai.networks.nets import vista3d

            self.vista3d = vista3d.Vista3D(
                in_channels=self.in_channels,
                encoder_name=self.encoder_name,
                feature_size=self.feature_size,
                **kwargs,
            )

            self.output_head = nn.Conv3d(
                self.feature_size,
                self.out_channels,
                kernel_size=1,
            )

            self._has_vista3d = True

        except (ImportError, AttributeError):
            warnings.warn(
                "Vista3D not available in this MONAI version. "
                "Using SegResNet backbone instead. "
                "Install MONAI >= 1.3.0 with vista3d extras for full support.",
                stacklevel=2,
            )

            from monai.networks.nets import SegResNet

            self.backbone = SegResNet(
                spatial_dims=3,
                in_channels=self.in_channels,
                out_channels=self.feature_size,
                init_filters=self.feature_size,
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
            )

            self.output_head = nn.Conv3d(
                self.feature_size,
                self.out_channels,
                kernel_size=1,
            )

            self._has_vista3d = False

    def forward(
        self,
        x: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        class_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional interactive prompts.

        Args:
            x: Input tensor [B, C, D, H, W].
            point_coords: Point prompt coordinates [B, N, 3] (z, y, x).
            point_labels: Point labels [B, N] (0=background, 1=foreground).
            boxes: Box prompts [B, M, 6] (z1, y1, x1, z2, y2, x2).
            class_ids: Class IDs for automatic mode [B] or [B, K].

        Returns:
            Dictionary with 'logits' and optionally 'features', 'embeddings'.
        """
        outputs: Dict[str, torch.Tensor] = {}

        if self._has_vista3d:
            if point_coords is not None or boxes is not None:
                vista_out = self.vista3d(
                    x,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    boxes=boxes,
                )
            else:
                vista_out = self.vista3d(x, class_ids=class_ids)

            if isinstance(vista_out, dict):
                features = vista_out.get("features", vista_out.get("logits"))
                embeddings = vista_out.get("embeddings")
                if embeddings is not None:
                    outputs["embeddings"] = embeddings
            else:
                features = vista_out

            outputs["features"] = features
            outputs["logits"] = self.output_head(features)

        else:
            features = self.backbone(x)
            outputs["features"] = features
            outputs["logits"] = self.output_head(features)

        return outputs

    def get_output_channels(self) -> int:
        return self.out_channels

    def freeze_encoder(self) -> None:
        """Freeze encoder/backbone parameters for fine-tuning."""
        if self._has_vista3d:
            for param in self.vista3d.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.freeze_encoder_flag = True

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder/backbone parameters."""
        if self._has_vista3d:
            for param in self.vista3d.encoder.parameters():
                param.requires_grad = True
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
        self.freeze_encoder_flag = False

    def load_pretrained_weights(self, weights_path: str) -> None:
        """
        Load pretrained weights from file.

        Args:
            weights_path: Path to weights file (.pt or .pth).
        """
        state_dict = torch.load(weights_path, map_location="cpu")

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        self.load_state_dict(state_dict, strict=False)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "vista3d_segresnet",
        **kwargs: Any,
    ) -> "Vista3DWrapper":
        """
        Load a pretrained Vista3D model.

        Args:
            model_name: Name of pretrained model configuration.
            **kwargs: Additional arguments passed to constructor.

        Returns:
            Initialized Vista3DWrapper with pretrained weights.
        """
        configs = {
            "vista3d_segresnet": {
                "encoder_name": "segresnet",
                "feature_size": 48,
            },
            "vista3d_swin": {
                "encoder_name": "swin",
                "feature_size": 48,
            },
        }

        if model_name in configs:
            config = configs[model_name]
            config.update(kwargs)
            return cls(pretrained=True, **config)
        else:
            raise ValueError(
                f"Unknown model name: {model_name}. "
                f"Available: {list(configs.keys())}"
            )

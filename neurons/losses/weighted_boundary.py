"""
Weighted boundary loss combining class-frequency balancing and boundary proximity.

Extends the boundary-aware losses with per-class inverse-frequency weighting
scaled by boundary proximity, giving rare classes near boundaries the highest
weight during training.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class WeightedBoundaryLoss(nn.Module):
    """
    Combined class-frequency and boundary-weighted cross entropy loss.

    This loss merges two objectives:
    1. **Class-frequency weighting**: Rare classes receive higher weight via
       inverse-frequency scaling so the model is not biased toward the dominant
       background class.
    2. **Boundary weighting**: Pixels near instance boundaries are up-weighted
       using morphological dilation/erosion so the model sharpens its edge
       predictions.

    The final per-pixel weight is:

        w(x) = class_weight(x) * (1 + (boundary_weight - 1) * boundary(x))

    Args:
        num_classes: Number of semantic classes.
        boundary_weight: Extra weight factor for boundary pixels (default: 5.0).
        kernel_size: Morphological kernel size for boundary detection (default: 3).
        class_weights: Optional manual per-class weights [C].  When None the loss
            computes inverse-frequency weights from each batch on the fly.
        ignore_index: Label index to ignore in loss computation (default: -100).

    Example:
        >>> loss_fn = WeightedBoundaryLoss(num_classes=4, boundary_weight=5.0)
        >>> logits = torch.randn(2, 4, 64, 128, 128)
        >>> labels = torch.randint(0, 4, (2, 64, 128, 128))
        >>> loss = loss_fn(logits, labels)
    """

    def __init__(
        self,
        num_classes: int = 2,
        boundary_weight: float = 5.0,
        kernel_size: int = 3,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.boundary_weight = boundary_weight
        self.kernel_size = kernel_size
        self.ignore_index = ignore_index

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Boundary detection
    # ------------------------------------------------------------------
    def _compute_boundaries_2d(self, labels: torch.Tensor) -> torch.Tensor:
        """Detect boundaries in a 2D label map [B, H, W] -> [B, H, W]."""
        labels_float = rearrange(labels.float(), "b h w -> b 1 h w")

        eroded = -F.max_pool2d(
            -labels_float,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        dilated = F.max_pool2d(
            labels_float,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        boundary = rearrange((dilated != eroded).float(), "b 1 h w -> b h w")
        return boundary

    def _compute_boundaries_3d(self, labels: torch.Tensor) -> torch.Tensor:
        """Detect boundaries in a 3D label volume [B, D, H, W] -> [B, D, H, W]."""
        labels_float = rearrange(labels.float(), "b d h w -> b 1 d h w")

        eroded = -F.max_pool3d(
            -labels_float,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        dilated = F.max_pool3d(
            labels_float,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        boundary = rearrange((dilated != eroded).float(), "b 1 d h w -> b d h w")
        return boundary

    # ------------------------------------------------------------------
    # Class-frequency weighting
    # ------------------------------------------------------------------
    def _compute_class_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse-frequency class weights from a label tensor.

        Args:
            labels: [B, ...] integer labels.

        Returns:
            Tensor of shape [num_classes] with weights.
        """
        flat = labels.reshape(-1)
        counts = torch.bincount(flat.clamp(min=0), minlength=self.num_classes).float()
        counts = counts[: self.num_classes]
        total = counts.sum()

        inv_freq = total / (counts + 1e-6)
        inv_freq = inv_freq / inv_freq.sum() * self.num_classes

        return inv_freq

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted boundary loss.

        Args:
            logits: Prediction logits [B, C, *spatial].
            labels: Ground truth labels [B, *spatial] (long).

        Returns:
            Scalar loss value.
        """
        spatial_dims = logits.dim() - 2
        labels = labels.long()

        # ---- class weights ----
        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
        else:
            cw = self._compute_class_weights(labels).to(logits.device)

        # per-pixel class weight
        safe_labels = labels.clamp(0, self.num_classes - 1)
        class_weight_map = cw[safe_labels]

        # ---- boundary weights ----
        if spatial_dims == 2:
            boundary = self._compute_boundaries_2d(labels)
        elif spatial_dims == 3:
            boundary = self._compute_boundaries_3d(labels)
        else:
            boundary = torch.zeros_like(labels, dtype=torch.float32)

        boundary_weight_map = 1.0 + (self.boundary_weight - 1.0) * boundary

        # ---- combined weight ----
        combined_weights = class_weight_map * boundary_weight_map

        # zero-out ignored pixels
        if self.ignore_index >= 0:
            ignore_mask = labels == self.ignore_index
            combined_weights = combined_weights * (~ignore_mask).float()

        # ---- cross entropy with per-pixel weighting ----
        ce_loss = F.cross_entropy(logits, labels, reduction="none", ignore_index=self.ignore_index)
        weighted_loss = (ce_loss * combined_weights).sum() / (combined_weights.sum() + 1e-8)

        return weighted_loss

"""
Vista2D Lightning Module for image-based segmentation training.

Uses the 3-head Vista2D model (semantic + instance + geometry) with
the Vista2DLoss for combined multi-task training.
"""

from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from einops import rearrange

from neurons.models.vista2d_model import Vista2DWrapper as _Model
from neurons.losses.vista2d_losses import Vista2DLoss as _Loss

_SPATIAL_DIMS = 2
_EXPAND_PATTERN = "b h w -> b 1 h w"
_SQUEEZE_PATTERN = "b 1 h w -> b h w"


class Vista2DModule(pl.LightningModule):
    """
    PyTorch Lightning module for Vista2D-based image segmentation.

    Three-head architecture:
    - semantic: per-pixel class logits  [B, 16, H, W]
    - instance: per-pixel embeddings    [B, 16, H, W]
    - geometry: affinity/grid/rgba      [B, 16, H, W]

    Args:
        model_config: Model configuration dict.
        optimizer_config: Optimizer configuration dict.
        loss_config: Loss function configuration dict.
    """

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        model_config = model_config or {}
        self.optimizer_config = optimizer_config or {}
        loss_config = loss_config or {}

        self.model = _Model(
            in_channels=model_config.get("in_channels", 1),
            num_classes=model_config.get("num_classes", 16),
            feature_size=model_config.get("feature_size", 48),
            encoder_name=model_config.get("encoder_name", "segresnet"),
        )

        self.criterion = _Loss(
            weight_pull=loss_config.get("weight_pull", 1.0),
            weight_push=loss_config.get("weight_push", 1.0),
            weight_norm=loss_config.get("weight_norm", 0.001),
            weight_edge=loss_config.get("weight_edge", 10.0),
            weight_bone=loss_config.get("weight_bone", 10.0),
            delta_v=loss_config.get("delta_v", 0.5),
            delta_d=loss_config.get("delta_d", 1.5),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through 3-head model."""
        return self.model(x)

    def _prepare_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract and reshape targets from batch dict."""
        labels = batch["label"]
        if labels.dim() == _SPATIAL_DIMS + 2:
            labels = rearrange(labels, _SQUEEZE_PATTERN)

        targets: Dict[str, torch.Tensor] = {
            "class_labels": batch.get("class_ids", (labels > 0).long()),
            "labels": labels,
        }

        spatial = labels.shape[1:]
        targets["gt_diff"] = batch.get("gt_diff", torch.zeros(labels.shape[0], 9, *spatial, device=labels.device))
        targets["gt_grid"] = batch.get("gt_grid", torch.zeros(labels.shape[0], 3, *spatial, device=labels.device))
        targets["gt_rgba"] = batch.get("gt_rgba", torch.zeros(labels.shape[0], 4, *spatial, device=labels.device))

        return targets

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images = batch["image"]
        if images.dim() == _SPATIAL_DIMS + 1:
            images = rearrange(images, _EXPAND_PATTERN)

        predictions = self(images)
        targets = self._prepare_targets(batch)
        losses = self.criterion(predictions, targets)

        bs = images.shape[0]
        for name, val in losses.items():
            self.log(f"train/{name}", val, prog_bar=(name == "loss"), batch_size=bs)

        return losses["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        images = batch["image"]
        if images.dim() == _SPATIAL_DIMS + 1:
            images = rearrange(images, _EXPAND_PATTERN)

        predictions = self(images)
        targets = self._prepare_targets(batch)
        losses = self.criterion(predictions, targets)

        bs = images.shape[0]
        for name, val in losses.items():
            self.log(f"val/{name}", val, prog_bar=(name == "loss"), sync_dist=True, batch_size=bs)

        preds = predictions["semantic"].argmax(dim=1)
        acc = (preds == targets["class_labels"]).float().mean()
        self.log("val/accuracy", acc, prog_bar=True, sync_dist=True, batch_size=bs)

        return losses["loss"]

    def configure_optimizers(self) -> Any:
        """Configure optimizer and scheduler."""
        lr = self.optimizer_config.get("lr", 1e-4)
        wd = self.optimizer_config.get("weight_decay", 1e-5)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

        sched_cfg = self.optimizer_config.get("scheduler", {})
        stype = sched_cfg.get("type", "cosine").lower()

        if stype == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=sched_cfg.get("T_max", 100), eta_min=sched_cfg.get("eta_min", 1e-7),
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer

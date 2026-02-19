"""
Vista3D Lightning Module for volumetric segmentation training.

Uses the 2-head Vista3D model (semantic + instance) with
the Vista3DLoss for combined multi-task training.
"""

from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from einops import rearrange

from neurons.models.vista3d_model import Vista3DWrapper as _Model
from neurons.losses.vista3d_losses import Vista3DLoss as _Loss

_SPATIAL_DIMS = 3
_EXPAND_PATTERN = "b d h w -> b 1 d h w"
_SQUEEZE_PATTERN = "b 1 d h w -> b d h w"


class Vista3DModule(pl.LightningModule):
    """
    PyTorch Lightning module for Vista3D-based volumetric segmentation.

    Two-head architecture:
    - semantic: per-voxel class logits  [B, C, D, H, W]
    - instance: per-voxel embeddings    [B, E, D, H, W]

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
            emb_dim=model_config.get("emb_dim", 16),
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
            ce_weight=loss_config.get("ce_weight", 1.0),
            dice_weight=loss_config.get("dice_weight", 0.0),
            class_weights=loss_config.get("class_weights"),
            ignore_index=loss_config.get("ignore_index", -100),
        )

    def forward(self, x: torch.Tensor, **kw: Any) -> Dict[str, torch.Tensor]:
        """Forward pass through 2-head model."""
        return self.model(x, **kw)

    def _prepare_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract and reshape targets from batch dict."""
        labels = batch["label"]
        if labels.dim() == _SPATIAL_DIMS + 2:
            labels = rearrange(labels, _SQUEEZE_PATTERN)

        targets: Dict[str, Any] = {
            "class_labels": batch.get("class_ids", (labels > 0).long()),
            "labels": labels,
        }
        if "class_ids" in batch:
            targets["class_ids"] = batch["class_ids"]
        return targets

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images = batch["image"]
        if images.dim() == _SPATIAL_DIMS + 1:
            images = rearrange(images, _EXPAND_PATTERN)

        targets = self._prepare_targets(batch)
        predictions = self.model(images, class_ids=targets.get("class_ids"))
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

        targets = self._prepare_targets(batch)
        predictions = self.model(images, class_ids=targets.get("class_ids"))
        losses = self.criterion(predictions, targets)

        bs = images.shape[0]
        for name, val in losses.items():
            self.log(f"val/{name}", val, prog_bar=(name == "loss"), sync_dist=True, batch_size=bs)

        preds = predictions["semantic"].argmax(dim=1)
        acc = (preds == targets["class_labels"]).float().mean()
        self.log("val/accuracy", acc, prog_bar=True, sync_dist=True, batch_size=bs)

        return losses["loss"]

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step (same metrics as validation)."""
        images = batch["image"]
        if images.dim() == _SPATIAL_DIMS + 1:
            images = rearrange(images, _EXPAND_PATTERN)

        targets = self._prepare_targets(batch)
        predictions = self.model(images, class_ids=targets.get("class_ids"))
        losses = self.criterion(predictions, targets)

        bs = images.shape[0]
        for name, val in losses.items():
            self.log(f"test/{name}", val, sync_dist=True, batch_size=bs)

        preds = predictions["semantic"].argmax(dim=1)
        acc = (preds == targets["class_labels"]).float().mean()
        self.log("test/accuracy", acc, sync_dist=True, batch_size=bs)

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

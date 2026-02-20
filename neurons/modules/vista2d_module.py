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
from neurons.inference.soft_clustering import SoftMeanShift
from neurons.metrics import (
    compute_per_batch_ari,
    compute_per_batch_ami,
    compute_per_batch_voi,
    compute_per_batch_ted,
    compute_per_batch_iou,
)

_SPATIAL_DIMS = 2
_EXPAND_PATTERN = "b h w -> b 1 h w"
_SQUEEZE_PATTERN = "b 1 h w -> b h w"


class Vista2DModule(pl.LightningModule):
    """
    PyTorch Lightning module for Vista2D-based image segmentation.

    Two-head architecture:
    - semantic: per-pixel class logits  [B, C, H, W]
    - instance: per-pixel embeddings    [B, E, H, W]

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
            feature_size=model_config.get("feature_size", 64),
            encoder_name=model_config.get("encoder_name", "vista3d"),
        )

        self.criterion = _Loss(
            weight_semantic=loss_config.get("weight_semantic", 1.0),
            weight_instance=loss_config.get("weight_instance", 1.0),
            weight_geometry=loss_config.get("weight_geometry", 0.0),
            weight_pull=loss_config.get("weight_pull", 1.0),
            weight_push=loss_config.get("weight_push", 1.0),
            weight_norm=loss_config.get("weight_norm", .001),
            weight_edge=loss_config.get("weight_edge", 10.0),
            weight_bone=loss_config.get("weight_bone", 10.0),
            delta_v=loss_config.get("delta_v", 0.5),
            delta_d=loss_config.get("delta_d", 1.5),
            weight_ce=loss_config.get("weight_ce", 1.0),
            weight_dice=loss_config.get("weight_dice", 0.0),
            class_weights=loss_config.get("class_weights"),
            ignore_index=loss_config.get("ignore_index", -100),
        )

        self._clusterer = SoftMeanShift(bandwidth=loss_config.get("delta_v", 0.5))

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

    @torch.no_grad()
    def _eval_metrics(
        self, predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor], prefix: str, bs: int,
    ) -> None:
        """Compute and log semantic + instance metrics."""
        sem_pred = predictions["semantic"].argmax(dim=1)
        sem_gt = targets["class_labels"]
        acc = (sem_pred == sem_gt).float().mean()
        iou = compute_per_batch_iou(sem_pred, sem_gt, num_classes=predictions["semantic"].shape[1])
        self.log(f"{prefix}/accuracy", acc, prog_bar=(prefix == "val"), sync_dist=True, batch_size=bs)
        self.log(f"{prefix}/iou", iou, sync_dist=True, batch_size=bs)

        fg_mask = targets["labels"] > 0
        inst_pred, _, _ = self._clusterer(predictions["instance"], fg_mask)
        inst_gt = targets["labels"]

        ari = compute_per_batch_ari(inst_pred, inst_gt)
        ami = compute_per_batch_ami(inst_pred, inst_gt)
        voi = compute_per_batch_voi(inst_pred, inst_gt)
        ted = compute_per_batch_ted(inst_pred, inst_gt)

        self.log(f"{prefix}/ari", ari, sync_dist=True, batch_size=bs)
        self.log(f"{prefix}/ami", ami, sync_dist=True, batch_size=bs)
        self.log(f"{prefix}/voi", voi.total, sync_dist=True, batch_size=bs)
        self.log(f"{prefix}/voi_split", voi.split, sync_dist=True, batch_size=bs)
        self.log(f"{prefix}/voi_merge", voi.merge, sync_dist=True, batch_size=bs)
        self.log(f"{prefix}/ted", ted, sync_dist=True, batch_size=bs)

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

        self._eval_metrics(predictions, targets, "val", bs)
        return losses["loss"]

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        images = batch["image"]
        if images.dim() == _SPATIAL_DIMS + 1:
            images = rearrange(images, _EXPAND_PATTERN)

        targets = self._prepare_targets(batch)
        predictions = self.model(images, class_ids=targets.get("class_ids"))
        losses = self.criterion(predictions, targets)

        bs = images.shape[0]
        for name, val in losses.items():
            self.log(f"test/{name}", val, sync_dist=True, batch_size=bs)

        self._eval_metrics(predictions, targets, "test", bs)
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

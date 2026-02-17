"""
Semantic Segmentation Lightning Module.

Cross-entropy based semantic segmentation for connectomics.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange

from neurons.models import SegResNetWrapper


class SemanticSegmentationModule(pl.LightningModule):
    """
    PyTorch Lightning module for semantic segmentation.

    Implements standard cross-entropy based training with:
    - Flexible backbone selection (SegResNet, Vista3D)
    - Class weighting for imbalanced data
    - IoU and Dice metric computation
    - TensorBoard / W&B logging integration

    Args:
        model_config: Model configuration dictionary.
        optimizer_config: Optimizer configuration dictionary.
        loss_config: Loss configuration dictionary.
        num_classes: Number of segmentation classes.
    """

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        model_config = model_config or {}
        optimizer_config = optimizer_config or {}
        loss_config = loss_config or {}

        self.num_classes = num_classes
        self.optimizer_config = optimizer_config
        self.loss_config = loss_config

        model_config.setdefault("out_channels", num_classes)
        self.model = SegResNetWrapper(**model_config)

        class_weights = loss_config.get("class_weights")
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights: Optional[torch.Tensor] = None

        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning model outputs."""
        return self.model(x)

    def _compute_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute segmentation metrics.

        Args:
            logits: Prediction logits [B, C, ...].
            targets: Ground truth labels [B, ...].

        Returns:
            Dictionary with IoU, Dice, and accuracy metrics.
        """
        preds = logits.argmax(dim=1)
        metrics: Dict[str, torch.Tensor] = {}

        metrics["accuracy"] = (preds == targets).float().mean()

        for cls in range(self.num_classes):
            pred_mask = preds == cls
            target_mask = targets == cls

            intersection = (pred_mask & target_mask).float().sum()
            union = (pred_mask | target_mask).float().sum()

            iou = intersection / (union + 1e-8)
            dice = 2 * intersection / (pred_mask.float().sum() + target_mask.float().sum() + 1e-8)

            metrics[f"iou_class_{cls}"] = iou
            metrics[f"dice_class_{cls}"] = dice

        if self.num_classes > 1:
            fg_ious = [metrics[f"iou_class_{i}"] for i in range(1, self.num_classes)]
            metrics["mean_iou"] = (
                torch.stack(fg_ious).mean() if fg_ious else metrics["iou_class_0"]
            )

        return metrics

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step."""
        images = batch["image"]
        labels = batch["label"]

        if labels.dim() == images.dim():
            labels = rearrange(labels, "b 1 ... -> b ...")
        labels = labels.long()

        outputs = self(images)
        logits = outputs["logits"]

        loss = self.ce_loss(logits, labels)

        with torch.no_grad():
            metrics = self._compute_metrics(logits, labels)

        batch_size = images.shape[0]
        self.log("train/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("train/accuracy", metrics["accuracy"], batch_size=batch_size)
        if "mean_iou" in metrics:
            self.log("train/mean_iou", metrics["mean_iou"], batch_size=batch_size)

        return loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Validation step."""
        images = batch["image"]
        labels = batch["label"]

        if labels.dim() == images.dim():
            labels = rearrange(labels, "b 1 ... -> b ...")
        labels = labels.long()

        outputs = self(images)
        logits = outputs["logits"]

        loss = self.ce_loss(logits, labels)
        metrics = self._compute_metrics(logits, labels)

        batch_size = images.shape[0]
        self.log("val/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("val/accuracy", metrics["accuracy"], batch_size=batch_size)
        if "mean_iou" in metrics:
            self.log("val/mean_iou", metrics["mean_iou"], prog_bar=True, batch_size=batch_size)

        for cls in range(self.num_classes):
            self.log(f"val/iou_class_{cls}", metrics[f"iou_class_{cls}"], batch_size=batch_size)
            self.log(f"val/dice_class_{cls}", metrics[f"dice_class_{cls}"], batch_size=batch_size)

        return loss

    def configure_optimizers(self) -> Any:
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config.get("lr", 1e-3),
            weight_decay=self.optimizer_config.get("weight_decay", 1e-4),
        )

        scheduler_config = self.optimizer_config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "cosine")

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get("T_max", 100),
                eta_min=scheduler_config.get("eta_min", 1e-6),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "val/loss"},
            }
        elif scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_config.get("factor", 0.5),
                patience=scheduler_config.get("patience", 10),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "val/loss"},
            }

        return optimizer

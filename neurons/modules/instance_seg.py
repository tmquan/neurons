"""
Instance Segmentation Lightning Module.

Combines discriminative loss for instance embeddings with semantic segmentation.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange

from neurons.models import SegResNetWrapper
from neurons.losses import DiscriminativeLoss


class InstanceSegmentationModule(pl.LightningModule):
    """
    PyTorch Lightning module for instance segmentation.

    Uses a multi-task approach combining:
    - Semantic segmentation (foreground/background classification)
    - Instance embeddings (discriminative loss for clustering)

    Based on "Semantic Instance Segmentation with a Discriminative Loss Function"
    by De Brabandere et al. (2017).

    Args:
        model_config: Model configuration dictionary.
        optimizer_config: Optimizer configuration dictionary.
        loss_config: Loss configuration dictionary with discriminative params.
    """

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        model_config = model_config or {}
        optimizer_config = optimizer_config or {}
        loss_config = loss_config or {}

        self.optimizer_config = optimizer_config
        self.loss_config = loss_config

        model_config.setdefault("use_ins_head", True)
        model_config.setdefault("out_channels", 2)

        self.model = SegResNetWrapper(**model_config)

        ins_config = loss_config.get("discriminative", {})
        self.instance_loss = DiscriminativeLoss(
            delta_var=ins_config.get("delta_var", 0.5),
            delta_dst=ins_config.get("delta_dst", 1.5),
            norm=ins_config.get("norm", 2),
            A=ins_config.get("A", 1.0),
            B=ins_config.get("B", 1.0),
            R=ins_config.get("R", 0.001),
        )

        self.semantic_loss = nn.CrossEntropyLoss()

        self.semantic_weight = loss_config.get("semantic_weight", 1.0)
        self.instance_weight = loss_config.get("instance_weight", 1.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning model outputs."""
        return self.model(x)

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            outputs: Model outputs with 'logits' and 'embedding'.
            labels: Instance segmentation labels [B, H, W] or [B, 1, H, W].

        Returns:
            Dictionary with individual and total losses.
        """
        if labels.dim() == 4:
            labels_squeezed = rearrange(labels, "b 1 h w -> b h w")
        else:
            labels_squeezed = labels

        semantic_logits = outputs["logits"]
        semantic_target = (labels_squeezed > 0).long()
        loss_sem = self.semantic_loss(semantic_logits, semantic_target)

        embedding = outputs["embedding"]
        loss_ins, loss_var, loss_dst, loss_reg = self.instance_loss(
            embedding, labels_squeezed
        )

        total_loss = self.semantic_weight * loss_sem + self.instance_weight * loss_ins

        return {
            "loss": total_loss,
            "loss_sem": loss_sem,
            "loss_ins": loss_ins,
            "loss_var": loss_var,
            "loss_dst": loss_dst,
            "loss_reg": loss_reg,
        }

    def _compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute segmentation metrics."""
        if labels.dim() == 4:
            labels_squeezed = rearrange(labels, "b 1 h w -> b h w")
        else:
            labels_squeezed = labels

        semantic_pred = outputs["logits"].argmax(dim=1)
        semantic_target = (labels_squeezed > 0).long()

        intersection = ((semantic_pred == 1) & (semantic_target == 1)).float().sum()
        union = ((semantic_pred == 1) | (semantic_target == 1)).float().sum()
        iou = intersection / (union + 1e-8)

        accuracy = (semantic_pred == semantic_target).float().mean()

        return {"iou": iou, "accuracy": accuracy}

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step."""
        images = batch["image"]
        labels = batch["label"]

        outputs = self(images)
        loss_dict = self._compute_losses(outputs, labels)
        loss = loss_dict["loss"]

        batch_size = images.shape[0]
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/loss_sem", loss_dict["loss_sem"], on_epoch=True, batch_size=batch_size)
        self.log("train/loss_ins", loss_dict["loss_ins"], on_epoch=True, batch_size=batch_size)
        self.log("train/loss_var", loss_dict["loss_var"], on_epoch=True, batch_size=batch_size)
        self.log("train/loss_dst", loss_dict["loss_dst"], on_epoch=True, batch_size=batch_size)

        return loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Validation step."""
        images = batch["image"]
        labels = batch["label"]

        outputs = self(images)
        loss_dict = self._compute_losses(outputs, labels)
        metrics = self._compute_metrics(outputs, labels)

        batch_size = images.shape[0]
        self.log("val/loss", loss_dict["loss"], prog_bar=True, batch_size=batch_size)
        self.log("val/loss_sem", loss_dict["loss_sem"], batch_size=batch_size)
        self.log("val/loss_ins", loss_dict["loss_ins"], batch_size=batch_size)
        self.log("val/iou", metrics["iou"], prog_bar=True, batch_size=batch_size)
        self.log("val/accuracy", metrics["accuracy"], batch_size=batch_size)

        return loss_dict["loss"]

    def configure_optimizers(self) -> Any:
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config.get("lr", 1e-3),
            weight_decay=self.optimizer_config.get("weight_decay", 1e-4),
        )

        scheduler_config = self.optimizer_config.get("scheduler", {})
        if scheduler_config.get("type") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get("T_max", 100),
                eta_min=scheduler_config.get("eta_min", 1e-6),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "val/loss"},
            }

        return optimizer

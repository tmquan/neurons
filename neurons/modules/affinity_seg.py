"""
Affinity Segmentation Lightning Module.

Affinity-based boundary prediction for connectomics segmentation.
Used in methods like Flood-Filling Networks and waterz watershed.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange

from neurons.models import SegResNetWrapper


class AffinitySegmentationModule(pl.LightningModule):
    """
    PyTorch Lightning module for affinity-based segmentation.

    Predicts affinity maps between neighboring voxels, which can then be
    used with watershed or agglomeration algorithms to produce instance
    segmentation.

    Args:
        model_config: Model configuration dictionary.
        optimizer_config: Optimizer configuration dictionary.
        loss_config: Loss configuration dictionary.
        affinity_offsets: List of offset tuples defining affinity neighbors.
        spatial_dims: Spatial dimensions (2 or 3).
    """

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        affinity_offsets: Optional[List[Tuple[int, ...]]] = None,
        spatial_dims: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        model_config = model_config or {}
        optimizer_config = optimizer_config or {}
        loss_config = loss_config or {}

        self.optimizer_config = optimizer_config
        self.loss_config = loss_config
        self.spatial_dims = spatial_dims

        if affinity_offsets is None:
            if spatial_dims == 2:
                self.affinity_offsets: List[Tuple[int, ...]] = [(0, 1), (1, 0)]
            else:
                self.affinity_offsets = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
        else:
            self.affinity_offsets = affinity_offsets

        self.num_affinities = len(self.affinity_offsets)

        model_config.setdefault("out_channels", self.num_affinities)
        model_config.setdefault("spatial_dims", spatial_dims)

        self.model = SegResNetWrapper(**model_config)

        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([loss_config.get("pos_weight", 1.0)])
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning affinity predictions."""
        outputs = self.model(x)
        outputs["affinity_logits"] = outputs["logits"]
        outputs["affinity"] = torch.sigmoid(outputs["logits"])
        return outputs

    def _compute_affinity_targets(
        self,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute affinity targets from instance labels.

        Affinity is 1 if neighboring voxels belong to the same instance,
        0 if they belong to different instances.

        Args:
            labels: Instance segmentation labels [B, H, W] or [B, D, H, W].

        Returns:
            Affinity targets [B, num_affinities, ...spatial...].
        """
        batch_size = labels.shape[0]
        spatial_shape = labels.shape[1:]
        device = labels.device

        affinity_shape = (batch_size, self.num_affinities) + spatial_shape
        affinities = torch.zeros(affinity_shape, device=device, dtype=torch.float32)

        for i, offset in enumerate(self.affinity_offsets):
            if self.spatial_dims == 3:
                dz, dy, dx = offset
                labels_shifted = torch.roll(labels, shifts=(-dz, -dy, -dx), dims=(1, 2, 3))

                valid_mask = torch.ones_like(labels, dtype=torch.bool)
                if dz > 0:
                    valid_mask[:, -dz:, :, :] = False
                elif dz < 0:
                    valid_mask[:, : -dz, :, :] = False
                if dy > 0:
                    valid_mask[:, :, -dy:, :] = False
                elif dy < 0:
                    valid_mask[:, :, : -dy, :] = False
                if dx > 0:
                    valid_mask[:, :, :, -dx:] = False
                elif dx < 0:
                    valid_mask[:, :, :, : -dx] = False

                same_instance = (labels == labels_shifted) & (labels > 0) & valid_mask
                affinities[:, i] = same_instance.float()
            else:
                dy, dx = offset[0], offset[1]
                labels_shifted = torch.roll(labels, shifts=(-dy, -dx), dims=(1, 2))

                valid_mask = torch.ones_like(labels, dtype=torch.bool)
                if dy > 0:
                    valid_mask[:, -dy:, :] = False
                elif dy < 0:
                    valid_mask[:, : -dy, :] = False
                if dx > 0:
                    valid_mask[:, :, -dx:] = False
                elif dx < 0:
                    valid_mask[:, :, : -dx] = False

                same_instance = (labels == labels_shifted) & (labels > 0) & valid_mask
                affinities[:, i] = same_instance.float()

        return affinities

    def _compute_metrics(
        self,
        affinity_pred: torch.Tensor,
        affinity_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute affinity prediction metrics."""
        pred_binary = (affinity_pred > 0.5).float()

        accuracy = (pred_binary == affinity_target).float().mean()

        true_pos = ((pred_binary == 1) & (affinity_target == 1)).float().sum()
        pred_pos = (pred_binary == 1).float().sum()
        actual_pos = (affinity_target == 1).float().sum()

        precision = true_pos / (pred_pos + 1e-8)
        recall = true_pos / (actual_pos + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

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

        outputs = self(images)
        affinity_logits = outputs["affinity_logits"]

        affinity_targets = self._compute_affinity_targets(labels)

        loss = self.loss_fn(affinity_logits, affinity_targets)

        batch_size = images.shape[0]
        self.log("train/loss", loss, prog_bar=True, batch_size=batch_size)

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

        outputs = self(images)
        affinity_logits = outputs["affinity_logits"]
        affinity_pred = outputs["affinity"]

        affinity_targets = self._compute_affinity_targets(labels)
        loss = self.loss_fn(affinity_logits, affinity_targets)

        metrics = self._compute_metrics(affinity_pred, affinity_targets)

        batch_size = images.shape[0]
        self.log("val/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("val/accuracy", metrics["accuracy"], batch_size=batch_size)
        self.log("val/precision", metrics["precision"], batch_size=batch_size)
        self.log("val/recall", metrics["recall"], batch_size=batch_size)
        self.log("val/f1", metrics["f1"], prog_bar=True, batch_size=batch_size)

        return loss

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

"""
Vista3D Lightning Module for training and inference.

Supports both automatic segmentation and interactive point-based segmentation.
Features:
- Semantic head for foreground/background classification
- Instance head for pixel embeddings with discriminative loss
- ARI/AMI metrics for instance segmentation evaluation
- MONAI MeanDice and MeanIoU metrics
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from monai.metrics import DiceMetric, MeanIoU


class Vista3DModule(pl.LightningModule):
    """
    PyTorch Lightning module for Vista3D-based segmentation.

    Supports:
    - Automatic segmentation mode (class label prompts)
    - Interactive segmentation mode (point click prompts)
    - Mixed training with both modes
    - Instance segmentation with discriminative loss

    Args:
        model_config: Model configuration dict.
        optimizer_config: Optimizer configuration dict.
        loss_config: Loss function configuration dict.
        training_mode: 'auto', 'interactive', or 'mixed'.
        num_point_prompts: Number of point prompts to sample during training.
        patch_size: Size of training patches (D, H, W).
    """

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        training_mode: str = "auto",
        num_point_prompts: int = 5,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model_config = model_config or {}
        self.optimizer_config = optimizer_config or {}
        self.loss_config = loss_config or {}
        self.training_mode = training_mode
        self.num_point_prompts = num_point_prompts
        self.patch_size = patch_size

        self.sem_head_dim: int = self.model_config.get("sem_head", 16)
        self.ins_head_dim: int = self.model_config.get("ins_head", 16)
        self.use_sem_head: bool = self.model_config.get("use_sem_head", True)
        self.use_ins_head: bool = self.model_config.get("use_ins_head", True)

        self._build_model()
        self._build_loss()

    def _build_model(self) -> None:
        """Build Vista3D model with optional semantic and instance heads."""
        from neurons.models.vista3d import Vista3DWrapper

        feature_size: int = self.model_config.get("feature_size", 48)
        num_classes: int = self.model_config.get("num_classes", 2)

        self.model = Vista3DWrapper(
            in_channels=self.model_config.get("in_channels", 1),
            num_classes=num_classes,
            pretrained=self.model_config.get("pretrained", False),
            freeze_encoder=self.model_config.get("freeze_encoder", False),
            encoder_name=self.model_config.get("encoder_name", "segresnet"),
            feature_size=feature_size,
            use_point_prompts=self.training_mode in ["interactive", "mixed"],
            use_automatic_mode=self.training_mode in ["auto", "mixed"],
        )

        if self.use_sem_head:
            self.sem_head = nn.Sequential(
                nn.Conv3d(num_classes, self.sem_head_dim, kernel_size=3, padding=1),
                nn.BatchNorm3d(self.sem_head_dim),
                nn.ReLU(inplace=True),
                nn.Conv3d(self.sem_head_dim, num_classes, kernel_size=1),
            )
        else:
            self.sem_head = nn.Identity()

        if self.use_ins_head:
            self.ins_head = nn.Sequential(
                nn.Conv3d(num_classes, self.ins_head_dim * 2, kernel_size=3, padding=1),
                nn.BatchNorm3d(self.ins_head_dim * 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(self.ins_head_dim * 2, self.ins_head_dim, kernel_size=1),
            )

    def _build_loss(self) -> None:
        """Build loss functions including discriminative loss."""
        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor(self.loss_config.get("class_weights", [1.0, 1.0])),
            ignore_index=self.loss_config.get("ignore_index", -100),
        )

        self.dice_loss_weight: float = self.loss_config.get("dice_weight", 0.5)
        self.ce_loss_weight: float = self.loss_config.get("ce_weight", 0.5)
        self.boundary_loss_weight: float = self.loss_config.get("boundary_weight", 0.0)

        if self.use_ins_head:
            from neurons.losses.discriminative import DiscriminativeLossVectorized

            ins_config = self.loss_config.get("discriminative", {})
            self.instance_loss = DiscriminativeLossVectorized(
                delta_var=ins_config.get("delta_var", 0.5),
                delta_dst=ins_config.get("delta_dst", 1.5),
                norm=ins_config.get("norm", 2),
                A=ins_config.get("A", 1.0),
                B=ins_config.get("B", 1.0),
                R=ins_config.get("R", 0.001),
            )
            self.ins_loss_weight: float = self.loss_config.get("ins_weight", 1.0)

        num_classes: int = self.model_config.get("num_classes", 2)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.iou_metric = MeanIoU(include_background=False, reduction="mean", get_not_nans=False)

    def forward(
        self,
        x: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        class_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional semantic and instance heads.

        Returns:
            Dict with 'logits' and optionally 'embeds'.
        """
        base_result = self.model(
            x, point_coords=point_coords, point_labels=point_labels, class_ids=class_ids,
        )

        logits = base_result["logits"]
        semantic = self.sem_head(logits)

        result: Dict[str, torch.Tensor] = {"logits": semantic, "labels": semantic}

        if self.use_ins_head:
            embeds = self.ins_head(logits)
            result["embeds"] = embeds

        return result

    def _compute_dice_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, smooth: float = 1e-5,
    ) -> torch.Tensor:
        """Compute Dice loss."""
        probs = F.softmax(logits, dim=1)

        num_classes = logits.shape[1]
        labels_clamped = torch.clamp(labels.long(), 0, num_classes - 1)
        labels_one_hot = F.one_hot(labels_clamped, num_classes)
        labels_one_hot = rearrange(labels_one_hot, "b d h w c -> b c d h w").float()

        intersection = (probs * labels_one_hot).sum(dim=(2, 3, 4))
        union = probs.sum(dim=(2, 3, 4)) + labels_one_hot.sum(dim=(2, 3, 4))

        dice = (2 * intersection + smooth) / (union + smooth)

        if num_classes > 1:
            return 1 - dice[:, 1:].mean()
        return 1 - dice.mean()

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeds: Optional[torch.Tensor] = None,
        instance_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss."""
        losses: Dict[str, torch.Tensor] = {}

        num_classes = logits.shape[1]
        labels = labels.long()
        labels = torch.clamp(labels, 0, num_classes - 1)

        ce_loss = self.ce_loss(logits, labels)
        losses["ce_loss"] = ce_loss

        dice_loss = self._compute_dice_loss(logits, labels)
        losses["dice_loss"] = dice_loss

        total_loss = self.ce_loss_weight * ce_loss + self.dice_loss_weight * dice_loss

        if self.use_ins_head and embeds is not None and instance_labels is not None:
            ins_loss, loss_var, loss_dst, loss_reg = self.instance_loss(embeds, instance_labels)
            losses["ins_loss"] = ins_loss
            losses["loss_var"] = loss_var
            losses["loss_dst"] = loss_dst
            losses["loss_reg"] = loss_reg
            total_loss = total_loss + self.ins_loss_weight * ins_loss

        losses["loss"] = total_loss
        return losses

    def _compute_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute segmentation metrics using MONAI."""
        metrics: Dict[str, torch.Tensor] = {}

        num_classes = logits.shape[1]

        preds = logits.argmax(dim=1)
        preds_one_hot = F.one_hot(preds, num_classes)
        preds_one_hot = rearrange(preds_one_hot, "b d h w c -> b c d h w")

        labels_clamped = torch.clamp(labels.long(), 0, num_classes - 1)
        labels_one_hot = F.one_hot(labels_clamped, num_classes)
        labels_one_hot = rearrange(labels_one_hot, "b d h w c -> b c d h w")

        self.dice_metric.reset()
        self.dice_metric(y_pred=preds_one_hot, y=labels_one_hot)
        mean_dice = self.dice_metric.aggregate()

        self.iou_metric.reset()
        self.iou_metric(y_pred=preds_one_hot, y=labels_one_hot)
        mean_iou = self.iou_metric.aggregate()

        accuracy = (preds == labels_clamped).float().mean()

        metrics["accuracy"] = accuracy
        metrics["dice"] = mean_dice
        metrics["iou"] = mean_iou

        return metrics

    def _relabel_after_crop(self, labels: torch.Tensor) -> torch.Tensor:
        """Relabel instance labels after cropping."""
        from neurons.utils.labels import relabel_after_crop
        return relabel_after_crop(labels, spatial_dims=3)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int,
    ) -> torch.Tensor:
        """Training step with instance segmentation support."""
        images = batch["image"]
        instance_label = batch["label"]
        class_ids = batch["class_ids"]

        if images.dim() == 4:
            images = rearrange(images, "b d h w -> b 1 d h w")
        if instance_label.dim() == 5:
            instance_label = rearrange(instance_label, "b 1 d h w -> b d h w")
        if class_ids.dim() == 5:
            class_ids = rearrange(class_ids, "b 1 d h w -> b d h w")

        instance_labels = self._relabel_after_crop(instance_label)
        semantic_labels = class_ids.long()

        result = self.forward(images)

        logits = result["logits"]
        embeds = result.get("embeds")

        losses = self._compute_loss(logits, semantic_labels, embeds=embeds, instance_labels=instance_labels)

        batch_size = images.shape[0]
        for name, value in losses.items():
            self.log(f"train/{name}", value, prog_bar=(name == "loss"), batch_size=batch_size)

        return losses["loss"]

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Validation step with instance segmentation metrics."""
        images = batch["image"]
        instance_label = batch["label"]
        class_ids = batch["class_ids"]

        if images.dim() == 4:
            images = rearrange(images, "b d h w -> b 1 d h w")
        if instance_label.dim() == 5:
            instance_label = rearrange(instance_label, "b 1 d h w -> b d h w")
        if class_ids.dim() == 5:
            class_ids = rearrange(class_ids, "b 1 d h w -> b d h w")

        instance_labels = self._relabel_after_crop(instance_label)
        semantic_labels = class_ids.long()

        result = self.forward(images)
        logits = result["logits"]
        embeds = result.get("embeds")

        losses = self._compute_loss(logits, semantic_labels, embeds=embeds, instance_labels=instance_labels)
        metrics = self._compute_metrics(logits, semantic_labels)

        batch_size = images.shape[0]
        for name, value in losses.items():
            self.log(f"val/{name}", value, prog_bar=(name == "loss"), sync_dist=True, batch_size=batch_size)
        for name, value in metrics.items():
            self.log(f"val/{name}", value, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return losses

    def configure_optimizers(self) -> Any:
        """Configure optimizer and scheduler."""
        opt_type = self.optimizer_config.get("type", "adamw").lower()
        lr = self.optimizer_config.get("lr", 1e-4)
        weight_decay = self.optimizer_config.get("weight_decay", 1e-5)

        if opt_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=weight_decay,
                betas=tuple(self.optimizer_config.get("betas", (0.9, 0.999))),
            )
        elif opt_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=lr, weight_decay=weight_decay,
                momentum=self.optimizer_config.get("momentum", 0.9),
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_cfg = self.optimizer_config.get("scheduler", {})
        scheduler_type = scheduler_cfg.get("type", "cosine").lower()

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=scheduler_cfg.get("T_max", 100), eta_min=scheduler_cfg.get("eta_min", 1e-7),
            )
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=scheduler_cfg.get("factor", 0.5),
                patience=scheduler_cfg.get("patience", 10),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}
        else:
            return optimizer

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

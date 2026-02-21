"""
Vista3D Lightning Module for volumetric segmentation training.

Supports two training modes that can be combined in a single step:

- **automatic**: predict everything from the image alone.
- **proofread**: additional context (fractionary labels or interactive
  point prompts) is provided.  Sub-modes:

  - *fractionary*: partial annotation exists — resolve labels and forward
    with ``semantic_ids``.
  - *interactive*: fully annotated — simulate point prompts sampled from GT.

When ``training_modes`` contains both, every batch runs both forward passes
and the losses are summed.
"""

from typing import Any, Dict, List, Optional

import torch
import pytorch_lightning as pl
from einops import rearrange

from neurons.models.vista3d_model import Vista3DWrapper as _Model
from neurons.losses.vista3d_losses import Vista3DLoss as _Loss
from neurons.inference.soft_clustering import SoftMeanShift
from neurons.metrics import (
    compute_per_batch_ari,
    compute_per_batch_ami,
    compute_per_batch_voi,
    compute_per_batch_ted,
    compute_per_batch_iou,
)
from neurons.utils.point_sampling import sample_point_prompts

_SPATIAL_DIMS = 3
_EXPAND_PATTERN = "b d h w -> b 1 d h w"
_SQUEEZE_PATTERN = "b 1 d h w -> b d h w"

_DEFAULT_TRAINING_MODES: List[str] = ["automatic"]


class Vista3DModule(pl.LightningModule):
    """
    PyTorch Lightning module for Vista3D-based volumetric segmentation.

    Three-head architecture:
    - semantic: per-voxel class logits  [B, C, D, H, W]
    - instance: per-voxel embeddings    [B, E, D, H, W]
    - geometry: per-voxel geometry      [B, G, D, H, W]

    Args:
        model_config: Model configuration dict.
        optimizer_config: Optimizer configuration dict.
        loss_config: Loss function configuration dict.
        training_config: Training configuration dict (contains
            ``training_modes``, ``num_pos_points``, etc.).
    """

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        model_config = model_config or {}
        self.optimizer_config = optimizer_config or {}
        loss_config = loss_config or {}
        training_config = training_config or {}

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
            weight_norm=loss_config.get("weight_norm", 0.001),
            weight_edge=loss_config.get("weight_edge", 10.0),
            weight_bone=loss_config.get("weight_bone", 10.0),
            delta_v=loss_config.get("delta_v", 0.5),
            delta_d=loss_config.get("delta_d", 1.5),
            weight_ce=loss_config.get("weight_ce", 1.0),
            weight_iou=loss_config.get("weight_iou", 0.0),
            weight_dice=loss_config.get("weight_dice", 0.0),
            class_weights=loss_config.get("class_weights"),
            ignore_index=loss_config.get("ignore_index", -100),
            weight_dir=loss_config.get("weight_dir", 1.0),
            weight_cov=loss_config.get("weight_cov", 1.0),
            weight_raw=loss_config.get("weight_raw", 1.0),
            dir_target=loss_config.get("dir_target", "centroid"),
        )

        self._clusterer = SoftMeanShift(bandwidth=loss_config.get("delta_v", 0.5))
        self._ignore_index = loss_config.get("ignore_index", -100)

        self.training_modes: List[str] = list(
            training_config.get("training_modes", _DEFAULT_TRAINING_MODES)
        )
        self._num_pos_points: int = training_config.get("num_pos_points", 5)
        self._num_neg_points: int = training_config.get("num_neg_points", 5)
        self._point_sample_mode: str = training_config.get("point_sample_mode", "class")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, **kw: Any) -> Dict[str, torch.Tensor]:
        return self.model(x, **kw)

    # ------------------------------------------------------------------
    # Target preparation
    # ------------------------------------------------------------------

    def _prepare_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract and reshape targets from batch dict."""
        labels = batch["label"]
        if labels.dim() == _SPATIAL_DIMS + 2:
            labels = rearrange(labels, _SQUEEZE_PATTERN)

        targets: Dict[str, Any] = {
            "semantic_labels": batch.get("semantic_ids", (labels > 0).long()),
            "labels": labels,
        }
        if "semantic_ids" in batch:
            targets["semantic_ids"] = batch["semantic_ids"]
        if "image" in batch:
            targets["raw_image"] = batch["image"]
        return targets

    # ------------------------------------------------------------------
    # Proofread helpers
    # ------------------------------------------------------------------

    def _get_proofread_sub_mode(
        self, targets: Dict[str, torch.Tensor],
    ) -> str:
        """Determine proofread sub-mode for the current batch.

        Returns ``"fractionary"`` when the labels contain a mix of valid
        values and ``ignore_index`` (partial annotation), otherwise
        ``"interactive"``.
        """
        labels = targets["labels"]
        has_ignore = (labels == self._ignore_index).any()
        has_valid_fg = (labels > 0).any() & (labels != self._ignore_index).any()
        if has_ignore and has_valid_fg:
            return "fractionary"
        return "interactive"

    def _resolve_fractionary_labels(
        self, targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Prepare targets for a fractionary-annotated patch.

        - Sets ``semantic_labels`` to ``ignore_index`` where annotation is
          missing.
        - Builds ``semantic_ids`` from the known voxels.
        - Remaps instance ``labels`` to contiguous IDs for the known region.
        """
        targets = dict(targets)
        labels = targets["labels"]
        ignore = self._ignore_index
        unknown = labels == ignore

        sem = targets["semantic_labels"].clone()
        sem[unknown] = ignore
        targets["semantic_labels"] = sem

        semantic_ids = sem.clone()
        targets["semantic_ids"] = semantic_ids

        inst = labels.clone()
        inst[unknown] = 0
        known_ids = inst.unique()
        known_ids = known_ids[known_ids > 0]
        remap = torch.zeros(int(known_ids.max().item()) + 1 if known_ids.numel() > 0 else 1,
                            dtype=torch.long, device=labels.device)
        for new_id, old_id in enumerate(known_ids, start=1):
            remap[old_id] = new_id
        flat = inst.view(-1)
        mask = flat > 0
        flat[mask] = remap[flat[mask]]
        targets["labels"] = inst
        return targets

    # ------------------------------------------------------------------
    # Per-mode forward + loss
    # ------------------------------------------------------------------

    def _run_automatic(
        self,
        images: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Automatic mode: forward without prompts."""
        predictions = self.model(images)
        return self.criterion(predictions, targets)

    def _run_proofread(
        self,
        images: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Proofread mode: fractionary or interactive sub-mode."""
        sub_mode = self._get_proofread_sub_mode(targets)

        if sub_mode == "fractionary":
            targets = self._resolve_fractionary_labels(targets)
            predictions = self.model(
                images, semantic_ids=targets.get("semantic_ids"),
            )
        else:
            point_prompts = sample_point_prompts(
                targets["semantic_labels"],
                targets["labels"],
                num_pos=self._num_pos_points,
                num_neg=self._num_neg_points,
                sample_mode=self._point_sample_mode,
            )
            predictions = self.model(images, point_prompts=point_prompts)

        return self.criterion(predictions, targets)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        if images.dim() == _SPATIAL_DIMS + 1:
            images = rearrange(images, _EXPAND_PATTERN)

        targets = self._prepare_targets(batch)

        all_losses: Dict[str, torch.Tensor] = {}
        mode_losses = []

        for mode in self.training_modes:
            if mode == "automatic":
                losses = self._run_automatic(images, targets)
            elif mode == "proofread":
                losses = self._run_proofread(images, targets)
            else:
                raise ValueError(f"Unknown training mode: {mode}")
            mode_losses.append(losses["loss"])
            for k, v in losses.items():
                all_losses[f"train/{mode}/{k}"] = v

        total_loss = sum(mode_losses) / len(mode_losses)

        bs = images.shape[0]
        for name, val in all_losses.items():
            self.log(name, val, batch_size=bs)
        self.log("train/loss", total_loss, prog_bar=True, batch_size=bs)

        return total_loss

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _eval_metrics(
        self, predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor], prefix: str, bs: int,
    ) -> None:
        sem_pred = predictions["semantic"].argmax(dim=1)
        sem_gt = targets["semantic_labels"]
        acc = (sem_pred == sem_gt).float().mean()
        iou = compute_per_batch_iou(sem_pred, sem_gt, num_classes=predictions["semantic"].shape[1])
        ari = compute_per_batch_ari(sem_pred, sem_gt)
        self.log(f"{prefix}/acc", acc, prog_bar=(prefix == "val"), sync_dist=True, batch_size=bs)
        self.log(f"{prefix}/ari", ari, prog_bar=(prefix == "val"), sync_dist=True, batch_size=bs)
        self.log(f"{prefix}/iou", iou, prog_bar=(prefix == "val"), sync_dist=True, batch_size=bs)

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
        images = batch["image"]
        if images.dim() == _SPATIAL_DIMS + 1:
            images = rearrange(images, _EXPAND_PATTERN)

        targets = self._prepare_targets(batch)
        predictions = self.model(images, semantic_ids=targets.get("semantic_ids"))
        losses = self.criterion(predictions, targets)

        bs = images.shape[0]
        for name, val in losses.items():
            self.log(f"val/{name}", val, prog_bar=(name == "loss"), sync_dist=True, batch_size=bs)

        self._eval_metrics(predictions, targets, "val", bs)
        return losses["loss"]

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        if images.dim() == _SPATIAL_DIMS + 1:
            images = rearrange(images, _EXPAND_PATTERN)

        targets = self._prepare_targets(batch)
        predictions = self.model(images, semantic_ids=targets.get("semantic_ids"))
        losses = self.criterion(predictions, targets)

        bs = images.shape[0]
        for name, val in losses.items():
            self.log(f"test/{name}", val, sync_dist=True, batch_size=bs)

        self._eval_metrics(predictions, targets, "test", bs)
        return losses["loss"]

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Any:
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

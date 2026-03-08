"""
Base Vista Lightning Module for 2D/3D segmentation training.

Provides the shared implementation for :class:`Vista2DModule` and
:class:`Vista3DModule`.  Subclasses set ``_SPATIAL_DIMS``, ``_model_cls``
and ``_loss_cls``; all training, evaluation and optimiser logic lives here.

Supports two training modes that can be combined in a single step:

- **automatic**: predict everything from the image alone.
- **proofread**: additional context (fractionary labels or interactive
  point prompts) is provided.
"""

import warnings
from typing import Any, Dict, List, Optional

import torch
import pytorch_lightning as pl
from einops import rearrange, reduce

from neurons.inference.soft_clustering import SoftMeanShift
from neurons.metrics import (
    compute_per_batch_ari,
    compute_per_batch_ami,
    compute_per_batch_dice,
    compute_per_batch_iou,
    compute_per_batch_voi,
    compute_per_batch_ted,
)
from neurons.utils.point_sampling import sample_point_prompts

_SPATIAL_AXES = {2: "h w", 3: "d h w"}
_DEFAULT_TRAINING_MODES: List[str] = ["automatic"]


class BaseVistaModule(pl.LightningModule):
    """Abstract base for Vista 2D / 3D modules.

    Subclasses **must** define:

    - ``_SPATIAL_DIMS`` (``int``): 2 or 3.
    - ``_model_cls`` (``type``): Model wrapper class.
    - ``_loss_cls`` (``type``): Loss class.

    Args:
        model_config: Model configuration dict.
        optimizer_config: Optimizer configuration dict.
        loss_config: Loss function configuration dict (passed as
            ``**loss_config`` to ``_loss_cls``).
        training_config: Training configuration dict (contains
            ``training_modes``, ``num_pos_points``, etc.).
    """

    _SPATIAL_DIMS: int
    _model_cls: type
    _loss_cls: type

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        dims = getattr(cls, "_SPATIAL_DIMS", None)
        if dims is not None and dims in _SPATIAL_AXES:
            axes = _SPATIAL_AXES[dims]
            cls._EXPAND_PATTERN = f"b {axes} -> b 1 {axes}"
            cls._SQUEEZE_PATTERN = f"b 1 {axes} -> b {axes}"

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if kwargs:
            warnings.warn(
                f"{type(self).__name__} received unknown keyword arguments "
                f"that will be ignored: {sorted(kwargs)}",
                stacklevel=2,
            )

        self.save_hyperparameters()

        model_config = model_config or {}
        self.optimizer_config = optimizer_config or {}
        loss_config = loss_config or {}
        training_config = training_config or {}

        self.model = self._model_cls(
            in_channels=model_config.get("in_channels", 1),
            num_classes=model_config.get("num_classes", 16),
            emb_dim=model_config.get("emb_dim", 16),
            feature_size=model_config.get("feature_size", 64),
            encoder_name=model_config.get("encoder_name", "vista3d"),
            dropout=model_config.get("dropout", 0.0),
        )

        self.criterion = self._loss_cls(**loss_config)

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

    @torch.no_grad()
    def _prepare_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract and reshape targets from *batch*.

        Squeezes the unit channel dim added by ``EnsureChannelFirstd``
        so every spatial target is ``[B, *spatial]``.
        """
        ndim_with_channel = self._SPATIAL_DIMS + 2
        squeeze = self._SQUEEZE_PATTERN

        labels = batch["label"]
        if labels.dim() == ndim_with_channel:
            labels = rearrange(labels, squeeze)

        sem = batch.get("semantic_ids", (labels > 0).long())
        if sem.dim() == ndim_with_channel:
            sem = rearrange(sem, squeeze)

        targets: Dict[str, Any] = {
            "semantic_labels": sem,
            "labels": labels,
        }
        if "semantic_ids" in batch:
            sid = batch["semantic_ids"]
            targets["semantic_ids"] = (
                rearrange(sid, squeeze) if sid.dim() == ndim_with_channel else sid
            )
        if "image" in batch and self.criterion.weight_geometry > 0:
            targets["raw_image"] = batch["image"]
        if "label_direction" in batch:
            targets["label_direction"] = batch["label_direction"]
        if "label_covariance" in batch:
            targets["label_covariance"] = batch["label_covariance"]
        return targets

    # ------------------------------------------------------------------
    # Proofread helpers
    # ------------------------------------------------------------------

    def _get_proofread_sub_mode(self, targets: Dict[str, torch.Tensor]) -> str:
        """Return ``"fractionary"`` for partial annotations, else ``"interactive"``."""
        labels = targets["labels"]
        has_ignore = (labels == self._ignore_index).any()
        has_valid_fg = (labels > 0).any() & (labels != self._ignore_index).any()
        if has_ignore and has_valid_fg:
            return "fractionary"
        return "interactive"

    def _resolve_fractionary_labels(
        self, targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Remap a fractionary-annotated patch to contiguous instance IDs."""
        targets = dict(targets)
        labels = targets["labels"]
        unknown = labels == self._ignore_index

        sem = targets["semantic_labels"].clone()
        sem[unknown] = self._ignore_index
        targets["semantic_labels"] = sem
        targets["semantic_ids"] = sem

        inst = labels.clone()
        inst[unknown] = 0
        known_ids = inst.unique()
        known_ids = known_ids[known_ids > 0]
        remap = torch.zeros(
            int(known_ids.max().item()) + 1 if known_ids.numel() > 0 else 1,
            dtype=torch.long, device=labels.device,
        )
        remap[known_ids] = torch.arange(1, known_ids.numel() + 1, device=labels.device)
        flat = rearrange(inst, "... -> (...)")
        mask = flat > 0
        flat[mask] = remap[flat[mask]]
        targets["labels"] = inst
        return targets

    # ------------------------------------------------------------------
    # Per-mode forward + loss
    # ------------------------------------------------------------------

    def _run_automatic(
        self, images: torch.Tensor, targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        predictions = self.model(images)
        return self.criterion(predictions, targets)

    def _run_proofread(
        self, images: torch.Tensor, targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        sub_mode = self._get_proofread_sub_mode(targets)
        if sub_mode == "fractionary":
            targets = self._resolve_fractionary_labels(targets)
            predictions = self.model(images, semantic_ids=targets.get("semantic_ids"))
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
        if images.dim() == self._SPATIAL_DIMS + 1:
            images = rearrange(images, self._EXPAND_PATTERN)

        targets = self._prepare_targets(batch)

        if len(self.training_modes) > 1:
            cached = self.criterion._compute_targets(targets["labels"], targets)
            targets["_cached_weights"] = cached

        all_losses: Dict[str, torch.Tensor] = {}
        mode_losses: List[torch.Tensor] = []

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

        total_loss = torch.stack(mode_losses).mean()

        bs = images.shape[0]
        for name, val in all_losses.items():
            self.log(name, val, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/loss", total_loss, prog_bar=True, on_step=True, batch_size=bs)

        return total_loss

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _eval_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        prefix: str,
        bs: int,
    ) -> None:
        sem_pred = predictions["semantic"].argmax(dim=1)
        sem_gt = targets["semantic_labels"]
        n_cls = predictions["semantic"].shape[1]

        sem_acc = reduce((sem_pred == sem_gt).float(), "b ... -> ", "mean")
        sem_iou = compute_per_batch_iou(sem_pred, sem_gt, num_classes=n_cls)
        sem_dice = compute_per_batch_dice(sem_pred, sem_gt, num_classes=n_cls)

        self.log(f"{prefix}/sem_acc", sem_acc, prog_bar=(prefix == "val"), sync_dist=True, batch_size=bs)
        self.log(f"{prefix}/sem_iou", sem_iou, prog_bar=(prefix == "val"), sync_dist=True, batch_size=bs)
        self.log(f"{prefix}/sem_dice", sem_dice, sync_dist=True, batch_size=bs)

        fg_mask = targets["labels"] > 0
        ins_pred, _, _ = self._clusterer(predictions["instance"], fg_mask)
        ins_gt = targets["labels"]

        ins_ari = compute_per_batch_ari(ins_pred, ins_gt)
        ins_ami = compute_per_batch_ami(ins_pred, ins_gt)
        ins_voi = compute_per_batch_voi(ins_pred, ins_gt)
        ins_ted = compute_per_batch_ted(ins_pred, ins_gt)

        self.log(f"{prefix}/ins_ari", ins_ari, prog_bar=(prefix == "val"), sync_dist=True, batch_size=bs)
        self.log(f"{prefix}/ins_ami", ins_ami, sync_dist=True, batch_size=bs)
        self.log(f"{prefix}/ins_voi", ins_voi.total, sync_dist=True, batch_size=bs)
        self.log(f"{prefix}/ins_voi_split", ins_voi.split, sync_dist=True, batch_size=bs)
        self.log(f"{prefix}/ins_voi_merge", ins_voi.merge, sync_dist=True, batch_size=bs)
        self.log(f"{prefix}/ins_ted", ins_ted, sync_dist=True, batch_size=bs)

    def _eval_step(self, batch: Dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        """Shared evaluation logic for validation and test steps."""
        images = batch["image"]
        if images.dim() == self._SPATIAL_DIMS + 1:
            images = rearrange(images, self._EXPAND_PATTERN)

        targets = self._prepare_targets(batch)
        predictions = self.model(images, semantic_ids=targets.get("semantic_ids"))
        losses = self.criterion(predictions, targets)

        bs = images.shape[0]
        for name, val in losses.items():
            self.log(
                f"{prefix}/{name}", val,
                prog_bar=(name == "loss" and prefix == "val"),
                sync_dist=True, batch_size=bs,
            )

        self._eval_metrics(predictions, targets, prefix, bs)
        return losses["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, "test")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Any:
        lr = self.optimizer_config.get("lr", 1e-4)
        wd = self.optimizer_config.get("weight_decay", 1e-5)

        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        param_groups = [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)

        sched_cfg = self.optimizer_config.get("scheduler", {})
        stype = sched_cfg.get("type", "cosine").lower()

        if stype == "cosine":
            from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

            warmup_epochs = sched_cfg.get("warmup_epochs", 5)
            T_max = sched_cfg.get("T_max", 100)
            eta_min = sched_cfg.get("eta_min", 1e-7)

            warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(
                optimizer, T_max=max(T_max - warmup_epochs, 1), eta_min=eta_min,
            )
            scheduler = SequentialLR(
                optimizer, [warmup, cosine], milestones=[warmup_epochs],
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        if stype:
            warnings.warn(
                f"Unknown scheduler type '{stype}', using no scheduler. "
                f"Supported: 'cosine'.",
                stacklevel=2,
            )
        return optimizer

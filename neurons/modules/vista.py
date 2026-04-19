"""
Base Vista Lightning module for 2-D / 3-D connectomics segmentation.

Shared implementation for :class:`Vista2DModule` and :class:`Vista3DModule`.
Subclasses just set ``_SPATIAL_DIMS``, ``_model_cls`` and ``_loss_cls``;
all training, evaluation and optimiser logic lives here.

Only the **automatic** training mode is supported (predict from the
image alone).
"""

import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import pytorch_lightning as pl
from einops import rearrange, reduce

from neurons.inference.clusterer import build_clusterer
from neurons.metrics import (
    compute_per_batch_ari,
    compute_per_batch_ami,
    compute_per_batch_dice,
    compute_per_batch_iou,
    compute_per_batch_voi,
    compute_per_batch_ted,
)

_SPATIAL_AXES = {2: "h w", 3: "d h w"}


class BaseVistaModule(pl.LightningModule):
    """Abstract base for Vista 2-D / 3-D modules.

    Subclasses **must** define:
      - ``_SPATIAL_DIMS`` (``int``): 2 or 3.
      - ``_model_cls``    (``type``): Model wrapper class.
      - ``_loss_cls``     (``type``): Loss class (typically :class:`CombinedLoss`).
    """

    _SPATIAL_DIMS: int
    _model_cls: type
    _loss_cls: type

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        dims = getattr(cls, "_SPATIAL_DIMS", None)
        if dims is None:
            return
        if dims not in _SPATIAL_AXES:
            raise ValueError(
                f"{cls.__name__}._SPATIAL_DIMS={dims} is invalid. "
                f"Must be one of {sorted(_SPATIAL_AXES)}."
            )
        axes = _SPATIAL_AXES[dims]
        cls._EXPAND_PATTERN = f"b {axes} -> b 1 {axes}"
        cls._SQUEEZE_PATTERN = f"b 1 {axes} -> b {axes}"

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

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
                f"{type(self).__name__} ignoring unknown kwargs: {sorted(kwargs)}",
                stacklevel=2,
            )

        self.save_hyperparameters()

        model_config = model_config or {}
        self.optimizer_config = optimizer_config or {}
        training_config = training_config or {}
        loss_config = loss_config or {}

        self.model = self._model_cls(
            in_channels=model_config.get("in_channels", 1),
            num_classes=model_config.get("num_classes", 16),
            emb_dim=model_config.get("emb_dim", 16),
            feature_size=model_config.get("feature_size", 64),
            encoder_name=model_config.get("encoder_name", "vista3d"),
            dropout=model_config.get("dropout", 0.0),
        )

        self.criterion = self._loss_cls(spatial_dims=self._SPATIAL_DIMS, **loss_config)

        clusterer_config = dict(training_config.get("clusterer", {}) or {})
        clusterer_name = clusterer_config.pop("name", "soft_meanshift")
        clusterer_config.setdefault("bandwidth", loss_config.get("delta_v", 0.5))
        clusterer_config.setdefault(
            "normalize_embeddings", loss_config.get("normalize_embeddings", False),
        )
        self.clusterer = build_clusterer(clusterer_name, **clusterer_config)

        # Vista models include a PointPromptEncoder for future proofread
        # support; keep it frozen since no config activates it.
        if hasattr(self.model, "point_encoder"):
            self.model.point_encoder.requires_grad_(False)

        self._eval_accum: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, **kw: Any) -> Dict[str, torch.Tensor]:
        return self.model(x, **kw)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_meta_tensor(batch: Dict[str, Any]) -> Dict[str, Any]:
        """Strip MONAI MetaTensor subclasses at the batch boundary."""
        return {
            k: v.as_subclass(torch.Tensor) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    @torch.no_grad()
    def _prepare_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Build the targets dict consumed by ``self.criterion``.

        Boundary/membrane voxels are applied upstream by the data pipeline
        (``data.find_boundaries`` → ``FindBoundariesd``), not here.
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
        for key in ("label_direction", "label_covariance"):
            if key in batch:
                targets[key] = batch[key]
        return targets

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        batch = self._strip_meta_tensor(batch)
        images = batch["image"]
        if images.dim() == self._SPATIAL_DIMS + 1:
            images = rearrange(images, self._EXPAND_PATTERN)

        targets = self._prepare_targets(batch)
        targets["_cached_weights"] = self.criterion._compute_targets(
            targets["labels"], targets,
        )

        predictions = self.model(images)
        losses = self.criterion(predictions, targets)
        total_loss = losses["loss"].float()

        bs = images.shape[0]
        for name, value in losses.items():
            self.log(f"train/automatic/{name}", value,
                     on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/loss", total_loss, prog_bar=True, on_step=True, batch_size=bs)

        return total_loss

    # ------------------------------------------------------------------
    # Evaluation — accumulate metrics locally, all-reduce once per epoch
    # ------------------------------------------------------------------

    def _accum(self, name: str, value, weight: float) -> None:
        v = value.item() if isinstance(value, torch.Tensor) else float(value)
        acc = self._eval_accum[name]
        acc[0] += v * weight
        acc[1] += weight

    @torch.no_grad()
    def _eval_step_and_accumulate(
        self, batch: Dict[str, torch.Tensor], prefix: str,
    ) -> None:
        batch = self._strip_meta_tensor(batch)
        images = batch["image"]
        if images.dim() == self._SPATIAL_DIMS + 1:
            images = rearrange(images, self._EXPAND_PATTERN)

        targets = self._prepare_targets(batch)
        predictions = self.model(images, semantic_ids=targets.get("semantic_ids"))
        losses = self.criterion(predictions, targets)

        bs = float(images.shape[0])
        for name, val in losses.items():
            self._accum(f"{prefix}/{name}", val, bs)

        sem_logits = predictions["semantic"]
        active = getattr(self.criterion.semantic_loss, "active_classes", None)
        if active is not None and active < sem_logits.shape[1]:
            sem_logits = sem_logits[:, :active]
        sem_pred = sem_logits.argmax(dim=1)
        sem_gt = targets["semantic_labels"]
        n_cls = sem_logits.shape[1]

        self._accum(f"{prefix}/sem_acc",
                    reduce((sem_pred == sem_gt).float(), "b ... -> ", "mean"), bs)
        self._accum(f"{prefix}/sem_iou",
                    compute_per_batch_iou(sem_pred, sem_gt, num_classes=n_cls), bs)
        self._accum(f"{prefix}/sem_dice",
                    compute_per_batch_dice(sem_pred, sem_gt, num_classes=n_cls), bs)

        fg_mask = targets["labels"] > 0
        if fg_mask.any():
            ins_pred, _, _ = self.clusterer(predictions["instance"], fg_mask)
            ins_gt = targets["labels"]
            self._accum(f"{prefix}/ins_ari", compute_per_batch_ari(ins_pred, ins_gt), bs)
            self._accum(f"{prefix}/ins_ami", compute_per_batch_ami(ins_pred, ins_gt), bs)
            voi = compute_per_batch_voi(ins_pred, ins_gt)
            self._accum(f"{prefix}/ins_voi", voi.total, bs)
            self._accum(f"{prefix}/ins_voi_split", voi.split, bs)
            self._accum(f"{prefix}/ins_voi_merge", voi.merge, bs)
            self._accum(f"{prefix}/ins_ted", compute_per_batch_ted(ins_pred, ins_gt), bs)
            del ins_pred

        del predictions, losses

    def _reduce_and_log_accum(self, prefix: str) -> None:
        if not self._eval_accum:
            return

        names = sorted(self._eval_accum)
        sums = torch.tensor([self._eval_accum[n][0] for n in names], device=self.device)
        counts = torch.tensor([self._eval_accum[n][1] for n in names], device=self.device)

        if self.trainer.world_size > 1:
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

        prog_bar_names = {
            f"{prefix}/loss",
            f"{prefix}/sem_acc",
            f"{prefix}/sem_iou",
            f"{prefix}/ins_ari",
        }
        for i, name in enumerate(names):
            if counts[i] > 0:
                avg = (sums[i] / counts[i]).item()
                self.log(name, avg, prog_bar=(name in prog_bar_names),
                         sync_dist=False, rank_zero_only=True)

        self._eval_accum.clear()

    # Validation
    def on_validation_epoch_start(self) -> None:
        self._eval_accum: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._eval_step_and_accumulate(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self._reduce_and_log_accum("val")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Test
    def on_test_epoch_start(self) -> None:
        self._eval_accum = defaultdict(lambda: [0.0, 0.0])

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._eval_step_and_accumulate(batch, "test")

    def on_test_epoch_end(self) -> None:
        self._reduce_and_log_accum("test")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            {"params": decay,    "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)

        sched_cfg = self.optimizer_config.get("scheduler", {})
        stype = sched_cfg.get("type", "cosine").lower()

        if stype == "cosine":
            from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

            warmup_epochs = sched_cfg.get("warmup_epochs", 5)
            t_max = sched_cfg.get("T_max", 100)
            eta_min = sched_cfg.get("eta_min", 1e-7)

            warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(
                optimizer, T_max=max(t_max - warmup_epochs, 1), eta_min=eta_min,
            )
            scheduler = SequentialLR(
                optimizer, [warmup, cosine], milestones=[warmup_epochs],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        if stype:
            warnings.warn(
                f"Unknown scheduler type '{stype}', using no scheduler. Supported: 'cosine'.",
                stacklevel=2,
            )
        return optimizer

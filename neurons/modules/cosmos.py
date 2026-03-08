"""
Base Cosmos Lightning Module for 3D volumetric segmentation training.

Provides the shared implementation for :class:`CosmosPredict3DModule` and
:class:`CosmosTransfer3DModule`.  Subclasses set ``_model_cls``,
``_loss_cls`` and ``_verify_fit_fn``; all training, evaluation, freeze
scheduling and optimiser logic lives here.

Supports two training modes that can be combined in a single step:

- **automatic**: predict from the volume alone.
- **proofread**: additional context via fractionary labels or interactive
  point prompts.
"""

import logging
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

logger = logging.getLogger(__name__)

_SPATIAL_DIMS = 3
_EXPAND_PATTERN = "b d h w -> b 1 d h w"
_SQUEEZE_PATTERN = "b 1 d h w -> b d h w"

_DEFAULT_TRAINING_MODES: List[str] = ["automatic"]


class BaseCosmosModule(pl.LightningModule):
    """Abstract base for Cosmos Predict / Transfer 3D modules.

    Subclasses **must** define:

    - ``_model_cls`` (``type``): Model wrapper class.
    - ``_loss_cls`` (``type``): Loss class.
    - ``_verify_fit_fn`` (``staticmethod``): The ``verify_fit`` callable
      from the corresponding model module.

    Args:
        model_config: Forwarded to ``_model_cls``.
        optimizer_config: Optimizer / scheduler settings.
        loss_config: Forwarded as ``**loss_config`` to ``_loss_cls``.
        training_config: Training behaviour (modes, point sampling, ...).
    """

    _model_cls: type
    _loss_cls: type
    _verify_fit_fn: Any  # staticmethod wrapping a Callable

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

        model_config = dict(model_config or {})
        _hf_token = model_config.pop("hf_token", None)
        self.save_hyperparameters()
        if _hf_token is not None:
            model_config["hf_token"] = _hf_token

        self.optimizer_config = optimizer_config or {}
        loss_config = loss_config or {}
        training_config = training_config or {}

        self.model = self._model_cls(
            in_channels=model_config.get("in_channels", 1),
            num_classes=model_config.get("num_classes", 16),
            emb_dim=model_config.get("emb_dim", 16),
            feature_size=model_config.get("feature_size", 64),
            variant=model_config.get("variant", "2B"),
            dtype=model_config.get("dtype", "bf16"),
            freeze_dit_backbone=model_config.get("freeze_dit_backbone", False),
            freeze_vae_decoder=model_config.get("freeze_vae_decoder", False),
            freeze_vae_encoder=model_config.get("freeze_vae_encoder", True),
            feature_layers=model_config.get("feature_layers"),
            cache_dir=model_config.get("cache_dir"),
            hf_token=model_config.get("hf_token"),
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

        self._variant = model_config.get("variant", "2B")

        self._freeze_schedule = {
            "vae_encoder": model_config.get("freeze_vae_encoder", True),
            "dit_backbone": model_config.get("freeze_dit_backbone", False),
            "vae_decoder": model_config.get("freeze_vae_decoder", False),
        }

    # ------------------------------------------------------------------
    # Phased freeze / unfreeze
    # ------------------------------------------------------------------

    @staticmethod
    def _should_freeze(value: "bool | int", epoch: int) -> bool:
        if isinstance(value, bool):
            return value
        return epoch < int(value)

    def on_train_epoch_start(self) -> None:
        epoch = self.current_epoch
        _METHODS = {
            "vae_encoder": (self.model.freeze_vae_encoder, self.model.unfreeze_vae_encoder),
            "dit_backbone": (self.model.freeze_dit_backbone, self.model.unfreeze_dit_backbone),
            "vae_decoder": (self.model.freeze_vae_decoder, self.model.unfreeze_vae_decoder),
        }
        _FLAGS = {
            "vae_encoder": "_freeze_vae_encoder",
            "dit_backbone": "_freeze_dit_backbone",
            "vae_decoder": "_freeze_vae_decoder",
        }
        for name, value in self._freeze_schedule.items():
            if isinstance(value, bool):
                continue
            want_frozen = self._should_freeze(value, epoch)
            is_frozen = getattr(self.model, _FLAGS[name])
            if want_frozen and not is_frozen:
                _METHODS[name][0]()
            elif not want_frozen and is_frozen:
                _METHODS[name][1]()

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
        """Extract and reshape targets from *batch*."""
        labels = batch["label"]
        if labels.dim() == _SPATIAL_DIMS + 2:
            labels = rearrange(labels, _SQUEEZE_PATTERN)

        sem = batch.get("semantic_ids", (labels > 0).long())
        if sem.dim() == _SPATIAL_DIMS + 2:
            sem = rearrange(sem, _SQUEEZE_PATTERN)

        targets: Dict[str, Any] = {
            "semantic_labels": sem,
            "labels": labels,
        }
        if "semantic_ids" in batch:
            targets["semantic_ids"] = (
                rearrange(batch["semantic_ids"], _SQUEEZE_PATTERN)
                if batch["semantic_ids"].dim() == _SPATIAL_DIMS + 2
                else batch["semantic_ids"]
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
        if images.dim() == _SPATIAL_DIMS + 1:
            images = rearrange(images, _EXPAND_PATTERN)

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
        if images.dim() == _SPATIAL_DIMS + 1:
            images = rearrange(images, _EXPAND_PATTERN)

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

        backbone_lr = self.optimizer_config.get("backbone_lr")
        if backbone_lr is not None and backbone_lr != lr:
            backbone_decay, backbone_no_decay = [], []
            head_decay, head_no_decay = [], []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                is_backbone = name.startswith("model.dit.")
                if param.dim() <= 1 or name.endswith(".bias"):
                    (backbone_no_decay if is_backbone else head_no_decay).append(param)
                else:
                    (backbone_decay if is_backbone else head_decay).append(param)
            param_groups = [
                {"params": backbone_decay, "lr": backbone_lr, "weight_decay": wd},
                {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
                {"params": head_decay, "lr": lr, "weight_decay": wd},
                {"params": head_no_decay, "lr": lr, "weight_decay": 0.0},
            ]
            optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)
        else:
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

        if stype in ("cosine", "cosine_warmup"):
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
                f"Supported: 'cosine', 'cosine_warmup'.",
                stacklevel=2,
            )
        return optimizer

    # ------------------------------------------------------------------
    # Compatibility check
    # ------------------------------------------------------------------

    def compatibility_check(
        self,
        input_shape: tuple = (1, 1, 32, 64, 64),
    ) -> Dict[str, Any]:
        """Verify the 3-D model fits the 3-stage volumetric segmentation task."""
        result = self._verify_fit_fn(
            variant=self._variant,
            input_shape=input_shape,
            num_classes=self.model.num_classes,
            emb_dim=self.model.emb_dim,
            feature_size=self.model.feature_size,
        )

        for mode in self.training_modes:
            if mode not in ("automatic", "proofread"):
                result["errors"].append(f"Invalid training mode: {mode}")
                result["compatible"] = False

        if self._variant == "14B" and self.criterion.weight_geometry > 0:
            result["warnings"].append(
                "14B + 3-D geometry head is extremely memory-intensive.  "
                "Consider weight_geometry=0 or gradient checkpointing."
            )

        if not self.model._backbone_loaded:
            result["warnings"].append(
                "Backbone weights not loaded -- model is randomly initialised."
            )

        result["checks"]["backbone_loaded"] = self.model._backbone_loaded
        result["checks"]["backbone_frozen"] = self.model._freeze_dit_backbone
        result["checks"]["training_modes"] = self.training_modes
        result["checks"]["variant"] = self._variant
        return result

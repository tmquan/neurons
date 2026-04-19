"""
Base Cosmos Lightning module for 3-D volumetric segmentation training.

Shared implementation for :class:`CosmosPredict3DModule` and
:class:`CosmosTransfer3DModule`.  Subclasses just set ``_model_cls`` and
``_loss_cls``; every training, evaluation, freeze-scheduling and
optimiser hook lives here.

Only the **automatic** training mode is supported (predict from the
volume alone).  Point-prompt / proofread training is a Vista-only path.
"""

import logging
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

logger = logging.getLogger(__name__)

_SPATIAL_DIMS = 3
_EXPAND_PATTERN = "b d h w -> b 1 d h w"
_SQUEEZE_PATTERN = "b 1 d h w -> b d h w"


class BaseCosmosModule(pl.LightningModule):
    """Abstract base for Cosmos-Predict / Cosmos-Transfer 3-D modules.

    Subclasses **must** define:
      - ``_model_cls`` (``type``): Model wrapper class.
      - ``_loss_cls``  (``type``): Loss class (typically :class:`CombinedLoss`).

    Args:
        model_config: Forwarded to ``_model_cls``.
        optimizer_config: Optimizer / scheduler settings.
        loss_config: Forwarded as ``**loss_config`` to ``_loss_cls``.
        training_config: Training behaviour (clusterer, freeze schedule, ...).
    """

    _model_cls: type
    _loss_cls: type

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

        model_config = dict(model_config or {})
        # ``hf_token`` is intentionally not persisted via save_hyperparameters.
        hf_token = model_config.pop("hf_token", None)
        self.save_hyperparameters()
        if hf_token is not None:
            model_config["hf_token"] = hf_token

        self.optimizer_config = optimizer_config or {}
        self.training_config = training_config or {}
        loss_config = loss_config or {}

        disabled_heads = {
            name for name in ("semantic", "instance", "geometry")
            if loss_config.get(f"weight_{name}", 1.0) == 0
        }
        self._disabled_heads = frozenset(disabled_heads)
        if disabled_heads:
            logger.info("Heads disabled (weight=0): %s", sorted(disabled_heads))

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
            gradient_checkpointing=model_config.get("gradient_checkpointing", False),
            feature_layers=model_config.get("feature_layers"),
            cache_dir=model_config.get("cache_dir"),
            hf_token=model_config.get("hf_token"),
            dropout=model_config.get("dropout", 0.0),
            disabled_heads=self._disabled_heads or None,
        )

        self.criterion = self._loss_cls(spatial_dims=_SPATIAL_DIMS, **loss_config)

        clusterer_config = dict(self.training_config.get("clusterer", {}) or {})
        clusterer_name = clusterer_config.pop("name", "soft_meanshift")
        clusterer_config.setdefault("bandwidth", loss_config.get("delta_v", 0.5))
        clusterer_config.setdefault(
            "normalize_embeddings", loss_config.get("normalize_embeddings", False),
        )
        self.clusterer = build_clusterer(clusterer_name, **clusterer_config)
        logger.info(
            "Validation clusterer: %s (%s)",
            clusterer_name, type(self.clusterer).__name__,
        )

        # Cosmos doesn't train the point encoder (proofread unsupported),
        # but the wrapper still instantiates one so ``ddp find_unused_parameters``
        # works; simply never flow grads through it.
        if hasattr(self.model, "point_encoder"):
            self.model.point_encoder.requires_grad_(False)
        if self.model._backbone_loaded and self.model.vae_encoder is not None:
            self.model._fallback_down.requires_grad_(False)

        self._eval_accum: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])

        # Phased freeze schedule: value is either a bool ("permanently
        # frozen / permanently trainable") or an int ("frozen for the
        # first N epochs, then unfreeze").
        self._freeze_schedule = {
            "vae_encoder": model_config.get("freeze_vae_encoder", True),
            "dit_backbone": model_config.get("freeze_dit_backbone", False),
            "vae_decoder": model_config.get("freeze_vae_decoder", False),
        }

    # ------------------------------------------------------------------
    # Phased freeze / unfreeze
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        methods = {
            "vae_encoder": (self.model.freeze_vae_encoder, self.model.unfreeze_vae_encoder),
            "dit_backbone": (self.model.freeze_dit_backbone, self.model.unfreeze_dit_backbone),
            "vae_decoder": (self.model.freeze_vae_decoder, self.model.unfreeze_vae_decoder),
        }
        flags = {
            "vae_encoder": "_freeze_vae_encoder",
            "dit_backbone": "_freeze_dit_backbone",
            "vae_decoder": "_freeze_vae_decoder",
        }
        needs_rebuild = False
        for name, schedule in self._freeze_schedule.items():
            if isinstance(schedule, bool):
                continue  # permanently frozen / permanently trainable
            want_frozen = self.current_epoch < int(schedule)
            is_frozen = getattr(self.model, flags[name])
            if want_frozen and not is_frozen:
                methods[name][0]()
            elif not want_frozen and is_frozen:
                methods[name][1]()
                needs_rebuild = True
        if needs_rebuild and self.trainer is not None:
            self.trainer.strategy.setup_optimizers(self.trainer)

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
        """Strip MONAI MetaTensor subclasses at the batch boundary.

        MetaTensor's ``__torch_function__`` override can interfere with
        mixed-dtype backward passes; plain ``torch.Tensor`` is safer.
        """
        return {
            k: v.as_subclass(torch.Tensor) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    @torch.inference_mode()
    def _prepare_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Build the targets dict consumed by ``self.criterion``.

        Boundary/membrane voxels are applied upstream by the data pipeline
        (``data.find_boundaries`` → ``FindBoundariesd``), not here.
        """
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
            sid = batch["semantic_ids"]
            targets["semantic_ids"] = (
                rearrange(sid, _SQUEEZE_PATTERN)
                if sid.dim() == _SPATIAL_DIMS + 2 else sid
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

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Optional[torch.Tensor]:
        batch = self._strip_meta_tensor(batch)
        images = batch["image"]
        if images.dim() == _SPATIAL_DIMS + 1:
            images = rearrange(images, _EXPAND_PATTERN)

        targets = self._prepare_targets(batch)
        targets["_cached_weights"] = self.criterion._compute_targets(
            targets["labels"], targets,
        )

        predictions = self.model(images)
        losses = self.criterion(predictions, targets)
        total_loss = losses["loss"]

        if total_loss.isnan().any() or total_loss.isinf().any():
            nan_keys = [
                k for k, v in losses.items()
                if isinstance(v, torch.Tensor) and (v.isnan().any() or v.isinf().any())
            ]
            logger.warning(
                "NaN/Inf total loss at step %d — skipping backward (keys=%s).",
                self.global_step, nan_keys,
            )
            return None

        bs = images.shape[0]
        for name, value in losses.items():
            self.log(f"train/automatic/{name}", value,
                     on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/loss", total_loss,
                 prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)

        return total_loss

    # ------------------------------------------------------------------
    # Evaluation — accumulate metrics locally, all-reduce once per epoch
    # ------------------------------------------------------------------

    def _accum(self, name: str, value, weight: float) -> None:
        v = value.item() if isinstance(value, torch.Tensor) else float(value)
        acc = self._eval_accum[name]
        acc[0] += v * weight
        acc[1] += weight

    @torch.inference_mode()
    def _eval_step_and_accumulate(
        self, batch: Dict[str, torch.Tensor], prefix: str,
    ) -> None:
        batch = self._strip_meta_tensor(batch)
        images = batch["image"]
        if images.dim() == _SPATIAL_DIMS + 1:
            images = rearrange(images, _EXPAND_PATTERN)

        targets = self._prepare_targets(batch)
        predictions = self.model(images, semantic_ids=targets.get("semantic_ids"))
        losses = self.criterion(predictions, targets)

        bs = float(images.shape[0])
        for name, val in losses.items():
            self._accum(f"{prefix}/{name}", val, bs)

        if "semantic" in predictions:
            sem_logits = predictions["semantic"]
            sem_loss = self.criterion.semantic_loss
            active = getattr(sem_loss, "active_classes", None) if sem_loss else None
            if active is not None and active < sem_logits.shape[1]:
                sem_logits = sem_logits[:, :active]

            sem_mode = getattr(sem_loss, "mode", "softmax") if sem_loss else "softmax"
            if sem_mode == "sigmoid" and sem_logits.shape[1] == 1:
                sem_pred = (sem_logits[:, 0].sigmoid() > 0.5).long()
                n_cls = 2
            else:
                sem_pred = sem_logits.argmax(dim=1)
                n_cls = sem_logits.shape[1]

            sem_gt = targets["semantic_labels"]
            self._accum(f"{prefix}/sem_acc",
                        reduce((sem_pred == sem_gt).float(), "b ... -> ", "mean"), bs)
            self._accum(f"{prefix}/sem_iou",
                        compute_per_batch_iou(sem_pred, sem_gt, num_classes=n_cls), bs)
            self._accum(f"{prefix}/sem_dice",
                        compute_per_batch_dice(sem_pred, sem_gt, num_classes=n_cls), bs)

        if "instance" in predictions:
            fg_mask = targets["labels"] > 0
            if fg_mask.any():
                ins_pred, _, _ = self.clusterer(predictions["instance"].float(), fg_mask)
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
            f"{prefix}/sem_dice",
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

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None,
    ) -> None:
        """Zero NaN/Inf gradients before clipping so bad batches don't poison weights."""
        bad = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None and (p.grad.isnan().any() or p.grad.isinf().any()):
                    p.grad.zero_()
                    bad += 1
        if bad:
            logger.warning(
                "Zeroed NaN/Inf gradients in %d parameters at step %d.",
                bad, self.global_step,
            )
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def configure_optimizers(self) -> Any:
        lr = self.optimizer_config.get("lr", 1e-4)
        wd = self.optimizer_config.get("weight_decay", 1e-5)
        backbone_lr = self.optimizer_config.get("dit_backbone_lr") or lr

        backbone_decay, backbone_no_decay, head_decay, head_no_decay = [], [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_backbone = name.startswith("model.dit.")
            no_decay = param.dim() <= 1 or name.endswith(".bias")
            if is_backbone:
                (backbone_no_decay if no_decay else backbone_decay).append(param)
            else:
                (head_no_decay if no_decay else head_decay).append(param)

        param_groups = [
            {"params": backbone_decay,    "lr": backbone_lr, "weight_decay": wd},
            {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
            {"params": head_decay,        "lr": lr,          "weight_decay": wd},
            {"params": head_no_decay,     "lr": lr,          "weight_decay": 0.0},
        ]
        param_groups = [g for g in param_groups if g["params"]]

        clip_val = self.training_config.get("gradient_clip_val")
        use_fused = (
            not clip_val
            and torch.cuda.is_available()
            and all(p.is_cuda for g in param_groups for p in g["params"])
        )
        optimizer = torch.optim.AdamW(
            param_groups, lr=lr, weight_decay=wd, fused=use_fused,
        )

        sched_cfg = self.optimizer_config.get("scheduler", {})
        stype = sched_cfg.get("type", "cosine").lower()

        if stype in ("cosine", "cosine_warmup"):
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
                f"Unknown scheduler type '{stype}', using no scheduler. "
                "Supported: 'cosine', 'cosine_warmup'.",
                stacklevel=2,
            )
        return optimizer

"""
Cosmos-Transfer2.5 **3D** Lightning Module for volumetric segmentation training.

Supports two training modes (combinable in a single step):

- **automatic**: predict from the volume alone.
- **proofread**: additional context via fractionary labels or interactive
  point prompts.

Includes :meth:`compatibility_check` to verify that the selected 2B / 14B
variant can handle 3-D input within memory constraints.
"""

from typing import Any, Dict, List, Optional

import logging
import torch
import pytorch_lightning as pl
from einops import rearrange, reduce

from neurons.models.cosmostransfer3d_model import (
    CosmosTransfer3DWrapper as _Model,
    verify_fit,
)
from neurons.losses.cosmostransfer3d_losses import CosmosTransfer3DLoss as _Loss
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


class CosmosTransfer3DModule(pl.LightningModule):
    """PyTorch Lightning module for 3-D Cosmos-Transfer2.5 segmentation.

    Three-head architecture:

    - ``semantic``  [B, C, D, H, W]
    - ``instance``  [B, E, D, H, W]
    - ``geometry``  [B, G, D, H, W]

    Args:
        model_config: Forwarded to :class:`CosmosTransfer3DWrapper`.
        optimizer_config: Optimizer / scheduler settings.
        loss_config: Forwarded to :class:`CosmosTransfer3DLoss`.
        training_config: Training behaviour (modes, point sampling, ...).
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
            variant=model_config.get("variant", "2B"),
            checkpoint_variant=model_config.get("checkpoint_variant", "post-trained"),
            dtype=model_config.get("dtype", "bf16"),
            freeze_backbone=model_config.get("freeze_backbone", True),
            feature_layers=model_config.get("feature_layers"),
            cache_dir=model_config.get("cache_dir"),
            hf_token=model_config.get("hf_token"),
            dropout=model_config.get("dropout", 0.0),
        )

        self.criterion = _Loss(
            weight_semantic=loss_config.get("weight_semantic", 1.0),
            weight_instance=loss_config.get("weight_instance", 1.0),
            weight_geometry=loss_config.get("weight_geometry", 0.0),
            weight_flow_consistency=loss_config.get("weight_flow_consistency", 0.0),
            semantic_mode=loss_config.get("semantic_mode", "sigmoid"),
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
            active_classes=loss_config.get("active_classes"),
            weight_dir=loss_config.get("weight_dir", 1.0),
            weight_cov=loss_config.get("weight_cov", 1.0),
            weight_raw=loss_config.get("weight_raw", 1.0),
            dir_target=loss_config.get("dir_target", "centroid"),
            loss_dir=loss_config.get("loss_dir", "smooth_l1"),
            loss_cov=loss_config.get("loss_cov", "mse"),
            loss_raw=loss_config.get("loss_raw", "l1"),
        )

        self._clusterer = SoftMeanShift(bandwidth=loss_config.get("delta_v", 0.5))
        self._ignore_index = loss_config.get("ignore_index", -100)

        self.training_modes: List[str] = list(
            training_config.get("training_modes", _DEFAULT_TRAINING_MODES)
        )
        self._num_pos_points: int = training_config.get("num_pos_points", 5)
        self._num_neg_points: int = training_config.get("num_neg_points", 5)
        self._point_sample_mode: str = training_config.get("point_sample_mode", "class")

        self._variant = model_config.get("variant", "2B")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, **kw: Any) -> Dict[str, torch.Tensor]:
        return self.model(x, **kw)

    # ------------------------------------------------------------------
    # Target preparation
    # ------------------------------------------------------------------

    def _prepare_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        if "image" in batch:
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
        labels = targets["labels"]
        has_ignore = (labels == self._ignore_index).any()
        has_valid_fg = (labels > 0).any() & (labels != self._ignore_index).any()
        if has_ignore and has_valid_fg:
            return "fractionary"
        return "interactive"

    def _resolve_fractionary_labels(
        self, targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        targets = dict(targets)
        labels = targets["labels"]
        unknown = labels == self._ignore_index

        sem = targets["semantic_labels"].clone()
        sem[unknown] = self._ignore_index
        targets["semantic_labels"] = sem
        targets["semantic_ids"] = sem.clone()

        inst = labels.clone()
        inst[unknown] = 0
        known_ids = inst.unique()
        known_ids = known_ids[known_ids > 0]
        remap = torch.zeros(
            int(known_ids.max().item()) + 1 if known_ids.numel() > 0 else 1,
            dtype=torch.long, device=labels.device,
        )
        for new_id, old_id in enumerate(known_ids, start=1):
            remap[old_id] = new_id
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
            self.criterion._get_cached_targets(targets["labels"], targets)

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

        backbone_lr = self.optimizer_config.get("backbone_lr")
        if backbone_lr is not None and backbone_lr != lr:
            backbone_params = [
                p for p in self.model.dit.parameters() if p.requires_grad
            ]
            head_params = [
                p for n, p in self.model.named_parameters()
                if not n.startswith("dit.") and p.requires_grad
            ]
            param_groups = [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": head_params, "lr": lr},
            ]
            optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

        sched_cfg = self.optimizer_config.get("scheduler", {})
        stype = sched_cfg.get("type", "cosine").lower()

        if stype == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_cfg.get("T_max", 100),
                eta_min=sched_cfg.get("eta_min", 1e-7),
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        if stype == "cosine_warmup":
            from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
            warmup_epochs = sched_cfg.get("warmup_epochs", 5)
            warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=sched_cfg.get("T_max", 100) - warmup_epochs,
                eta_min=sched_cfg.get("eta_min", 1e-7),
            )
            scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        if stype:
            import warnings
            warnings.warn(
                f"Unknown scheduler type '{stype}', using no scheduler.  "
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
        """Verify the 3-D model fits the 3-stage volumetric segmentation task.

        **Stage mapping (3-D):**

        ================  ==============================  ============================
        Stage             Cosmos Component                Constraint / Mismatch
        ================  ==============================  ============================
        1 (Semantic)      DiT features -> semantic head    Channel adaptation needed
                                                          (EM 1-ch -> RGB 3-ch).
                                                          Depth maps to temporal axis.
        2 (Instance)      DiT features -> instance head    Embedding bandwidth limited
                                                          by hidden_dim projection.
                                                          3-D clustering is more
                                                          expensive than 2-D.
        3 (Geometry)      DiT features -> geometry head    16 output channels
                                                          (dir=3 + cov=9 + rgba=4).
                                                          **14B** variant may OOM on
                                                          single GPU with geometry.
        ================  ==============================  ============================
        """
        result = verify_fit(
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
        result["checks"]["backbone_frozen"] = self.model._freeze_backbone
        result["checks"]["training_modes"] = self.training_modes
        result["checks"]["variant"] = self._variant
        return result

"""
Vista2D combined loss for image-based segmentation.

Composes three standalone losses:
- SemanticLoss  (from ``neurons.losses.semantic``)
- InstanceLoss  (from ``neurons.losses.instance``)
- GeometryLoss  (from ``neurons.losses.geometry``)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from neurons.losses.semantic import SemanticLoss
from neurons.losses.instance import InstanceLoss
from neurons.losses.geometry import GeometryLoss

_SPATIAL_DIMS = 2


class Vista2DLoss(nn.Module):
    """Compose SemanticLoss + InstanceLoss + GeometryLoss for Vista2D."""

    def __init__(
        self,
        weight_semantic: float = 1.0,
        weight_instance: float = 1.0,
        weight_geometry: float = 0.0,
        semantic_mode: str = "sigmoid",
        weight_pull: float = 1.0, weight_push: float = 1.0,
        weight_norm: float = 0.001,
        weight_edge: float = 10.0, weight_bone: float = 10.0,
        delta_v: float = 0.5, delta_d: float = 1.5,
        weight_ce: float = 1.0, weight_iou: float = 0.0, weight_dice: float = 0.0,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -100,
        active_classes: Optional[int] = None,
        **geom_kwargs,
    ) -> None:
        super().__init__()
        self.weight_semantic = weight_semantic
        self.weight_instance = weight_instance
        self.weight_geometry = weight_geometry

        self.semantic_loss = SemanticLoss(
            mode=semantic_mode, weight_ce=weight_ce,
            weight_iou=weight_iou, weight_dice=weight_dice,
            class_weights=class_weights, ignore_index=ignore_index,
            active_classes=active_classes,
        )
        self.instance_loss = InstanceLoss(
            spatial_dims=_SPATIAL_DIMS,
            weight_pull=weight_pull, weight_push=weight_push,
            weight_norm=weight_norm, weight_edge=weight_edge,
            weight_bone=weight_bone, delta_v=delta_v, delta_d=delta_d,
        )
        self.geometry_loss = (
            GeometryLoss(spatial_dims=_SPATIAL_DIMS, **geom_kwargs)
            if weight_geometry > 0 else None
        )

        self._cache_key: Optional[Tuple] = None
        self._cached_ins_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._cached_geom_targets: Optional[Dict] = None

    @staticmethod
    def _label_fingerprint(labels):
        return (labels.shape, labels.data_ptr(), int(labels.sum().item()))

    def _get_cached_targets(self, labels, targets=None):
        key = self._label_fingerprint(labels)
        if key != self._cache_key:
            self._cache_key = key
            self._cached_ins_weights = self.instance_loss.compute_weights(labels)
            if (self.geometry_loss is not None
                    and targets is not None
                    and "label_direction" in targets
                    and "label_covariance" in targets):
                self._cached_geom_targets = self.geometry_loss.targets_from_pipeline(
                    targets["label_direction"], targets["label_covariance"],
                )
            elif self.geometry_loss is not None:
                self._cached_geom_targets = self.geometry_loss.compute_targets(labels)
            else:
                self._cached_geom_targets = None
        return self._cached_ins_weights, self._cached_geom_targets

    def forward(self, predictions, targets) -> Dict[str, torch.Tensor]:
        labels = targets["labels"]
        (w_edge, w_bone), geom_targets = self._get_cached_targets(labels, targets)

        sem = self.semantic_loss(predictions["semantic"], targets["semantic_labels"])

        sem_ids = targets.get("semantic_ids")
        if sem_ids is None:
            sem_ids = predictions.get("semantic_ids")

        ins = self.instance_loss(
            predictions["instance"], labels,
            semantic_ids=sem_ids,
            weight_edge=w_edge, weight_bone=w_bone,
        )

        total = self.weight_semantic * sem["loss"] + self.weight_instance * ins["loss"]

        out: Dict[str, torch.Tensor] = {
            "loss_sem":       sem["loss"],
            "loss_sem/ce":    sem["ce"],
            "loss_ins":       ins["loss"],
            "loss_ins/pull":  ins["pull"],
            "loss_ins/push":  ins["push"],
            "loss_ins/norm":  ins["norm"],
        }
        if self.semantic_loss.weight_iou > 0:
            out["loss_sem/iou"] = sem["iou"]
        if self.semantic_loss.weight_dice > 0:
            out["loss_sem/dice"] = sem["dice"]

        if self.geometry_loss is not None and "geometry" in predictions:
            geom = self.geometry_loss(
                predictions["geometry"], labels,
                raw_image=targets.get("raw_image"),
                cached_targets=geom_targets,
            )
            total = total + self.weight_geometry * geom["loss"]
            out["loss_geom"]      = geom["loss"]
            out["loss_geom/dir"]  = geom["dir"]
            out["loss_geom/cov"]  = geom["cov"]
            out["loss_geom/raw"]  = geom["raw"]

        out["loss"] = total
        return out

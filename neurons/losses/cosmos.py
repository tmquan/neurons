"""
Base combined loss for segmentation (2-D and 3-D).

Composes :class:`SemanticLoss`, :class:`InstanceLoss`, and optionally
:class:`GeometryLoss`.  Subclasses set ``_SPATIAL_DIMS`` and may add
extra terms (e.g. feature-consistency).
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurons.losses.semantic import SemanticLoss
from neurons.losses.instance import InstanceLoss
from neurons.losses.geometry import GeometryLoss


class BaseCombinedLoss(nn.Module):
    """Compose SemanticLoss + InstanceLoss + GeometryLoss.

    Subclasses must set ``_SPATIAL_DIMS`` (2 or 3).
    """

    _SPATIAL_DIMS: int = 3

    def __init__(
        self,
        weight_semantic: float = 1.0,
        weight_instance: float = 1.0,
        weight_geometry: float = 0.0,
        semantic_mode: str = "sigmoid",
        weight_pull: float = 1.0,
        weight_push: float = 1.0,
        weight_norm: float = 0.001,
        weight_edge: float = 10.0,
        weight_bone: float = 10.0,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        weight_ce: float = 1.0,
        weight_iou: float = 0.0,
        weight_dice: float = 0.0,
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
            mode=semantic_mode,
            weight_ce=weight_ce,
            weight_iou=weight_iou,
            weight_dice=weight_dice,
            class_weights=class_weights,
            ignore_index=ignore_index,
            active_classes=active_classes,
        )
        self.instance_loss = InstanceLoss(
            spatial_dims=self._SPATIAL_DIMS,
            weight_pull=weight_pull,
            weight_push=weight_push,
            weight_norm=weight_norm,
            weight_edge=weight_edge,
            weight_bone=weight_bone,
            delta_v=delta_v,
            delta_d=delta_d,
        )
        self.geometry_loss: Optional[GeometryLoss] = (
            GeometryLoss(spatial_dims=self._SPATIAL_DIMS, **geom_kwargs)
            if weight_geometry > 0
            else None
        )

    def _compute_targets(
        self,
        labels: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Derive instance weights and geometry targets from label maps."""
        ins_weights = self.instance_loss.compute_weights(labels)

        if (
            self.geometry_loss is not None
            and targets is not None
            and "label_direction" in targets
            and "label_covariance" in targets
        ):
            geom_targets = self.geometry_loss.targets_from_pipeline(
                targets["label_direction"],
                targets["label_covariance"],
            )
        elif self.geometry_loss is not None:
            geom_targets = self.geometry_loss.compute_targets(labels)
        else:
            geom_targets = None

        return ins_weights, geom_targets

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        labels = targets["labels"]
        cached = targets.get("_cached_weights")
        if cached is not None:
            (w_edge, w_bone), geom_targets = cached
        else:
            (w_edge, w_bone), geom_targets = self._compute_targets(labels, targets)

        sem = self.semantic_loss(
            predictions["semantic"], targets["semantic_labels"],
        )

        sem_ids = targets.get("semantic_ids")
        if sem_ids is None:
            sem_ids = predictions.get("semantic_ids")

        ins = self.instance_loss(
            predictions["instance"],
            labels,
            semantic_ids=sem_ids,
            weight_edge=w_edge,
            weight_bone=w_bone,
        )

        total = (
            self.weight_semantic * sem["loss"]
            + self.weight_instance * ins["loss"]
        )

        out: Dict[str, torch.Tensor] = {
            "loss_sem": sem["loss"],
            "loss_sem/ce": sem["ce"],
            "loss_sem/iou": sem["iou"],
            "loss_sem/dice": sem["dice"],
            "loss_ins": ins["loss"],
            "loss_ins/pull": ins["pull"],
            "loss_ins/push": ins["push"],
            "loss_ins/norm": ins["norm"],
        }

        if self.geometry_loss is not None and "geometry" in predictions:
            geom = self.geometry_loss(
                predictions["geometry"],
                labels,
                raw_image=targets.get("raw_image"),
                cached_targets=geom_targets,
            )
            total = total + self.weight_geometry * geom["loss"]
            out["loss_geom"] = geom["loss"]
            out["loss_geom/dir"] = geom["dir"]
            out["loss_geom/cov"] = geom["cov"]
            out["loss_geom/raw"] = geom["raw"]

        out["loss"] = total
        return out


class BaseCombinedLossWithConsistency(BaseCombinedLoss):
    """BaseCombinedLoss + optional feature-consistency term.

    Used by Cosmos-Predict3D and Cosmos-Transfer3D.
    """

    _SPATIAL_DIMS: int = 3

    def __init__(
        self,
        weight_semantic: float = 1.0,
        weight_instance: float = 1.0,
        weight_geometry: float = 0.0,
        weight_flow_consistency: float = 0.0,
        semantic_mode: str = "sigmoid",
        weight_pull: float = 1.0,
        weight_push: float = 1.0,
        weight_norm: float = 0.001,
        weight_edge: float = 10.0,
        weight_bone: float = 10.0,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        weight_ce: float = 1.0,
        weight_iou: float = 0.0,
        weight_dice: float = 0.0,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -100,
        active_classes: Optional[int] = None,
        weight_dir: float = 1.0,
        weight_cov: float = 1.0,
        weight_raw: float = 1.0,
        loss_dir: str = "smooth_l1",
        loss_cov: str = "mse",
        loss_raw: str = "l1",
        **geom_kwargs,
    ) -> None:
        super().__init__(
            weight_semantic=weight_semantic,
            weight_instance=weight_instance,
            weight_geometry=weight_geometry,
            semantic_mode=semantic_mode,
            weight_pull=weight_pull,
            weight_push=weight_push,
            weight_norm=weight_norm,
            weight_edge=weight_edge,
            weight_bone=weight_bone,
            delta_v=delta_v,
            delta_d=delta_d,
            weight_ce=weight_ce,
            weight_iou=weight_iou,
            weight_dice=weight_dice,
            class_weights=class_weights,
            ignore_index=ignore_index,
            active_classes=active_classes,
            weight_dir=weight_dir,
            weight_cov=weight_cov,
            weight_raw=weight_raw,
            loss_dir=loss_dir,
            loss_cov=loss_cov,
            loss_raw=loss_raw,
            **geom_kwargs,
        )
        self.weight_flow_consistency = weight_flow_consistency

    @staticmethod
    def _feature_consistency_loss(
        features_a: torch.Tensor,
        features_b: torch.Tensor,
    ) -> torch.Tensor:
        """Normalised L2 distance between two feature maps.

        Both tensors should have shape ``[B, C, D, H, W]``.  The loss
        is the mean over all elements of the squared difference between
        L2-normalised feature vectors.
        """
        from einops import rearrange, reduce

        a_flat = rearrange(features_a, "b c ... -> b c (...)")
        b_flat = rearrange(features_b, "b c ... -> b c (...)")

        a_norm = F.normalize(a_flat, p=2, dim=1, eps=1e-8)
        b_norm = F.normalize(b_flat, p=2, dim=1, eps=1e-8)

        return reduce((a_norm - b_norm) ** 2, "b c n -> ", "mean")

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        out = super().forward(predictions, targets)
        total = out["loss"]

        if self.semantic_loss.weight_iou <= 0:
            out.pop("loss_sem/iou", None)
        if self.semantic_loss.weight_dice <= 0:
            out.pop("loss_sem/dice", None)

        if (
            self.weight_flow_consistency > 0
            and "features_aug" in predictions
            and "features" in predictions
        ):
            fc = self._feature_consistency_loss(
                predictions["features"], predictions["features_aug"],
            )
            total = total + self.weight_flow_consistency * fc
            out["loss_flow_consistency"] = fc
            out["loss"] = total

        return out

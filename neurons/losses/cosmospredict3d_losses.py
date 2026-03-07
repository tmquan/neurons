"""
Cosmos-Predict2.5 **3D** combined loss for volumetric segmentation.

Composes three standalone losses:

- :class:`SemanticLoss`  -- CE + IoU + Dice
- :class:`InstanceLoss`  -- pull / push / norm
- :class:`GeometryLoss`  -- direction / covariance / raw reconstruction

Optionally adds a **feature-consistency loss** that penalises the L2
distance between DiT features at two augmented views of the same input,
encouraging the backbone representations to be invariant to data
augmentation.  This is useful when fine-tuning the frozen backbone.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, reduce

from neurons.losses.semantic import SemanticLoss
from neurons.losses.instance import InstanceLoss
from neurons.losses.geometry import GeometryLoss

_SPATIAL_DIMS = 3


class CosmosPredict3DLoss(nn.Module):
    """Composite loss for Cosmos-Predict2.5 3D volumetric segmentation heads.

    Mirrors the :class:`Vista3DLoss` pattern but adds an optional
    ``weight_flow_consistency`` term for feature-level self-consistency.

    Args:
        weight_semantic: Weight for the semantic loss component.
        weight_instance: Weight for the instance loss component.
        weight_geometry: Weight for the geometry loss component (0 = off).
        weight_flow_consistency: Weight for feature consistency (0 = off).
        semantic_mode: ``"sigmoid"`` or ``"softmax"``.
        weight_pull: Instance pull term weight.
        weight_push: Instance push term weight.
        weight_norm: Instance norm regularisation weight.
        weight_edge: Boundary-voxel weight multiplier.
        weight_bone: Medial-axis voxel weight multiplier.
        delta_v: Pull margin.
        delta_d: Push margin.
        weight_ce: Cross-entropy weight.
        weight_iou: IoU weight (0 = off).
        weight_dice: Dice weight (0 = off).
        class_weights: Per-class CE weights.
        ignore_index: Label value to exclude.
        active_classes: Number of leading channels to supervise.
        weight_dir: Geometry direction sub-loss weight.
        weight_cov: Geometry covariance sub-loss weight.
        weight_raw: Geometry raw-reconstruction sub-loss weight.
        dir_target: Direction target mode.
        loss_dir: Loss function for direction.
        loss_cov: Loss function for covariance.
        loss_raw: Loss function for raw reconstruction.
    """

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
        dir_target: str = "centroid",
        loss_dir: str = "smooth_l1",
        loss_cov: str = "mse",
        loss_raw: str = "l1",
        **geom_kwargs,
    ) -> None:
        super().__init__()
        self.weight_semantic = weight_semantic
        self.weight_instance = weight_instance
        self.weight_geometry = weight_geometry
        self.weight_flow_consistency = weight_flow_consistency

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
            spatial_dims=_SPATIAL_DIMS,
            weight_pull=weight_pull,
            weight_push=weight_push,
            weight_norm=weight_norm,
            weight_edge=weight_edge,
            weight_bone=weight_bone,
            delta_v=delta_v,
            delta_d=delta_d,
        )

        self.geometry_loss: Optional[GeometryLoss] = (
            GeometryLoss(
                spatial_dims=_SPATIAL_DIMS,
                weight_dir=weight_dir,
                weight_cov=weight_cov,
                weight_raw=weight_raw,
                loss_dir=loss_dir,
                loss_cov=loss_cov,
                loss_raw=loss_raw,
                **geom_kwargs,
            )
            if weight_geometry > 0
            else None
        )

        self._cache_key: Optional[Tuple] = None
        self._cached_ins_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._cached_geom_targets: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    @staticmethod
    def _label_fingerprint(labels: torch.Tensor) -> Tuple:
        return (labels.shape, labels.data_ptr(), int(labels.sum().item()))

    def _get_cached_targets(
        self,
        labels: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Optional[Dict],
    ]:
        key = self._label_fingerprint(labels)
        if key != self._cache_key:
            self._cache_key = key
            self._cached_ins_weights = self.instance_loss.compute_weights(
                labels,
            )

            if (
                self.geometry_loss is not None
                and targets is not None
                and "label_direction" in targets
                and "label_covariance" in targets
            ):
                self._cached_geom_targets = (
                    self.geometry_loss.targets_from_pipeline(
                        targets["label_direction"],
                        targets["label_covariance"],
                    )
                )
            elif self.geometry_loss is not None:
                self._cached_geom_targets = (
                    self.geometry_loss.compute_targets(labels)
                )
            else:
                self._cached_geom_targets = None

        return self._cached_ins_weights, self._cached_geom_targets  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Feature-consistency loss
    # ------------------------------------------------------------------

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
        a_flat = rearrange(features_a, "b c ... -> b c (...)")
        b_flat = rearrange(features_b, "b c ... -> b c (...)")

        a_norm = a_flat / (reduce(a_flat ** 2, "b c n -> b 1 n", "sum").sqrt() + 1e-8)
        b_norm = b_flat / (reduce(b_flat ** 2, "b c n -> b 1 n", "sum").sqrt() + 1e-8)

        return reduce((a_norm - b_norm) ** 2, "b c n -> ", "mean")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute composite loss.

        Args:
            predictions: Dict with ``"semantic"``, ``"instance"``,
                ``"geometry"`` (optional), and ``"features_aug"`` (optional
                second-view features for consistency loss).
            targets: Dict with ``"labels"``, ``"semantic_labels"``, and
                optionally ``"semantic_ids"``, ``"raw_image"``,
                ``"label_direction"``, ``"label_covariance"``.

        Returns:
            Dict with ``"loss"`` (total) and per-component sub-losses.
        """
        labels = targets["labels"]
        (w_edge, w_bone), geom_targets = self._get_cached_targets(
            labels, targets,
        )

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
            "loss_ins": ins["loss"],
            "loss_ins/pull": ins["pull"],
            "loss_ins/push": ins["push"],
            "loss_ins/norm": ins["norm"],
        }
        if self.semantic_loss.weight_iou > 0:
            out["loss_sem/iou"] = sem["iou"]
        if self.semantic_loss.weight_dice > 0:
            out["loss_sem/dice"] = sem["dice"]

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

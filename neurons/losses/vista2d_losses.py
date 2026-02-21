"""
Vista2D losses for image-based segmentation.

Public classes:
- SemanticLoss:   CE + IoU + Dice on semantic logits
- InstanceLoss:   pull/push/norm discriminative on instance embeddings (2D)
- GeometryLoss:   dir/cov/raw regression (imported from discriminative.py)
- Vista2DLoss:    composes SemanticLoss + InstanceLoss + GeometryLoss
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from neurons.losses.discriminative import GeometryLoss

_SPATIAL_DIMS = 2
_POOL_FN = F.max_pool2d
_PAD_TUPLE = (1, 1, 1, 1)


# ======================================================================
# 1.  Semantic loss  (CE + IoU + Dice)
# ======================================================================

class SemanticLoss(nn.Module):
    """Semantic segmentation loss supporting both multi-label (sigmoid)
    and mutually-exclusive (softmax) modes.

    * **sigmoid** (default):  per-class binary cross-entropy.  Each class
      is an independent binary classifier — a pixel can belong to multiple
      classes simultaneously (e.g. neuron AND mitochondrion).
      Target: ``[B, C, *spatial]`` multi-hot float or ``[B, *spatial]`` int
      (auto-converted to one-hot).
    * **softmax**:  standard cross-entropy with softmax.  Classes are
      mutually exclusive — each pixel belongs to exactly one class.
      Target: ``[B, *spatial]`` integer class labels.

    Both modes support optional soft IoU and soft Dice auxiliary losses.

    Args:
        mode: ``"sigmoid"`` (multi-label, default) or ``"softmax"``(mutually exclusive).
        weight_ce: scalar weight for CE/BCE term (default 1.0).
        weight_iou: scalar weight for IoU term (default 0.0).
        weight_dice: scalar weight for Dice term (default 0.0).
        class_weights: per-class weights (optional).
        ignore_index: label value to ignore in softmax mode (default -100).
    """

    def __init__(
        self,
        mode: str = "sigmoid",
        weight_ce: float = 1.0,
        weight_iou: float = 0.0,
        weight_dice: float = 0.0,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if mode not in ("sigmoid", "softmax"):
            raise ValueError(f"mode must be 'sigmoid' or 'softmax', got '{mode}'")
        self.mode = mode
        self.weight_ce = weight_ce
        self.weight_iou = weight_iou
        self.weight_dice = weight_dice
        self.ignore_index = ignore_index

        cw = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        if mode == "softmax":
            self.ce_loss = nn.CrossEntropyLoss(weight=cw, ignore_index=ignore_index)
        else:
            self.ce_loss = nn.BCEWithLogitsLoss(
                pos_weight=cw, reduction="none",
            )

    def _to_probs_and_target(
        self,
        logits: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> tuple:
        """Convert logits + labels to (probs [B,C,*], target [B,C,*])."""
        C = logits.shape[1]

        if self.mode == "softmax":
            probs = F.softmax(logits, dim=1)
            valid = class_labels != self.ignore_index
            target_safe = class_labels.clone()
            target_safe[~valid] = 0
            one_hot = F.one_hot(target_safe.long(), C).float()
            one_hot = rearrange(one_hot, "b ... c -> b c ...")
            valid_mask = rearrange(valid.float(), "b ... -> b 1 ...")
            return probs * valid_mask, one_hot * valid_mask
        else:
            probs = torch.sigmoid(logits)
            if class_labels.dim() == logits.dim():
                target = class_labels.float()
            else:
                target_safe = class_labels.clone().long()
                target_safe[target_safe < 0] = 0
                target = F.one_hot(target_safe, C).float()
                target = rearrange(target, "b ... c -> b c ...")
            return probs, target

    def _compute_ce(
        self,
        logits: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "softmax":
            return self.ce_loss(logits, class_labels)
        else:
            _, target = self._to_probs_and_target(logits, class_labels)
            return self.ce_loss(logits, target).mean()

    def _iou_loss(
        self,
        logits: torch.Tensor,
        class_labels: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Soft IoU loss averaged over classes (1 - IoU)."""
        probs, target = self._to_probs_and_target(logits, class_labels)
        spatial = tuple(range(2, probs.dim()))
        intersection = (probs * target).sum(dim=spatial)
        union = probs.sum(dim=spatial) + target.sum(dim=spatial) - intersection
        iou = (intersection + eps) / (union + eps)
        return 1.0 - iou.mean()

    def _dice_loss(
        self,
        logits: torch.Tensor,
        class_labels: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Soft Dice loss averaged over classes (1 - Dice)."""
        probs, target = self._to_probs_and_target(logits, class_labels)
        spatial = tuple(range(2, probs.dim()))
        intersection = (probs * target).sum(dim=spatial)
        card_p = probs.sum(dim=spatial)
        card_g = target.sum(dim=spatial)
        dice = (2.0 * intersection + eps) / (card_p + card_g + eps)
        return 1.0 - dice.mean()

    def forward(
        self,
        logits: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [B, C, *spatial] semantic logits.
            class_labels: [B, C, *spatial] multi-hot (sigmoid mode) or
                [B, *spatial] integer labels (softmax mode).  Integer
                labels are auto-converted to one-hot in sigmoid mode.

        Returns:
            Dict with ``loss``, ``ce``, ``iou``, ``dice``.
        """
        loss_ce = self._compute_ce(logits, class_labels)
        loss_iou = (
            self._iou_loss(logits, class_labels)
            if self.weight_iou > 0
            else torch.tensor(0.0, device=logits.device)
        )
        loss_dice = (
            self._dice_loss(logits, class_labels)
            if self.weight_dice > 0
            else torch.tensor(0.0, device=logits.device)
        )
        loss = (
            self.weight_ce * loss_ce
            + self.weight_iou * loss_iou
            + self.weight_dice * loss_dice
        )
        return {"loss": loss, "ce": loss_ce, "iou": loss_iou, "dice": loss_dice}


# ======================================================================
# 2.  Instance loss  (pull / push / norm + boundary / skeleton weighting)
# ======================================================================

class InstanceLoss(nn.Module):
    """Weighted discriminative pull/push/norm loss on instance embeddings.

    Boundary and skeleton weighting boost gradients near edges and the
    medial axis respectively.

    Args:
        weight_pull: pull term weight.
        weight_push: push term weight.
        weight_norm: regularisation term weight.
        weight_edge: multiplicative boost at instance boundaries.
        weight_bone: multiplicative boost at skeleton / medial axis.
        delta_v: pull hinge margin.
        delta_d: push margin.
    """

    def __init__(
        self,
        weight_pull: float = 1.0,
        weight_push: float = 1.0,
        weight_norm: float = 0.001,
        weight_edge: float = 10.0,
        weight_bone: float = 10.0,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
    ) -> None:
        super().__init__()
        self.weight_pull = weight_pull
        self.weight_push = weight_push
        self.weight_norm = weight_norm
        self.weight_edge = weight_edge
        self.weight_bone = weight_bone
        self.delta_v = delta_v
        self.delta_d = delta_d

    # ---- weighting helpers ----

    def _get_weight_boundary(self, label: torch.Tensor) -> torch.Tensor:
        gt_label_float = rearrange(label, "b ... -> b 1 ...").float()
        padded_arr = F.pad(gt_label_float, _PAD_TUPLE, mode="replicate")
        pooled_max = _POOL_FN(+padded_arr, 3, stride=1, padding=0)
        pooled_min = _POOL_FN(-padded_arr, 3, stride=1, padding=0).neg_()
        boundary = rearrange(pooled_max != pooled_min, "b 1 ... -> b ...").float()
        return 1.0 + boundary * (self.weight_edge - 1.0)

    @torch.no_grad()
    def _get_weight_skeleton(self, label: torch.Tensor) -> torch.Tensor:
        from scipy.ndimage import distance_transform_edt as _scipy_edt

        weight_bone = torch.ones_like(label, dtype=torch.float32)
        label_np = label.cpu().numpy()

        for b in range(label.shape[0]):
            unique_ids = np.unique(label_np[b])
            for uid in unique_ids:
                if uid == 0:
                    continue
                mask = label_np[b] == uid
                dt = _scipy_edt(mask).astype(np.float32)
                max_d = dt.max()
                if max_d > 0:
                    dt /= max_d
                dt_t = torch.from_numpy(dt).to(label.device)
                inst_mask = label[b] == uid
                weight_bone[b][inst_mask] = 1.0 + dt_t[inst_mask] * (self.weight_bone - 1.0)

        return weight_bone

    # ---- core loss ----

    def _loss_single(
        self,
        embed: torch.Tensor,
        label: torch.Tensor,
        weight_edge: torch.Tensor,
        weight_bone: torch.Tensor,
    ) -> torch.Tensor:
        B = embed.shape[0]
        emb_flat = rearrange(embed, "b c ... -> b c (...)")
        lbl_flat = rearrange(label, "b ... -> b (...)")
        weight_flat = rearrange(weight_edge * weight_bone, "b ... -> b (...)")

        loss_pull = torch.tensor(0.0, device=embed.device)
        loss_push = torch.tensor(0.0, device=embed.device)
        loss_norm = torch.tensor(0.0, device=embed.device)
        valid = 0

        for b in range(B):
            ids = torch.unique(lbl_flat[b])
            ids = ids[ids > 0]
            if len(ids) == 0:
                continue
            valid += 1
            n_inst = len(ids)

            centers = []
            b_pull = torch.tensor(0.0, device=embed.device)
            for uid in ids:
                mask = lbl_flat[b] == uid
                w = weight_flat[b, mask]
                emb = emb_flat[b, :, mask]
                center = (emb * rearrange(w, "n -> 1 n")).sum(1) / (w.sum() + 1e-8)
                centers.append(center)

                dist = torch.norm(emb - rearrange(center, "e -> e 1"), dim=0)
                pull = F.relu(dist - self.delta_v) ** 2
                b_pull = b_pull + (pull * w).mean()

            loss_pull = loss_pull + b_pull / n_inst

            if len(centers) > 1:
                c = torch.stack(centers)
                ci = rearrange(c, "n e -> n 1 e")
                cj = rearrange(c, "n e -> 1 n e")
                pw = torch.norm(ci - cj, dim=2)
                triu = torch.triu_indices(len(ids), len(ids), offset=1, device=pw.device)
                push = F.relu(2 * self.delta_d - pw[triu[0], triu[1]]) ** 2
                loss_push = loss_push + push.mean()

            loss_norm = loss_norm + torch.stack([c.norm() for c in centers]).mean()

        n = max(valid, 1)
        l_pull = loss_pull / n
        l_push = loss_push / n
        l_norm = loss_norm / n
        total = self.weight_pull * l_pull + self.weight_push * l_push + self.weight_norm * l_norm
        return {"loss": total, "pull": l_pull, "push": l_push, "norm": l_norm}

    def forward(
        self,
        embed: torch.Tensor,
        label: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embed: [B, E, *spatial] instance embedding.
            label: [B, *spatial] instance labels (0 = background).
            class_ids: [B, *spatial] optional semantic class ids.

        Returns:
            Dict with ``loss``, ``pull``, ``push``, ``norm``.
        """
        weight_edge = self._get_weight_boundary(label) if self.weight_edge > 1.0 else torch.ones_like(label, dtype=torch.float32)
        weight_bone = self._get_weight_skeleton(label) if self.weight_bone > 1.0 else torch.ones_like(label, dtype=torch.float32)

        if class_ids is not None:
            unique_classes = torch.unique(class_ids)
            unique_classes = unique_classes[unique_classes > 0]
            if len(unique_classes) > 0:
                zero = torch.tensor(0.0, device=embed.device)
                accum = {"loss": zero.clone(), "pull": zero.clone(), "push": zero.clone(), "norm": zero.clone()}
                for cid in unique_classes:
                    class_mask = (class_ids == cid).long()
                    out = self._loss_single(embed, label * class_mask, weight_edge, weight_bone)
                    for k in accum:
                        accum[k] = accum[k] + out[k]
                nc = len(unique_classes)
                return {k: v / nc for k, v in accum.items()}

        return self._loss_single(embed, label, weight_edge, weight_bone)


# ======================================================================
# 3.  Combined loss  (composes semantic + instance + geometry)
# ======================================================================

class Vista2DLoss(nn.Module):
    """Combined loss for the Vista2D 3-head architecture.

    Composes:
    - ``SemanticLoss``:  CE + Dice on semantic logits.
    - ``InstanceLoss``:  pull/push/norm on instance embeddings.
    - ``GeometryLoss``:  dir/cov/raw on geometry head (optional).

    Args:
        weight_semantic: top-level weight for the semantic branch (default 1.0).
        weight_instance: top-level weight for the instance branch (default 1.0).
        weight_geometry: top-level weight for the geometry branch (default 0.0 = off).
        semantic_mode: ``"sigmoid"`` (multi-label, default) or ``"softmax"``(mutually exclusive).  Forwarded to ``SemanticLoss``.
        weight_pull, weight_push, weight_norm, weight_edge, weight_bone,
        delta_v, delta_d: forwarded to ``InstanceLoss``.
        weight_ce, weight_iou, weight_dice, class_weights, ignore_index: forwarded to ``SemanticLoss``.
        geom_kwargs: extra kwargs forwarded to ``GeometryLoss``.
    """

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
        )
        self.instance_loss = InstanceLoss(
            weight_pull=weight_pull,
            weight_push=weight_push,
            weight_norm=weight_norm,
            weight_edge=weight_edge,
            weight_bone=weight_bone,
            delta_v=delta_v,
            delta_d=delta_d,
        )
        self.geometry_loss = GeometryLoss(
            spatial_dims=_SPATIAL_DIMS, **geom_kwargs,
        ) if weight_geometry > 0 else None

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Dict with ``semantic``, ``instance``,
                and optionally ``geometry`` tensors.
            targets: Dict with ``semantic_labels`` and ``labels``.

        Returns:
            Dict with hierarchical keys: ``loss``, ``loss_sem``,
            ``loss_sem/ce``, ``loss_sem/iou``, ``loss_sem/dice``,
            ``loss_ins``, ``loss_ins/pull``, ``loss_ins/push``,
            ``loss_ins/norm``, and optionally ``loss_geom``,
            ``loss_geom/dir``, ``loss_geom/cov``, ``loss_geom/raw``.
        """
        sem_out = self.semantic_loss(
            predictions["semantic"], targets["semantic_labels"],
        )

        class_ids = targets.get("semantic_ids") or predictions.get("semantic_ids")
        ins_out = self.instance_loss(
            predictions["instance"], targets["labels"], class_ids,
        )

        loss_sem = sem_out["loss"]
        loss_ins = ins_out["loss"]
        total = self.weight_semantic * loss_sem + self.weight_instance * loss_ins

        out: Dict[str, torch.Tensor] = {
            "loss_sem":       loss_sem,
            "loss_sem/ce":    sem_out["ce"],
            "loss_sem/iou":   sem_out["iou"],
            "loss_sem/dice":  sem_out["dice"],
            "loss_ins":       loss_ins,
            "loss_ins/pull":  ins_out["pull"],
            "loss_ins/push":  ins_out["push"],
            "loss_ins/norm":  ins_out["norm"],
        }

        if self.geometry_loss is not None and "geometry" in predictions:
            geom_out = self.geometry_loss(
                predictions["geometry"], targets["labels"],
                raw_image=targets.get("raw_image"),
            )
            loss_geom = geom_out["loss"]
            total = total + self.weight_geometry * loss_geom
            out["loss_geom"]      = loss_geom
            out["loss_geom/dir"]  = geom_out["dir"]
            out["loss_geom/cov"]  = geom_out["cov"]
            out["loss_geom/raw"]  = geom_out["raw"]

        out["loss"] = total
        return out

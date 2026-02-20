"""
Vista3D losses for volumetric segmentation.

Public classes:
- SemanticLoss:   CE + Dice on semantic logits
- InstanceLoss:   pull/push/norm discriminative on instance embeddings (3D)
- GeometryLoss:   dir/cov/raw regression (imported from discriminative.py)
- Vista3DLoss:    composes SemanticLoss + InstanceLoss + GeometryLoss
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from neurons.losses.discriminative import GeometryLoss

_SPATIAL_DIMS = 3
_POOL_FN = F.max_pool3d
_PAD_TUPLE = (1, 1, 1, 1, 1, 1)


# ======================================================================
# 1.  Semantic loss  (CE + Dice)
# ======================================================================

class SemanticLoss(nn.Module):
    """Cross-entropy + optional Dice loss on semantic logits.

    Args:
        weight_ce: scalar weight for CE term (default 1.0).
        weight_dice: scalar weight for Dice term (default 0.0).
        class_weights: per-class CE weights (optional).
        ignore_index: label value to ignore (default -100).
    """

    def __init__(
        self,
        weight_ce: float = 1.0,
        weight_dice: float = 0.0,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ignore_index = ignore_index

        cw = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce_loss = nn.CrossEntropyLoss(weight=cw, ignore_index=ignore_index)

    @staticmethod
    def _dice_loss(
        logits: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int = -100,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Soft Dice loss averaged over classes (1 - Dice)."""
        C = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        valid = target != ignore_index
        target_safe = target.clone()
        target_safe[~valid] = 0
        one_hot = F.one_hot(target_safe.long(), C).float()
        one_hot = rearrange(one_hot, "b ... c -> b c ...")
        valid_mask = rearrange(valid.float(), "b ... -> b 1 ...")

        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        intersection = (probs * one_hot).sum(dim=tuple(range(2, probs.dim())))
        card_p = probs.sum(dim=tuple(range(2, probs.dim())))
        card_g = one_hot.sum(dim=tuple(range(2, one_hot.dim())))

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
            class_labels: [B, *spatial] integer class labels.

        Returns:
            Dict with ``loss``, ``loss_ce``, ``loss_dice``.
        """
        loss_ce = self.ce_loss(logits, class_labels)
        loss_dice = (
            self._dice_loss(logits, class_labels, self.ignore_index)
            if self.weight_dice > 0
            else torch.tensor(0.0, device=logits.device)
        )
        loss = self.weight_ce * loss_ce + self.weight_dice * loss_dice
        return {"loss": loss, "loss_ce": loss_ce, "loss_dice": loss_dice}


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

            centers = []
            for uid in ids:
                mask = lbl_flat[b] == uid
                w = weight_flat[b, mask]
                emb = emb_flat[b, :, mask]
                center = (emb * rearrange(w, "n -> 1 n")).sum(1) / (w.sum() + 1e-8)
                centers.append(center)

                dist = torch.norm(emb - rearrange(center, "e -> e 1"), dim=0)
                pull = F.relu(dist - self.delta_v) ** 2
                loss_pull = loss_pull + (pull * w).mean()

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
        return (
            self.weight_pull * loss_pull / n
            + self.weight_push * loss_push / n
            + self.weight_norm * loss_norm / n
        )

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
            Dict with ``loss``.
        """
        weight_edge = self._get_weight_boundary(label) if self.weight_edge > 1.0 else torch.ones_like(label, dtype=torch.float32)
        weight_bone = self._get_weight_skeleton(label) if self.weight_bone > 1.0 else torch.ones_like(label, dtype=torch.float32)

        if class_ids is not None:
            unique_classes = torch.unique(class_ids)
            unique_classes = unique_classes[unique_classes > 0]
            if len(unique_classes) > 0:
                total = torch.tensor(0.0, device=embed.device)
                for cid in unique_classes:
                    class_mask = (class_ids == cid).long()
                    total = total + self._loss_single(embed, label * class_mask, weight_edge, weight_bone)
                return {"loss": total / len(unique_classes)}

        return {"loss": self._loss_single(embed, label, weight_edge, weight_bone)}


# ======================================================================
# 3.  Combined loss  (composes semantic + instance + geometry)
# ======================================================================

class Vista3DLoss(nn.Module):
    """Combined loss for the Vista3D 3-head architecture.

    Composes:
    - ``SemanticLoss``:  CE + Dice on semantic logits.
    - ``InstanceLoss``:  pull/push/norm on instance embeddings.
    - ``GeometryLoss``:  dir/cov/raw on geometry head (optional).

    Args:
        weight_semantic: top-level weight for the semantic branch (default 1.0).
        weight_instance: top-level weight for the instance branch (default 1.0).
        weight_geometry: top-level weight for the geometry branch (default 0.0 = off).
        weight_pull, weight_push, weight_norm, weight_edge, weight_bone,
        delta_v, delta_d: forwarded to ``InstanceLoss``.
        weight_ce, weight_dice, class_weights, ignore_index:
            forwarded to ``SemanticLoss``.
        geom_kwargs: extra kwargs forwarded to ``GeometryLoss``.
    """

    def __init__(
        self,
        weight_semantic: float = 1.0,
        weight_instance: float = 1.0,
        weight_geometry: float = 0.0,
        weight_pull: float = 1.0,
        weight_push: float = 1.0,
        weight_norm: float = 0.001,
        weight_edge: float = 10.0,
        weight_bone: float = 10.0,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
        weight_ce: float = 1.0,
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
            weight_ce=weight_ce,
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
            targets: Dict with ``class_labels`` and ``labels``.

        Returns:
            Dict with ``loss``, ``loss_sem``, ``loss_ce``, ``loss_dice``,
            ``loss_ins``, and optionally ``loss_geom``.
        """
        sem_out = self.semantic_loss(
            predictions["semantic"], targets["class_labels"],
        )

        class_ids = targets.get("class_ids") or predictions.get("class_ids")
        ins_out = self.instance_loss(
            predictions["instance"], targets["labels"], class_ids,
        )

        loss_sem = sem_out["loss"]
        loss_ins = ins_out["loss"]
        total = self.weight_semantic * loss_sem + self.weight_instance * loss_ins

        out: Dict[str, torch.Tensor] = {
            "loss_sem": loss_sem,
            "loss_ce": sem_out["loss_ce"],
            "loss_dice": sem_out["loss_dice"],
            "loss_ins": loss_ins,
        }

        if self.geometry_loss is not None and "geometry" in predictions:
            geom_out = self.geometry_loss(
                predictions["geometry"], targets["labels"],
                raw_image=targets.get("raw_image"),
            )
            loss_geom = geom_out["loss"]
            total = total + self.weight_geometry * loss_geom
            out["loss_geom"] = loss_geom

        out["loss"] = total
        return out

"""
Vista3D combined loss for volumetric segmentation.

Computes semantic (CE + Dice) and instance (pull/push discriminative) losses
for the 2-head Vista3D model.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from scipy.ndimage import distance_transform_edt as _scipy_edt

_SPATIAL_DIMS = 3
_POOL_FN = F.max_pool3d
_PAD_TUPLE = (1, 1, 1, 1, 1, 1)
_REDUCE_PATTERN = "b ... -> b " + " ".join(["1"] * _SPATIAL_DIMS)


class Vista3DLoss(nn.Module):
    """
    Combined loss for the Vista3D 2-head architecture.

    Semantic branch:  ``ce_weight * CE  +  dice_weight * (1 - Dice)``
    Instance branch:  weighted pull/push/norm discriminative loss.

    Args:
        weight_pull: Pull (variance) weight for instance embed.
        weight_push: Push (distance) weight for instance embed.
        weight_norm: Regularisation weight for instance embed.
        weight_edge: Extra weight factor on boundary voxels.
        weight_bone: Extra weight factor on skeleton (medial axis) voxels.
        delta_v: Margin for the pull term.
        delta_d: Margin for the push term.
        ce_weight: Weight for CE loss in the semantic branch.
        dice_weight: Weight for Dice loss in the semantic branch.
        class_weights: Per-class weights for CE loss (length = num_classes).
        ignore_index: Label value to ignore in CE loss.
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
        ce_weight: float = 1.0,
        dice_weight: float = 0.0,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.weight_pull = weight_pull
        self.weight_push = weight_push
        self.weight_norm = weight_norm
        self.weight_edge = weight_edge
        self.weight_bone = weight_bone
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

        cw = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce_loss = nn.CrossEntropyLoss(weight=cw, ignore_index=ignore_index)

    # ------------------------------------------------------------------
    # Semantic helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Weighting helpers
    # ------------------------------------------------------------------
    def _get_weight_boundary(self, label: torch.Tensor) -> torch.Tensor:
        """Boundary weight map via morphological gradient."""
        gt_label_float = rearrange(label, "b ... -> b 1 ...").float()
        padded_arr = F.pad(gt_label_float, _PAD_TUPLE, mode="replicate")
        pooled_max = _POOL_FN(+padded_arr, 3, stride=1, padding=0)
        pooled_min = _POOL_FN(-padded_arr, 3, stride=1, padding=0).neg_()
        boundary = rearrange(pooled_max != pooled_min, "b 1 ... -> b ...").float()
        return 1.0 + boundary * (self.weight_edge - 1.0)

    @torch.no_grad()
    def _get_weight_skeleton(self, label: torch.Tensor) -> torch.Tensor:
        """Skeleton (medial axis) weight map via scipy EDT."""
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

    # ------------------------------------------------------------------
    # Instance loss (pull / push / norm)
    # ------------------------------------------------------------------
    def _instance_loss(
        self,
        embed: torch.Tensor,
        label: torch.Tensor,
        w_edge: torch.Tensor,
        w_bone: torch.Tensor,
    ) -> torch.Tensor:
        """Discriminative instance loss on flattened spatial dims."""
        B = embed.shape[0]
        emb_flat = rearrange(embed, "b c ... -> b c (...)")
        lbl_flat = rearrange(label, "b ... -> b (...)")
        w_combined = w_edge * w_bone
        w_flat = rearrange(w_combined, "b ... -> b (...)")

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
                w = w_flat[b, mask]
                emb = emb_flat[b, :, mask]
                center = (emb * w.unsqueeze(0)).sum(1) / (w.sum() + 1e-8)
                centers.append(center)

                dist = torch.norm(emb - center.unsqueeze(1), dim=0)
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            predictions: Dict with 'semantic' and 'instance' tensors.
            targets: Dict with 'class_labels' and 'labels'.

        Returns:
            Dict with 'loss', 'loss_sem', 'loss_ce', 'loss_dice', 'loss_ins'.
        """
        sem_logits = predictions["semantic"]
        class_labels = targets["class_labels"]

        loss_ce = self.ce_loss(sem_logits, class_labels)
        loss_dice = (
            self._dice_loss(sem_logits, class_labels, self.ignore_index)
            if self.dice_weight > 0
            else torch.tensor(0.0, device=sem_logits.device)
        )
        loss_sem = self.ce_weight * loss_ce + self.dice_weight * loss_dice

        labels = targets["labels"]
        w_edge = self._get_weight_boundary(labels) if self.weight_edge > 1.0 else torch.ones_like(labels, dtype=torch.float32)
        w_bone = self._get_weight_skeleton(labels) if self.weight_bone > 1.0 else torch.ones_like(labels, dtype=torch.float32)
        loss_ins = self._instance_loss(predictions["instance"], labels, w_edge, w_bone)

        total = loss_sem + loss_ins

        return {
            "loss": total,
            "loss_sem": loss_sem,
            "loss_ce": loss_ce,
            "loss_dice": loss_dice,
            "loss_ins": loss_ins,
        }

"""
Vista2D combined loss for image-based segmentation.

Computes semantic (CE), geometry (L1 on diff/grid/rgba), and
instance (pull/push discriminative) losses for the 3-head Vista2D model.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

try:
    from kornia.contrib import distance_transform
    _HAS_KORNIA = True
except ImportError:
    _HAS_KORNIA = False

_SPATIAL_DIMS = 2
_POOL_FN = F.max_pool2d
_PAD_TUPLE = (1, 1, 1, 1)
_REDUCE_PATTERN = "b ... -> b " + " ".join(["1"] * _SPATIAL_DIMS)


class Vista2DLoss(nn.Module):
    """
    Combined loss for the Vista2D 3-head architecture.

    Heads and their target slicing convention (16 geometry channels):
        channels  0--8  : gt_diff  (local displacement field, 9ch)
        channels  9--11 : gt_grid  (normalised spatial coordinates, 3ch)
        channels 12--15 : gt_rgba  (auxiliary colour/intensity, 4ch)

    Args:
        weight_pull: Pull (variance) weight for instance embed.
        weight_push: Push (distance) weight for instance embed.
        weight_norm: Regularisation weight for instance embed.
        weight_edge: Extra weight factor on boundary pixels.
        weight_bone: Extra weight factor on skeleton (medial axis) pixels.
        delta_v: Margin for the pull term.
        delta_d: Margin for the push term.
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
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()

    # ------------------------------------------------------------------
    # Weighting helpers
    # ------------------------------------------------------------------
    def _get_weight_boundary(self, label: torch.Tensor) -> torch.Tensor:
        """Compute boundary weight map via morphological gradient."""
        gt_label_float = rearrange(label, "b ... -> b 1 ...").float()
        padded_arr = F.pad(gt_label_float, _PAD_TUPLE, mode="replicate")
        pooled_max = _POOL_FN(+padded_arr, 3, stride=1, padding=0)
        pooled_min = _POOL_FN(-padded_arr, 3, stride=1, padding=0).neg_()
        boundary = rearrange(pooled_max != pooled_min, "b 1 ... -> b ...").float()
        return 1.0 + boundary * (self.weight_edge - 1.0)

    def _get_weight_skeleton(self, label: torch.Tensor) -> torch.Tensor:
        """Compute skeleton (medial axis) weight map via distance transform."""
        if not _HAS_KORNIA:
            return torch.ones_like(label, dtype=torch.float32)

        weight_bone = torch.ones_like(label, dtype=torch.float32)
        unique_ids = torch.unique(label)
        for uid in unique_ids:
            if uid == 0:
                continue
            inst_mask = rearrange(label == uid, "b ... -> b 1 ...").float()
            dt = distance_transform(inst_mask)
            dt = rearrange(dt, "b 1 ... -> b ...")
            max_d = reduce(dt, _REDUCE_PATTERN, "max")
            dt_norm = torch.where(max_d > 0, dt / max_d, dt)
            weight_bone = torch.where(
                label == uid,
                1.0 + dt_norm * (self.weight_bone - 1.0),
                weight_bone,
            )
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
            predictions: Dict with 'semantic', 'instance', 'geometry' tensors.
            targets: Dict with 'class_labels', 'labels', 'gt_diff', 'gt_grid', 'gt_rgba'.

        Returns:
            Dict with 'loss', 'loss_sem', 'loss_aff', 'loss_ins'.
        """
        loss_sem = self.ce_loss(predictions["semantic"], targets["class_labels"])

        aff = predictions["geometry"]
        loss_aff = (
            self.l1_loss(aff[:, 0:9], targets["gt_diff"])
            + self.l1_loss(aff[:, 9:12], targets["gt_grid"])
            + self.l1_loss(aff[:, 12:16], targets["gt_rgba"])
        )

        labels = targets["labels"]
        w_edge = self._get_weight_boundary(labels) if self.weight_edge > 1.0 else torch.ones_like(labels, dtype=torch.float32)
        w_bone = self._get_weight_skeleton(labels) if self.weight_bone > 1.0 else torch.ones_like(labels, dtype=torch.float32)
        loss_ins = self._instance_loss(predictions["instance"], labels, w_edge, w_bone)

        total = loss_sem + loss_aff + loss_ins

        return {
            "loss": total,
            "loss_sem": loss_sem,
            "loss_aff": loss_aff,
            "loss_ins": loss_ins,
        }

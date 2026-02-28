"""
Vista3D losses for volumetric segmentation.

Public classes:
- SemanticLoss:   CE + IoU + Dice on semantic logits
- InstanceLoss:   pull/push/norm discriminative on instance embeddings (3D)
- GeometryLoss:   dir/cov/raw regression (imported from discriminative.py)
- Vista3DLoss:    composes SemanticLoss + InstanceLoss + GeometryLoss
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from neurons.losses.discriminative import GeometryLoss
from neurons.utils.parallel import pmap

_SPATIAL_DIMS = 3
_POOL_FN = F.max_pool3d
_PAD_TUPLE = (1, 1, 1, 1, 1, 1)


# -----------------------------------------------------------------------
# EDT workers  (CPU subprocess via pmap — must use scipy directly)
# -----------------------------------------------------------------------

def _edt_worker(args):
    """Per-instance EDT for pmap subprocesses (CPU/scipy)."""
    from scipy.ndimage import distance_transform_edt
    label_np_b, uid = args
    mask = label_np_b == uid
    dt = distance_transform_edt(mask).astype(np.float32)
    max_d = dt.max()
    if max_d > 0:
        dt /= max_d
    return (uid, dt)


# ======================================================================
# 1.  Semantic loss  (CE + IoU + Dice)
# ======================================================================

class SemanticLoss(nn.Module):
    """Semantic segmentation loss: sigmoid (multi-label) or softmax (exclusive).

    loss = w_ce * CE  +  w_iou * (1 - SoftIoU)  +  w_dice * (1 - SoftDice)
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
            self.ce_loss = nn.BCEWithLogitsLoss(pos_weight=cw, reduction="none")

    # ---- internal helpers ----

    def _to_probs_and_target(self, logits, class_labels):
        """Convert logits + labels → (probs [B,C,*], target [B,C,*])."""
        C = logits.shape[1]

        if self.mode == "softmax":
            probs = F.softmax(logits, dim=1)
            valid = class_labels != self.ignore_index
            safe = class_labels.clone()
            safe[~valid] = 0
            one_hot = rearrange(
                F.one_hot(safe.long(), C).float(),
                "b ... c -> b c ...",
            )
            valid_mask = rearrange(valid.float(), "b ... -> b 1 ...")
            return probs * valid_mask, one_hot * valid_mask

        # sigmoid mode
        probs = torch.sigmoid(logits)
        if class_labels.dim() == logits.dim():
            return probs, class_labels.float()

        safe = class_labels.clone().long()
        neg = safe < 0
        safe[neg] = 0
        safe = safe.clamp(0, C - 1)
        target = rearrange(
            F.one_hot(safe, C).float(),
            "b ... c -> b c ...",
        )                                                          # [B, C, *spatial]
        neg_mask = rearrange(neg, "b ... -> b 1 ...")              # broadcast across C
        target[neg_mask.expand_as(target)] = 0.0
        return probs, target

    def _compute_ce(self, logits, class_labels):
        if self.mode == "softmax":
            return self.ce_loss(logits, class_labels)
        _, target = self._to_probs_and_target(logits, class_labels)
        return self.ce_loss(logits, target).mean()

    def _iou_loss(self, logits, class_labels, eps=1e-5):
        """1 − mean(IoU) over classes."""
        probs, target = self._to_probs_and_target(logits, class_labels)
        spatial = tuple(range(2, probs.dim()))
        inter = (probs * target).sum(dim=spatial)
        union = probs.sum(dim=spatial) + target.sum(dim=spatial) - inter
        return 1.0 - ((inter + eps) / (union + eps)).mean()

    def _dice_loss(self, logits, class_labels, eps=1e-5):
        """1 − mean(Dice) over classes."""
        probs, target = self._to_probs_and_target(logits, class_labels)
        spatial = tuple(range(2, probs.dim()))
        inter = (probs * target).sum(dim=spatial)
        card = probs.sum(dim=spatial) + target.sum(dim=spatial)
        return 1.0 - ((2.0 * inter + eps) / (card + eps)).mean()

    # ---- forward ----

    def forward(self, logits, class_labels) -> Dict[str, torch.Tensor]:
        ce = self._compute_ce(logits, class_labels)
        iou = self._iou_loss(logits, class_labels) if self.weight_iou > 0 else torch.tensor(0.0, device=logits.device)
        dice = self._dice_loss(logits, class_labels) if self.weight_dice > 0 else torch.tensor(0.0, device=logits.device)
        loss = self.weight_ce * ce + self.weight_iou * iou + self.weight_dice * dice
        return {"loss": loss, "ce": ce, "iou": iou, "dice": dice}


# ======================================================================
# 2.  Instance loss  (pull / push / norm + boundary / skeleton weighting)
# ======================================================================

class InstanceLoss(nn.Module):
    """Weighted discriminative pull/push/norm on instance embeddings.

    Boundary pixels and medial-axis pixels receive boosted weights so
    the model pays extra attention to separating touching instances and
    reconstructing the skeleton.
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

    @torch.no_grad()
    def _get_weight_boundary(self, label: torch.Tensor) -> torch.Tensor:
        """Boundary weight via morphological gradient (max_pool ≠ min_pool)."""
        label_4d = rearrange(label, "b ... -> b 1 ...")            # [B,1,D,H,W]
        padded = F.pad(label_4d.float(), _PAD_TUPLE, mode="replicate")
        dilated = _POOL_FN(+padded, 3, stride=1, padding=0)
        eroded = _POOL_FN(-padded, 3, stride=1, padding=0).neg_()
        boundary = rearrange(dilated != eroded, "b 1 ... -> b ...").float()
        return 1.0 + boundary * (self.weight_edge - 1.0)

    @torch.no_grad()
    def _get_weight_skeleton(self, label: torch.Tensor) -> torch.Tensor:
        """Per-instance EDT skeleton weight.

        GPU path:  DLPack zero-copy torch→cupy, all EDTs in cupy, zero-copy back.
        CPU path:  all instances via pmap (always parallel).
        """
        from neurons.utils.gpu_ndimage import _use_gpu as _cupy_ok

        B = label.shape[0]

        if _cupy_ok():
            import cupy as cp
            from cupyx.scipy.ndimage import distance_transform_edt as cp_edt
            from neurons.utils.gpu_ndimage import torch_to_cupy, cupy_to_torch

            label_cp = torch_to_cupy(label)                        # zero-copy GPU
            weight_cp = cp.ones(label_cp.shape, dtype=cp.float32)

            for b in range(B):
                fg_ids = cp.unique(label_cp[b])
                fg_ids = fg_ids[fg_ids > 0]
                for uid in fg_ids:
                    mask = label_cp[b] == uid
                    dt = cp_edt(mask).astype(cp.float32)
                    dt_max = dt.max()
                    if float(dt_max) > 0:
                        dt = dt / dt_max
                    weight_cp[b][mask] = 1.0 + dt[mask] * (self.weight_bone - 1.0)

            return cupy_to_torch(weight_cp, device=label.device).float()

        # CPU fallback — parallel via pmap
        label_np = label.cpu().numpy()
        weight_np = np.ones_like(label_np, dtype=np.float32)
        for b in range(B):
            fg_ids = np.unique(label_np[b])
            fg_ids = fg_ids[fg_ids > 0]
            if len(fg_ids) == 0:
                continue
            results = pmap(_edt_worker, [(label_np[b], int(u)) for u in fg_ids])
            for uid, dt in results:
                m = label_np[b] == uid
                weight_np[b][m] = 1.0 + dt[m] * (self.weight_bone - 1.0)

        return torch.from_numpy(weight_np).to(label.device)

    # ---- core discriminative loss ----

    def _loss_single(self, embed, label, w_edge, w_bone) -> Dict[str, torch.Tensor]:
        """Pull/push/norm over all instances in the batch.

        Shapes
        ------
        embed  : [B, E, *spatial]   instance embeddings
        label  : [B, *spatial]      instance ids (0 = background)
        w_edge : [B, *spatial]      boundary boost
        w_bone : [B, *spatial]      skeleton boost
        """
        emb_flat = rearrange(embed, "b e ... -> b e (...)")        # [B, E, N]
        lbl_flat = rearrange(label, "b ... -> b (...)")            # [B, N]
        wgt_flat = rearrange(w_edge * w_bone, "b ... -> b (...)")  # [B, N]

        dev = embed.device
        loss_pull = torch.tensor(0.0, device=dev)
        loss_push = torch.tensor(0.0, device=dev)
        loss_norm = torch.tensor(0.0, device=dev)
        n_valid = 0

        for b in range(embed.shape[0]):
            ids = torch.unique(lbl_flat[b])
            ids = ids[ids > 0]
            if len(ids) == 0:
                continue
            n_valid += 1
            K = len(ids)

            # --- weighted centroids + pull ---
            centers = []
            b_pull = torch.tensor(0.0, device=dev)
            for uid in ids:
                mask = lbl_flat[b] == uid                          # [N]
                w = wgt_flat[b, mask]                              # [M]
                e = emb_flat[b, :, mask]                           # [E, M]
                c = (e * rearrange(w, "m -> 1 m")).sum(1) / (w.sum() + 1e-8)  # [E]
                centers.append(c)
                dist = torch.norm(e - rearrange(c, "e -> e 1"), dim=0)
                b_pull = b_pull + (F.relu(dist - self.delta_v) ** 2 * w).mean()
            loss_pull = loss_pull + b_pull / K

            # --- push (pairwise centroid margin) ---
            if K > 1:
                c_stack = torch.stack(centers)                     # [K, E]
                pw = torch.norm(
                    rearrange(c_stack, "i e -> i 1 e") -
                    rearrange(c_stack, "j e -> 1 j e"),
                    dim=2,
                )                                                  # [K, K]
                triu = torch.triu_indices(K, K, offset=1, device=dev)
                loss_push = loss_push + reduce(
                    F.relu(2 * self.delta_d - pw[triu[0], triu[1]]) ** 2,
                    "n -> ", "mean",
                )

            # --- norm (centroid regularisation) ---
            loss_norm = loss_norm + torch.stack([c.norm() for c in centers]).mean()

        n = max(n_valid, 1)
        pull = loss_pull / n
        push = loss_push / n
        norm = loss_norm / n
        total = self.weight_pull * pull + self.weight_push * push + self.weight_norm * norm
        return {"loss": total, "pull": pull, "push": push, "norm": norm}

    # ---- public interface ----

    def compute_weights(self, label):
        """Pre-compute boundary + skeleton weights (cache-friendly)."""
        w_edge = self._get_weight_boundary(label) if self.weight_edge > 1.0 else torch.ones_like(label, dtype=torch.float32)
        w_bone = self._get_weight_skeleton(label) if self.weight_bone > 1.0 else torch.ones_like(label, dtype=torch.float32)
        return w_edge, w_bone

    def forward(self, embed, label, semantic_ids=None,
                weight_edge=None, weight_bone=None) -> Dict[str, torch.Tensor]:
        if weight_edge is None or weight_bone is None:
            weight_edge, weight_bone = self.compute_weights(label)

        if semantic_ids is not None:
            classes = torch.unique(semantic_ids)
            classes = classes[classes > 0]
            if len(classes) > 0:
                zero = torch.tensor(0.0, device=embed.device)
                acc = {k: zero.clone() for k in ("loss", "pull", "push", "norm")}
                for cid in classes:
                    out = self._loss_single(
                        embed, label * (semantic_ids == cid).long(),
                        weight_edge, weight_bone,
                    )
                    for k in acc:
                        acc[k] = acc[k] + out[k]
                return {k: v / len(classes) for k, v in acc.items()}

        return self._loss_single(embed, label, weight_edge, weight_bone)


# ======================================================================
# 3.  Combined loss  (semantic + instance + geometry)
# ======================================================================

class Vista3DLoss(nn.Module):
    """Compose SemanticLoss + InstanceLoss + GeometryLoss for Vista3D."""

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
        )
        self.instance_loss = InstanceLoss(
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

    def _get_cached_targets(self, labels):
        key = self._label_fingerprint(labels)
        if key != self._cache_key:
            self._cache_key = key
            self._cached_ins_weights = self.instance_loss.compute_weights(labels)
            self._cached_geom_targets = (
                self.geometry_loss.compute_targets(labels)
                if self.geometry_loss is not None else None
            )
        return self._cached_ins_weights, self._cached_geom_targets

    def forward(self, predictions, targets) -> Dict[str, torch.Tensor]:
        labels = targets["labels"]
        (w_edge, w_bone), geom_targets = self._get_cached_targets(labels)

        sem = self.semantic_loss(predictions["semantic"], targets["semantic_labels"])
        ins = self.instance_loss(
            predictions["instance"], labels,
            targets.get("semantic_ids") or predictions.get("semantic_ids"),
            weight_edge=w_edge, weight_bone=w_bone,
        )

        total = self.weight_semantic * sem["loss"] + self.weight_instance * ins["loss"]

        out: Dict[str, torch.Tensor] = {
            "loss_sem":       sem["loss"],
            "loss_sem/ce":    sem["ce"],
            "loss_sem/iou":   sem["iou"],
            "loss_sem/dice":  sem["dice"],
            "loss_ins":       ins["loss"],
            "loss_ins/pull":  ins["pull"],
            "loss_ins/push":  ins["push"],
            "loss_ins/norm":  ins["norm"],
        }

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

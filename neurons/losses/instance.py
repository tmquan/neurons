"""
Instance segmentation loss: pull / push / norm.

Dimension-agnostic — parameterized by ``spatial_dims`` to handle both
2-D (H, W) and 3-D (D, H, W) inputs with the appropriate pool function.

Boundary and skeleton pixels receive boosted weights so the model
pays extra attention to separating touching instances and
reconstructing the medial axis.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from neurons.utils.parallel import pmap


def _edt_worker(args):
    """Per-instance normalised EDT for pmap subprocesses (CPU/scipy)."""
    from scipy.ndimage import distance_transform_edt
    label_np_b, uid = args
    mask = label_np_b == uid
    dt = distance_transform_edt(mask).astype(np.float32)
    max_d = dt.max()
    if max_d > 0:
        dt /= max_d
    return (uid, dt)


def _pool_fn(spatial_dims: int):
    return F.max_pool3d if spatial_dims == 3 else F.max_pool2d


def _pad_tuple(spatial_dims: int):
    return (1, 1, 1, 1, 1, 1) if spatial_dims == 3 else (1, 1, 1, 1)


class InstanceLoss(nn.Module):
    """Weighted discriminative pull/push/norm on instance embeddings.

    Args:
        spatial_dims: 2 for images, 3 for volumes.
        weight_pull: Weight for the pull (intra-cluster) term.
        weight_push: Weight for the push (inter-cluster) term.
        weight_norm: Weight for the centroid norm regularisation term.
        weight_edge: Boundary pixel weight multiplier (1.0 = disabled).
        weight_bone: Medial-axis pixel weight multiplier (1.0 = disabled).
        delta_v: Pull margin (hinge threshold per embedding).
        delta_d: Push margin (half of the minimum centroid separation).
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        weight_pull: float = 1.0,
        weight_push: float = 1.0,
        weight_norm: float = 0.001,
        weight_edge: float = 10.0,
        weight_bone: float = 10.0,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.weight_pull = weight_pull
        self.weight_push = weight_push
        self.weight_norm = weight_norm
        self.weight_edge = weight_edge
        self.weight_bone = weight_bone
        self.delta_v = delta_v
        self.delta_d = delta_d

        self._pool = _pool_fn(spatial_dims)
        self._pad = _pad_tuple(spatial_dims)

    # ------------------------------------------------------------------
    # Weighting helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_weight_boundary(self, label: torch.Tensor) -> torch.Tensor:
        """Boundary weight via morphological gradient (max_pool != min_pool)."""
        label_4d = rearrange(label, "b ... -> b 1 ...")
        padded = F.pad(label_4d.float(), self._pad, mode="replicate")
        dilated = self._pool(+padded, 3, stride=1, padding=0)
        eroded = self._pool(-padded, 3, stride=1, padding=0).neg_()
        boundary = rearrange(dilated != eroded, "b 1 ... -> b ...").float()
        return 1.0 + boundary * (self.weight_edge - 1.0)

    @torch.no_grad()
    def _get_weight_skeleton(self, label: torch.Tensor) -> torch.Tensor:
        """Per-instance EDT skeleton weight.

        GPU path (cupy): DLPack zero-copy torch->cupy, exact L2 EDT.
        GPU path (torch): approximate L-inf EDT via morphological erosion.
        CPU path: all instances across batch via single pmap call.
        """
        from neurons.utils.gpu_ndimage import _use_gpu as _cupy_ok

        B = label.shape[0]

        if _cupy_ok():
            import cupy as cp
            from cupyx.scipy.ndimage import distance_transform_edt as cp_edt
            from neurons.utils.gpu_ndimage import torch_to_cupy, cupy_to_torch

            label_cp = torch_to_cupy(label)
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

        if label.is_cuda:
            return self._skeleton_weight_torch(label)

        return self._skeleton_weight_cpu(label)

    @torch.no_grad()
    def _skeleton_weight_torch(self, label: torch.Tensor) -> torch.Tensor:
        """Approximate skeleton weight via iterative morphological erosion on GPU."""
        B = label.shape[0]
        weight = torch.ones_like(label, dtype=torch.float32)
        _CHUNK = 16
        spatial_tail = label.shape[-self.spatial_dims:]
        max_iter = min(spatial_tail) // 2 + 1
        ones_pattern = " ".join(["1"] * self.spatial_dims)
        reshape_pattern = f"k c -> k c {ones_pattern}"

        for b in range(B):
            fg_ids = torch.unique(label[b])
            fg_ids = fg_ids[fg_ids > 0]
            if len(fg_ids) == 0:
                continue

            K = len(fg_ids)
            for start in range(0, K, _CHUNK):
                chunk_ids = fg_ids[start:start + _CHUNK]
                masks = rearrange(
                    torch.stack([(label[b] == uid).float() for uid in chunk_ids]),
                    "k ... -> k 1 ...",
                )

                remaining = masks.clone()
                dt = torch.zeros_like(masks)
                for layer_idx in range(1, max_iter + 1):
                    eroded = -self._pool(-remaining, 3, stride=1, padding=1)
                    removed = (remaining > 0.5) & (eroded < 0.5)
                    dt[removed] = float(layer_idx)
                    remaining = eroded * (eroded > 0.5).float()
                    if not remaining.any():
                        break

                max_d = dt.flatten(2).amax(dim=2).clamp(min=1.0)
                dt = dt / rearrange(max_d, reshape_pattern)

                for i, uid in enumerate(chunk_ids):
                    m = label[b] == uid
                    weight[b][m] = 1.0 + dt[i, 0][m] * (self.weight_bone - 1.0)

        return weight

    @torch.no_grad()
    def _skeleton_weight_cpu(self, label: torch.Tensor) -> torch.Tensor:
        """Skeleton weight via scipy EDT on CPU with batched pmap."""
        B = label.shape[0]
        label_np = label.cpu().numpy()
        weight_np = np.ones_like(label_np, dtype=np.float32)

        all_args = []
        all_meta = []
        for b in range(B):
            fg_ids = np.unique(label_np[b])
            fg_ids = fg_ids[fg_ids > 0]
            for uid in fg_ids:
                all_args.append((label_np[b], int(uid)))
                all_meta.append(b)

        if len(all_args) == 0:
            return torch.from_numpy(weight_np).to(label.device)

        results = pmap(_edt_worker, all_args)
        for batch_idx, (uid, dt) in zip(all_meta, results):
            m = label_np[batch_idx] == uid
            weight_np[batch_idx][m] = 1.0 + dt[m] * (self.weight_bone - 1.0)

        return torch.from_numpy(weight_np).to(label.device)

    # ------------------------------------------------------------------
    # Core discriminative loss
    # ------------------------------------------------------------------

    def _loss_single(self, embed, label, w_edge, w_bone) -> Dict[str, torch.Tensor]:
        """Pull/push/norm over all instances in the batch.

        Shapes:
            embed:  [B, E, *spatial]
            label:  [B, *spatial]
            w_edge: [B, *spatial]
            w_bone: [B, *spatial]
        """
        emb_flat = rearrange(embed, "b e ... -> b e (...)")
        lbl_flat = rearrange(label, "b ... -> b (...)")
        wgt_flat = rearrange(w_edge * w_bone, "b ... -> b (...)")

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

            centers = []
            b_pull = torch.tensor(0.0, device=dev)
            for uid in ids:
                mask = lbl_flat[b] == uid
                w = wgt_flat[b, mask]
                e = emb_flat[b, :, mask]
                c = (e * rearrange(w, "m -> 1 m")).sum(1) / (w.sum() + 1e-8)
                centers.append(c)
                dist = torch.norm(e - rearrange(c, "e -> e 1"), dim=0)
                b_pull = b_pull + (F.relu(dist - self.delta_v) ** 2 * w).mean()
            loss_pull = loss_pull + b_pull / K

            if K > 1:
                c_stack = torch.stack(centers)
                pw = torch.norm(
                    rearrange(c_stack, "i e -> i 1 e") -
                    rearrange(c_stack, "j e -> 1 j e"),
                    dim=2,
                )
                triu = torch.triu_indices(K, K, offset=1, device=dev)
                loss_push = loss_push + reduce(
                    F.relu(2 * self.delta_d - pw[triu[0], triu[1]]) ** 2,
                    "n -> ", "mean",
                )

            loss_norm = loss_norm + torch.stack([c.norm() for c in centers]).mean()

        n = max(n_valid, 1)
        pull = loss_pull / n
        push = loss_push / n
        norm = loss_norm / n
        total = self.weight_pull * pull + self.weight_push * push + self.weight_norm * norm
        return {"loss": total, "pull": pull, "push": push, "norm": norm}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_weights(self, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute boundary + skeleton weights (cache-friendly)."""
        w_edge = self._get_weight_boundary(label) if self.weight_edge > 1.0 else torch.ones_like(label, dtype=torch.float32)
        w_bone = self._get_weight_skeleton(label) if self.weight_bone > 1.0 else torch.ones_like(label, dtype=torch.float32)
        return w_edge, w_bone

    def forward(
        self,
        embed: torch.Tensor,
        label: torch.Tensor,
        semantic_ids: Optional[torch.Tensor] = None,
        weight_edge: Optional[torch.Tensor] = None,
        weight_bone: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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

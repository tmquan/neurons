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


def _edt_worker(args: Tuple[np.ndarray, int]) -> Tuple[int, np.ndarray]:
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

                max_d = reduce(dt, "k c ... -> k c", "max").clamp(min=1.0)
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

    @staticmethod
    def _scatter_weighted_mean(emb, lbl, wgt, K):
        """Compute weighted centroid per instance using scatter.

        Args:
            emb: [E, N] embeddings for one batch element.
            lbl: [N] zero-based instance indices (0..K-1), -1 for background.
            wgt: [N] per-pixel weights.
            K: number of instances.

        Returns:
            centers: [K, E] weighted centroids.
        """
        E = emb.shape[0]
        fg = lbl >= 0
        emb_fg = emb[:, fg].float()                   # [E, M] always float32
        lbl_fg = lbl[fg]                               # [M]
        if wgt is not None:
            wgt_fg = wgt[fg].float()                   # [M] always float32
            weighted_emb = emb_fg * wgt_fg.unsqueeze(0)
            w_sum = torch.zeros(K, device=emb.device, dtype=torch.float32)
            w_sum.scatter_add_(0, lbl_fg, wgt_fg)
        else:
            weighted_emb = emb_fg
            w_sum = torch.bincount(lbl_fg, minlength=K).float().clamp(min=1)

        c_sum = torch.zeros(E, K, device=emb.device, dtype=torch.float32)
        lbl_expand = lbl_fg.unsqueeze(0).expand(E, -1)
        c_sum.scatter_add_(1, lbl_expand, weighted_emb)

        centers = c_sum / (w_sum.unsqueeze(0) + 1e-8)  # [E, K]
        return centers.T                                # [K, E]

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
        if w_edge is not None and w_bone is not None:
            wgt_flat = rearrange(w_edge * w_bone, "b ... -> b (...)")
        elif w_edge is not None:
            wgt_flat = rearrange(w_edge, "b ... -> b (...)")
        elif w_bone is not None:
            wgt_flat = rearrange(w_bone, "b ... -> b (...)")
        else:
            wgt_flat = None

        device = embed.device
        loss_pull = torch.tensor(0.0, device=device)
        loss_push = torch.tensor(0.0, device=device)
        loss_norm = torch.tensor(0.0, device=device)
        n_valid = 0

        for b in range(embed.shape[0]):
            lbl_b = lbl_flat[b]                        # [N]
            fg = lbl_b > 0

            if not fg.any():
                continue

            fg_labels = lbl_b[fg]
            unique_ids, inverse = torch.unique(fg_labels, return_inverse=True)
            K = unique_ids.shape[0]
            n_valid += 1

            remap = torch.full_like(lbl_b, -1, dtype=torch.long)
            remap[fg] = inverse

            emb_b = emb_flat[b]                        # [E, N]
            wgt_b = wgt_flat[b] if wgt_flat is not None else None

            centers = self._scatter_weighted_mean(emb_b, remap, wgt_b, K)  # [K, E]

            center_per_pixel = centers[inverse]        # [M, E]
            emb_fg = emb_b[:, fg].T                    # [M, E]

            dist = ((emb_fg - center_per_pixel) ** 2).sum(dim=1).clamp(min=1e-12).sqrt()  # [M]
            pull_per_pixel = F.relu(dist - self.delta_v) ** 2
            if wgt_b is not None:
                pull_per_pixel = pull_per_pixel * wgt_b[fg]

            pull_sum = torch.zeros(K, device=device, dtype=torch.float32)
            pull_sum.scatter_add_(0, inverse, pull_per_pixel)
            pull_count = torch.bincount(inverse, minlength=K).float().clamp(min=1)
            b_pull = (pull_sum / pull_count).mean()
            loss_pull = loss_pull + b_pull

            if K > 1:
                pw_diff = (rearrange(centers, "i e -> i 1 e") -
                           rearrange(centers, "j e -> 1 j e"))
                pw = (pw_diff ** 2).sum(dim=2).clamp(min=1e-12).sqrt()
                triu = torch.triu_indices(K, K, offset=1, device=device)
                loss_push = loss_push + reduce(
                    F.relu(2 * self.delta_d - pw[triu[0], triu[1]]) ** 2,
                    "n -> ", "mean",
                )

            loss_norm = loss_norm + (centers ** 2).sum(dim=1).clamp(min=1e-12).sqrt().mean()

        n = max(n_valid, 1)
        pull = loss_pull / n
        push = loss_push / n
        norm = loss_norm / n
        total = self.weight_pull * pull + self.weight_push * push + self.weight_norm * norm
        return {"loss": total, "pull": pull, "push": push, "norm": norm}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_weights(self, label: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Pre-compute boundary + skeleton weights (cache-friendly).

        Returns ``None`` for a weight component when the corresponding
        multiplier is <= 1.0 (disabled), avoiding a full-size ones allocation.
        """
        w_edge = self._get_weight_boundary(label) if self.weight_edge > 1.0 else None
        w_bone = self._get_weight_skeleton(label) if self.weight_bone > 1.0 else None
        return w_edge, w_bone

    def forward(
        self,
        embed: torch.Tensor,
        label: torch.Tensor,
        semantic_ids: Optional[torch.Tensor] = None,
        weight_edge: Optional[torch.Tensor] = None,
        weight_bone: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if weight_edge is None and weight_bone is None:
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

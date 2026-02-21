"""
Discriminative Loss for Instance Segmentation.

Three loss modules in logical order:

1. **CentroidEmbeddingLoss** — classic De Brabandere et al. (2017).
   Pull embeddings toward their instance *centroid* (mean embedding),
   push centroids apart, regularise norms.

2. **SkeletonEmbeddingLoss** — geometry-aware variant.
   Pull embeddings toward the nearest *skeleton* point, push instance
   centres apart, and add two geometric terms:
     * *boundary penalty*  — cosine alignment with the DT gradient
     * *skeleton benefit*  — differentiable sampling of the normalised DT

3. **GeometryLoss** — regression loss for the geometry head
   (direction, covariance, raw reconstruction).  Each sub-loss supports
   configurable loss type: ``mse``, ``l1``, or ``smooth_l1``.
   Defaults: Smooth-L1 for direction, MSE for covariance, L1 for raw.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from scipy.ndimage import distance_transform_edt as _scipy_edt, gaussian_filter as _gauss

from neurons.losses.skeletonize import Skeletonize


# ======================================================================
# Shared helpers
# ======================================================================

def _flatten_spatial(
    embedding: torch.Tensor,
    ins_label: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Flatten spatial dims for both 2-D and 3-D inputs.

    Returns (embed_flat [B, E, N], label_flat [B, N], is_3d).
    """
    is_3d = embedding.dim() == 5
    if is_3d:
        embed_flat = rearrange(embedding, "b e d h w -> b e (d h w)")
        if ins_label.dim() == 5:
            label_flat = rearrange(ins_label, "b 1 d h w -> b (d h w)")
        else:
            label_flat = rearrange(ins_label, "b d h w -> b (d h w)")
    else:
        embed_flat = rearrange(embedding, "b e h w -> b e (h w)")
        if ins_label.dim() == 4:
            label_flat = rearrange(ins_label, "b 1 h w -> b (h w)")
        else:
            label_flat = rearrange(ins_label, "b h w -> b (h w)")
    return embed_flat, label_flat, is_3d


def _build_instance_index(
    ins: torch.Tensor,
    unique_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Map raw instance ids to contiguous 0..K-1 indices.

    Returns (cluster_indices [N], valid_mask [N], num_instances).
    """
    num = len(unique_ids)
    max_id = int(ins.max().item()) + 1
    id2idx = torch.full((max_id,), -1, device=ins.device, dtype=torch.long)
    for i, uid in enumerate(unique_ids):
        id2idx[uid.long()] = i
    cluster_idx = id2idx[ins.long()]
    valid = cluster_idx >= 0
    return cluster_idx, valid, num


def _scatter_mean(
    emb: torch.Tensor,
    idx: torch.Tensor,
    valid: torch.Tensor,
    num_clusters: int,
) -> torch.Tensor:
    """Compute per-cluster mean embedding via scatter_add.

    Args:
        emb: [E, N] embeddings.
        idx: [N] cluster index (-1 for invalid).
        valid: [N] bool mask.
        num_clusters: K.

    Returns:
        [K, E] cluster means.
    """
    E = emb.shape[0]
    device, dtype = emb.device, emb.dtype

    v_emb = emb[:, valid]                    # [E, M]
    v_idx = idx[valid]                        # [M]
    v_emb_t = rearrange(v_emb, "e m -> m e") # [M, E]

    sums = torch.zeros((num_clusters, E), device=device, dtype=dtype)
    counts = torch.zeros(num_clusters, device=device, dtype=dtype)

    for e in range(E):
        sums[:, e].scatter_add_(0, v_idx, v_emb_t[:, e])
    counts.scatter_add_(0, v_idx, torch.ones(v_idx.shape[0], device=device, dtype=dtype))
    counts = counts.clamp(min=1)
    return sums / rearrange(counts, "k -> k 1")



def _make_coord_grid(spatial_shape: Tuple, device: torch.device) -> torch.Tensor:
    """Build coordinate grid [S, N] in (x, y[, z]) order, flattened."""
    ranges = [torch.arange(s, device=device, dtype=torch.float32)
              for s in spatial_shape]
    grids = torch.meshgrid(*ranges, indexing="ij")
    stacked = torch.stack(list(reversed(grids)), dim=0)  # [S, *spatial]
    S = len(spatial_shape)
    return stacked.reshape(S, -1)


def _spatial_gradient(x: torch.Tensor) -> List[torch.Tensor]:
    """Central-difference spatial gradient (matches ``numpy.gradient``).

    Args:
        x: [*spatial] tensor (2-D or 3-D).

    Returns:
        List of *S* tensors (one per spatial dim, in dim-order).
    """
    grads: List[torch.Tensor] = []
    for d in range(x.dim()):
        g = torch.zeros_like(x)
        pre  = [slice(None)] * x.dim()
        post = [slice(None)] * x.dim()
        ctr  = [slice(None)] * x.dim()
        pre[d]  = slice(None, -2)
        post[d] = slice(2, None)
        ctr[d]  = slice(1, -1)
        g[tuple(ctr)] = (x[tuple(post)] - x[tuple(pre)]) / 2.0
        s0 = [slice(None)] * x.dim()
        s1 = [slice(None)] * x.dim()
        s0[d], s1[d] = slice(0, 1), slice(1, 2)
        g[tuple(s0)] = x[tuple(s1)] - x[tuple(s0)]
        sm1 = [slice(None)] * x.dim()
        sm2 = [slice(None)] * x.dim()
        sm1[d], sm2[d] = slice(-1, None), slice(-2, -1)
        g[tuple(sm1)] = x[tuple(sm1)] - x[tuple(sm2)]
        grads.append(g)
    return grads


def _flat_indices(
    coords_ij: torch.Tensor, spatial_shape: Tuple,
) -> torch.Tensor:
    """Convert [P, S] dim-order coordinates to flat linear indices."""
    S = len(spatial_shape)
    stride = 1
    strides: List[int] = []
    for d in reversed(range(S)):
        strides.append(stride)
        stride *= spatial_shape[d]
    strides_t = torch.tensor(
        list(reversed(strides)), device=coords_ij.device, dtype=torch.long,
    )
    return (coords_ij.long() * strides_t.unsqueeze(0)).sum(dim=1)


_SKEL_MODULE: Optional[Skeletonize] = None


@torch.no_grad()
def _skeletonize_mask(
    mask: torch.Tensor, num_iter: int = 50,
) -> torch.Tensor:
    """Thin a binary mask to a 1-pixel-wide skeleton (topology-preserving).

    Uses the Menten et al. (ICCV 2023) iterative boundary-peeling algorithm
    implemented entirely with convolutions.

    Args:
        mask: [*spatial] boolean or float binary mask (2-D or 3-D).
        num_iter: peeling iterations (should be >= max inscribed radius).

    Returns:
        [*spatial] boolean skeleton mask.
    """
    global _SKEL_MODULE
    if _SKEL_MODULE is None or _SKEL_MODULE.num_iter != num_iter:
        _SKEL_MODULE = Skeletonize(probabilistic=False, num_iter=num_iter)
        _SKEL_MODULE.eval()
    mod = _SKEL_MODULE.to(mask.device)
    inp = mask.float()[None, None]
    skel = mod(inp)
    return rearrange(skel, "1 1 ... -> ...") > 0.5


# ======================================================================
# On-the-fly target computation helpers
# ======================================================================

@torch.no_grad()
def _compute_centroid_offsets(
    lbl_flat: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """Compute unit-normalised per-pixel direction toward the instance centroid.

    Args:
        lbl_flat: [N] instance labels (0 = background).
        coords: [S, N] pixel coordinates.

    Returns:
        [S, N] unit direction vectors (0 for background pixels).
    """
    S, N = coords.shape
    device = coords.device
    offsets = torch.zeros_like(coords)

    uids = torch.unique(lbl_flat)
    uids = uids[uids > 0]
    for uid in uids:
        mask = lbl_flat == uid
        centroid = coords[:, mask].mean(dim=1)               # [S]
        offsets[:, mask] = centroid.unsqueeze(1) - coords[:, mask]

    norms = offsets.norm(dim=0, keepdim=True).clamp(min=1e-6)
    offsets = offsets / norms
    offsets[:, lbl_flat == 0] = 0.0

    return offsets


@torch.no_grad()
def _compute_skeleton_offsets(
    lbl_flat: torch.Tensor,
    coords: torch.Tensor,
    spatial_shape: Tuple,
) -> torch.Tensor:
    """Compute unit-normalised per-pixel direction toward the nearest skeleton point.

    Skeleton is extracted via the Menten et al. (ICCV 2023) topology-preserving
    skeletonization (pure PyTorch convolutions, 2-D and 3-D).

    Args:
        lbl_flat: [N] instance labels (0 = background).
        coords: [S, N] pixel coordinates.
        spatial_shape: original spatial dims, e.g. (H, W) or (D, H, W).

    Returns:
        [S, N] unit direction vectors (0 for background pixels).
    """
    S = len(spatial_shape)
    N = coords.shape[1]
    device = coords.device

    labels = lbl_flat.reshape(spatial_shape)
    offsets = torch.zeros(S, N, device=device, dtype=torch.float32)

    uids = torch.unique(labels)
    uids = uids[uids > 0]

    for uid in uids:
        mask = labels == uid
        if mask.sum() < 2:
            continue

        skeleton = _skeletonize_mask(mask)
        if skeleton.sum() == 0:
            skeleton = mask

        skel_xy = torch.nonzero(skeleton, as_tuple=False).flip(1).float()
        pixel_ij = torch.nonzero(mask, as_tuple=False)
        pixel_xy = pixel_ij.flip(1).float()

        dists = torch.cdist(pixel_xy, skel_xy)                # [P, R]
        nearest_skel = skel_xy[dists.argmin(dim=1)]            # [P, S]
        off_xy = nearest_skel - pixel_xy                       # [P, S]

        fi = _flat_indices(pixel_ij, spatial_shape)
        for s in range(S):
            offsets[s, fi] = off_xy[:, s]

    norms = offsets.norm(dim=0, keepdim=True).clamp(min=1e-6)
    offsets = offsets / norms
    offsets[:, lbl_flat.reshape(-1) == 0] = 0.0

    return offsets


@torch.no_grad()
def _compute_covariance(
    lbl_flat: torch.Tensor,
    coords: torch.Tensor,
    spatial_shape: Optional[Tuple] = None,
    sigma: float = 5.0,
) -> torch.Tensor:
    """EDT structure tensor per foreground pixel (morphology-aware).

    For each instance the Euclidean distance transform is computed, its
    gradient is obtained via Gaussian derivatives, and the smoothed outer
    product of the gradient (the *structure tensor*) is stored per pixel.

    Gradients are masked to the instance interior and the Gaussian
    integration is normalised by the mask coverage to avoid boundary
    leakage from the zero-padded exterior.

    After smoothing, the tensor is blended toward isotropy in proportion
    to the normalised EDT depth: ``w = (EDT / max_EDT)^2``.  Near the
    boundary (``w ~ 0``) the raw anisotropic tensor is preserved; at the
    medial axis (``w ~ 1``) eigenvalues converge to their mean, yielding
    a round glyph.

    Args:
        lbl_flat: [N] instance labels (0 = background).
        coords: [S, N] pixel coordinates.
        spatial_shape: e.g. (H, W) or (D, H, W).  Required.
        sigma: Integration scale for structure tensor smoothing.

    Returns:
        [S*S, N] structure tensor flattened row-major per pixel
        (0 for background), in (x, y[, z]) coordinate order
        where x = col, y = row.
    """
    if spatial_shape is None:
        raise ValueError("spatial_shape is required")

    S, N = coords.shape
    device = coords.device
    sigma_d = max(1.0, sigma / 3.0)

    labels_np = lbl_flat.cpu().numpy().reshape(spatial_shape)
    st_np = np.zeros((S * S,) + spatial_shape, dtype=np.float32)

    uids = np.unique(labels_np)
    uids = uids[uids > 0]

    for uid in uids:
        mask = labels_np == uid
        if mask.sum() < 2:
            continue
        dt = _scipy_edt(mask).astype(np.float64)
        mask_f = mask.astype(np.float64)
        edt_max = dt.max()

        norm = np.maximum(_gauss(mask_f, sigma=sigma), 1e-10)

        grads = []
        for i in range(S):
            order = [0] * S
            order[S - 1 - i] = 1
            g = _gauss(dt, sigma=sigma_d, order=order)
            g *= mask_f
            grads.append(g)

        idx = 0
        for i in range(S):
            for j in range(S):
                st_np[idx][mask] = (
                    _gauss(grads[i] * grads[j], sigma=sigma) / norm
                )[mask]
                idx += 1

        # Blend toward isotropy based on normalised EDT depth.
        w = np.zeros_like(dt)
        if edt_max > 1e-6:
            w[mask] = (dt[mask] / edt_max) ** 2

        trace = np.zeros(mask.shape, dtype=np.float64)
        for i in range(S):
            trace += st_np[i * S + i]
        iso_val = trace / S

        idx = 0
        for i in range(S):
            for j in range(S):
                if i == j:
                    st_np[idx][mask] = (
                        (1.0 - w[mask]) * st_np[idx][mask] + w[mask] * iso_val[mask]
                    ).astype(np.float32)
                else:
                    st_np[idx][mask] = (
                        (1.0 - w[mask]) * st_np[idx][mask]
                    ).astype(np.float32)
                idx += 1

    st_flat = torch.from_numpy(
        st_np.reshape(S * S, N)
    ).to(device=device, dtype=torch.float32)
    return st_flat


@torch.no_grad()
def _compute_skeleton_targets(
    gt_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute skeleton-based targets from instance labels.

    Skeleton via Menten et al. (ICCV 2023) topology-preserving peeling;
    EDT via torch-distmap for normalised DT and DT gradient.
    Stays on-device, works for 2-D and 3-D.

    Args:
        gt_labels: [B, *spatial] instance labels (0 = background).

    Returns:
        nr_skel:      [B, S, *spatial] absolute coords of nearest skeleton
                      point in (x, y[, z]) order.
        dt_norm:      [B, 1, *spatial] normalised distance transform
                      (0 at boundary, 1 at skeleton ridge).
        dt_grad:      [B, S, *spatial] spatial gradient of the DT
                      in (x, y[, z]) order.
    """
    B = gt_labels.shape[0]
    spatial_shape = gt_labels.shape[1:]
    S = len(spatial_shape)
    device = gt_labels.device

    nr_skel = torch.zeros((B, S) + spatial_shape, device=device, dtype=torch.float32)
    dt_norm = torch.zeros((B, 1) + spatial_shape, device=device, dtype=torch.float32)
    dt_grad = torch.zeros((B, S) + spatial_shape, device=device, dtype=torch.float32)

    for b in range(B):
        labels_b = gt_labels[b]
        uids = torch.unique(labels_b)
        uids = uids[uids > 0]

        for uid in uids:
            mask = labels_b == uid
            if mask.sum() < 2:
                continue

            skeleton = _skeletonize_mask(mask)
            if skeleton.sum() == 0:
                skeleton = mask

            skel_xy = torch.nonzero(skeleton, as_tuple=False).flip(1).float()
            pixel_ij = torch.nonzero(mask, as_tuple=False)
            pixel_xy = pixel_ij.flip(1).float()

            dists = torch.cdist(pixel_xy, skel_xy)                 # [P, R]
            nearest_ridge = skel_xy[dists.argmin(dim=1)]            # [P, S]

            fi = _flat_indices(pixel_ij, spatial_shape)
            nr_skel_flat = nr_skel[b].reshape(S, -1)
            for s in range(S):
                nr_skel_flat[s, fi] = nearest_ridge[:, s]

            dt = torch.from_numpy(
                _scipy_edt(mask.cpu().numpy())
            ).to(device=device, dtype=torch.float32)
            dt_max = dt[mask].max()
            normed = dt / dt_max if dt_max > 0 else dt
            dt_norm[b, 0][mask] = normed[mask]

            grads_dim = _spatial_gradient(dt)
            grads_xy = list(reversed(grads_dim))
            dt_grad_flat = dt_grad[b].reshape(S, -1)
            for s in range(S):
                dt_grad_flat[s, fi] = grads_xy[s].reshape(-1)[fi]

    return nr_skel, dt_norm, dt_grad


# ======================================================================
# 1.  Centroid variant  (classic De Brabandere pull/push/reg)
# ======================================================================

class CentroidEmbeddingLoss(nn.Module):
    """Discriminative pull/push/regularisation loss on instance embeddings.

    * **L_pull** (variance):  hinged L2 from each pixel embedding to its
      instance centroid (mean embedding).  Pulls same-instance pixels together.
    * **L_push** (distance):  pairwise margin on instance centroids.
      Pushes different instances apart.
    * **L_reg**:  L-p norm on centroids, keeps embeddings near the origin.

    Args:
        delta_pull: pull hinge margin (default 0.5).
        delta_push: push margin — centroids closer than ``2 * delta_push``
            are penalised (default 1.5).
        norm: p-norm for all distance computations (default 2).
        w_pull, w_push, w_reg: scalar weights for the three terms.
    """

    def __init__(
        self,
        delta_pull: float = 0.5,
        delta_push: float = 1.5,
        norm: int = 2,
        w_pull: float = 1.0,
        w_push: float = 1.0,
        w_reg: float = 0.001,
        delta_var: Optional[float] = None,
        delta_dst: Optional[float] = None,
        A: Optional[float] = None,
        B: Optional[float] = None,
        R: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.delta_pull = delta_var if delta_var is not None else delta_pull
        self.delta_push = delta_dst if delta_dst is not None else delta_push
        self.norm = norm
        self.w_pull = A if A is not None else w_pull
        self.w_push = B if B is not None else w_push
        self.w_reg  = R if R is not None else w_reg

    def _pull_loss(
        self,
        emb: torch.Tensor,
        idx: torch.Tensor,
        valid: torch.Tensor,
        centers: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Per-instance mean of hinged L2 from pixels to their centroid."""
        if K == 0:
            return torch.tensor(0.0, device=emb.device, dtype=torch.float32)

        v_emb = emb[:, valid]
        v_idx = idx[valid]
        gathered = rearrange(centers[v_idx], "m e -> e m")

        dist = torch.norm(v_emb.float() - gathered.float(), p=self.norm, dim=0)
        hinged = F.relu(dist - self.delta_pull) ** 2

        cl = torch.zeros(K, device=emb.device, dtype=torch.float32)
        cc = torch.zeros(K, device=emb.device, dtype=torch.float32)
        cl.scatter_add_(0, v_idx, hinged.float())
        cc.scatter_add_(0, v_idx, torch.ones_like(hinged, dtype=torch.float32))
        return reduce(cl / cc.clamp(min=1), "k -> ", "mean")

    def _push_loss(self, centers: torch.Tensor) -> torch.Tensor:
        """Pairwise hinge on centroid distances — pushes instances apart."""
        K = centers.shape[0]
        if K <= 1:
            return torch.tensor(0.0, device=centers.device, dtype=torch.float32)

        ci = rearrange(centers.float(), "k e -> k 1 e")
        cj = rearrange(centers.float(), "k e -> 1 k e")
        pw = torch.norm(ci - cj, p=self.norm, dim=2)

        triu = torch.triu_indices(K, K, offset=1, device=centers.device)
        hinged = F.relu(2 * self.delta_push - pw[triu[0], triu[1]]) ** 2
        return reduce(hinged, "n -> ", "mean")

    def _reg_loss(self, centers: torch.Tensor) -> torch.Tensor:
        """L-p norm regularisation on centroid embeddings."""
        if centers.shape[0] == 0:
            return torch.tensor(0.0, device=centers.device, dtype=torch.float32)
        return torch.norm(centers.float(), p=self.norm, dim=1).mean()

    def forward(
        self,
        embedding: torch.Tensor,
        ins_label: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embedding: [B, E, *spatial] instance embedding.
            ins_label: [B, *spatial] instance labels (0 = background).

        Returns:
            Dict with keys ``loss``, ``l_pull``, ``l_push``, ``l_reg``.
        """
        emb_flat, lbl_flat, _ = _flatten_spatial(embedding, ins_label)
        dev = embedding.device
        zero = torch.tensor(0.0, device=dev, dtype=torch.float32)

        L_pull, L_push, L_reg = zero.clone(), zero.clone(), zero.clone()
        valid_b = 0

        for b in range(embedding.shape[0]):
            uids = torch.unique(lbl_flat[b])
            uids = uids[uids > 0]
            if len(uids) == 0:
                continue
            valid_b += 1
            idx, mask, K = _build_instance_index(lbl_flat[b], uids)
            centers = _scatter_mean(emb_flat[b].float(), idx, mask, K)
            L_pull = L_pull + self._pull_loss(emb_flat[b], idx, mask, centers, K)
            L_push = L_push + self._push_loss(centers)
            L_reg = L_reg + self._reg_loss(centers)

        n = max(valid_b, 1)
        L_pull, L_push, L_reg = L_pull / n, L_push / n, L_reg / n
        total = self.w_pull * L_pull + self.w_push * L_push + self.w_reg * L_reg

        return {"loss": total, "l_pull": L_pull, "l_push": L_push, "l_reg": L_reg}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"delta_pull={self.delta_pull}, delta_push={self.delta_push}, "
            f"norm={self.norm}, "
            f"w_pull={self.w_pull}, w_push={self.w_push}, w_reg={self.w_reg})"
        )


# Backward-compatible aliases
DiscriminativeLoss = CentroidEmbeddingLoss
DiscriminativeLossVectorized = CentroidEmbeddingLoss


# ======================================================================
# 2.  Skeleton variant  (geometry-aware)
# ======================================================================

class SkeletonEmbeddingLoss(nn.Module):
    """Discriminative loss that pulls **offset-based embeddings** toward
    the instance *skeleton*.

    Supports both **2-D** and **3-D** inputs (auto-detected from
    ``offsets.dim()``).

    * 2-D: offsets ``[B, 2, H, W]``, embedding ``e_i = (x,y)_i + offset_i``
    * 3-D: offsets ``[B, 3, D, H, W]``, embedding ``e_i = (x,y,z)_i + offset_i``

    Four loss terms (all differentiable):

    * **L_pull**    — L2 between ``e_i`` and the nearest GT skeleton point.
    * **L_push**    — pairwise margin on per-instance mean embeddings.
    * **L_penalty** — cosine alignment between the offset and the DT
      gradient (repels from boundaries).
    * **L_benefit** — ``F.grid_sample`` of the normalised DT at ``e_i``
      (attracts toward the skeleton ridge).

    Args:
        delta_push: push margin between instance centres (default 20.0).
        w_pull: weight for L_pull   (default 1.0).
        w_push: weight for L_push   (default 1.0).
        w_penalty: weight for L_penalty (default 1.0).
        w_benefit: weight for L_benefit (default 5.0).
    """

    def __init__(
        self,
        delta_push: float = 20.0,
        w_pull: float = 1.0,
        w_push: float = 1.0,
        w_penalty: float = 1.0,
        w_benefit: float = 5.0,
    ) -> None:
        super().__init__()
        self.delta_push = delta_push
        self.w_pull = w_pull
        self.w_push = w_push
        self.w_penalty = w_penalty
        self.w_benefit = w_benefit

    @staticmethod
    def _make_coords(spatial_shape: Tuple, device: torch.device) -> torch.Tensor:
        """Build absolute coordinate tensor [1, S, *spatial_shape]."""
        ranges = [torch.arange(s, device=device, dtype=torch.float32)
                  for s in spatial_shape]
        grids = torch.meshgrid(*ranges, indexing="ij")
        return torch.stack(list(reversed(grids)), dim=0).unsqueeze(0)

    def forward(
        self,
        offsets: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_nr_skel: Optional[torch.Tensor] = None,
        gt_dt_norm: Optional[torch.Tensor] = None,
        gt_dt_grad: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            offsets:         [B, S, *spatial]  predicted offset field (S=2 or 3).
            gt_labels:       [B, *spatial]     instance labels (0 = background).
            gt_nr_skel:      [B, S, *spatial]  coords of nearest skeleton point.
                             Computed on the fly from *gt_labels* when ``None``.
            gt_dt_norm:      [B, 1, *spatial]  normalised DT (0=boundary, 1=skeleton).
                             Computed on the fly from *gt_labels* when ``None``.
            gt_dt_grad:      [B, S, *spatial]  gradient of the DT.
                             Computed on the fly from *gt_labels* when ``None``.

        Returns:
            Dict with ``loss``, ``l_pull``, ``l_push``,
            ``l_penalty``, ``l_benefit``, and optionally
            ``gt_nr_skel``, ``gt_dt_norm``, ``gt_dt_grad``
            (present when any target was computed on the fly).
        """
        computed_targets = (
            gt_nr_skel is None
            or gt_dt_norm is None
            or gt_dt_grad is None
        )
        if computed_targets:
            _skel, _dt_n, _dt_g = _compute_skeleton_targets(gt_labels)
            if gt_nr_skel is None:
                gt_nr_skel = _skel
            if gt_dt_norm is None:
                gt_dt_norm = _dt_n
            if gt_dt_grad is None:
                gt_dt_grad = _dt_g

        is_3d = offsets.dim() == 5
        B = offsets.shape[0]
        S = offsets.shape[1]
        spatial = offsets.shape[2:]
        device = offsets.device

        coords = self._make_coords(spatial, device).expand(B, -1, *spatial)
        embeddings = coords + offsets

        fg = gt_labels > 0
        N_fg = fg.sum().float().clamp(min=1.0)

        emb_flat = rearrange(embeddings, "b s ... -> b s (...)")
        skel_flat = rearrange(gt_nr_skel, "b s ... -> b s (...)")
        off_flat = rearrange(offsets, "b s ... -> b s (...)")
        grad_flat = rearrange(gt_dt_grad, "b s ... -> b s (...)")
        fg_flat = rearrange(fg, "b ... -> b (...)")
        mask_flat = rearrange(gt_labels, "b ... -> b (...)")

        pull_diff = emb_flat - skel_flat
        l_pull = (pull_diff ** 2).sum(dim=1)
        l_pull = (l_pull * fg_flat.float()).sum() / N_fg

        norm_off = F.normalize(off_flat, p=2, dim=1, eps=1e-5)
        norm_grad = F.normalize(grad_flat, p=2, dim=1, eps=1e-5)
        cos_sim = (norm_off * norm_grad).sum(dim=1)
        l_penalty = ((1.0 - cos_sim) * fg_flat.float()).sum() / N_fg

        if is_3d:
            D, H, W = spatial
            grid_x = (embeddings[:, 0] / (W - 1)) * 2.0 - 1.0
            grid_y = (embeddings[:, 1] / (H - 1)) * 2.0 - 1.0
            grid_z = (embeddings[:, 2] / (D - 1)) * 2.0 - 1.0
            sample_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        else:
            H, W = spatial
            grid_x = (embeddings[:, 0] / (W - 1)) * 2.0 - 1.0
            grid_y = (embeddings[:, 1] / (H - 1)) * 2.0 - 1.0
            sample_grid = torch.stack([grid_x, grid_y], dim=-1)

        sampled_dt = F.grid_sample(
            gt_dt_norm, sample_grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        sampled_flat = rearrange(sampled_dt, "b 1 ... -> b (...)")
        l_benefit = ((1.0 - sampled_flat) * fg_flat.float()).sum() / N_fg

        l_push = torch.tensor(0.0, device=device, dtype=torch.float32)
        for b in range(B):
            ids = torch.unique(mask_flat[b])
            ids = ids[ids > 0]
            K = len(ids)
            if K <= 1:
                continue
            centers = []
            for uid in ids:
                m = mask_flat[b] == uid
                centers.append(emb_flat[b, :, m].mean(dim=1))
            centers_t = torch.stack(centers)
            pw = torch.cdist(centers_t, centers_t, p=2)
            push_pen = F.relu(self.delta_push - pw) ** 2
            off_diag = ~torch.eye(K, dtype=torch.bool, device=device)
            l_push = l_push + push_pen[off_diag].sum() / (K * (K - 1))
        l_push = l_push / B

        total = (
            self.w_pull * l_pull
            + self.w_push * l_push
            + self.w_penalty * l_penalty
            + self.w_benefit * l_benefit
        )

        out = {
            "loss": total,
            "l_pull": l_pull,
            "l_push": l_push,
            "l_penalty": l_penalty,
            "l_benefit": l_benefit,
        }
        if computed_targets:
            out["gt_nr_skel"] = gt_nr_skel
            out["gt_dt_norm"] = gt_dt_norm
            out["gt_dt_grad"] = gt_dt_grad
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"delta_push={self.delta_push}, "
            f"w_pull={self.w_pull}, w_push={self.w_push}, "
            f"w_penalty={self.w_penalty}, w_benefit={self.w_benefit})"
        )


# ======================================================================
# 3.  Geometry head loss  (dir + cov + raw)
# ======================================================================

_LOSS_FN_REGISTRY = {
    "mse": "mse",
    "l2": "mse",
    "l1": "l1",
    "mae": "l1",
    "smooth_l1": "smooth_l1",
    "huber": "smooth_l1",
}


def _resolve_loss_fn(name: str) -> str:
    key = name.lower().replace("-", "_")
    if key not in _LOSS_FN_REGISTRY:
        raise ValueError(
            f"Unknown loss type '{name}'. "
            f"Choose from: {sorted(set(_LOSS_FN_REGISTRY.values()))}"
        )
    return _LOSS_FN_REGISTRY[key]


class GeometryLoss(nn.Module):
    """Regression loss for the geometry head output.

    Supervises three groups of channels produced by the model's
    ``head_geometry`` against on-the-fly computed targets:

    * **L_dir**  (first S channels):  per-pixel offset toward instance
      centroid or nearest skeleton point.
    * **L_cov**  (next S*S channels):  EDT structure tensor with
      depth-blended isotropy (see ``_compute_covariance``).
    * **L_raw**  (last 4 channels):  RGBA reconstruction of the input
      image (requires ``raw_image``).

    The channel layout of the geometry tensor is ``[dir, cov, raw]``
    with a total of ``S + S*S + 4`` channels.

    Each sub-loss supports a configurable loss type:

    * ``"mse"`` / ``"l2"``  — mean squared error
    * ``"l1"`` / ``"mae"``  — mean absolute error (sharper)
    * ``"smooth_l1"`` / ``"huber"``  — Smooth-L1 (robust to outliers)

    Defaults: Smooth-L1 for direction (outlier-robust offset regression),
    MSE for covariance (Frobenius-norm on tensors), L1 for raw
    reconstruction (sharper image quality).

    Args:
        spatial_dims: 2 or 3 (default 2).
        dir_target: ``"centroid"`` or ``"skeleton"`` (default ``"centroid"``).
        weight_dir: weight for L_dir (default 1.0).
        weight_cov: weight for L_cov (default 1.0).
        weight_raw: weight for L_raw (default 1.0).
        loss_dir: loss type for L_dir (default ``"smooth_l1"``).
        loss_cov: loss type for L_cov (default ``"mse"``).
        loss_raw: loss type for L_raw (default ``"l1"``).
        smooth_l1_beta: beta parameter for Smooth-L1 (default 1.0).
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        dir_target: str = "centroid",
        weight_dir: float = 1.0,
        weight_cov: float = 1.0,
        weight_raw: float = 1.0,
        loss_dir: str = "smooth_l1",
        loss_cov: str = "mse",
        loss_raw: str = "l1",
        smooth_l1_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.dir_target = dir_target
        self.weight_dir = weight_dir
        self.weight_cov = weight_cov
        self.weight_raw = weight_raw
        self.loss_dir = _resolve_loss_fn(loss_dir)
        self.loss_cov = _resolve_loss_fn(loss_cov)
        self.loss_raw = _resolve_loss_fn(loss_raw)
        self.smooth_l1_beta = smooth_l1_beta

        S = spatial_dims
        self._ch_dir = S
        self._ch_cov = S * S
        self._ch_raw = 4

    def _fg_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fg: torch.Tensor,
        loss_type: str,
    ) -> torch.Tensor:
        """Foreground-masked regression loss for one sample.

        Args:
            pred: [C, N] predicted channels (flattened spatial).
            target: [C, N] target channels.
            fg: [N] bool foreground mask.
            loss_type: one of ``"mse"``, ``"l1"``, ``"smooth_l1"``.
        """
        N_fg = fg.sum().float().clamp(min=1.0)
        diff = pred[:, fg] - target[:, fg]
        numel = N_fg * diff.shape[0]

        if loss_type == "mse":
            return (diff ** 2).sum() / numel
        elif loss_type == "l1":
            return diff.abs().sum() / numel
        else:
            return F.smooth_l1_loss(
                diff, torch.zeros_like(diff),
                beta=self.smooth_l1_beta, reduction="sum",
            ) / numel

    def forward(
        self,
        geometry: torch.Tensor,
        ins_label: torch.Tensor,
        raw_image: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            geometry: [B, S+S*S+4, *spatial] geometry head prediction.
            ins_label: [B, *spatial] instance labels (0 = background).
            raw_image: [B, 1, *spatial] (optional, for L_raw target).

        Returns:
            Dict with keys ``loss``, ``dir``, ``cov``, ``raw``.
        """
        B = geometry.shape[0]
        spatial_shape = geometry.shape[2:]
        dev = geometry.device
        zero = torch.tensor(0.0, device=dev, dtype=torch.float32)

        L_dir, L_cov, L_raw = zero.clone(), zero.clone(), zero.clone()
        valid_b = 0

        geom_flat = rearrange(geometry, "b c ... -> b c (...)")
        c1 = self._ch_dir
        c2 = c1 + self._ch_cov
        c3 = c2 + self._ch_raw
        pred_dir = geom_flat[:, :c1]
        pred_cov = geom_flat[:, c1:c2]
        pred_raw = geom_flat[:, c2:c3]

        lbl_flat = rearrange(ins_label, "b ... -> b (...)").long()
        coords = _make_coord_grid(spatial_shape, dev)

        for b in range(B):
            uids = torch.unique(lbl_flat[b])
            uids = uids[uids > 0]
            if len(uids) == 0:
                continue
            valid_b += 1
            fg = lbl_flat[b] > 0

            if self.weight_dir > 0:
                if self.dir_target == "centroid":
                    dir_tgt = _compute_centroid_offsets(lbl_flat[b], coords)
                else:
                    dir_tgt = _compute_skeleton_offsets(
                        lbl_flat[b], coords, spatial_shape,
                    )
                L_dir = L_dir + self._fg_loss(pred_dir[b], dir_tgt, fg, self.loss_dir)

            if self.weight_cov > 0:
                cov_tgt = _compute_covariance(lbl_flat[b], coords, spatial_shape)
                L_cov = L_cov + self._fg_loss(pred_cov[b], cov_tgt, fg, self.loss_cov)

            if self.weight_raw > 0 and raw_image is not None:
                img_flat = rearrange(raw_image[b], "c ... -> c (...)").clamp(0.0, 1.0)
                rgba_tgt = torch.cat([
                    img_flat.expand(3, -1),
                    fg.unsqueeze(0).float(),
                ], dim=0)
                raw_pred_b = torch.sigmoid(pred_raw[b])
                L_raw = L_raw + self._fg_loss(raw_pred_b, rgba_tgt, fg, self.loss_raw)

        n = max(valid_b, 1)
        L_dir, L_cov, L_raw = L_dir / n, L_cov / n, L_raw / n

        total = self.weight_dir * L_dir + self.weight_cov * L_cov + self.weight_raw * L_raw

        return {"loss": total, "dir": L_dir, "cov": L_cov, "raw": L_raw}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"spatial_dims={self.spatial_dims}, "
            f"dir_target='{self.dir_target}', "
            f"loss_dir='{self.loss_dir}', loss_cov='{self.loss_cov}', loss_raw='{self.loss_raw}', "
            f"weight_dir={self.weight_dir}, weight_cov={self.weight_cov}, weight_raw={self.weight_raw})"
        )

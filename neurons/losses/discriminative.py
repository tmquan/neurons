"""
Discriminative Loss for Instance Segmentation.

Two embedding loss modules:

1. **CentroidEmbeddingLoss** -- classic De Brabandere et al. (2017).
   Pull embeddings toward their instance *centroid* (mean embedding),
   push centroids apart, regularise norms.

2. **SkeletonEmbeddingLoss** -- geometry-aware variant.
   Pull embeddings toward the nearest *skeleton* point, push instance
   centres apart, and add two geometric terms:
     * *boundary penalty*  -- cosine alignment with the DT gradient
     * *skeleton benefit*  -- differentiable sampling of the normalised DT
"""

from typing import Dict, List, Optional, Tuple

import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from neurons.transforms.edt import (
    distance_transform_edt as _gpu_edt,
    _use_gpu as _cupy_available,
)

from neurons.losses.skeletonize import Skeletonize
from neurons.utils.parallel import pmap


# ======================================================================
# Shared helpers
# ======================================================================

def _flatten_spatial(
    embedding: torch.Tensor,
    ins_label: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Flatten spatial dims → ``(embed_flat [B,E,N], label_flat [B,N], is_3d)``.

    Works identically for 2-D (``[B,E,H,W]``) and 3-D (``[B,E,D,H,W]``)
    embeddings.  ``ins_label`` may carry an optional unit channel dim.
    """
    is_3d = embedding.dim() == 5
    embed_flat = rearrange(embedding, "b e ... -> b e (...)")         # [B, E, N]
    if ins_label.dim() == embedding.dim():
        label_flat = rearrange(ins_label, "b 1 ... -> b (...)")      # strip channel
    else:
        label_flat = rearrange(ins_label, "b ... -> b (...)")        # no channel
    return embed_flat, label_flat, is_3d


def _build_instance_index(
    ins: torch.Tensor,
    unique_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Map raw instance ids → contiguous 0..K-1.  Returns (idx, valid_mask, K)."""
    num = len(unique_ids)
    max_id = int(ins.max().item()) + 1
    id2idx = torch.full((max_id,), -1, device=ins.device, dtype=torch.long)
    id2idx[unique_ids.long()] = torch.arange(num, device=ins.device)
    cluster_idx = id2idx[ins.long()]
    valid = cluster_idx >= 0
    return cluster_idx, valid, num


def _scatter_mean(
    emb: torch.Tensor,
    idx: torch.Tensor,
    valid: torch.Tensor,
    num_clusters: int,
) -> torch.Tensor:
    """Per-cluster mean embedding via scatter_add.  Returns [K, E]."""
    E = emb.shape[0]
    device = emb.device

    v_emb = emb[:, valid]
    v_idx = idx[valid]
    v_emb_t = rearrange(v_emb, "e m -> m e").float()

    sums = torch.zeros((num_clusters, E), device=device, dtype=torch.float32)
    counts = torch.zeros(num_clusters, device=device, dtype=torch.float32)

    idx_for_scatter = repeat(v_idx, "m -> m e", e=E)
    sums.scatter_add_(0, idx_for_scatter, v_emb_t)
    counts.scatter_add_(0, v_idx, torch.ones(v_idx.shape[0], device=device, dtype=torch.float32))
    counts = counts.clamp(min=1)
    return sums / rearrange(counts, "k -> k 1")



def _make_coord_grid(spatial_shape: Tuple, device: torch.device) -> torch.Tensor:
    """Build flattened coordinate grid [S, N] in reversed (x,y[,z]) order."""
    ranges = [torch.arange(s, device=device, dtype=torch.float32)
              for s in spatial_shape]
    grids = torch.meshgrid(*ranges, indexing="ij")
    stacked = torch.stack(list(reversed(grids)), dim=0)
    S = len(spatial_shape)
    return rearrange(stacked, "s ... -> s (...)")


def _spatial_gradient(x: torch.Tensor) -> List[torch.Tensor]:
    """Central-difference spatial gradient, one tensor per dim.

    Matches ``numpy.gradient``:  interior uses (f[i+1] - f[i-1]) / 2,
    boundaries use forward/backward first-order differences.
    Returns a list of S tensors, one per spatial dimension.
    """
    grads: List[torch.Tensor] = []
    for d in range(x.dim()):
        g = torch.zeros_like(x)

        # Interior: central difference  (f[i+1] - f[i-1]) / 2
        pre  = [slice(None)] * x.dim()
        post = [slice(None)] * x.dim()
        ctr  = [slice(None)] * x.dim()
        pre[d]  = slice(None, -2)
        post[d] = slice(2, None)
        ctr[d]  = slice(1, -1)
        g[tuple(ctr)] = (x[tuple(post)] - x[tuple(pre)]) / 2.0

        # First element: forward difference  f[1] - f[0]
        s0 = [slice(None)] * x.dim()
        s1 = [slice(None)] * x.dim()
        s0[d], s1[d] = slice(0, 1), slice(1, 2)
        g[tuple(s0)] = x[tuple(s1)] - x[tuple(s0)]

        # Last element: backward difference  f[-1] - f[-2]
        sm1 = [slice(None)] * x.dim()
        sm2 = [slice(None)] * x.dim()
        sm1[d], sm2[d] = slice(-1, None), slice(-2, -1)
        g[tuple(sm1)] = x[tuple(sm1)] - x[tuple(sm2)]

        grads.append(g)
    return grads


def _flat_indices(
    coords_ij: torch.Tensor, spatial_shape: Tuple,
) -> torch.Tensor:
    """Convert [P, S] dim-order coordinates → flat linear indices.

    Computes row-major strides for *spatial_shape* and dot-products
    each coordinate row with the stride vector.
    """
    S = len(spatial_shape)
    # Row-major strides: last dim has stride 1, second-to-last has stride W, etc.
    stride = 1
    strides: List[int] = []
    for d in reversed(range(S)):
        strides.append(stride)
        stride *= spatial_shape[d]
    strides_t = torch.tensor(
        list(reversed(strides)), device=coords_ij.device, dtype=torch.long,
    )
    return (coords_ij.long() * rearrange(strides_t, "s -> 1 s")).sum(dim=1)


@functools.lru_cache(maxsize=8)
def _get_skel_module(device_str: str, num_iter: int) -> Skeletonize:
    """Cached Skeletonize module per (device, num_iter) pair."""
    mod = Skeletonize(probabilistic=False, num_iter=num_iter)
    mod.eval()
    return mod.to(device_str)


@torch.no_grad()
def _skeletonize_mask(
    mask: torch.Tensor, num_iter: int = 50,
) -> torch.Tensor:
    """Thin a binary mask to a 1-pixel-wide skeleton (topology-preserving)."""
    mod = _get_skel_module(str(mask.device), num_iter)
    inp = rearrange(mask.float(), "... -> 1 1 ...")               # [1, 1, *spatial]
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
    """Per-pixel direction toward instance centroid with global magnitude."""
    S, N = coords.shape
    offsets = torch.zeros_like(coords)

    uids = torch.unique(lbl_flat)
    uids = uids[uids > 0]
    for uid in uids:
        mask = lbl_flat == uid
        centroid = coords[:, mask].mean(dim=1)
        raw_off = rearrange(centroid, "s -> s 1") - coords[:, mask]
        max_dist = raw_off.norm(dim=0).max().clamp(min=1e-6)
        offsets[:, mask] = raw_off / max_dist

    offsets[:, lbl_flat == 0] = 0.0
    return offsets


@torch.no_grad()
def _compute_skeleton_offsets(
    lbl_flat: torch.Tensor,
    coords: torch.Tensor,
    spatial_shape: Tuple,
) -> torch.Tensor:
    """Per-pixel direction toward nearest skeleton point with global magnitude."""
    S = len(spatial_shape)
    N = coords.shape[1]
    device = coords.device

    labels = lbl_flat.reshape(spatial_shape)                         # [*spatial]
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

        P = pixel_xy.shape[0]
        CHUNK = 4096
        if P <= CHUNK:
            dists = torch.cdist(pixel_xy, skel_xy)
            nearest_skel = skel_xy[dists.argmin(dim=1)]
        else:
            nearest_skel = torch.empty_like(pixel_xy)
            for i in range(0, P, CHUNK):
                chunk = pixel_xy[i:i + CHUNK]
                d = torch.cdist(chunk, skel_xy)
                nearest_skel[i:i + CHUNK] = skel_xy[d.argmin(dim=1)]
        off_xy = nearest_skel - pixel_xy

        max_dist = off_xy.norm(dim=1).max().clamp(min=1e-6)
        off_xy = off_xy / max_dist

        fi = _flat_indices(pixel_ij, spatial_shape)
        for s in range(S):
            offsets[s, fi] = off_xy[:, s]

    offsets[:, lbl_flat == 0] = 0.0
    return offsets


# -----------------------------------------------------------------------
# Covariance target: cupy-native (GPU) and scipy (CPU/pmap) paths
# -----------------------------------------------------------------------

def _covariance_one_gpu(labels_np, uid, S, sigma):
    """Per-instance EDT structure tensor via cucim/scipy (numpy arrays).

    Five phases:
    1. EDT of the binary instance mask
    2. Smoothed gradient estimation (derivative-order gaussian_filter)
    3. Structure tensor: outer product of gradients, smoothed
    4. Isotropy blending: lerp toward isotropic tensor at the medial axis
    5. EDT magnitude scaling: centre voxels carry larger tensors

    Returns (uid, mask, st) — all numpy arrays.
    """
    from neurons.transforms.edt import distance_transform_edt, gaussian_filter

    sigma_d = max(1.0, sigma / 3.0)
    spatial_shape = labels_np.shape

    mask = labels_np == uid
    if int(mask.sum()) < 2:
        return None

    dt = distance_transform_edt(mask).astype(np.float64)
    mask_f = mask.astype(np.float64)
    edt_max = float(dt.max())
    norm = np.maximum(gaussian_filter(mask_f, sigma=sigma), 1e-10)

    grads = []
    for i in range(S):
        order = [0] * S
        order[S - 1 - i] = 1
        g = gaussian_filter(dt, sigma=sigma_d, order=order)
        g = g * mask_f
        grads.append(g)

    st_inst = np.zeros((S * S,) + spatial_shape, dtype=np.float32)
    idx = 0
    for i in range(S):
        for j in range(S):
            st_inst[idx][mask] = (gaussian_filter(grads[i] * grads[j], sigma=sigma) / norm)[mask]
            idx += 1

    w = np.zeros_like(dt)
    if edt_max > 1e-6:
        w[mask] = (dt[mask] / edt_max) ** 2

    trace = sum(st_inst[i * S + i] for i in range(S))
    iso_val = trace / S

    idx = 0
    for i in range(S):
        for j in range(S):
            if i == j:
                st_inst[idx][mask] = ((1.0 - w[mask]) * st_inst[idx][mask] + w[mask] * iso_val[mask]).astype(np.float32)
            else:
                st_inst[idx][mask] = ((1.0 - w[mask]) * st_inst[idx][mask]).astype(np.float32)
            idx += 1

    edt_scale = np.zeros_like(dt, dtype=np.float32)
    if edt_max > 1e-6:
        edt_scale[mask] = (dt[mask] / edt_max).astype(np.float32)
    for c in range(S * S):
        st_inst[c][mask] = st_inst[c][mask] * edt_scale[mask]

    return (int(uid), mask, st_inst)


def _covariance_worker(args):
    """Per-instance structure tensor using scipy (CPU) -- for pmap."""
    from neurons.transforms.edt import distance_transform_edt, gaussian_filter
    labels_np, uid, S, sigma = args
    sigma_d = max(1.0, sigma / 3.0)
    spatial_shape = labels_np.shape

    mask = labels_np == uid
    if mask.sum() < 2:
        return None

    dt = distance_transform_edt(mask).astype(np.float64)
    mask_f = mask.astype(np.float64)
    edt_max = dt.max()
    norm = np.maximum(gaussian_filter(mask_f, sigma=sigma), 1e-10)

    grads = []
    for i in range(S):
        order = [0] * S
        order[S - 1 - i] = 1
        g = gaussian_filter(dt, sigma=sigma_d, order=order)
        g *= mask_f
        grads.append(g)

    st_inst = np.zeros((S * S,) + spatial_shape, dtype=np.float32)
    idx = 0
    for i in range(S):
        for j in range(S):
            st_inst[idx][mask] = (gaussian_filter(grads[i] * grads[j], sigma=sigma) / norm)[mask]
            idx += 1

    w = np.zeros_like(dt)
    if edt_max > 1e-6:
        w[mask] = (dt[mask] / edt_max) ** 2

    trace = sum(st_inst[i * S + i] for i in range(S))
    iso_val = trace / S

    idx = 0
    for i in range(S):
        for j in range(S):
            if i == j:
                st_inst[idx][mask] = ((1.0 - w[mask]) * st_inst[idx][mask] + w[mask] * iso_val[mask]).astype(np.float32)
            else:
                st_inst[idx][mask] = ((1.0 - w[mask]) * st_inst[idx][mask]).astype(np.float32)
            idx += 1

    edt_scale = np.zeros_like(dt, dtype=np.float32)
    if edt_max > 1e-6:
        edt_scale[mask] = (dt[mask] / edt_max).astype(np.float32)
    for c in range(S * S):
        st_inst[c][mask] *= edt_scale[mask]

    return (uid, mask, st_inst)


@torch.no_grad()
def _compute_covariance(
    lbl_flat: torch.Tensor,
    coords: torch.Tensor,
    spatial_shape: Optional[Tuple] = None,
    sigma: float = 5.0,
) -> torch.Tensor:
    """EDT structure tensor per foreground pixel.

    GPU path: cucim-backed EDT + gaussian_filter via edt.py.
    CPU path: all instances processed in parallel via pmap.
    """
    if spatial_shape is None:
        raise ValueError("spatial_shape is required")

    S, N = coords.shape
    device = coords.device

    if _cupy_available():
        labels_np = lbl_flat.cpu().numpy().reshape(spatial_shape)
        st_np = np.zeros((S * S,) + spatial_shape, dtype=np.float32)

        uids = np.unique(labels_np)
        uids = uids[uids > 0]

        for uid in uids:
            res = _covariance_one_gpu(labels_np, int(uid), S, sigma)
            if res is None:
                continue
            _, mask_np, st_inst_np = res
            for c in range(S * S):
                st_np[c][mask_np] = st_inst_np[c][mask_np]

        return rearrange(
            torch.from_numpy(st_np), "c ... -> c (...)",
        ).to(device=device, dtype=torch.float32)

    labels_np = lbl_flat.cpu().numpy().reshape(spatial_shape)
    st_np = np.zeros((S * S,) + spatial_shape, dtype=np.float32)

    uids = np.unique(labels_np)
    uids = uids[uids > 0]

    if len(uids) > 0:
        results = pmap(
            _covariance_worker,
            [(labels_np, int(uid), S, sigma) for uid in uids],
        )
        for res in results:
            if res is None:
                continue
            _, mask, st_inst = res
            for c in range(S * S):
                st_np[c][mask] = st_inst[c][mask]

    return rearrange(
        torch.from_numpy(st_np), "c ... -> c (...)",
    ).to(device=device, dtype=torch.float32)


@torch.no_grad()
def _compute_skeleton_targets(
    gt_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute skeleton-based targets from instance labels."""
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

            dists = torch.cdist(pixel_xy, skel_xy)
            nearest_ridge = skel_xy[dists.argmin(dim=1)]

            fi = _flat_indices(pixel_ij, spatial_shape)
            nr_skel_flat = rearrange(nr_skel[b], "s ... -> s (...)")
            for s in range(S):
                nr_skel_flat[s, fi] = nearest_ridge[:, s]

            dt = torch.from_numpy(
                _gpu_edt(mask.cpu().numpy())
            ).to(device=device, dtype=torch.float32)
            dt_max = dt[mask].max()
            normed = dt / dt_max if dt_max > 0 else dt
            dt_norm[b, 0][mask] = normed[mask]

            grads_dim = _spatial_gradient(dt)
            grads_xy = list(reversed(grads_dim))
            dt_grad_flat = rearrange(dt_grad[b], "s ... -> s (...)")
            for s in range(S):
                grad_flat = rearrange(grads_xy[s], "... -> (...)")
                dt_grad_flat[s, fi] = grad_flat[fi]

    return nr_skel, dt_norm, dt_grad


# ======================================================================
# 1.  Centroid variant  (classic De Brabandere pull/push/reg)
# ======================================================================

class CentroidEmbeddingLoss(nn.Module):
    """Discriminative pull/push/regularisation loss on instance embeddings."""

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

    def _pull_loss(self, emb, idx, valid, centers, K):
        if K == 0:
            return torch.tensor(0.0, device=emb.device, dtype=torch.float32)
        v_emb = emb[:, valid]
        v_idx = idx[valid]
        gathered = rearrange(centers[v_idx], "m e -> e m")
        diff = v_emb.float() - gathered.float()
        dist = (diff ** 2).sum(dim=0).clamp(min=1e-12).sqrt()
        hinged = F.relu(dist - self.delta_pull) ** 2
        cl = torch.zeros(K, device=emb.device, dtype=torch.float32)
        cc = torch.zeros(K, device=emb.device, dtype=torch.float32)
        cl.scatter_add_(0, v_idx, hinged.float())
        cc.scatter_add_(0, v_idx, torch.ones_like(hinged, dtype=torch.float32))
        return reduce(cl / cc.clamp(min=1), "k -> ", "mean")

    def _push_loss(self, centers):
        K = centers.shape[0]
        if K <= 1:
            return torch.tensor(0.0, device=centers.device, dtype=torch.float32)
        ci = rearrange(centers.float(), "k e -> k 1 e")
        cj = rearrange(centers.float(), "k e -> 1 k e")
        pw = ((ci - cj) ** 2).sum(dim=2).clamp(min=1e-12).sqrt()
        triu = torch.triu_indices(K, K, offset=1, device=centers.device)
        hinged = F.relu(2 * self.delta_push - pw[triu[0], triu[1]]) ** 2
        return reduce(hinged, "n -> ", "mean")

    def _reg_loss(self, centers):
        if centers.shape[0] == 0:
            return torch.tensor(0.0, device=centers.device, dtype=torch.float32)
        return (centers.float() ** 2).sum(dim=1).clamp(min=1e-12).sqrt().mean()

    def forward(self, embedding, ins_label):
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


DiscriminativeLoss = CentroidEmbeddingLoss
DiscriminativeLossVectorized = CentroidEmbeddingLoss


# ======================================================================
# 2.  Skeleton variant  (geometry-aware)
# ======================================================================

class SkeletonEmbeddingLoss(nn.Module):
    """Discriminative loss that pulls offset-based embeddings toward
    the instance skeleton."""

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
    def _make_coords(spatial_shape, device):
        ranges = [torch.arange(s, device=device, dtype=torch.float32)
                  for s in spatial_shape]
        grids = torch.meshgrid(*ranges, indexing="ij")
        return rearrange(torch.stack(list(reversed(grids)), dim=0), "s ... -> 1 s ...")

    def forward(self, offsets, gt_labels, gt_nr_skel=None, gt_dt_norm=None, gt_dt_grad=None):
        computed_targets = (
            gt_nr_skel is None or gt_dt_norm is None or gt_dt_grad is None
        )
        if computed_targets:
            _skel, _dt_n, _dt_g = _compute_skeleton_targets(gt_labels)
            if gt_nr_skel is None:
                gt_nr_skel = _skel
            if gt_dt_norm is None:
                gt_dt_norm = _dt_n
            if gt_dt_grad is None:
                gt_dt_grad = _dt_g

        B = offsets.shape[0]
        S = offsets.shape[1]
        spatial = offsets.shape[2:]
        device = offsets.device

        coords = repeat(self._make_coords(spatial, device), "1 s ... -> b s ...", b=B)
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
        l_pull = reduce(pull_diff.float() ** 2, "b s n -> b n", "sum")
        l_pull = reduce(l_pull * fg_flat.float(), "b n -> ", "sum") / N_fg

        norm_off = F.normalize(off_flat, p=2, dim=1, eps=1e-5)
        norm_grad = F.normalize(grad_flat, p=2, dim=1, eps=1e-5)
        cos_sim = reduce(norm_off * norm_grad, "b s n -> b n", "sum")
        l_penalty = reduce((1.0 - cos_sim) * fg_flat.float(), "b n -> ", "sum") / N_fg

        # Build normalised grid for F.grid_sample (works for both 2D and 3D).
        # grid_sample expects coords in (x, y[, z]) order, each in [-1, 1].
        # `spatial` is (D, H, W) for 3D or (H, W) for 2D — reversed so
        # spatial[-1] = W, spatial[-2] = H, etc.
        grid_coords = []
        for s_idx in range(S):
            extent = max(spatial[-(s_idx + 1)] - 1, 1)
            grid_coords.append(embeddings[:, s_idx] / extent * 2.0 - 1.0)
        sample_grid = torch.stack(grid_coords, dim=-1)

        sampled_dt = F.grid_sample(
            gt_dt_norm, sample_grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        sampled_flat = rearrange(sampled_dt, "b 1 ... -> b (...)")
        l_benefit = reduce((1.0 - sampled_flat) * fg_flat.float(), "b n -> ", "sum") / N_fg

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

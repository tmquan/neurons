"""
Label utilities for connectomics segmentation.

Organised in four sections:

1. **Boundary detection** — ``find_boundaries``, a pure-torch
   reimplementation of ``skimage.segmentation.find_boundaries``.
2. **Sequential relabelling** — ``relabel_sequential``.
3. **Connected-component relabelling** — ``relabel_connected_components``
   (GPU via cupy, CPU via scipy + pmap).  ``relabel_after_crop`` is a
   convenience wrapper.
4. **Embedding clustering** — ``cluster_embeddings_meanshift``,
   ``cluster_embeddings_soft``, ``cluster_offsets_hough``.

All public functions work identically on 2-D and 3-D inputs.
Metrics live in ``neurons.metrics``.
"""

from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


# ═══════════════════════════════════════════════════════════════════════
# §1  Boundary detection  (torch-accelerated, 2D / 3D)
# ═══════════════════════════════════════════════════════════════════════

# ── 1a. Shift primitive ──────────────────────────────────────────────

def _shift(t: torch.Tensor, dim: int, direction: int) -> torch.Tensor:
    """Shift *t* by one pixel along *dim*, replicating the boundary.

    ``direction = +1`` → ``result[i] = t[i+1]``  (next neighbour)
    ``direction = −1`` → ``result[i] = t[i−1]``  (prev neighbour)

    Works for any dtype (float, long, bool) and any number of dims.
    """
    if direction == 0:
        return t
    # body : everything except the trailing/leading edge
    # edge : the single boundary slice that gets replicated
    slc_body = [slice(None)] * t.dim()
    slc_edge = [slice(None)] * t.dim()
    if direction > 0:                          # look *forward*
        slc_body[dim] = slice(1, None)         #   body = t[1:]
        slc_edge[dim] = slice(-1, None)        #   edge = t[-1:]
    else:                                      # look *backward*
        slc_body[dim] = slice(None, -1)        #   body = t[:-1]
        slc_edge[dim] = slice(None, 1)         #   edge = t[:1]
    # Concatenate:  [body, edge]  for +1,  [edge, body]  for −1
    if direction > 0:
        return torch.cat([t[tuple(slc_body)], t[tuple(slc_edge)]], dim=dim)
    return torch.cat([t[tuple(slc_edge)], t[tuple(slc_body)]], dim=dim)


# ── 1b. Pooling-based dilation / erosion ─────────────────────────────

_POOL = {2: F.max_pool2d, 3: F.max_pool3d}


def _pool_nd(
    t: torch.Tensor, ndim: int, *, negate: bool = False,
) -> torch.Tensor:
    """Max-pool with a 3^ndim kernel on a ``[*spatial]`` tensor.

    When *negate* is True the result is ``−max_pool(−t)`` which gives
    the local **minimum** (morphological erosion).
    """
    x = rearrange(t, "... -> 1 1 ...")               # [1, 1, *spatial]
    if negate:
        x = -x
    x = F.pad(x, pad=(1, 1) * ndim, mode="replicate")
    x = _POOL[ndim](x, kernel_size=3, stride=1, padding=0)
    if negate:
        x = -x
    return rearrange(x, "1 1 ... -> ...")             # [*spatial]


# ── 1c. Dilation + erosion  (unified 2D / 3D) ───────────────────────

def _dilate_erode(
    lbl: torch.Tensor,
    ndim: int,
    connectivity: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Morphological dilation (local max) and erosion (local min).

    connectivity < ndim → face-adjacent neighbourhood  (shift-based).
    connectivity ≥ ndim → full 3^ndim neighbourhood    (pool-based).

    Returns ``(dilated, eroded)`` with the same shape as *lbl*.
    """
    if connectivity >= ndim:
        return _pool_nd(lbl, ndim), _pool_nd(lbl, ndim, negate=True)

    dilated = lbl.clone()
    eroded = lbl.clone()
    for d in range(ndim):
        fwd = _shift(lbl, d, +1)
        bwd = _shift(lbl, d, -1)
        dilated = torch.max(dilated, torch.max(fwd, bwd))
        eroded = torch.min(eroded, torch.min(fwd, bwd))
    return dilated, eroded


# ── 1d. Neighbour offsets ────────────────────────────────────────────

def _neighbor_offsets(ndim: int, connectivity: int) -> List[Tuple[int, ...]]:
    """Offset tuples for the requested neighbourhood.

    connectivity < ndim → face-adjacent  (2·ndim offsets)
    connectivity ≥ ndim → full           (3^ndim − 1 offsets)

    Examples (2-D)::

        face : [(-1,0), (1,0), (0,-1), (0,1)]
        full : [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), …]
    """
    if connectivity < ndim:
        offsets: List[Tuple[int, ...]] = []
        for d in range(ndim):
            for s in (-1, 1):
                off = [0] * ndim
                off[d] = s
                offsets.append(tuple(off))
        return offsets
    zero = (0,) * ndim
    return [o for o in product((-1, 0, 1), repeat=ndim) if o != zero]


# ── 1e. Thin inter-instance boundary ────────────────────────────────

def _has_different_fg_neighbor(
    labels: torch.Tensor,
    is_fg: torch.Tensor,
    ndim: int,
    connectivity: int,
) -> torch.Tensor:
    """True where a **foreground** pixel has a **foreground** neighbour
    carrying a different label.

    Background → foreground transitions are *ignored*, so the result is
    the thinnest possible inter-instance boundary (one pixel on each
    side of the interface).
    """
    offsets = _neighbor_offsets(ndim, connectivity)
    boundary = torch.zeros_like(labels, dtype=torch.bool)

    for offset in offsets:
        nbr_lbl = labels
        nbr_fg = is_fg
        for d, s in enumerate(offset):
            if s != 0:
                nbr_lbl = _shift(nbr_lbl, d, s)
                nbr_fg = _shift(nbr_fg, d, s)
        boundary |= is_fg & nbr_fg & (labels != nbr_lbl)

    return boundary


# ── 1f. Public API ───────────────────────────────────────────────────

def find_boundaries(
    labels: torch.Tensor,
    connectivity: int = 1,
    mode: str = "inner",
    background: int = 0,
) -> torch.Tensor:
    """Return bool tensor where boundaries between labeled regions are True.

    Pure-torch reimplementation of ``skimage.segmentation.find_boundaries``.
    Works on 2-D ``[H, W]`` and 3-D ``[D, H, W]`` labels alike (an
    optional leading unit channel ``[1, *spatial]`` is stripped then
    restored).

    Modes
    -----
    thick
        ``dilation(labels) ≠ erosion(labels)`` — every pixel whose
        neighbourhood contains a different label.  Two pixels wide.
    inner *(default)*
        Foreground pixels with a differently-labelled **foreground**
        neighbour.  Excludes the foreground / background interface →
        thinnest possible inter-instance boundary.
    outer
        Background pixels adjacent to objects, plus the foreground
        pixels where two different objects touch (``skimage`` convention).

    Args:
        labels: ``[*spatial]`` or ``[1, *spatial]`` integer labels.
        connectivity: ``1`` (face-adjacent, default) or ``ndim`` (full).
        mode: ``"thick"``, ``"inner"``, or ``"outer"``.
        background: Background label value (default ``0``).

    Returns:
        Bool tensor, same shape as *labels*.
    """
    # ---- strip optional channel dim ----
    has_channel = labels.dim() in (3, 4) and labels.shape[0] == 1
    work = rearrange(labels, "1 ... -> ...") if has_channel else labels
    ndim = work.dim()
    if ndim not in (2, 3):
        raise ValueError(f"Expected 2-D or 3-D spatial labels, got {ndim}-D")

    lbl_f = work.float()
    is_fg = work != background

    # ---- mode dispatch (identical structure for 2D and 3D) ----
    if mode == "thick":
        dilated, eroded = _dilate_erode(lbl_f, ndim, connectivity)
        boundaries = dilated != eroded

    elif mode == "inner":
        boundaries = _has_different_fg_neighbor(work, is_fg, ndim, connectivity)

    elif mode == "outer":
        dilated, eroded = _dilate_erode(lbl_f, ndim, connectivity)
        thick = dilated != eroded
        is_bg = ~is_fg
        # full connectivity for adjacent-object detection (skimage convention)
        dil_full, _ = _dilate_erode(lbl_f, ndim, ndim)
        inv = lbl_f.clone()
        inv[is_bg] = float(work.max().item()) + 1
        _, ero_inv = _dilate_erode(inv, ndim, ndim)
        adjacent_objs = (dil_full != ero_inv) & is_fg
        boundaries = thick & (is_bg | adjacent_objs)

    else:
        raise ValueError(
            f"mode must be 'thick', 'inner', or 'outer', got '{mode}'"
        )

    # ---- restore channel dim ----
    return rearrange(boundaries, "... -> 1 ...") if has_channel else boundaries


# ═══════════════════════════════════════════════════════════════════════
# §2  Sequential relabelling
# ═══════════════════════════════════════════════════════════════════════

def relabel_sequential(
    labels: torch.Tensor,
    start_label: int = 1,
) -> torch.Tensor:
    """Map foreground labels to consecutive integers starting at *start_label*.

    Background (``0``) is preserved.  Negative values pass through unchanged.
    """
    device, dtype = labels.device, labels.dtype
    fg = torch.unique(labels)
    fg = fg[fg > 0]
    if len(fg) == 0:
        return labels.clone()

    max_id = int(labels.max().item()) + 1
    lut = torch.zeros(max_id, device=device, dtype=dtype)
    for new, old in enumerate(fg):
        lut[old.long()] = start_label + new

    safe = labels.long().clamp(0, max_id - 1)
    out = lut[safe]
    out[labels < 0] = labels[labels < 0]
    return out


# ═══════════════════════════════════════════════════════════════════════
# §3  Connected-component relabelling  (unified 2D / 3D)
# ═══════════════════════════════════════════════════════════════════════

# ── 3a. Scipy worker (CPU — runs inside pmap subprocesses) ───────────

def _cc_worker(args):
    """Per-instance CC labelling via ``scipy.ndimage.label``.

    Identical logic for 2-D and 3-D — the dimensionality is encoded in
    the *structure* element passed through *args*.

    Args:
        args: ``(labels_np, old_label, structure)``

    Returns:
        ``(old_label, labeled_mask, num_features)``
    """
    from scipy import ndimage
    labels_np, old_label, structure = args
    labeled_mask, num_features = ndimage.label(
        labels_np == old_label, structure=structure,
    )
    return (old_label, labeled_mask, num_features)


# ── 3b. GPU path (cupy + DLPack zero-copy) ──────────────────────────

def _relabel_cc_gpu(
    labels_t: torch.Tensor,
    spatial_dims: int,
    connectivity: int,
) -> torch.Tensor:
    """Relabel via cupy connected components — zero-copy for CUDA tensors."""
    import cupy as cp
    from neurons.utils.gpu_ndimage import (
        cupy_label, cupy_gen_struct, torch_to_cupy, cupy_to_torch,
    )

    labels_cp = torch_to_cupy(labels_t.long().contiguous())
    structure_cp = cupy_gen_struct(
        spatial_dims, 1 if connectivity <= spatial_dims else connectivity,
    )

    fg_ids = cp.unique(labels_cp)
    fg_ids = fg_ids[fg_ids > 0]
    relabeled = cp.zeros_like(labels_cp, dtype=cp.int64)
    next_id = 1

    for old in fg_ids:
        cc_map, n = cupy_label(labels_cp == old, structure_cp)
        for i in range(1, int(n) + 1):
            relabeled[cc_map == i] = next_id
            next_id += 1

    return cupy_to_torch(relabeled, device=labels_t.device)


# ── 3c. Connectivity → scipy structure rank mapping ──────────────────

_CONN_TO_RANK: Dict[int, int] = {
    # 2-D
    4: 1, 8: 2,
    # 3-D
    6: 1, 18: 2, 26: 3,
}

_DEFAULT_CONN: Dict[int, int] = {2: 4, 3: 6}


# ── 3d. Unified public function ──────────────────────────────────────

def relabel_connected_components(
    labels: torch.Tensor,
    spatial_dims: int = 3,
    connectivity: Optional[int] = None,
) -> torch.Tensor:
    """Relabel by finding per-instance connected components.

    Works identically for 2-D and 3-D inputs.

    Dispatch order:

    1. **GPU** — ``cupy.ndimage.label`` via DLPack zero-copy.
    2. **Main-process CPU** — ``scipy.ndimage.label`` per instance via
       ``pmap`` (parallel).
    3. **DataLoader-worker CPU** — sequential scipy (safe after ``fork``).

    Args:
        labels: ``[*spatial]`` or ``[batch, *spatial]`` integer labels.
        spatial_dims: ``2`` or ``3``.
        connectivity: Scipy-style (default ``4`` for 2-D, ``6`` for 3-D).

    Returns:
        Relabelled tensor, same shape and dtype as *labels*.
    """
    if connectivity is None:
        connectivity = _DEFAULT_CONN[spatial_dims]

    # ---- recurse over optional batch dimension ----
    if labels.dim() == spatial_dims + 1:
        return torch.stack([
            relabel_connected_components(labels[b], spatial_dims, connectivity)
            for b in range(labels.shape[0])
        ])

    # ---- GPU path ----
    from neurons.utils.gpu_ndimage import _use_gpu

    if _use_gpu():
        return _relabel_cc_gpu(labels, spatial_dims, connectivity).to(
            dtype=labels.dtype,
        )

    # ---- CPU path ----
    device = labels.device
    labels_np = labels.cpu().numpy().astype(np.int64)

    from scipy import ndimage

    struct_rank = _CONN_TO_RANK.get(connectivity, connectivity)
    structure = ndimage.generate_binary_structure(spatial_dims, struct_rank)

    fg_ids = np.unique(labels_np)
    fg_ids = fg_ids[fg_ids > 0]
    if len(fg_ids) == 0:
        return labels.clone()

    import os
    from neurons.utils.gpu_ndimage import _MAIN_PID

    in_worker = _MAIN_PID is not None and os.getpid() != _MAIN_PID
    args = [(labels_np, int(uid), structure) for uid in fg_ids]

    if in_worker:
        results = [_cc_worker(a) for a in args]
    else:
        from neurons.utils.parallel import pmap
        results = pmap(_cc_worker, args)

    relabeled = np.zeros(labels_np.shape, dtype=np.int64)
    next_id = 1
    for _, cc_map, n in results:
        for i in range(1, n + 1):
            relabeled[cc_map == i] = next_id
            next_id += 1

    return torch.from_numpy(relabeled).to(device=device, dtype=labels.dtype)


# ── 3e. Convenience wrappers (backward-compatible) ───────────────────

def relabel_connected_components_3d(
    labels: torch.Tensor, connectivity: int = 6,
) -> torch.Tensor:
    """Relabel 3-D labels by connected components.  See ``relabel_connected_components``."""
    return relabel_connected_components(labels, spatial_dims=3, connectivity=connectivity)


def relabel_connected_components_2d(
    labels: torch.Tensor, connectivity: int = 4,
) -> torch.Tensor:
    """Relabel 2-D labels by connected components.  See ``relabel_connected_components``."""
    return relabel_connected_components(labels, spatial_dims=2, connectivity=connectivity)


def relabel_after_crop(
    labels: torch.Tensor,
    spatial_dims: int = 3,
    connectivity: Optional[int] = None,
) -> torch.Tensor:
    """Relabel instance labels after cropping.

    After cropping, instances may be split into disconnected fragments.
    This function finds connected components and renumbers them.
    """
    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
    return relabel_connected_components(labels, spatial_dims, connectivity)


# ═══════════════════════════════════════════════════════════════════════
# §4  Embedding clustering
# ═══════════════════════════════════════════════════════════════════════

# ── 4a. Sklearn MeanShift (CPU, evaluation only) ─────────────────────

def _cluster_with_sklearn(
    emb_fg: np.ndarray,
    bandwidth: float,
    min_cluster_size: int,
) -> np.ndarray:
    """Cluster foreground embeddings using sklearn MeanShift (CPU)."""
    try:
        from sklearn.cluster import MeanShift
    except ImportError:
        return np.ones(len(emb_fg), dtype=np.int64)

    try:
        clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        labels_fg = clusterer.fit_predict(emb_fg) + 1
    except ValueError:
        labels_fg = np.ones(len(emb_fg), dtype=np.int64)

    unique_labels, counts = np.unique(labels_fg, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if label > 0 and count < min_cluster_size:
            labels_fg[labels_fg == label] = 0

    remaining = np.unique(labels_fg)
    remaining = remaining[remaining > 0]
    label_map = {int(old): new + 1 for new, old in enumerate(remaining)}
    label_map[0] = 0
    labels_fg = np.array(
        [label_map.get(int(l), 0) for l in labels_fg], dtype=np.int64,
    )
    return labels_fg


def cluster_embeddings_meanshift(
    embedding: torch.Tensor,
    foreground_mask: Optional[torch.Tensor] = None,
    bandwidth: float = 0.5,
    min_cluster_size: int = 50,
) -> torch.Tensor:
    """Cluster pixel embeddings via sklearn MeanShift.

    **Warning** — O(N²); use only for evaluation, not during training.

    Args:
        embedding: ``[E, *spatial]`` embedding tensor (2-D or 3-D).
        foreground_mask: Optional ``[*spatial]`` bool mask.

    Returns:
        ``[*spatial]`` integer cluster labels.
    """
    device = embedding.device
    spatial_shape = embedding.shape[1:]                   # works for 2D and 3D

    # einops "e ... -> (...) e" flattens all spatial dims regardless of count
    emb_flat = rearrange(embedding, "e ... -> (...) e")

    if foreground_mask is not None:
        fg_flat = rearrange(foreground_mask, "... -> (...)") > 0
    else:
        fg_flat = torch.ones(emb_flat.shape[0], dtype=torch.bool, device=device)

    fg_idx = torch.where(fg_flat)[0]
    if len(fg_idx) == 0:
        return torch.zeros(spatial_shape, device=device, dtype=torch.long)

    labels_fg = _cluster_with_sklearn(
        emb_flat[fg_idx].cpu().numpy(), bandwidth, min_cluster_size,
    )
    labels_fg = torch.from_numpy(labels_fg).to(device=device, dtype=torch.long)

    labels_full = torch.zeros(emb_flat.shape[0], device=device, dtype=torch.long)
    labels_full[fg_idx] = labels_fg
    return labels_full.view(spatial_shape)


# ── 4b. Differentiable clustering wrappers ───────────────────────────

def cluster_embeddings_soft(
    embedding: torch.Tensor,
    foreground_mask: Optional[torch.Tensor] = None,
    bandwidth: float = 0.5,
    num_iters: int = 10,
    temperature: float = 1.0,
    min_cluster_size: int = 50,
) -> torch.Tensor:
    """Cluster pixel embeddings using differentiable soft mean-shift."""
    from neurons.inference.soft_clustering import SoftMeanShift

    batched = embedding.dim() >= 4
    if not batched:
        embedding = rearrange(embedding, "... -> 1 ...")
        if foreground_mask is not None:
            foreground_mask = rearrange(foreground_mask, "... -> 1 ...")

    clusterer = SoftMeanShift(
        bandwidth=bandwidth,
        num_iters=num_iters,
        temperature=temperature,
        min_cluster_size=min_cluster_size,
    )
    labels, _, _ = clusterer(embedding, foreground_mask)

    if not batched:
        labels = rearrange(labels, "1 ... -> ...")
    return labels


def cluster_offsets_hough(
    offsets: torch.Tensor,
    foreground_mask: Optional[torch.Tensor] = None,
    bin_size: float = 2.0,
    sigma: float = 2.0,
    threshold: float = 0.3,
    min_votes: int = 50,
) -> torch.Tensor:
    """Cluster via Hough voting on predicted spatial offsets."""
    from neurons.inference.soft_clustering import HoughVoting

    batched = offsets.dim() >= 4
    if not batched:
        offsets = rearrange(offsets, "... -> 1 ...")
        if foreground_mask is not None:
            foreground_mask = rearrange(foreground_mask, "... -> 1 ...")

    voter = HoughVoting(
        bin_size=bin_size,
        sigma=sigma,
        threshold=threshold,
        min_votes=min_votes,
    )
    labels = voter(offsets, foreground_mask)

    if not batched:
        labels = rearrange(labels, "1 ... -> ...")
    return labels

"""
Label utilities for connectomics segmentation.

Organised in three sections:

1. **Sequential relabelling** — ``relabel_sequential``
   (wraps ``skimage.segmentation.relabel_sequential``).
2. **Connected-component relabelling** — ``relabel_connected_components``
   (wraps ``skimage.measure.label``).  ``relabel_after_crop`` is a
   convenience wrapper.
3. **Embedding clustering** — ``cluster_embeddings_meanshift``,
   ``cluster_embeddings_soft``, ``cluster_offsets_hough``.

All public functions work identically on 2-D and 3-D inputs.
Metrics live in ``neurons.metrics``.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from skimage.measure import label as _skimage_label
from skimage.segmentation import relabel_sequential as _skimage_relabel_sequential


# ═══════════════════════════════════════════════════════════════════════
# §1  Sequential relabelling
# ═══════════════════════════════════════════════════════════════════════

def relabel_sequential(
    labels: torch.Tensor,
    start_label: int = 1,
) -> torch.Tensor:
    """Map foreground labels to consecutive integers starting at *start_label*.

    Wraps ``skimage.segmentation.relabel_sequential``.
    Background (``0``) is preserved.  Negative values pass through unchanged.
    """
    device, dtype = labels.device, labels.dtype
    labels_np = labels.cpu().numpy()

    neg_mask = labels_np < 0
    safe_np = labels_np.copy()
    safe_np[neg_mask] = 0

    relabeled_np, _, _ = _skimage_relabel_sequential(
        safe_np.astype(np.intp), offset=start_label,
    )
    relabeled_np = relabeled_np.astype(labels_np.dtype)
    relabeled_np[neg_mask] = labels_np[neg_mask]

    return torch.from_numpy(relabeled_np).to(device=device, dtype=dtype)


# ═══════════════════════════════════════════════════════════════════════
# §2  Connected-component relabelling  (unified 2D / 3D)
# ═══════════════════════════════════════════════════════════════════════

_CONN_TO_SK: Dict[int, int] = {
    # scipy-style → skimage connectivity
    4: 1, 8: 2,       # 2-D
    6: 1, 18: 2, 26: 3,  # 3-D
}

_DEFAULT_CONN: Dict[int, int] = {2: 4, 3: 6}


def relabel_connected_components(
    labels: torch.Tensor,
    spatial_dims: int = 3,
    connectivity: Optional[int] = None,
) -> torch.Tensor:
    """Relabel by finding per-instance connected components.

    Wraps ``skimage.measure.label``.  Two pixels are in the same
    component when they are neighbours **and** share the same value.

    Args:
        labels: ``[*spatial]`` or ``[batch, *spatial]`` integer labels.
        spatial_dims: ``2`` or ``3``.
        connectivity: Scipy-style (default ``4`` for 2-D, ``6`` for 3-D).

    Returns:
        Relabelled tensor, same shape and dtype as *labels*.
    """
    if connectivity is None:
        connectivity = _DEFAULT_CONN[spatial_dims]

    if labels.dim() == spatial_dims + 1:
        return torch.stack([
            relabel_connected_components(labels[b], spatial_dims, connectivity)
            for b in range(labels.shape[0])
        ])

    device, dtype = labels.device, labels.dtype
    labels_np = labels.cpu().numpy()
    sk_conn = _CONN_TO_SK.get(connectivity, connectivity)

    relabeled_np = _skimage_label(labels_np, background=0, connectivity=sk_conn)
    return torch.from_numpy(relabeled_np).to(device=device, dtype=dtype)


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

    Wraps ``skimage.measure.label``.  After cropping, instances may be
    split into disconnected fragments.  This function assigns a unique
    ID to each connected component.
    """
    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
    return relabel_connected_components(labels, spatial_dims, connectivity)


# ═══════════════════════════════════════════════════════════════════════
# §3  Embedding clustering
# ═══════════════════════════════════════════════════════════════════════

# ── 3a. Sklearn MeanShift (CPU, evaluation only) ─────────────────────

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


# ── 3b. Differentiable clustering wrappers ───────────────────────────

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

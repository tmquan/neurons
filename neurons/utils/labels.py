"""
Label utilities for connectomics segmentation.

Provides functions for:
- Relabeling/reindexing instance labels after cropping
- Connected component relabeling
- Embedding clustering for instance prediction

Metrics live in ``neurons.metrics``.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from einops import rearrange


def find_boundaries(
    labels: torch.Tensor,
    connectivity: int = 1,
    mode: str = "inner",
    background: int = 0,
) -> torch.Tensor:
    """Return bool tensor where boundaries between labeled regions are True.

    Pure-torch reimplementation of ``skimage.segmentation.find_boundaries``.
    Supports 2-D and 3-D label tensors.

    Args:
        labels: Integer label tensor [*spatial] or [C, *spatial].
        connectivity: 1 = face-adjacent (thin), ``labels.ndim`` = include
            corners (thick).  Maps to the structuring element radius.
        mode: ``"thick"`` — any pixel not fully surrounded by same label.
            ``"inner"`` — boundary pixels inside foreground only.
            ``"outer"`` — boundary pixels in background around objects;
            also marks where two objects touch.
        background: Label value treated as background (default 0).

    Returns:
        Bool tensor same shape as labels.
    """
    import torch.nn.functional as F

    has_channel = labels.dim() == 4 and labels.shape[0] == 1
    spatial_dims = labels.dim() - (1 if has_channel else 0)

    if has_channel:
        lbl = rearrange(labels.float(), "c ... -> 1 c ...")
    else:
        lbl = rearrange(labels.float(), "... -> 1 1 ...")

    if spatial_dims == 3:
        pool_fn = F.max_pool3d
        ks = 3
        pad = (1, 1, 1, 1, 1, 1)
    else:
        pool_fn = F.max_pool2d
        ks = 3
        pad = (1, 1, 1, 1)

    if connectivity < spatial_dims:
        # Face-adjacent only: use per-axis shift-and-compare
        if has_channel:
            core = labels
            dims = list(range(1, spatial_dims + 1))
        else:
            core = labels
            dims = list(range(spatial_dims))

        dilated = core.clone()
        eroded = core.clone()
        for d in dims:
            fwd = torch.roll(core, shifts=-1, dims=d)
            bwd = torch.roll(core, shifts=1, dims=d)
            slc_last = [slice(None)] * core.dim()
            slc_first = [slice(None)] * core.dim()
            slc_last[d] = slice(-1, None)
            slc_first[d] = slice(0, 1)
            fwd[tuple(slc_last)] = core[tuple(slc_last)]
            bwd[tuple(slc_first)] = core[tuple(slc_first)]
            dilated = torch.max(dilated, torch.max(fwd, bwd))
            eroded = torch.min(eroded, torch.min(fwd, bwd))
    else:
        # Full connectivity: use max_pool / min_pool (3x3x3 kernel)
        padded = F.pad(lbl, pad, mode="replicate")
        dilated_t = pool_fn(padded, kernel_size=ks, stride=1, padding=0)
        eroded_t = pool_fn(-padded, kernel_size=ks, stride=1, padding=0).neg_()

        if has_channel:
            dilated = rearrange(dilated_t, "1 c ... -> c ...")
            eroded = rearrange(eroded_t, "1 c ... -> c ...")
        else:
            dilated = rearrange(dilated_t, "1 1 ... -> ...")
            eroded = rearrange(eroded_t, "1 1 ... -> ...")

    boundaries = dilated != eroded

    if mode == "inner":
        boundaries = boundaries & (labels != background)
    elif mode == "outer":
        is_bg = labels == background
        # Where two different objects touch (via full connectivity)
        padded_full = F.pad(lbl, pad, mode="replicate")
        if spatial_dims == 3:
            dil_full = F.max_pool3d(padded_full, kernel_size=3, stride=1, padding=0)
            inv_bg = lbl.clone()
            inv_bg[lbl == background] = float(labels.max().item()) + 1
            padded_inv = F.pad(inv_bg, pad, mode="replicate")
            ero_inv = F.max_pool3d(-padded_inv, kernel_size=3, stride=1, padding=0).neg_()
        else:
            dil_full = F.max_pool2d(padded_full, kernel_size=3, stride=1, padding=0)
            inv_bg = lbl.clone()
            inv_bg[lbl == background] = float(labels.max().item()) + 1
            padded_inv = F.pad(inv_bg, pad, mode="replicate")
            ero_inv = F.max_pool2d(-padded_inv, kernel_size=3, stride=1, padding=0).neg_()

        if has_channel:
            adj = rearrange(dil_full != ero_inv, "1 c ... -> c ...") & ~is_bg
        else:
            adj = rearrange(dil_full != ero_inv, "1 1 ... -> ...") & ~is_bg
        boundaries = boundaries & (is_bg | adj)

    return boundaries


def relabel_sequential(
    labels: torch.Tensor,
    start_label: int = 1,
) -> torch.Tensor:
    """
    Relabel instance labels to be sequential starting from start_label.

    Background (0) is preserved. All other unique labels are mapped to
    consecutive integers starting from start_label.

    Args:
        labels: Instance labels tensor of any shape.
        start_label: Starting label for foreground instances (default: 1).

    Returns:
        Relabeled tensor with sequential labels.

    Example:
        >>> labels = torch.tensor([0, 5, 0, 5, 12, 12, 0])
        >>> relabel_sequential(labels)
        tensor([0, 1, 0, 1, 2, 2, 0])
    """
    device = labels.device
    dtype = labels.dtype

    unique_labels = torch.unique(labels)
    fg_labels = unique_labels[unique_labels > 0]

    if len(fg_labels) == 0:
        return labels.clone()

    max_label = int(labels.max().item()) + 1
    label_map = torch.zeros(max_label, device=device, dtype=dtype)

    for new_idx, old_label in enumerate(fg_labels):
        label_map[old_label.long()] = start_label + new_idx

    safe = labels.long().clamp(0, max_label - 1)
    relabeled = label_map[safe]
    relabeled[labels < 0] = labels[labels < 0]
    return relabeled


def relabel_connected_components_3d(
    labels: torch.Tensor,
    connectivity: int = 6,
) -> torch.Tensor:
    """
    Relabel 3D volume by finding connected components.

    After cropping, a single instance label might represent multiple
    disconnected components. This function assigns unique labels to
    each connected component.

    Args:
        labels: 3D label volume [D, H, W] or [B, D, H, W].
        connectivity: Connectivity for finding components (6, 18, or 26).

    Returns:
        Relabeled volume with unique labels for each connected component.
    """
    if labels.dim() == 4:
        batch_results = []
        for b in range(labels.shape[0]):
            result = relabel_connected_components_3d(labels[b], connectivity)
            batch_results.append(result)
        return torch.stack(batch_results)

    device = labels.device
    labels_np = labels.cpu().numpy().astype(np.int64)

    try:
        from scipy import ndimage

        unique_labels = np.unique(labels_np)
        unique_labels = unique_labels[unique_labels > 0]

        relabeled = np.zeros(labels_np.shape, dtype=np.int64)
        next_label = 1

        if connectivity == 6:
            structure = ndimage.generate_binary_structure(3, 1)
        elif connectivity == 18:
            structure = ndimage.generate_binary_structure(3, 2)
        else:
            structure = ndimage.generate_binary_structure(3, 3)

        for old_label in unique_labels:
            mask = labels_np == old_label
            labeled_mask, num_features = ndimage.label(mask, structure=structure)

            for i in range(1, num_features + 1):
                relabeled[labeled_mask == i] = next_label
                next_label += 1

        return torch.from_numpy(relabeled).to(device=device, dtype=labels.dtype)

    except ImportError:
        return relabel_sequential(labels)


def relabel_connected_components_2d(
    labels: torch.Tensor,
    connectivity: int = 4,
) -> torch.Tensor:
    """
    Relabel 2D image by finding connected components.

    Args:
        labels: 2D label image [H, W] or [B, H, W].
        connectivity: Connectivity for finding components (4 or 8).

    Returns:
        Relabeled image with unique labels for each connected component.
    """
    if labels.dim() == 3:
        batch_results = []
        for b in range(labels.shape[0]):
            result = relabel_connected_components_2d(labels[b], connectivity)
            batch_results.append(result)
        return torch.stack(batch_results)

    device = labels.device
    labels_np = labels.cpu().numpy().astype(np.int64)

    try:
        from scipy import ndimage

        unique_labels = np.unique(labels_np)
        unique_labels = unique_labels[unique_labels > 0]

        relabeled = np.zeros(labels_np.shape, dtype=np.int64)
        next_label = 1

        if connectivity == 4:
            structure = ndimage.generate_binary_structure(2, 1)
        else:
            structure = ndimage.generate_binary_structure(2, 2)

        for old_label in unique_labels:
            mask = labels_np == old_label
            labeled_mask, num_features = ndimage.label(mask, structure=structure)

            for i in range(1, num_features + 1):
                relabeled[labeled_mask == i] = next_label
                next_label += 1

        return torch.from_numpy(relabeled).to(device=device, dtype=labels.dtype)

    except ImportError:
        return relabel_sequential(labels)


def relabel_after_crop(
    labels: torch.Tensor,
    spatial_dims: int = 3,
    connectivity: Optional[int] = None,
) -> torch.Tensor:
    """
    Relabel instance labels after cropping.

    After cropping a volume/image, some instances may be split into
    disconnected components, or some may be entirely removed. This
    function:
    1. Finds connected components (to separate split instances)
    2. Relabels sequentially (to have consecutive IDs)

    Args:
        labels: Label tensor [D, H, W], [B, D, H, W], [H, W], or [B, H, W].
        spatial_dims: Number of spatial dimensions (2 or 3).
        connectivity: Connectivity for component detection
            (default: 6 for 3D, 4 for 2D).

    Returns:
        Relabeled tensor with sequential unique labels per component.
    """
    if spatial_dims == 3:
        if connectivity is None:
            connectivity = 6
        return relabel_connected_components_3d(labels, connectivity)
    elif spatial_dims == 2:
        if connectivity is None:
            connectivity = 4
        return relabel_connected_components_2d(labels, connectivity)
    else:
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")


def cluster_embeddings_meanshift(
    embedding: torch.Tensor,
    foreground_mask: Optional[torch.Tensor] = None,
    bandwidth: float = 0.5,
    min_cluster_size: int = 50,
) -> torch.Tensor:
    """
    Cluster pixel embeddings using mean-shift clustering.

    WARNING: This is slow for large volumes. Only use for evaluation,
    not during training.

    Args:
        embedding: Pixel embeddings [E, D, H, W] or [E, H, W].
        foreground_mask: Binary mask, same spatial shape as embedding.
        bandwidth: Mean-shift bandwidth (related to delta_var).
        min_cluster_size: Minimum pixels per cluster.

    Returns:
        Instance labels with same spatial shape as embedding.
    """
    device = embedding.device
    is_3d = embedding.dim() == 4

    if is_3d:
        E, D, H, W = embedding.shape
        emb_flat = rearrange(embedding, "e d h w -> (d h w) e")
        spatial_shape = (D, H, W)
    else:
        E, H, W = embedding.shape
        emb_flat = rearrange(embedding, "e h w -> (h w) e")
        spatial_shape = (H, W)

    if foreground_mask is not None:
        fg_flat = rearrange(foreground_mask, "... -> (...)") > 0
    else:
        fg_flat = torch.ones(emb_flat.shape[0], dtype=torch.bool, device=device)

    fg_indices = torch.where(fg_flat)[0]

    if len(fg_indices) == 0:
        return torch.zeros(spatial_shape, device=device, dtype=torch.long)

    emb_fg = emb_flat[fg_indices]

    labels_fg = _cluster_with_sklearn(
        emb_fg.cpu().numpy(), bandwidth, min_cluster_size
    )
    labels_fg = torch.from_numpy(labels_fg).to(device=device, dtype=torch.long)

    labels_full = torch.zeros(emb_flat.shape[0], device=device, dtype=torch.long)
    labels_full[fg_indices] = labels_fg

    labels_out = labels_full.reshape(spatial_shape)
    return labels_out


def _cluster_with_sklearn(
    emb_fg: np.ndarray,
    bandwidth: float,
    min_cluster_size: int,
) -> np.ndarray:
    """
    Cluster embeddings using sklearn MeanShift (CPU fallback).

    Args:
        emb_fg: Foreground embeddings [N, E] as numpy array.
        bandwidth: Mean-shift bandwidth.
        min_cluster_size: Minimum cluster size.

    Returns:
        Cluster labels [N] starting from 1 (0 = filtered out).
    """
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
    labels_fg = np.array([label_map.get(int(l), 0) for l in labels_fg], dtype=np.int64)

    return labels_fg


# ---------------------------------------------------------------------------
# Differentiable clustering wrappers
# ---------------------------------------------------------------------------

def cluster_embeddings_soft(
    embedding: torch.Tensor,
    foreground_mask: Optional[torch.Tensor] = None,
    bandwidth: float = 0.5,
    num_iters: int = 10,
    temperature: float = 1.0,
    min_cluster_size: int = 50,
) -> torch.Tensor:
    """Cluster pixel embeddings using differentiable soft mean-shift.

    This is the GPU-friendly, gradient-preserving alternative to
    ``cluster_embeddings_meanshift``.

    Args:
        embedding: Pixel embeddings [E, *spatial] or [B, E, *spatial].
        foreground_mask: Binary mask, same spatial shape.
        bandwidth: Gaussian kernel bandwidth.
        num_iters: Mean-shift iterations.
        temperature: Softmax temperature (lower = harder assignments).
        min_cluster_size: Minimum pixels per cluster.

    Returns:
        Instance labels with same spatial shape.
    """
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
    """Cluster via Hough voting on predicted spatial offsets.

    Args:
        offsets: Predicted offsets [S, *spatial] or [B, S, *spatial].
        foreground_mask: Binary mask, same spatial shape.
        bin_size: Spatial bin size for vote accumulator.
        sigma: Gaussian smoothing sigma for votes.
        threshold: Relative peak threshold.
        min_votes: Minimum votes for a valid peak.

    Returns:
        Instance labels with same spatial shape.
    """
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

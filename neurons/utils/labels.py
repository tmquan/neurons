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

    relabeled = label_map[labels.long().clamp(0, max_label - 1)]
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
    labels_np = labels.cpu().numpy().astype(np.int32)

    try:
        from scipy import ndimage

        unique_labels = np.unique(labels_np)
        unique_labels = unique_labels[unique_labels > 0]

        relabeled = np.zeros_like(labels_np)
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
    labels_np = labels.cpu().numpy().astype(np.int32)

    try:
        from scipy import ndimage

        unique_labels = np.unique(labels_np)
        unique_labels = unique_labels[unique_labels > 0]

        relabeled = np.zeros_like(labels_np)
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
        fg_flat = foreground_mask.reshape(-1) > 0
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
        embedding = embedding.unsqueeze(0)
        if foreground_mask is not None:
            foreground_mask = foreground_mask.unsqueeze(0)

    clusterer = SoftMeanShift(
        bandwidth=bandwidth,
        num_iters=num_iters,
        temperature=temperature,
        min_cluster_size=min_cluster_size,
    )
    labels, _, _ = clusterer(embedding, foreground_mask)

    if not batched:
        labels = labels.squeeze(0)
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
        offsets = offsets.unsqueeze(0)
        if foreground_mask is not None:
            foreground_mask = foreground_mask.unsqueeze(0)

    voter = HoughVoting(
        bin_size=bin_size,
        sigma=sigma,
        threshold=threshold,
        min_votes=min_votes,
    )
    labels = voter(offsets, foreground_mask)

    if not batched:
        labels = labels.squeeze(0)
    return labels

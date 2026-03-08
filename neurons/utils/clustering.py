"""
Embedding clustering utilities for connectomics segmentation.

- ``cluster_embeddings_meanshift`` — sklearn MeanShift (CPU, evaluation only).
- ``cluster_embeddings_soft``      — differentiable soft mean-shift.
- ``cluster_offsets_hough``        — Hough voting on predicted spatial offsets.

All public functions work identically on 2-D and 3-D inputs.
"""

from typing import Optional

import numpy as np
import torch
from einops import rearrange


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
    spatial_shape = embedding.shape[1:]

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
    return labels_full.reshape(spatial_shape)


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

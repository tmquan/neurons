"""
Differentiable clustering modules for instance segmentation.

Two strategies:

1. **SoftMeanShift** -- temperature-scaled Gaussian kernel soft assignment
   over iteratively refined modes.  Works on high-dim embedding spaces
   produced by the instance head.

2. **HoughVoting** -- spatial binning of offset-based embeddings
   (coords + predicted offsets).  Natural companion to SkeletonEmbeddingLoss.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SoftMeanShift(nn.Module):
    """Differentiable mean-shift clustering on pixel/voxel embeddings.

    Iteratively refines K mode estimates using Gaussian-kernel weighted
    averages.  Soft assignments allow gradient flow during training;
    temperature annealing sharpens assignments for inference.

    Args:
        bandwidth: Gaussian kernel bandwidth (related to delta_var).
        num_iters: Number of mean-shift refinement iterations.
        temperature: Softmax temperature for assignment (lower = harder).
        min_cluster_size: Discard clusters smaller than this.
    """

    def __init__(
        self,
        bandwidth: float = 0.5,
        num_iters: int = 10,
        temperature: float = 1.0,
        min_cluster_size: int = 50,
    ) -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.num_iters = num_iters
        self.temperature = temperature
        self.min_cluster_size = min_cluster_size

    def _init_seeds(
        self,
        emb_flat: torch.Tensor,
        fg_mask: torch.Tensor,
        max_seeds: int = 256,
    ) -> torch.Tensor:
        """Subsample foreground embeddings as initial mode seeds.

        Args:
            emb_flat: [E, N] embeddings.
            fg_mask: [N] boolean foreground mask.
            max_seeds: Maximum number of seeds.

        Returns:
            [K, E] initial mode estimates.
        """
        fg_idx = torch.where(fg_mask)[0]
        if len(fg_idx) == 0:
            return emb_flat[:, :1].T
        n = min(max_seeds, len(fg_idx))
        perm = torch.randperm(len(fg_idx), device=fg_idx.device)[:n]
        seeds = emb_flat[:, fg_idx[perm]].T
        return seeds

    def forward(
        self,
        embedding: torch.Tensor,
        foreground_mask: Optional[torch.Tensor] = None,
        max_seeds: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cluster embeddings via differentiable mean-shift.

        Args:
            embedding: [B, E, *spatial] embedding tensor.
            foreground_mask: [B, *spatial] boolean mask (optional).
            max_seeds: Max initial seeds per sample.

        Returns:
            labels:       [B, *spatial] integer instance labels (0 = bg).
            soft_assign:  [B, K, *spatial] soft assignment probabilities.
            centers:      [B, K, E] final cluster centres.
        """
        B = embedding.shape[0]
        E = embedding.shape[1]
        spatial_shape = embedding.shape[2:]
        device = embedding.device

        emb_flat = rearrange(embedding, "b e ... -> b e (...)")
        N = emb_flat.shape[2]

        if foreground_mask is not None:
            fg_flat = rearrange(foreground_mask, "b ... -> b (...)") > 0
        else:
            fg_flat = torch.ones(B, N, device=device, dtype=torch.bool)

        all_labels = []
        all_soft = []
        all_centers = []

        for b in range(B):
            fg_b = fg_flat[b]
            emb_b = emb_flat[b]

            if fg_b.sum() == 0:
                all_labels.append(torch.zeros(N, device=device, dtype=torch.long))
                all_soft.append(torch.zeros(1, N, device=device))
                all_centers.append(torch.zeros(1, E, device=device))
                continue

            modes = self._init_seeds(emb_b, fg_b, max_seeds)
            K = modes.shape[0]

            for _ in range(self.num_iters):
                emb_fg = emb_b[:, fg_b]                           # [E, M]
                diff = (rearrange(emb_fg, "e n -> 1 e n")
                        - rearrange(modes, "k e -> k e 1"))       # [K, E, M]
                sq_dist = (diff ** 2).sum(dim=1)                   # [K, M]
                weights = torch.exp(-sq_dist / (2 * self.bandwidth ** 2))
                weights_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
                modes = (rearrange(weights, "k n -> k 1 n")
                         * rearrange(emb_fg, "e n -> 1 e n")).sum(dim=2) / weights_sum

                merged = self._merge_modes(modes)
                if merged.shape[0] < modes.shape[0]:
                    modes = merged
                    K = modes.shape[0]

            diff_all = (rearrange(emb_b, "e n -> 1 e n")
                        - rearrange(modes, "k e -> k e 1"))       # [K, E, N]
            sq_dist_all = (diff_all ** 2).sum(dim=1)
            logits = -sq_dist_all / (2 * self.bandwidth ** 2 * self.temperature)
            soft = F.softmax(logits, dim=0)

            hard = soft.argmax(dim=0) + 1
            hard[~fg_b] = 0

            hard = self._filter_small_clusters(hard, K)

            all_labels.append(hard)
            all_soft.append(soft)
            all_centers.append(modes)

        labels = torch.stack(all_labels).reshape(B, *spatial_shape)

        max_K = max(s.shape[0] for s in all_soft)
        padded_soft = []
        padded_centers = []
        for s, c in zip(all_soft, all_centers):
            k = s.shape[0]
            if k < max_K:
                s = F.pad(s, (0, 0, 0, max_K - k))
                c = F.pad(c, (0, 0, 0, max_K - k))
            padded_soft.append(s)
            padded_centers.append(c)

        soft_assign = torch.stack(padded_soft).reshape(B, max_K, *spatial_shape)
        centers = torch.stack(padded_centers)

        return labels, soft_assign, centers

    def _merge_modes(
        self, modes: torch.Tensor, factor: float = 0.5,
    ) -> torch.Tensor:
        """Merge modes closer than factor * bandwidth."""
        if modes.shape[0] <= 1:
            return modes
        pw = torch.cdist(modes, modes)
        threshold = self.bandwidth * factor
        K = modes.shape[0]
        keep = torch.ones(K, device=modes.device, dtype=torch.bool)
        for i in range(K):
            if not keep[i]:
                continue
            for j in range(i + 1, K):
                if keep[j] and pw[i, j] < threshold:
                    keep[j] = False
        return modes[keep]

    def _filter_small_clusters(
        self, labels: torch.Tensor, K: int,
    ) -> torch.Tensor:
        """Remove clusters smaller than min_cluster_size."""
        for uid in range(1, K + 1):
            mask = labels == uid
            if mask.sum() < self.min_cluster_size:
                labels[mask] = 0
        return labels


class HoughVoting(nn.Module):
    """Differentiable Hough voting for offset-based instance segmentation.

    Each foreground pixel votes for a spatial location by adding its
    predicted offset to its coordinate.  Votes are accumulated into a
    smooth vote map via Gaussian splatting, then peaks are detected as
    instance centres.

    Args:
        bin_size: Spatial bin size for the vote accumulator.
        sigma: Gaussian sigma for vote splatting (in voxels).
        threshold: Relative peak threshold (fraction of max vote).
        min_votes: Minimum votes for a valid peak.
    """

    def __init__(
        self,
        bin_size: float = 2.0,
        sigma: float = 2.0,
        threshold: float = 0.3,
        min_votes: int = 50,
    ) -> None:
        super().__init__()
        self.bin_size = bin_size
        self.sigma = sigma
        self.threshold = threshold
        self.min_votes = min_votes

    @staticmethod
    def _make_coords(spatial_shape, device):
        """Build [S, *spatial] coordinate grid."""
        ranges = [torch.arange(s, device=device, dtype=torch.float32)
                  for s in spatial_shape]
        grids = torch.meshgrid(*ranges, indexing="ij")
        return torch.stack(list(reversed(grids)), dim=0)

    def forward(
        self,
        offsets: torch.Tensor,
        foreground_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cluster via Hough voting on predicted offsets.

        Args:
            offsets: [B, S, *spatial] predicted spatial offsets.
            foreground_mask: [B, *spatial] boolean mask (optional).

        Returns:
            labels: [B, *spatial] integer instance labels (0 = bg).
        """
        B, S = offsets.shape[:2]
        spatial_shape = offsets.shape[2:]
        device = offsets.device

        coords = self._make_coords(spatial_shape, device)
        coords = rearrange(coords, "s ... -> 1 s ...").expand(B, -1, *spatial_shape)
        votes = coords + offsets

        if foreground_mask is None:
            foreground_mask = torch.ones(B, *spatial_shape, device=device, dtype=torch.bool)

        all_labels = []
        for b in range(B):
            fg = foreground_mask[b]
            vote_flat = rearrange(votes[b], "s ... -> s (...)")[
                :, rearrange(fg, "... -> (...)")]

            if vote_flat.shape[1] == 0:
                all_labels.append(torch.zeros(spatial_shape, device=device, dtype=torch.long))
                continue

            bins = (vote_flat / self.bin_size).round().long()

            bin_min = bins.min(dim=1).values
            bins_shifted = bins - rearrange(bin_min, "s -> s 1")
            bin_max = bins_shifted.max(dim=1).values + 1

            acc_shape = tuple(bin_max.tolist())
            accumulator = torch.zeros(acc_shape, device=device, dtype=torch.float32)

            if S == 2:
                accumulator[bins_shifted[0], bins_shifted[1]] += 1.0
            elif S == 3:
                accumulator[bins_shifted[0], bins_shifted[1], bins_shifted[2]] += 1.0

            if self.sigma > 0:
                k = int(3 * self.sigma) * 2 + 1
                if S == 2:
                    acc_4d = rearrange(accumulator, "h w -> 1 1 h w")
                    kernel = torch.ones(1, 1, k, k, device=device) / (k * k)
                    smoothed = F.conv2d(acc_4d, kernel, padding=k // 2)
                    accumulator = rearrange(smoothed, "1 1 h w -> h w")
                elif S == 3:
                    acc_5d = rearrange(accumulator, "d h w -> 1 1 d h w")
                    kernel = torch.ones(1, 1, k, k, k, device=device) / (k ** 3)
                    smoothed = F.conv3d(acc_5d, kernel, padding=k // 2)
                    accumulator = rearrange(smoothed, "1 1 d h w -> d h w")

            peak_threshold = accumulator.max() * self.threshold
            peaks_mask = accumulator >= max(peak_threshold, self.min_votes)

            if peaks_mask.sum() == 0:
                all_labels.append(torch.zeros(spatial_shape, device=device, dtype=torch.long))
                continue

            peak_coords = torch.nonzero(peaks_mask, as_tuple=False).float()

            fg_indices = torch.where(rearrange(fg, "... -> (...)"))[0]
            fg_bins = bins_shifted.float().T

            dists = torch.cdist(fg_bins, peak_coords)
            nearest = dists.argmin(dim=1) + 1

            label_flat = torch.zeros(fg.numel(), device=device, dtype=torch.long)
            label_flat[fg_indices] = nearest
            all_labels.append(label_flat.reshape(spatial_shape))

        return torch.stack(all_labels)

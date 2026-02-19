"""
Embedding-aware stitcher for reconciling instance IDs across patches.

After sliding window inference + per-patch clustering, different patches
may assign different IDs to the same neuron.  ``EmbeddingStitcher``
resolves this by:

1. Comparing instance embeddings in overlap regions between adjacent patches
2. Merging IDs whose mean embeddings are close (Hungarian matching)
3. Splitting IDs that became disconnected after merging (connected components)
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange

from neurons.utils.labels import (
    relabel_connected_components_3d,
    relabel_sequential,
)


class _UnionFind:
    """Simple union-find (disjoint set) for label merging."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


class EmbeddingStitcher:
    """Reconcile instance labels across overlapping patches.

    Args:
        merge_threshold: Maximum embedding distance for merging two IDs.
            Should be close to ``delta_var`` used during training.
        min_instance_size: Remove instances smaller than this (voxels).
        spatial_dims: 2 or 3.
    """

    def __init__(
        self,
        merge_threshold: float = 0.5,
        min_instance_size: int = 50,
        spatial_dims: int = 3,
    ) -> None:
        self.merge_threshold = merge_threshold
        self.min_instance_size = min_instance_size
        self.spatial_dims = spatial_dims

    def stitch(
        self,
        labels: torch.Tensor,
        embeddings: torch.Tensor,
        patch_positions: List[Tuple[int, ...]],
        patch_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """Merge and split instance labels across patches.

        Args:
            labels: [*volume_shape] integer instance labels from per-patch
                clustering (each patch used local IDs; overlap regions have
                the ID from whichever patch wrote last).
            embeddings: [E, *volume_shape] embedding tensor (Gaussian-blended).
            patch_positions: List of (d, h, w) start positions.
            patch_size: (pd, ph, pw) patch dimensions.

        Returns:
            [*volume_shape] stitched instance labels with clean sequential IDs.
        """
        device = labels.device
        vol_shape = labels.shape

        max_id = int(labels.max().item())
        if max_id == 0:
            return labels

        uf = _UnionFind(max_id + 1)

        overlaps = self._find_overlapping_pairs(patch_positions, patch_size, vol_shape)

        for (posA, posB), overlap_slices in overlaps:
            sl = tuple(slice(s, e) for s, e in overlap_slices)
            lab_overlap = labels[sl]
            emb_overlap = embeddings[(slice(None),) + sl]

            ids_in_overlap = torch.unique(lab_overlap)
            ids_in_overlap = ids_in_overlap[ids_in_overlap > 0]
            if len(ids_in_overlap) < 2:
                continue

            centers = []
            id_list = []
            emb_flat = rearrange(emb_overlap, "e ... -> e (...)")
            for uid in ids_in_overlap:
                mask = lab_overlap == uid
                if mask.sum() == 0:
                    continue
                mask_flat = rearrange(mask, "... -> (...)")
                center = emb_flat[:, mask_flat].mean(dim=1)
                centers.append(center)
                id_list.append(int(uid.item()))

            if len(centers) < 2:
                continue

            centers_t = torch.stack(centers)
            pw_dist = torch.cdist(centers_t, centers_t)

            K = len(id_list)
            for i in range(K):
                for j in range(i + 1, K):
                    if pw_dist[i, j] < self.merge_threshold:
                        uf.union(id_list[i], id_list[j])

        label_map = torch.zeros(max_id + 1, device=device, dtype=labels.dtype)
        for old_id in range(1, max_id + 1):
            label_map[old_id] = uf.find(old_id)
        merged = label_map[labels.long()]

        if self.spatial_dims == 3 and merged.dim() == 3:
            split = rearrange(
                relabel_connected_components_3d(rearrange(merged, "d h w -> 1 d h w")),
                "1 d h w -> d h w",
            )
        else:
            split = merged

        if self.min_instance_size > 0:
            for uid in torch.unique(split):
                if uid == 0:
                    continue
                if (split == uid).sum() < self.min_instance_size:
                    split[split == uid] = 0

        return relabel_sequential(split)

    def _find_overlapping_pairs(
        self,
        positions: List[Tuple[int, ...]],
        patch_size: Tuple[int, ...],
        vol_shape: Tuple[int, ...],
    ) -> List[Tuple[Tuple, List[Tuple[int, int]]]]:
        """Find pairs of patches that overlap and return their overlap regions."""
        S = len(patch_size)
        pairs = []
        n = len(positions)

        for i in range(n):
            for j in range(i + 1, n):
                overlap = []
                has_overlap = True
                for d in range(S):
                    start_i, end_i = positions[i][d], positions[i][d] + patch_size[d]
                    start_j, end_j = positions[j][d], positions[j][d] + patch_size[d]
                    o_start = max(start_i, start_j)
                    o_end = min(end_i, end_j)
                    if o_start >= o_end:
                        has_overlap = False
                        break
                    overlap.append((o_start, o_end))
                if has_overlap:
                    pairs.append(((positions[i], positions[j]), overlap))

        return pairs

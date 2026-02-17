"""
Discriminative Loss for Instance Segmentation.

Based on "Semantic Instance Segmentation with a Discriminative Loss Function"
by De Brabandere et al. (2017).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


class DiscriminativeLoss(nn.Module):
    """
    Discriminative loss for learning pixel embeddings for instance segmentation.

    The loss encourages:
    - L_var: Embeddings of same instance to be close (within delta_var)
    - L_dst: Embeddings of different instances to be far apart (beyond delta_dst)
    - L_reg: Embeddings to stay close to origin (regularization)

    Supports both 2D and 3D inputs.

    Args:
        delta_var: Margin for variance term (default: 0.5).
        delta_dst: Margin for distance term (default: 1.5).
        norm: Norm type for distance computation (default: 2).
        A: Weight for variance term (default: 1.0).
        B: Weight for distance term (default: 1.0).
        R: Weight for regularization term (default: 0.001).

    Example:
        >>> loss_fn = DiscriminativeLoss(delta_var=0.5, delta_dst=1.5)
        >>> embedding = torch.randn(4, 16, 256, 256)  # [B, E, H, W]
        >>> labels = torch.randint(0, 10, (4, 256, 256))  # [B, H, W]
        >>> total_loss, L_var, L_dst, L_reg = loss_fn(embedding, labels)
    """

    def __init__(
        self,
        delta_var: float = 0.5,
        delta_dst: float = 1.5,
        norm: int = 2,
        A: float = 1.0,
        B: float = 1.0,
        R: float = 0.001,
    ) -> None:
        super().__init__()
        self.delta_var = delta_var
        self.delta_dst = delta_dst
        self.norm = norm
        self.A = A
        self.B = B
        self.R = R

    def _flatten_spatial(
        self,
        embedding: torch.Tensor,
        ins_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Flatten spatial dimensions for both 2D and 3D inputs.

        Args:
            embedding: [B, E, H, W] for 2D or [B, E, D, H, W] for 3D.
            ins_label: [B, H, W] or [B, 1, H, W] for 2D;
                           [B, D, H, W] or [B, 1, D, H, W] for 3D.

        Returns:
            Tuple of (embed_flat [B, E, N], value_flat [B, N], is_3d).
        """
        is_3d = embedding.dim() == 5

        if is_3d:
            embed_flat = rearrange(embedding, "b e d h w -> b e (d h w)")
            if ins_label.dim() == 5:
                value_flat = rearrange(ins_label, "b 1 d h w -> b (d h w)")
            else:
                value_flat = rearrange(ins_label, "b d h w -> b (d h w)")
        else:
            embed_flat = rearrange(embedding, "b e h w -> b e (h w)")
            if ins_label.dim() == 4:
                value_flat = rearrange(ins_label, "b 1 h w -> b (h w)")
            else:
                value_flat = rearrange(ins_label, "b h w -> b (h w)")

        return embed_flat, value_flat, is_3d

    def _compute_cluster_means(
        self,
        emb: torch.Tensor,
        ins: torch.Tensor,
        unique_insances: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cluster centers for each insance."""
        centers = []

        for ins_id in unique_insances:
            ins_mask = ins == ins_id
            if ins_mask.sum() == 0:
                continue
            ins_embeddings = emb[:, ins_mask]
            center = ins_embeddings.mean(dim=1)
            centers.append(center)

        if len(centers) == 0:
            return torch.zeros((0, emb.shape[0]), device=emb.device)

        return torch.stack(centers)

    def _var_loss(
        self,
        emb: torch.Tensor,
        ins: torch.Tensor,
        unique_instances: torch.Tensor,
        cluster_centers: torch.Tensor,
    ) -> torch.Tensor:
        """Compute variance loss: pull embeddings toward their cluster center."""
        num_instances = len(unique_instances)

        if num_instances == 0:
            return torch.tensor(0.0, device=emb.device, dtype=torch.float32)

        emb = emb.float()
        cluster_centers = cluster_centers.float()

        loss_var = torch.tensor(0.0, device=emb.device, dtype=torch.float32)

        for idx, ins_id in enumerate(unique_instances):
            ins_mask = ins == ins_id
            if ins_mask.sum() == 0:
                continue

            ins_embeddings = emb[:, ins_mask]
            center = cluster_centers[idx]
            center_broadcast = rearrange(center, "e -> e 1")

            distances = torch.norm(ins_embeddings - center_broadcast, p=self.norm, dim=0)
            hinged = F.relu(distances - self.delta_var) ** 2
            loss_var = loss_var + hinged.mean()

        return loss_var / num_instances

    def _dst_loss(self, cluster_centers: torch.Tensor) -> torch.Tensor:
        """Compute distance loss: push different instance centers apart."""
        num_instances = cluster_centers.shape[0]

        if num_instances <= 1:
            return torch.tensor(0.0, device=cluster_centers.device, dtype=torch.float32)

        cluster_centers = cluster_centers.float()

        loss_dst = torch.tensor(0.0, device=cluster_centers.device, dtype=torch.float32)
        n_pairs = 0

        for i in range(num_instances):
            for j in range(i + 1, num_instances):
                dst = torch.norm(cluster_centers[i] - cluster_centers[j], p=self.norm)
                hinged = F.relu(2 * self.delta_dst - dst) ** 2
                loss_dst = loss_dst + hinged
                n_pairs += 1

        if n_pairs > 0:
            loss_dst = loss_dst / n_pairs

        return loss_dst

    def _reg_loss(self, cluster_centers: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss: keep centers near origin."""
        if cluster_centers.shape[0] == 0:
            return torch.tensor(0.0, device=cluster_centers.device, dtype=torch.float32)

        cluster_centers = cluster_centers.float()
        norms = torch.norm(cluster_centers, p=self.norm, dim=1)
        return norms.mean()

    def forward(
        self,
        embedding: torch.Tensor,
        ins_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute discriminative loss.

        Args:
            embedding: Pixel embeddings [B, E, H, W] (2D) or [B, E, D, H, W] (3D).
            ins_label: Instance labels [B, H, W] or [B, D, H, W].
                           Value 0 = background, >0 = instance IDs.

        Returns:
            Tuple of (total_loss, L_var, L_dst, L_reg).
        """
        batch_size = embedding.shape[0]
        embed_flat, value_flat, _ = self._flatten_spatial(embedding, ins_label)

        loss_var_total = torch.tensor(0.0, device=embedding.device, dtype=torch.float32)
        loss_dst_total = torch.tensor(0.0, device=embedding.device, dtype=torch.float32)
        loss_reg_total = torch.tensor(0.0, device=embedding.device, dtype=torch.float32)

        valid_batches = 0

        for b in range(batch_size):
            emb = embed_flat[b]
            inst = value_flat[b]

            unique_instances = torch.unique(inst)
            unique_instances = unique_instances[unique_instances > 0]

            if len(unique_instances) == 0:
                continue

            valid_batches += 1

            cluster_centers = self._compute_cluster_means(emb, inst, unique_instances)

            loss_var = self._var_loss(emb, inst, unique_instances, cluster_centers)
            loss_dst = self._dst_loss(cluster_centers)
            loss_reg = self._reg_loss(cluster_centers)

            loss_var_total = loss_var_total + loss_var
            loss_dst_total = loss_dst_total + loss_dst
            loss_reg_total = loss_reg_total + loss_reg

        if valid_batches > 0:
            loss_var_total = loss_var_total / valid_batches
            loss_dst_total = loss_dst_total / valid_batches
            loss_reg_total = loss_reg_total / valid_batches

        total_loss = (
            self.A * loss_var_total
            + self.B * loss_dst_total
            + self.R * loss_reg_total
        )

        return total_loss, loss_var_total, loss_dst_total, loss_reg_total

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"delta_var={self.delta_var}, "
            f"delta_dst={self.delta_dst}, "
            f"norm={self.norm}, "
            f"A={self.A}, "
            f"B={self.B}, "
            f"R={self.R})"
        )


class DiscriminativeLossVectorized(DiscriminativeLoss):
    """
    Vectorized implementation using einops for better GPU efficiency.

    Uses scatter operations instead of explicit Python loops where possible.
    """

    def _compute_cluster_means(
        self,
        emb: torch.Tensor,
        ins: torch.Tensor,
        unique_instances: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cluster means using scatter operations."""
        num_instances = len(unique_instances)
        emb_dim = emb.shape[0]

        emb = emb.float()

        max_ins = int(ins.max().item()) + 1
        ins2idx = torch.full((max_ins,), -1, device=emb.device, dtype=torch.long)

        for idx, ins_id in enumerate(unique_instances):
            ins2idx[ins_id.long()] = idx

        cluster_indices = ins2idx[ins.long()]
        valid_mask = cluster_indices >= 0

        if not valid_mask.any():
            return torch.zeros((num_instances, emb_dim), device=emb.device, dtype=emb.dtype)

        valid_emb = emb[:, valid_mask]
        valid_idx = cluster_indices[valid_mask]

        cluster_sums = torch.zeros((num_instances, emb_dim), device=emb.device, dtype=emb.dtype)
        cluster_counts = torch.zeros(num_instances, device=emb.device, dtype=emb.dtype)

        valid_emb_t = rearrange(valid_emb, "e n -> n e")
        for e in range(emb_dim):
            cluster_sums[:, e].scatter_add_(0, valid_idx, valid_emb_t[:, e])
        cluster_counts.scatter_add_(
            0, valid_idx, torch.ones(valid_idx.shape[0], device=emb.device, dtype=emb.dtype)
        )

        cluster_counts = torch.clamp(cluster_counts, min=1)
        cluster_means = cluster_sums / rearrange(cluster_counts, "c -> c 1")

        return cluster_means

    def _var_loss(
        self,
        emb: torch.Tensor,
        ins: torch.Tensor,
        unique_instances: torch.Tensor,
        cluster_means: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized variance loss using scatter operations."""
        num_instances = len(unique_instances)

        if num_instances == 0:
            return torch.tensor(0.0, device=emb.device, dtype=torch.float32)

        emb = emb.float()
        cluster_means = cluster_means.float()

        max_ins = int(ins.max().item()) + 1
        ins2idx = torch.full((max_ins,), -1, device=emb.device, dtype=torch.long)

        for idx, ins_id in enumerate(unique_instances):
            ins2idx[ins_id.long()] = idx

            cluster_indices = ins2idx[ins.long()]
        valid_mask = cluster_indices >= 0

        if not valid_mask.any():
            return torch.tensor(0.0, device=emb.device, dtype=torch.float32)

        valid_emb = emb[:, valid_mask]
        valid_idx = cluster_indices[valid_mask]

        gathered_means = cluster_means[valid_idx]
        gathered_means = rearrange(gathered_means, "n e -> e n")

        diff = valid_emb - gathered_means
        distances = torch.norm(diff, p=self.norm, dim=0)

        hinged = F.relu(distances - self.delta_var) ** 2

        cluster_losses = torch.zeros(num_instances, device=emb.device, dtype=torch.float32)
        cluster_counts = torch.zeros(num_instances, device=emb.device, dtype=torch.float32)

        cluster_losses.scatter_add_(0, valid_idx, hinged.float())
        cluster_counts.scatter_add_(0, valid_idx, torch.ones_like(hinged, dtype=torch.float32))

        cluster_counts = torch.clamp(cluster_counts, min=1)
        per_cluster_loss = cluster_losses / cluster_counts
        var_loss = reduce(per_cluster_loss, "c -> ", "mean")

        return var_loss

    def _dst_loss(self, cluster_means: torch.Tensor) -> torch.Tensor:
        """Vectorized distance loss between cluster centers."""
        num_instances = cluster_means.shape[0]

        if num_instances <= 1:
            return torch.tensor(0.0, device=cluster_means.device, dtype=torch.float32)

        cluster_means = cluster_means.float()

        means_i = rearrange(cluster_means, "c e -> c 1 e")
        means_j = rearrange(cluster_means, "c e -> 1 c e")

        diff = means_i - means_j
        pairwise_dst = torch.norm(diff, p=self.norm, dim=2)

        triu_indices = torch.triu_indices(
            num_instances, num_instances, offset=1, device=cluster_means.device
        )
        upper_dsts = pairwise_dst[triu_indices[0], triu_indices[1]]

        hinged = F.relu(2 * self.delta_dst - upper_dsts) ** 2
        dst_loss = reduce(hinged, "n -> ", "mean")

        return dst_loss

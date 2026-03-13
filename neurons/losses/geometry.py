"""
Geometry head regression loss: direction + covariance + raw reconstruction.

Dimension-agnostic — parameterized by ``spatial_dims``.

Supervises three groups of channels predicted by the geometry head:

* **direction** -- first ``S`` channels: unit vectors toward instance centroid.
* **covariance** -- next ``S*S`` channels: full symmetric spatial covariance.
* **raw** -- last 4 channels: RGBA reconstruction of the input image.

All three sub-losses are foreground-only and support configurable loss type:
``mse``, ``l1``, or ``smooth_l1``.

Expected geometry head output has ``S + S*S + 4`` channels:
  2-D:  2 + 4 + 4 = 10
  3-D:  3 + 9 + 4 = 16
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


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

    Args:
        spatial_dims: 2 for images, 3 for volumes.
        weight_dir: Weight for the direction sub-loss.
        weight_cov: Weight for the covariance sub-loss.
        weight_raw: Weight for the raw-reconstruction sub-loss.
        loss_dir: Loss function for direction (``mse``, ``l1``, ``smooth_l1``).
        loss_cov: Loss function for covariance.
        loss_raw: Loss function for raw reconstruction.
        smooth_l1_beta: Beta parameter for smooth-L1 when used.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        weight_dir: float = 1.0,
        weight_cov: float = 1.0,
        weight_raw: float = 1.0,
        loss_dir: str = "l1",
        loss_cov: str = "l1",
        loss_raw: str = "l1",
        smooth_l1_beta: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
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

    @property
    def geometry_channels(self) -> int:
        """Total number of geometry head output channels."""
        return self._ch_dir + self._ch_cov + self._ch_raw

    # ------------------------------------------------------------------
    # Foreground-masked regression loss
    # ------------------------------------------------------------------

    def _fg_loss(self, pred, target, fg, loss_type):
        """Regression loss over foreground pixels only.

        Args:
            pred:      [C, N] predicted channels (flattened spatial).
            target:    [C, N] target channels.
            fg:        [N] boolean foreground mask.
            loss_type: "mse" | "l1" | "smooth_l1".
        """
        n_fg = fg.sum().float().clamp(min=1.0)
        numel = n_fg * pred.shape[0]

        p, t = pred[:, fg], target[:, fg]

        if loss_type == "mse":
            diff = p - t
            return (diff ** 2).sum() / numel
        elif loss_type == "l1":
            return (p - t).abs().sum() / numel
        else:
            return F.smooth_l1_loss(
                p, t, beta=self.smooth_l1_beta, reduction="sum",
            ) / numel

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------

    def _upper_tri_to_full(self, tri: torch.Tensor) -> torch.Tensor:
        """Expand upper-triangle [S*(S+1)/2, N] -> full [S*S, N]."""
        S = self.spatial_dims
        full = torch.zeros(S * S, tri.shape[1], device=tri.device, dtype=tri.dtype)
        ch = 0
        for i in range(S):
            for j in range(i, S):
                full[i * S + j] = tri[ch]
                full[j * S + i] = tri[ch]
                ch += 1
        return full

    def targets_from_pipeline(
        self,
        direction: torch.Tensor,
        covariance: torch.Tensor,
    ) -> dict:
        """Build cached_targets from datamodule-precomputed tensors.

        Args:
            direction:  [B, S, *spatial] unit direction vectors.
            covariance: [B, S*(S+1)//2, *spatial] upper-triangle covariance.
        """
        B = direction.shape[0]
        dir_targets: List[Optional[torch.Tensor]] = []
        cov_targets: List[Optional[torch.Tensor]] = []

        for b in range(B):
            dir_flat = rearrange(direction[b], "c ... -> c (...)")
            cov_flat = rearrange(covariance[b], "c ... -> c (...)")

            has_fg = dir_flat.abs().sum() > 0
            if not has_fg:
                dir_targets.append(None)
                cov_targets.append(None)
                continue

            dir_targets.append(dir_flat if self.weight_dir > 0 else None)
            cov_targets.append(
                self._upper_tri_to_full(cov_flat) if self.weight_cov > 0 else None
            )

        return {"dir_targets": dir_targets, "cov_targets": cov_targets}

    @torch.no_grad()
    def compute_targets(self, ins_label: torch.Tensor) -> dict:
        """On-the-fly target computation from instance labels."""
        from neurons.transforms.direction import compute_direction_field
        from neurons.transforms.covariance import compute_covariance_field

        B = ins_label.shape[0]
        dir_targets: List[Optional[torch.Tensor]] = []
        cov_targets: List[Optional[torch.Tensor]] = []

        for b in range(B):
            lbl_b = ins_label[b]
            has_fg = (lbl_b > 0).any()
            if not has_fg:
                dir_targets.append(None)
                cov_targets.append(None)
                continue

            if self.weight_dir > 0:
                d = compute_direction_field(lbl_b, normalize=True)
                dir_targets.append(rearrange(d, "c ... -> c (...)"))
            else:
                dir_targets.append(None)

            if self.weight_cov > 0:
                c = compute_covariance_field(lbl_b, normalized=True)
                c_flat = rearrange(c, "c ... -> c (...)")
                cov_targets.append(self._upper_tri_to_full(c_flat))
            else:
                cov_targets.append(None)

        return {"dir_targets": dir_targets, "cov_targets": cov_targets}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, geometry, ins_label, raw_image=None, cached_targets=None):
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

        if cached_targets is None:
            cached_targets = self.compute_targets(ins_label)

        B = geometry.shape[0]
        for b in range(B):
            dir_tgt = cached_targets["dir_targets"][b]
            cov_tgt = cached_targets["cov_targets"][b]
            if dir_tgt is None and cov_tgt is None:
                continue
            valid_b += 1
            fg = lbl_flat[b] > 0

            if self.weight_dir > 0 and dir_tgt is not None:
                L_dir = L_dir + self._fg_loss(pred_dir[b], dir_tgt, fg, self.loss_dir)

            if self.weight_cov > 0 and cov_tgt is not None:
                L_cov = L_cov + self._fg_loss(pred_cov[b], cov_tgt, fg, self.loss_cov)

            if self.weight_raw > 0 and raw_image is not None:
                img_flat = rearrange(raw_image[b].detach(), "c ... -> c (...)").clamp(0.0, 1.0)
                c_in = img_flat.shape[0]
                if c_in == 1:
                    rgb = repeat(img_flat, "1 n -> 3 n")
                elif c_in >= 3:
                    rgb = img_flat[:3]
                else:
                    rgb = img_flat.expand(3, -1)[:3]
                rgba_tgt = torch.cat([
                    rgb,
                    rearrange(fg.float(), "n -> 1 n"),
                ], dim=0)
                L_raw = L_raw + self._fg_loss(
                    pred_raw[b], rgba_tgt, fg, self.loss_raw,
                )

        n = max(valid_b, 1)
        L_dir, L_cov, L_raw = L_dir / n, L_cov / n, L_raw / n
        total = self.weight_dir * L_dir + self.weight_cov * L_cov + self.weight_raw * L_raw

        return {"loss": total, "dir": L_dir, "cov": L_cov, "raw": L_raw}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"spatial_dims={self.spatial_dims}, "
            f"channels={self.geometry_channels} "
            f"(dir={self._ch_dir}+cov={self._ch_cov}+raw={self._ch_raw}), "
            f"loss_dir='{self.loss_dir}', loss_cov='{self.loss_cov}', loss_raw='{self.loss_raw}', "
            f"weight_dir={self.weight_dir}, weight_cov={self.weight_cov}, weight_raw={self.weight_raw})"
        )

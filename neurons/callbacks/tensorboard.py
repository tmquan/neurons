"""
TensorBoard image logger callback.

Logs a visual grid at the end of each training epoch:
  raw image, instance label, semantic prediction,
  instance embedding (PCA-projected), and geometry channels (dir / cov / raw).

Works for both 2-D slices and 3-D volumes (takes a central slice).
"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange


def _to_2d(t: torch.Tensor) -> torch.Tensor:
    """If *t* has a depth dimension, take the central slice."""
    if t.dim() == 5:
        return t[:, :, t.shape[2] // 2]
    return t


def _normalise(t: torch.Tensor) -> torch.Tensor:
    """Min-max normalise to [0, 1] per image in the batch."""
    B = t.shape[0]
    flat = t.reshape(B, -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    denom = (hi - lo).clamp(min=1e-5)
    return ((flat - lo) / denom).reshape_as(t)


def _label_to_rgb(labels: torch.Tensor) -> torch.Tensor:
    """Map integer instance labels to a deterministic RGB image.

    Args:
        labels: [B, H, W] long tensor.

    Returns:
        [B, 3, H, W] float tensor in [0, 1].
    """
    B, H, W = labels.shape
    flat = labels.reshape(-1).long()
    torch.manual_seed(0)
    palette = torch.rand(flat.max().item() + 1, 3, device=labels.device)
    palette[0] = 0.0
    rgb = palette[flat].reshape(B, H, W, 3)
    return rearrange(rgb, "b h w c -> b c h w")


def _pca_project(emb: torch.Tensor, n_components: int = 3) -> torch.Tensor:
    """Project [B, E, H, W] embedding to [B, n_components, H, W] via PCA.

    Each image in the batch is projected independently so colours are
    locally meaningful.
    """
    B, E, H, W = emb.shape
    flat = rearrange(emb, "b e h w -> b e (h w)").float()
    mean = flat.mean(dim=2, keepdim=True)
    centered = flat - mean
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    proj = Vh[:, :n_components]
    proj = rearrange(proj, "b c (h w) -> b c h w", h=H, w=W)
    return _normalise(proj)


class ImageLogger(pl.Callback):
    """Log sample images to TensorBoard at the end of every *n*-th epoch.

    Hooks into ``on_train_epoch_end`` to run a single forward pass on the
    first training batch and log a visual grid of:

    1. **image** -- raw EM input (grayscale).
    2. **label** -- ground-truth instance labels (colour-coded).
    3. **semantic** -- argmax of the semantic logits (colour-coded).
    4. **instance** -- PCA projection of the embedding to 3 channels (RGB).
    5. **geometry/dir** -- first S channels (normalised).
    6. **geometry/cov** -- next S*S channels, trace shown as heatmap.
    7. **geometry/raw** -- last 4 channels (predicted RGBA).

    Args:
        every_n_epochs: log every *n* epochs (default 1).
        max_images: maximum batch elements to log (default 4).
        spatial_dims: 2 or 3 — controls central-slice extraction for 3-D.
    """

    def __init__(
        self,
        every_n_epochs: int = 1,
        max_images: int = 4,
        spatial_dims: int = 2,
    ) -> None:
        super().__init__()
        self.every_n_epochs = max(every_n_epochs, 1)
        self.max_images = max_images
        self.spatial_dims = spatial_dims
        self._train_batch: Optional[Dict[str, torch.Tensor]] = None

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        if batch_idx == 0:
            self._train_batch = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

    @torch.no_grad()
    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return
        if self._train_batch is None:
            return
        logger = trainer.logger
        if logger is None:
            return
        tb = getattr(logger, "experiment", None)
        if tb is None or not hasattr(tb, "add_images"):
            return

        batch = self._train_batch
        images = batch["image"].to(pl_module.device)
        if images.dim() == self.spatial_dims + 1:
            images = images.unsqueeze(1)

        labels = batch["label"].to(pl_module.device)
        if labels.dim() == self.spatial_dims + 2:
            labels = labels.squeeze(1)

        preds = pl_module.model(images)

        n = min(images.shape[0], self.max_images)
        images = images[:n]
        labels = labels[:n]
        sem = preds["semantic"][:n]
        inst = preds["instance"][:n]
        geom = preds["geometry"][:n]

        images = _to_2d(images)
        labels = _to_2d(labels.unsqueeze(1)).squeeze(1)
        sem = _to_2d(sem)
        inst = _to_2d(inst)
        geom = _to_2d(geom)

        S = self.spatial_dims
        ch_dir = S
        ch_cov = S * S

        img_gray = _normalise(images).expand(-1, 3, -1, -1)

        lbl_rgb = _label_to_rgb(labels.long())

        sem_ids = sem.argmax(dim=1)
        sem_rgb = _label_to_rgb(sem_ids)

        inst_rgb = _pca_project(inst, n_components=3)

        g_dir = _normalise(geom[:, :ch_dir])
        if ch_dir < 3:
            pad = torch.zeros(n, 3 - ch_dir, g_dir.shape[2], g_dir.shape[3],
                              device=g_dir.device)
            g_dir = torch.cat([g_dir, pad], dim=1)

        trace = sum(geom[:, ch_dir + i * S + i: ch_dir + i * S + i + 1]
                    for i in range(S))
        cov_heat = _normalise(trace).expand(-1, 3, -1, -1)

        g_raw = geom[:, ch_dir + ch_cov:]
        g_raw_rgb = _normalise(g_raw[:, :3])

        tag = "train_vis"
        tb.add_images(f"{tag}/image", img_gray, global_step=epoch)
        tb.add_images(f"{tag}/label", lbl_rgb, global_step=epoch)
        tb.add_images(f"{tag}/semantic", sem_rgb, global_step=epoch)
        tb.add_images(f"{tag}/instance_pca", inst_rgb, global_step=epoch)
        tb.add_images(f"{tag}/geometry_dir", g_dir, global_step=epoch)
        tb.add_images(f"{tag}/geometry_cov_trace", cov_heat, global_step=epoch)
        tb.add_images(f"{tag}/geometry_raw", g_raw_rgb, global_step=epoch)

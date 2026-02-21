"""
TensorBoard image logger callback.

Logs visual grids at the end of each training epoch for both automatic
and proofread modes:
  raw image, instance label, semantic prediction,
  instance embedding (PCA-projected), geometry channels (dir / cov / raw),
  and point prompt overlay (proofread only).

Works for both 2-D slices and 3-D volumes (takes a central slice).
"""

from typing import Any, Dict, List, Optional

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


def _draw_points_on_image(
    img_rgb: torch.Tensor,
    pos_points: List[torch.Tensor],
    neg_points: List[torch.Tensor],
    spatial_dims: int,
    center_depth: Optional[int] = None,
    radius: int = 2,
) -> torch.Tensor:
    """Overlay sampled prompt points on an RGB image.

    Args:
        img_rgb: [B, 3, H, W] image to draw on (will be cloned).
        pos_points: list of [N_pos, spatial_dims] coordinate tensors.
        neg_points: list of [N_neg, spatial_dims] coordinate tensors.
        spatial_dims: 2 or 3.
        center_depth: for 3-D, the depth index of the displayed slice.
            Points within ``radius`` slices of center are drawn.
        radius: marker radius in pixels.

    Returns:
        [B, 3, H, W] image with green (pos) and red (neg) markers.
    """
    out = img_rgb.clone()
    B, _, H, W = out.shape

    for b in range(min(B, len(pos_points))):
        for pts, color in [(pos_points[b], (0.0, 1.0, 0.0)),
                           (neg_points[b], (1.0, 0.0, 0.0))]:
            if pts.numel() == 0:
                continue
            coords = pts.long()
            if spatial_dims == 3:
                if center_depth is None:
                    continue
                depth_idx = coords[:, 0]
                near = (depth_idx - center_depth).abs() <= radius
                coords = coords[near]
                if coords.shape[0] == 0:
                    continue
                ys, xs = coords[:, 1], coords[:, 2]
            else:
                ys, xs = coords[:, 0], coords[:, 1]

            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dy * dy + dx * dx > radius * radius:
                        continue
                    cy = (ys + dy).clamp(0, H - 1)
                    cx = (xs + dx).clamp(0, W - 1)
                    out[b, 0, cy, cx] = color[0]
                    out[b, 1, cy, cx] = color[1]
                    out[b, 2, cy, cx] = color[2]
    return out


def _log_predictions(
    tb: Any,
    tag: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    preds: Dict[str, torch.Tensor],
    spatial_dims: int,
    n: int,
    epoch: int,
) -> None:
    """Log a standard set of prediction visualizations.

    Args:
        tb: TensorBoard SummaryWriter.
        tag: tag prefix (e.g. ``"train_vis"`` or ``"train_vis_proofread"``).
        images: [n, 1, H, W] input images (already 2-D sliced).
        labels: [n, H, W] instance labels (already 2-D sliced).
        preds: model output dict with ``semantic``, ``instance``, ``geometry``.
        spatial_dims: 2 or 3 (controls geometry channel layout).
        n: number of images.
        epoch: global step for TensorBoard.
    """
    sem = preds["semantic"][:n]
    inst = preds["instance"][:n]
    geom = preds["geometry"][:n]

    sem = _to_2d(sem)
    inst = _to_2d(inst)
    geom = _to_2d(geom)

    S = spatial_dims
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

    g_raw = torch.sigmoid(geom[:, ch_dir + ch_cov:])
    g_raw_rgb = g_raw[:, :3].clamp(0.0, 1.0)

    tb.add_images(f"{tag}/image", img_gray, global_step=epoch)
    tb.add_images(f"{tag}/label", lbl_rgb, global_step=epoch)
    tb.add_images(f"{tag}/semantic", sem_rgb, global_step=epoch)
    tb.add_images(f"{tag}/instance_pca", inst_rgb, global_step=epoch)
    tb.add_images(f"{tag}/geometry_dir", g_dir, global_step=epoch)
    tb.add_images(f"{tag}/geometry_cov_trace", cov_heat, global_step=epoch)
    tb.add_images(f"{tag}/geometry_raw", g_raw_rgb, global_step=epoch)

    return img_gray


class ImageLogger(pl.Callback):
    """Log sample images to TensorBoard at the end of every *n*-th epoch.

    Logs visualizations for **automatic** mode (image-only forward) and,
    when the module has ``"proofread"`` in its ``training_modes``, also
    for **proofread** mode (with sampled point prompts overlaid).

    Automatic-mode images are logged under ``train_vis/``.
    Proofread-mode images are logged under ``train_vis_proofread/``,
    with an extra ``prompt_overlay`` panel showing positive (green) and
    negative (red) points on the input image.

    Args:
        every_n_epochs: log every *n* epochs (default 1).
        max_images: maximum batch elements to log (default 4).
        spatial_dims: 2 or 3 -- controls central-slice extraction for 3-D.
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

        n = min(images.shape[0], self.max_images)

        # --- Automatic mode ---
        preds_auto = pl_module.model(images)

        images_2d = _to_2d(images[:n])
        labels_2d = _to_2d(labels[:n].unsqueeze(1)).squeeze(1)

        _log_predictions(
            tb, "train_vis", images_2d, labels_2d,
            preds_auto, self.spatial_dims, n, epoch,
        )

        # --- Proofread mode ---
        training_modes = getattr(pl_module, "training_modes", [])
        if "proofread" not in training_modes:
            return

        from neurons.utils.point_sampling import sample_point_prompts

        sem_labels = (labels > 0).long()
        num_pos = getattr(pl_module, "_num_pos_points", 5)
        num_neg = getattr(pl_module, "_num_neg_points", 5)
        sample_mode = getattr(pl_module, "_point_sample_mode", "class")

        point_prompts = sample_point_prompts(
            sem_labels, labels,
            num_pos=num_pos,
            num_neg=num_neg,
            sample_mode=sample_mode,
        )

        preds_proof = pl_module.model(images, point_prompts=point_prompts)

        img_gray = _log_predictions(
            tb, "train_vis_proofread", images_2d, labels_2d,
            preds_proof, self.spatial_dims, n, epoch,
        )

        center_d = images.shape[2] // 2 if self.spatial_dims == 3 else None
        overlay = _draw_points_on_image(
            img_gray,
            point_prompts["pos_points"],
            point_prompts["neg_points"],
            spatial_dims=self.spatial_dims,
            center_depth=center_d,
        )
        tb.add_images("train_vis_proofread/prompt_overlay", overlay, global_step=epoch)

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

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat


def _to_2d(t: torch.Tensor) -> torch.Tensor:
    """If *t* has a depth dimension, take the central slice."""
    if t.dim() == 5:
        return t[:, :, t.shape[2] // 2]
    return t


def _normalise(t: torch.Tensor) -> torch.Tensor:
    """Min-max normalise to [0, 1] per image in the batch."""
    flat = rearrange(t, "b ... -> b (...)")
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    denom = (hi - lo).clamp(min=1e-5)
    return ((rearrange(t, "b ... -> b (...)") - lo) / denom).reshape_as(t)


def _label_to_rgb(labels: torch.Tensor) -> torch.Tensor:
    """Map integer instance labels to a deterministic RGB image.

    Args:
        labels: [B, H, W] long tensor.

    Returns:
        [B, 3, H, W] float tensor in [0, 1].
    """
    B, H, W = labels.shape
    flat = rearrange(labels, "b h w -> (b h w)").long()
    gen = torch.Generator(device=labels.device).manual_seed(0)
    palette = torch.rand(flat.max().item() + 1, 3, device=labels.device, generator=gen)
    palette[0] = 0.0
    rgb = palette[flat]
    return rearrange(rgb, "(b h w) c -> b c h w", b=B, h=H, w=W)


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


def _render_cov_glyphs(
    cov_mat: torch.Tensor,
    img_rgb: torch.Tensor,
    labels: torch.Tensor,
    S: int,
    step: int = 8,
) -> torch.Tensor:
    """Render structure-tensor ellipse glyphs on the EM image.

    Args:
        cov_mat: [B, H, W, s1, s2] predicted covariance matrices (2D-sliced).
        img_rgb: [B, 3, H, W] grayscale EM repeated to 3 channels.
        labels: [B, H, W] instance labels for coloring glyphs.
        S: spatial_dims (2 or 3). Only the last 2 eigenvectors are used for 2D glyphs.
        step: grid spacing for glyph placement.

    Returns:
        [B, 3, H, W] tensor with ellipse glyphs overlaid on the EM image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    B, _, H, W = img_rgb.shape
    device = img_rgb.device
    glyph_radius = step * 0.45

    gen = torch.Generator(device="cpu").manual_seed(0)
    max_id = max(int(labels.max().item()), 1)
    palette = torch.rand(max_id + 1, 3, generator=gen).numpy()
    palette[0] = 0.5

    result = []
    for b in range(B):
        bg = img_rgb[b].detach().cpu().permute(1, 2, 0).numpy().copy()
        lbl = labels[b].detach().cpu().numpy()
        mat = cov_mat[b].detach().cpu().numpy()  # [H, W, s1, s2]

        fig, ax = plt.subplots(1, 1, figsize=(W / 64, H / 64), dpi=64)
        ax.imshow(bg, aspect="equal", interpolation="nearest")
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        rows_sub = np.arange(step // 2, H, step)
        cols_sub = np.arange(step // 2, W, step)

        for r in rows_sub:
            for c in cols_sub:
                if lbl[r, c] == 0:
                    continue
                T = mat[r, c]  # [s1, s2]
                if S == 3:
                    T = T[1:, 1:]  # use YX submatrix for 2D visualization
                eigvals, eigvecs = np.linalg.eigh(T)
                abs_eig = np.abs(eigvals)
                if abs_eig.max() < 1e-8:
                    continue
                ratio = abs_eig.min() / max(abs_eig.max(), 1e-8)
                idx_max = int(abs_eig.argmax())
                angle = np.degrees(np.arctan2(
                    eigvecs[1, idx_max], eigvecs[0, idx_max],
                ))
                color = palette[int(lbl[r, c]) % len(palette)]
                ax.add_patch(Ellipse(
                    xy=(c, r),
                    width=2 * glyph_radius,
                    height=2 * glyph_radius * ratio,
                    angle=angle,
                    fill=True, facecolor=color, edgecolor=color,
                    linewidth=0.5, alpha=0.7,
                ))

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        arr = np.asarray(buf)[:, :, :3].copy()
        plt.close(fig)

        arr_resized = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        arr_resized = F.interpolate(
            arr_resized.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze(0)
        result.append(arr_resized)

    return torch.stack(result).to(device)


def _render_dir_quiver(
    dir_val: torch.Tensor,
    img_rgb: torch.Tensor,
    labels: torch.Tensor,
    S: int,
    dir_target: str = "centroid",
    step: int = 8,
) -> torch.Tensor:
    """Render direction vectors as quiver arrows on the EM image.

    Args:
        dir_val: [B, S, H, W] predicted direction channels (2D-sliced).
        img_rgb: [B, 3, H, W] grayscale EM repeated to 3 channels.
        labels: [B, H, W] instance labels for coloring arrows.
        S: spatial_dims (2 or 3). For 3D, uses last 2 channels (Y, X).
        dir_target: ``"centroid"`` or ``"skeleton"`` (for title only).
        step: grid spacing for arrow placement.

    Returns:
        [B, 3, H, W] tensor with quiver arrows overlaid on the EM image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    B, _, H, W = img_rgb.shape
    device = img_rgb.device

    gen = torch.Generator(device="cpu").manual_seed(0)
    max_id = max(int(labels.max().item()), 1)
    palette = torch.rand(max_id + 1, 4, generator=gen).numpy()
    palette[:, 3] = 1.0
    palette[0] = [0.5, 0.5, 0.5, 0.0]

    rows_sub = np.arange(step // 2, H, step)
    cols_sub = np.arange(step // 2, W, step)
    CC, RR = np.meshgrid(cols_sub, rows_sub)

    result = []
    for b in range(B):
        bg = img_rgb[b].detach().cpu().permute(1, 2, 0).numpy().copy()
        lbl = labels[b].detach().cpu().numpy()
        d = dir_val[b].detach().cpu().numpy()  # [S, H, W]

        if S == 3:
            U = d[2][RR, CC]   # X direction
            V = -d[1][RR, CC]  # -Y direction (quiver V+ = screen-up)
        else:
            U = d[0][RR, CC]
            V = -d[1][RR, CC]

        fg = lbl[RR, CC] > 0
        mag = np.sqrt(U ** 2 + V ** 2)
        mag = np.where(fg & (mag > 0), mag, 1.0)
        U_n, V_n = U / mag, V / mag

        arrow_colors = palette[lbl[RR, CC].ravel().astype(int) % len(palette)]

        fig, ax = plt.subplots(1, 1, figsize=(W / 64, H / 64), dpi=64)
        ax.imshow(bg, aspect="equal", interpolation="nearest")
        m = fg.ravel()
        if m.any():
            ax.quiver(
                CC.ravel()[m], RR.ravel()[m],
                U_n.ravel()[m], V_n.ravel()[m],
                color=arrow_colors[m],
                scale=30, width=0.005, headwidth=2, headlength=3,
            )
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        arr = np.asarray(buf)[:, :, :3].copy()
        plt.close(fig)

        arr_t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        arr_t = F.interpolate(
            arr_t.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze(0)
        result.append(arr_t)

    return torch.stack(result).to(device)


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
    clusterer: Any = None,
    dir_target: str = "centroid",
) -> None:
    """Log a standard set of prediction visualizations.

    Args:
        tb: TensorBoard SummaryWriter.
        tag: tag prefix (e.g. ``"train_vis_automatic"`` or ``"train_vis_proofread"``).
        images: [n, 1, H, W] input images (already 2-D sliced).
        labels: [n, H, W] instance labels (already 2-D sliced).
        preds: model output dict with ``semantic``, ``instance``, ``geometry``.
        spatial_dims: 2 or 3 (controls geometry channel layout).
        n: number of images.
        epoch: global step for TensorBoard.
        clusterer: optional clustering module (e.g. SoftMeanShift) for
            producing ``instance_pred`` from embeddings.
        dir_target: ``"centroid"`` or ``"skeleton"`` (controls direction vis title).
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

    img_gray = repeat(_normalise(images), "b 1 h w -> b 3 h w")
    lbl_rgb = _label_to_rgb(labels.long())
    sem_ids = sem.argmax(dim=1)
    sem_rgb = _label_to_rgb(sem_ids)
    inst_rgb = _pca_project(inst, n_components=3)

    g_dir_rgb = _render_dir_quiver(
        geom[:, :ch_dir], img_gray, labels, S, dir_target=dir_target,
    )

    cov_val = geom[:, ch_dir:ch_dir + ch_cov]  # [n, S*S, H, W]
    cov_mat = rearrange(cov_val, "b (s1 s2) h w -> b h w s1 s2", s1=S, s2=S)
    g_cov_rgb = _render_cov_glyphs(cov_mat, img_gray, labels, S)

    g_raw = torch.sigmoid(geom[:, ch_dir + ch_cov:])
    g_raw_rgb = g_raw[:, :3].clamp(0.0, 1.0)

    tb.add_images(f"{tag}/image", img_gray, global_step=epoch)
    tb.add_images(f"{tag}/label", lbl_rgb, global_step=epoch)
    tb.add_images(f"{tag}/semantic", sem_rgb, global_step=epoch)
    tb.add_images(f"{tag}/instance_pca", inst_rgb, global_step=epoch)

    if clusterer is not None:
        fg_mask = labels > 0
        if inst.dim() == 5:
            fg_mask_full = rearrange(_to_2d(rearrange(fg_mask, "b ... -> b 1 ...")), "b 1 ... -> b ...")
        else:
            fg_mask_full = fg_mask
        ins_pred, _, _ = clusterer(inst, rearrange(fg_mask_full, "b ... -> b 1 ...") if fg_mask_full.dim() == 3 else fg_mask_full)
        ins_pred_2d = rearrange(_to_2d(rearrange(ins_pred, "b ... -> b 1 ...")), "b 1 ... -> b ...") if ins_pred.dim() > 3 else ins_pred
        ins_pred_rgb = _label_to_rgb(ins_pred_2d.long())
        tb.add_images(f"{tag}/instance_pred", ins_pred_rgb, global_step=epoch)

    tb.add_images(f"{tag}/geometry_dir_{dir_target}", g_dir_rgb, global_step=epoch)
    tb.add_images(f"{tag}/geometry_cov", g_cov_rgb, global_step=epoch)
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
        with torch.no_grad():
            images = batch["image"].to(pl_module.device)
            if images.dim() == self.spatial_dims + 1:
                images = rearrange(images, "b ... -> b 1 ...")

            labels = batch["label"].to(pl_module.device)
            if labels.dim() == self.spatial_dims + 2:
                labels = rearrange(labels, "b 1 ... -> b ...")

            n = min(images.shape[0], self.max_images)

            # --- Automatic mode ---
            preds_auto = pl_module.model(images[:n])
            clusterer = getattr(pl_module, "_clusterer", None)

        criterion = getattr(pl_module, "criterion", None)
        geom_loss = getattr(criterion, "geometry_loss", None) if criterion else None
        dir_target = getattr(geom_loss, "dir_target", "centroid") if geom_loss else "centroid"

        images_2d = _to_2d(images[:n])
        labels_2d = rearrange(_to_2d(rearrange(labels[:n], "b ... -> b 1 ...")), "b 1 ... -> b ...")

        _log_predictions(
            tb, "train_vis_automatic", images_2d, labels_2d,
            preds_auto, self.spatial_dims, n, epoch,
            clusterer=clusterer,
            dir_target=dir_target,
        )

        # --- Proofread mode ---
        training_modes = getattr(pl_module, "training_modes", [])
        if "proofread" not in training_modes:
            return

        from neurons.utils.point_sampling import sample_point_prompts

        with torch.no_grad():
            sem_labels = (labels[:n] > 0).long()
            num_pos = getattr(pl_module, "_num_pos_points", 5)
            num_neg = getattr(pl_module, "_num_neg_points", 5)
            sample_mode = getattr(pl_module, "_point_sample_mode", "class")

            point_prompts = sample_point_prompts(
                sem_labels, labels[:n],
                num_pos=num_pos,
                num_neg=num_neg,
                sample_mode=sample_mode,
            )

            preds_proof = pl_module.model(images[:n], point_prompts=point_prompts)

        img_gray = _log_predictions(
            tb, "train_vis_proofread", images_2d, labels_2d,
            preds_proof, self.spatial_dims, n, epoch,
            clusterer=clusterer,
            dir_target=dir_target,
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

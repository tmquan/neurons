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
from einops import rearrange, reduce, repeat


# ======================================================================
# Visualisation helpers
# ======================================================================

def _to_2d(t: torch.Tensor) -> torch.Tensor:
    """Extract the central depth-slice from a 5-D tensor [B,C,D,H,W].

    Returns *t* unchanged if it is already 4-D [B,C,H,W].
    """
    if t.dim() == 5:
        return t[:, :, t.shape[2] // 2]
    return t


def _normalise(t: torch.Tensor) -> torch.Tensor:
    """Per-image min-max normalisation to [0, 1].

    Each image in the batch is normalised independently so that
    its minimum becomes 0 and its maximum becomes 1.
    """
    flat = rearrange(t, "b ... -> b (...)")                        # [B, N]
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    denom = (hi - lo).clamp(min=1e-5)
    normed = (flat - lo) / denom                                   # [B, N]
    return rearrange(normed, "b (c h w) -> b c h w",
                     c=t.shape[1], h=t.shape[2], w=t.shape[3])


def _hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Vectorised HSV→RGB.  All inputs/outputs in [0, 1], shape [N]."""
    h6 = h * 6.0
    sector = h6.long() % 6
    f = h6 - h6.floor()
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    lut = [(v, t, p), (q, v, p), (p, v, t),
           (p, q, v), (t, p, v), (v, p, q)]
    r, g, b = torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h)
    for i, (ri, gi, bi) in enumerate(lut):
        mask = sector == i
        r = torch.where(mask, ri, r)
        g = torch.where(mask, gi, g)
        b = torch.where(mask, bi, b)
    return torch.stack([r, g, b], dim=-1)


def _label_to_rgb(labels: torch.Tensor) -> torch.Tensor:
    """Map integer instance labels → pastel, deterministic RGB image.

    Background (0) is black.  Non-zero labels are coloured via a
    golden-ratio hue-spaced HSV palette with moderate saturation and
    high value for soft, PCA-like colours.

    Args:
        labels: [B, H, W] long tensor.

    Returns:
        [B, 3, H, W] float tensor in [0, 1].
    """
    B, H, W = labels.shape
    flat = rearrange(labels, "b h w -> (b h w)").long()
    n_labels = flat.max().item() + 1

    GOLDEN_RATIO = 0.618033988749895
    ids = torch.arange(n_labels, device=labels.device, dtype=torch.float32)
    hue = (ids * GOLDEN_RATIO) % 1.0

    gen = torch.Generator(device=labels.device).manual_seed(0)
    rand = torch.rand(n_labels, 2, device=labels.device, generator=gen)
    sat = 0.20 + 0.25 * rand[:, 0]                                # [0.20, 0.45]
    val = 0.75 + 0.25 * rand[:, 1]                                # [0.75, 1.00]

    palette = _hsv_to_rgb(hue, sat, val)                           # [n_labels, 3]
    palette[0] = 0.0                                               # background → black

    rgb = palette[flat]
    return rearrange(rgb, "(b h w) c -> b c h w", b=B, h=H, w=W)


def _pca_project(emb: torch.Tensor, n_components: int = 3) -> torch.Tensor:
    """Project [B, E, H, W] embeddings → [B, 3, H, W] via per-image PCA.

    The top-3 principal components are used as RGB channels.  Each image
    in the batch is projected independently so colours are locally
    meaningful (nearby pixels with similar embeddings get similar colours).

    Falls back to the first 3 channels when SVD fails to converge
    (ill-conditioned matrices are common in early training).
    """
    B, E, H, W = emb.shape
    flat = rearrange(emb, "b e h w -> b e (h w)").float()         # [B, E, N]
    mean = flat.mean(dim=2, keepdim=True)
    centered = flat - mean

    try:
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        proj = Vh[:, :n_components]                                # [B, 3, N]
    except (torch._C._LinAlgError, RuntimeError):
        proj = centered[:, :n_components]                          # [B, 3, N]

    proj = rearrange(proj, "b c (h w) -> b c h w", h=H, w=W)
    return _normalise(proj)


# ======================================================================
# Matplotlib-based geometry renderers
# ======================================================================

def _render_cov_glyphs(
    cov_mat: torch.Tensor,
    img_rgb: torch.Tensor,
    labels: torch.Tensor,
    S: int,
    step: int = 4,
) -> torch.Tensor:
    """Render EDT structure-tensor ellipse glyphs on the EM image.

    Each foreground pixel on a subsampled grid gets an ellipse whose:
    - **size** reflects the maximum eigenvalue relative to the global max
      (large near instance centres where EDT is high, small near boundaries).
    - **aspect ratio** reflects the eigenvalue ratio (elongated near
      boundaries, round near the medial axis).
    - **angle** is aligned with the major eigenvector (boundary tangent
      direction for anisotropic regions).

    Args:
        cov_mat: [B, H, W, s1, s2] predicted covariance matrices (2D-sliced).
        img_rgb: [B, 3, H, W] grayscale EM repeated to 3 channels.
        labels: [B, H, W] fg/bg mask (any int tensor; 0 = background, >0 = foreground).
        S: spatial_dims (2 or 3).  For 3D the last 2x2 submatrix is used.
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
    max_glyph_radius = step * 1.2
    COLOR = (0.0, 0.8, 1.0)

    result = []
    for b in range(B):
        bg = rearrange(img_rgb[b].detach().cpu().float(), "c h w -> h w c").numpy().copy()
        lbl = labels[b].detach().cpu().numpy()
        mat = cov_mat[b].detach().cpu().float().numpy()

        rows_sub = np.arange(step // 2, H, step)
        cols_sub = np.arange(step // 2, W, step)

        # First pass: find global max eigenvalue for normalisation
        max_eig_global = 0.0
        for r in rows_sub:
            for c in cols_sub:
                if lbl[r, c] == 0:
                    continue
                T = mat[r, c]
                if S == 3:
                    T = T[1:, 1:]                                  # project 3x3 → 2x2 (YX plane)
                e = np.abs(np.linalg.eigvalsh(T)).max()
                if e > max_eig_global:
                    max_eig_global = e
        if max_eig_global < 1e-8:
            max_eig_global = 1.0

        fig, ax = plt.subplots(1, 1, figsize=(W / 64, H / 64), dpi=64)
        ax.imshow(bg, aspect="equal", interpolation="nearest")
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Second pass: draw glyphs
        for r in rows_sub:
            for c in cols_sub:
                if lbl[r, c] == 0:
                    continue
                T = mat[r, c]
                if S == 3:
                    T = T[1:, 1:]

                eigvals, eigvecs = np.linalg.eigh(T)
                abs_eig = np.abs(eigvals)
                if abs_eig.max() < 1e-8:
                    continue

                scale = abs_eig.max() / max_eig_global
                glyph_radius = max_glyph_radius * np.clip(scale, 0.1, 1.0)

                ratio = abs_eig.min() / max(abs_eig.max(), 1e-8)

                idx_max = int(abs_eig.argmax())
                angle = np.degrees(np.arctan2(
                    eigvecs[1, idx_max], eigvecs[0, idx_max],
                ))

                ax.add_patch(Ellipse(
                    xy=(c, r),
                    width=2 * glyph_radius,
                    height=2 * glyph_radius * ratio,
                    angle=angle,
                    fill=True, facecolor=COLOR, edgecolor=COLOR,
                    linewidth=1.2, alpha=0.8,
                ))

        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)

        rendered = rearrange(
            torch.from_numpy(arr).float() / 255.0,
            "h w c -> c h w",
        )
        rendered = rearrange(
            F.interpolate(
                rearrange(rendered, "c h w -> 1 c h w"),
                size=(H, W), mode="bilinear", align_corners=False,
            ),
            "1 c h w -> c h w",
        )
        result.append(rendered)

    return torch.stack(result).to(device)


def _render_dir_quiver(
    dir_val: torch.Tensor,
    img_rgb: torch.Tensor,
    labels: torch.Tensor,
    S: int,
    dir_target: str = "centroid",
    step: int = 4,
) -> torch.Tensor:
    """Render direction vectors as quiver arrows on the EM image.

    Arrow length reflects the global magnitude of each direction vector:
    boundary pixels (far from centroid/skeleton) produce long arrows,
    centre pixels produce short ones.

    Args:
        dir_val: [B, S, H, W] predicted direction channels (2D-sliced).
        img_rgb: [B, 3, H, W] grayscale EM repeated to 3 channels.
        labels: [B, H, W] fg/bg mask (any int tensor; 0 = background, >0 = foreground).
        S: spatial_dims (2 or 3).
        dir_target: ``"centroid"`` or ``"skeleton"`` (cosmetic only).
        step: grid spacing for arrow placement.

    Returns:
        [B, 3, H, W] tensor with quiver arrows overlaid on the EM image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    B, _, H, W = img_rgb.shape
    device = img_rgb.device
    COLOR = (1.0, 0.4, 0.0, 0.9)

    rows_sub = np.arange(step // 2, H, step)
    cols_sub = np.arange(step // 2, W, step)
    CC, RR = np.meshgrid(cols_sub, rows_sub)

    result = []
    for b in range(B):
        bg = rearrange(img_rgb[b].detach().cpu().float(), "c h w -> h w c").numpy().copy()
        lbl = labels[b].detach().cpu().numpy()
        d = dir_val[b].detach().cpu().float().numpy()

        if S == 3:
            U = d[2][RR, CC]
            V = d[1][RR, CC]
        else:
            U = d[0][RR, CC]
            V = d[1][RR, CC]

        fg = lbl[RR, CC] > 0

        fig, ax = plt.subplots(1, 1, figsize=(W / 64, H / 64), dpi=64)
        ax.imshow(bg, aspect="equal", interpolation="nearest")
        m = fg.ravel()
        if m.any():
            ax.quiver(
                CC.ravel()[m], RR.ravel()[m],
                U.ravel()[m], V.ravel()[m],
                color=COLOR,
                angles="xy", scale_units="xy", scale=1.0 / (step * 2.0),
                width=0.014, headwidth=4.0, headlength=4.5,
            )
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)

        rendered = rearrange(
            torch.from_numpy(arr).float() / 255.0,
            "h w c -> c h w",
        )
        rendered = rearrange(
            F.interpolate(
                rearrange(rendered, "c h w -> 1 c h w"),
                size=(H, W), mode="bilinear", align_corners=False,
            ),
            "1 c h w -> c h w",
        )
        result.append(rendered)

    return torch.stack(result).to(device)


# ======================================================================
# Point prompt overlay
# ======================================================================

def _draw_points_on_image(
    img_rgb: torch.Tensor,
    pos_points: List[torch.Tensor],
    neg_points: List[torch.Tensor],
    spatial_dims: int,
    center_depth: Optional[int] = None,
    radius: int = 2,
) -> torch.Tensor:
    """Overlay sampled prompt points on an RGB image.

    Positive points are drawn in green, negative in red.  For 3-D data
    only points within ``radius`` slices of ``center_depth`` are shown.

    Args:
        img_rgb: [B, 3, H, W] image to draw on (will be cloned).
        pos_points: list of [N_pos, spatial_dims] coordinate tensors.
        neg_points: list of [N_neg, spatial_dims] coordinate tensors.
        spatial_dims: 2 or 3.
        center_depth: depth index of the displayed slice (3-D only).
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
                near = (coords[:, 0] - center_depth).abs() <= radius
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


# ======================================================================
# Prediction logger (assembles all panels for one mode)
# ======================================================================

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
    active_classes: Optional[int] = None,
) -> torch.Tensor:
    """Log a standard set of prediction visualisations to TensorBoard.

    Panels logged: image, label, semantic, instance_pca, instance_pred,
    geometry_dir, geometry_cov, geometry_raw.

    Args:
        tb: TensorBoard SummaryWriter.
        tag: tag prefix (e.g. ``"train_vis_automatic"``).
        images: [n, 1, H, W] input images (already 2-D sliced).
        labels: [n, H, W] instance labels (already 2-D sliced).
        preds: model output dict with ``semantic``, ``instance``, ``geometry``.
        spatial_dims: 2 or 3 (controls geometry channel layout).
        n: number of images.
        epoch: global step for TensorBoard.
        clusterer: optional SoftMeanShift for producing instance_pred.
        dir_target: ``"centroid"`` or ``"skeleton"``.
        active_classes: number of active semantic channels (from config).

    Returns:
        [n, 3, H, W] grayscale image repeated to RGB (for prompt overlay).
    """
    sem = _to_2d(preds["semantic"][:n])
    if active_classes is not None and active_classes < sem.shape[1]:
        sem = sem[:, :active_classes]
    inst = _to_2d(preds["instance"][:n])
    geom = _to_2d(preds["geometry"][:n])

    S = spatial_dims
    ch_dir = S
    ch_cov = S * S

    img_gray = _normalise(images).expand(-1, 3, -1, -1).contiguous()
    lbl_rgb = _label_to_rgb(labels.long())
    if sem.shape[1] == 1:
        sem_fg = sem[:, :1].sigmoid()
        sem_ids = (sem_fg[:, 0] > 0.5).long()
    else:
        sem_ids = sem.argmax(dim=1)
        sem_fg = sem[:, :2].softmax(dim=1)[:, 1:]
    sem_gray = repeat(sem_fg, "b 1 h w -> b 3 h w")
    inst_rgb = _pca_project(inst, n_components=3)

    fg_mask_pred = (sem_ids > 0).long()

    g_dir_rgb = _render_dir_quiver(
        geom[:, :ch_dir], img_gray, fg_mask_pred, S, dir_target=dir_target,
    )

    cov_val = geom[:, ch_dir:ch_dir + ch_cov]                     # [n, S*S, H, W]
    cov_mat = rearrange(cov_val, "b (s1 s2) h w -> b h w s1 s2", s1=S, s2=S)
    g_cov_rgb = _render_cov_glyphs(cov_mat, img_gray, fg_mask_pred, S)

    g_raw = geom[:, ch_dir + ch_cov:]
    g_raw_rgb = g_raw[:, :3].clamp(0.0, 1.0)

    tb.add_images(f"{tag}/image", img_gray, global_step=epoch)
    tb.add_images(f"{tag}/label", lbl_rgb, global_step=epoch)
    tb.add_images(f"{tag}/semantic", sem_gray, global_step=epoch)
    tb.add_images(f"{tag}/instance_pca", inst_rgb, global_step=epoch)

    if clusterer is not None:
        fg_mask_pred = sem_ids > 0
        if inst.dim() == 5:
            fg_mask_full = rearrange(
                _to_2d(rearrange(fg_mask_pred, "b ... -> b 1 ...")),
                "b 1 ... -> b ...",
            )
        else:
            fg_mask_full = fg_mask_pred
        fg_input = (rearrange(fg_mask_full, "b ... -> b 1 ...")
                    if fg_mask_full.dim() == 3 else fg_mask_full)
        ins_pred, _, _ = clusterer(inst, fg_input)
        if ins_pred.dim() > 3:
            ins_pred = rearrange(
                _to_2d(rearrange(ins_pred, "b ... -> b 1 ...")),
                "b 1 ... -> b ...",
            )
        ins_rgb = _label_to_rgb(ins_pred.long()) * sem_fg
        tb.add_images(f"{tag}/instance_pred", ins_rgb, global_step=epoch)

    tb.add_images(f"{tag}/geometry_dir_{dir_target}", g_dir_rgb, global_step=epoch)
    tb.add_images(f"{tag}/geometry_cov", g_cov_rgb, global_step=epoch)
    tb.add_images(f"{tag}/geometry_raw", g_raw_rgb, global_step=epoch)

    return img_gray


# ======================================================================
# Lightning Callback
# ======================================================================

class ImageLogger(pl.Callback):
    """Log sample images to TensorBoard at the end of every *n*-th epoch.

    Logs visualisations for both **training** and **validation** batches
    using **automatic** mode (image-only forward).  When the module has
    ``"proofread"`` in its ``training_modes``, training visualisations
    also include proofread mode with sampled point prompts.

    Tag prefixes:
      - ``train_vis_automatic/``   — training batch, automatic mode
      - ``train_vis_proofread/``   — training batch, proofread mode
      - ``val_vis_automatic/``     — validation batch, automatic mode

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
        self._val_batch: Optional[Dict[str, torch.Tensor]] = None

    @staticmethod
    def _detach_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        if batch_idx == 0 and trainer.global_rank == 0:
            self._train_batch = self._detach_batch(batch)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0 and trainer.global_rank == 0:
            self._val_batch = self._detach_batch(batch)

    def _get_tb(self, trainer: pl.Trainer):
        """Return TensorBoard SummaryWriter or None."""
        logger = trainer.logger
        if logger is None:
            return None
        tb = getattr(logger, "experiment", None)
        if tb is None or not hasattr(tb, "add_images"):
            return None
        return tb

    @torch.no_grad()
    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if trainer.global_rank != 0:
            self._train_batch = None
            return
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return
        if self._train_batch is None:
            return
        tb = self._get_tb(trainer)
        if tb is None:
            return

        batch = self._train_batch
        was_training = pl_module.training
        pl_module.eval()
        try:
            self._run_visualization(tb, pl_module, batch, prefix="train_vis")
        finally:
            self._train_batch = None
            if was_training:
                pl_module.train()

    @torch.no_grad()
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if trainer.global_rank != 0:
            self._val_batch = None
            return
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return
        if self._val_batch is None:
            return
        tb = self._get_tb(trainer)
        if tb is None:
            return

        was_training = pl_module.training
        pl_module.eval()
        try:
            self._run_visualization(
                tb, pl_module, self._val_batch, prefix="val_vis",
                include_proofread=False,
            )
        finally:
            self._val_batch = None
            if was_training:
                pl_module.train()

    def _run_visualization(
        self, tb, pl_module, batch, *, prefix: str = "train_vis",
        include_proofread: bool = True,
    ):
        epoch = pl_module.current_epoch
        device_type = str(pl_module.device).split(":")[0]
        with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=torch.cuda.is_available()):
            images = batch["image"].to(pl_module.device)
            if images.dim() == self.spatial_dims + 1:
                images = rearrange(images, "b ... -> b 1 ...")

            labels = batch["label"].to(pl_module.device)
            if labels.dim() == self.spatial_dims + 2:
                labels = rearrange(labels, "b 1 ... -> b ...")

            n = min(images.shape[0], self.max_images)
            preds_auto = pl_module.model(images[:n])

        preds_auto = {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in preds_auto.items()}
        clusterer = getattr(pl_module, "clusterer", None) or getattr(pl_module, "_clusterer", None)

        criterion = getattr(pl_module, "criterion", None)
        geom_loss = getattr(criterion, "geometry_loss", None) if criterion else None
        dir_target = getattr(geom_loss, "dir_target", "centroid") if geom_loss else "centroid"
        sem_loss = getattr(criterion, "semantic_loss", None) if criterion else None
        active_classes = getattr(sem_loss, "active_classes", None) if sem_loss else None

        images_2d = _to_2d(images[:n])
        labels_2d = rearrange(
            _to_2d(rearrange(labels[:n], "b ... -> b 1 ...")),
            "b 1 ... -> b ...",
        )

        _log_predictions(
            tb, f"{prefix}_automatic", images_2d, labels_2d,
            preds_auto, self.spatial_dims, n, epoch,
            clusterer=clusterer, dir_target=dir_target,
            active_classes=active_classes,
        )
        del preds_auto

        # --- Proofread mode (only for training, when enabled) ---
        if not include_proofread:
            return

        training_modes = getattr(pl_module, "training_modes", [])
        if "proofread" not in training_modes:
            return

        from neurons.utils.point_sampling import sample_point_prompts

        with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=torch.cuda.is_available()):
            sem_labels = (labels[:n] > 0).long()
            point_prompts = sample_point_prompts(
                sem_labels, labels[:n],
                num_pos=getattr(pl_module, "_num_pos_points", 5),
                num_neg=getattr(pl_module, "_num_neg_points", 5),
                sample_mode=getattr(pl_module, "_point_sample_mode", "class"),
            )
            preds_proof = pl_module.model(images[:n], point_prompts=point_prompts)

        preds_proof = {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in preds_proof.items()}

        img_gray = _log_predictions(
            tb, f"{prefix}_proofread", images_2d, labels_2d,
            preds_proof, self.spatial_dims, n, epoch,
            clusterer=clusterer, dir_target=dir_target,
            active_classes=active_classes,
        )
        del preds_proof

        center_d = images.shape[2] // 2 if self.spatial_dims == 3 else None
        overlay = _draw_points_on_image(
            img_gray,
            point_prompts["pos_points"],
            point_prompts["neg_points"],
            spatial_dims=self.spatial_dims,
            center_depth=center_d,
        )
        tb.add_images(f"{prefix}_proofread/prompt_overlay", overlay, global_step=epoch)

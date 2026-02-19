"""
Generate a teaser image for the Neurons project using real SNEMI3D EM data
and the Vista2D boundary-detection / instance-segmentation pipeline.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter

import torch
import torch.nn.functional as F

from neurons.utils.io import load_volume


# -- data path (symlinked to /scratch/SNEMI3D) --
SNEMI_DIR = "data/SNEMI3D"
SLICE_IDX = 42


# --------------------------------------------------------------------------
# Helpers that mirror vista2d_losses._get_weight_boundary
# --------------------------------------------------------------------------

def compute_boundary_weight(label: np.ndarray, weight_edge: float = 10.0) -> np.ndarray:
    """Morphological-gradient boundary weight (mirrors Vista2DLoss)."""
    t = torch.from_numpy(label.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    padded_arr = F.pad(t, (1, 1, 1, 1), mode="replicate")
    pooled_max = +F.max_pool2d(+padded_arr, 3, stride=1, padding=0)
    pooled_min = -F.max_pool2d(-padded_arr, 3, stride=1, padding=0)
    boundary = (pooled_max != pooled_min).float().squeeze().numpy()
    weight = 1.0 + boundary * (weight_edge - 1.0)
    return boundary, weight


def make_instance_cmap(n: int = 128) -> ListedColormap:
    """Perceptually-distinct neon palette for instance labels."""
    rng = np.random.RandomState(0)
    base = [
        "#00e5ff", "#ff3d00", "#76ff03", "#d500f9",
        "#ffea00", "#00b0ff", "#ff6d00", "#00e676",
        "#e040fb", "#ffd740", "#18ffff", "#ff9100",
        "#69f0ae", "#ea80fc", "#ffe57f", "#00b8d4",
        "#dd2c00", "#64dd17", "#aa00ff", "#ffd600",
        "#0091ea", "#ff1744", "#00c853", "#d50000",
        "#aeea00", "#304ffe", "#ff6d00", "#00bfa5",
        "#c51162", "#f4ff81", "#1de9b6", "#f50057",
    ]
    colors = ["#00000000"] + base
    while len(colors) < n:
        colors.append("#%06x" % rng.randint(0x222222, 0xffffff))
    return ListedColormap(colors[:n])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def generate_teaser(out_path: str = "teaser.png") -> None:
    # ── Load real SNEMI3D data ──
    img_vol = load_volume(f"{SNEMI_DIR}/AC4_inputs.h5")      # (100, 1024, 1024)
    seg_vol = load_volume(f"{SNEMI_DIR}/AC4_labels.h5")

    # Pick a nice 2D slice and crop a square region
    img_slice = img_vol[SLICE_IDX].astype(np.float32)
    seg_slice = seg_vol[SLICE_IDX].astype(np.int64)

    SZ = 512
    r0, c0 = 200, 200
    img_crop = img_slice[r0:r0 + SZ, c0:c0 + SZ]
    seg_crop = seg_slice[r0:r0 + SZ, c0:c0 + SZ]

    img_norm = (img_crop - img_crop.min()) / (img_crop.max() - img_crop.min() + 1e-8)

    # ── Vista2D-style boundary detection ──
    boundary, weight_map = compute_boundary_weight(seg_crop, weight_edge=10.0)

    # ── Figure ──
    fig = plt.figure(figsize=(15.0, 6.4), facecolor="#0d1117")

    gs = fig.add_gridspec(
        1, 5,
        left=0.012, right=0.988, top=0.78, bottom=0.12,
        wspace=0.035,
        width_ratios=[1, 0.06, 1, 0.06, 1],
    )

    # ── Title ──
    fig.text(
        0.50, 0.945, "NEURONS",
        fontsize=42, fontweight="bold", color="white",
        ha="center", va="center", fontfamily="monospace",
        path_effects=[pe.withStroke(linewidth=4, foreground="#0077b6")],
    )
    fig.text(
        0.50, 0.875,
        "Modular PyTorch Lightning Infrastructure for Connectomics",
        fontsize=13, color="#8b949e",
        ha="center", va="center", fontfamily="sans-serif",
    )

    # ── Panel 1: Raw EM ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_norm, cmap="gray", vmin=0, vmax=1, aspect="equal",
               interpolation="bilinear")
    ax1.set_title("Electron Microscopy", color="white", fontsize=13,
                  fontweight="bold", pad=10)
    ax1.axis("off")

    # ── Arrow 1 ──
    ax_a1 = fig.add_subplot(gs[0, 1])
    ax_a1.set_xlim(0, 1); ax_a1.set_ylim(0, 1)
    ax_a1.annotate(
        "", xy=(0.95, 0.5), xytext=(0.05, 0.5),
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.5,head_length=0.4",
            color="#00b0ff", lw=2.8,
        ),
    )
    ax_a1.axis("off")
    ax_a1.set_facecolor("#0d1117")

    # ── Panel 2: Vista2D boundary weight map ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(img_norm, cmap="gray", vmin=0, vmax=1, aspect="equal",
               interpolation="bilinear")

    boundary_smooth = gaussian_filter(boundary, sigma=0.6)
    boundary_smooth = np.clip(boundary_smooth / (boundary_smooth.max() + 1e-8), 0, 1)
    boundary_rgba = np.zeros((*boundary_smooth.shape, 4))
    boundary_rgba[..., 0] = 0.0
    boundary_rgba[..., 1] = 0.71
    boundary_rgba[..., 2] = 1.0
    boundary_rgba[..., 3] = boundary_smooth * 0.82
    ax2.imshow(boundary_rgba, aspect="equal", interpolation="bilinear")

    ax2.set_title("Boundary Detection", color="white", fontsize=13,
                  fontweight="bold", pad=10)
    ax2.axis("off")

    # ── Arrow 2 ──
    ax_a2 = fig.add_subplot(gs[0, 3])
    ax_a2.set_xlim(0, 1); ax_a2.set_ylim(0, 1)
    ax_a2.annotate(
        "", xy=(0.95, 0.5), xytext=(0.05, 0.5),
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.5,head_length=0.4",
            color="#00b0ff", lw=2.8,
        ),
    )
    ax_a2.axis("off")
    ax_a2.set_facecolor("#0d1117")

    # ── Panel 3: Instance segmentation ──
    ax3 = fig.add_subplot(gs[0, 4])
    ax3.imshow(img_norm * 0.30, cmap="gray", vmin=0, vmax=1, aspect="equal",
               interpolation="bilinear")

    cmap = make_instance_cmap(128)
    n_labels = int(seg_crop.max()) + 1
    masked_seg = np.ma.masked_where(seg_crop == 0, seg_crop % 127 + 1)
    ax3.imshow(masked_seg, cmap=cmap, alpha=0.62, vmin=0, vmax=127,
               aspect="equal", interpolation="nearest")

    boundary_glow = np.zeros((*boundary.shape, 4))
    boundary_glow[boundary > 0] = [1.0, 1.0, 1.0, 0.45]
    ax3.imshow(boundary_glow, aspect="equal")

    ax3.set_title("Instance Segmentation", color="white", fontsize=13,
                  fontweight="bold", pad=10)
    ax3.axis("off")

    # ── Feature pills ──
    labels_row = ["Datasets", "Models", "Losses", "Metrics"]
    features = [
        "SNEMI3D  /  CREMI3D  /  MICRONS  /  MitoEM2",
        "SegResNet  /  Vista3D  /  Vista2D",
        "Discriminative  /  Boundary  /  Skeleton",
        "ARI  /  AMI  /  Dice  /  IoU",
    ]
    for i, (lbl, feat) in enumerate(zip(labels_row, features)):
        x = 0.125 + i * 0.25
        fig.text(
            x, 0.072, lbl,
            fontsize=8.5, color="#58a6ff", ha="center", va="center",
            fontfamily="sans-serif", fontweight="bold",
        )
        fig.text(
            x, 0.035, feat,
            fontsize=8.2, color="#c9d1d9", ha="center", va="center",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#161b22", edgecolor="#30363d", linewidth=0.8,
            ),
        )

    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved teaser → {out_path}")


if __name__ == "__main__":
    generate_teaser()

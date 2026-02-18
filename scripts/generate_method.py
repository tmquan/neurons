"""
Generate method.png — Vista2D architecture diagram.

Pipeline:  EM slice -> SegResNet -> 3 heads -> 3 losses -> total loss
Grid-snapped layout with Manhattan (right-angle) arrow routing.
Uses real SNEMI3D AC4 data for the image patches.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, ArrowStyle
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter

import torch
import torch.nn.functional as F

from neurons.utils.io import load_volume

SNEMI_DIR = "data/SNEMI3D"
SLICE_IDX = 42
CROP = (200, 200, 400, 400)

# ── Palette ──
BG       = "#0d1117"
CARD_BG  = "#161b22"
EDGE     = "#30363d"
TXT      = "#c9d1d9"
DIM      = "#8b949e"
BLUE     = "#58a6ff"
CYAN     = "#00b0ff"
GREEN    = "#3fb950"
ORANGE   = "#d29922"
PURPLE   = "#bc8cff"
RED      = "#f85149"
WHITE    = "#ffffff"

# ──────────────────────────────────────────────────────────────────────
# Grid definition  (all values in axes-fraction [0..1])
# ──────────────────────────────────────────────────────────────────────
# Columns (left edge of each element)
C_IMG   = 0.020   # input image
C_BB    = 0.155   # backbone box
C_FEAT  = 0.300   # feature box
C_HEAD  = 0.435   # head boxes
C_OUT   = 0.570   # output patches
C_LOSS  = 0.700   # loss boxes
C_TOTAL = 0.840   # total-loss box

# Rows (vertical centre of each lane)
R_TOP = 0.78      # semantic lane
R_MID = 0.50      # instance lane
R_BOT = 0.22      # geometry lane

# Standard element sizes
BW = 0.115; BH = 0.12          # backbone / feature box
HW = 0.105; HH = 0.12          # head box
LW = 0.105; LH = 0.085         # loss box
TW = 0.145; TH = 0.14          # total-loss box
PW = 0.090; PH = 0.165         # image patch (axes fraction)
GAP = 0.012                    # arrow gap from edge


# ── Data ──

def _load():
    em = load_volume(f"{SNEMI_DIR}/AC4_inputs.h5")
    sg = load_volume(f"{SNEMI_DIR}/AC4_labels.h5")
    r0, c0, r1, c1 = CROP
    e = em[SLICE_IDX, r0:r1, c0:c1].astype(np.float32)
    s = sg[SLICE_IDX, r0:r1, c0:c1].astype(np.int64)
    e = (e - e.min()) / (e.max() - e.min() + 1e-8)
    return e, s

def _boundary(seg):
    t = torch.from_numpy(seg.astype(np.float32))[None, None]
    p = F.pad(t, (1, 1, 1, 1), mode="replicate")
    return ((F.max_pool2d(p, 3, 1, 0) != (-F.max_pool2d(-p, 3, 1, 0)))
            .float().squeeze().numpy())

def _icmap(n=128):
    rng = np.random.RandomState(0)
    base = [
        "#00e5ff", "#ff3d00", "#76ff03", "#d500f9", "#ffea00", "#00b0ff",
        "#ff6d00", "#00e676", "#e040fb", "#ffd740", "#18ffff", "#ff9100",
        "#69f0ae", "#ea80fc", "#ffe57f", "#00b8d4", "#dd2c00", "#64dd17",
        "#aa00ff", "#ffd600", "#0091ea", "#ff1744", "#00c853", "#d50000",
        "#aeea00", "#304ffe", "#ff6d00", "#00bfa5", "#c51162", "#f4ff81",
    ]
    c = ["#00000000"] + base
    while len(c) < n:
        c.append("#%06x" % rng.randint(0x333333, 0xffffff))
    return ListedColormap(c[:n])


# ── Drawing helpers ──

def _box(ax, cx, cy, w, h, fc, text, fs=10, tc=WHITE, ec=None, lw=1.4):
    """Rounded box centred at (cx, cy)."""
    x0 = cx - w / 2
    y0 = cy - h / 2
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0,rounding_size=0.012",
        facecolor=fc, edgecolor=ec or EDGE, linewidth=lw,
        transform=ax.transAxes, clip_on=False))
    ax.text(cx, cy, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fs, color=tc,
            fontfamily="monospace", fontweight="bold")


def _harrow(ax, x0, y, x1, color=CYAN, lw=1.8):
    """Horizontal arrow at constant y from x0 to x1."""
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y),
        arrowstyle=ArrowStyle("-|>", head_length=6, head_width=4),
        color=color, lw=lw, transform=ax.transAxes, clip_on=False))


def _manhattan(ax, x0, y0, x1, y1, color=CYAN, lw=1.6, vfirst=True):
    """Right-angle (Manhattan) arrow from (x0,y0) to (x1,y1).
    vfirst=True:  go vertical first, then horizontal.
    vfirst=False: go horizontal first, then vertical."""
    if vfirst:
        mid_y = y1
        verts = [(x0, y0), (x0, mid_y), (x1, mid_y)]
    else:
        mid_x = x1
        verts = [(x0, y0), (mid_x, y0), (mid_x, y1)]
    path = matplotlib.path.Path(verts)
    ax.add_patch(FancyArrowPatch(
        path=path,
        arrowstyle=ArrowStyle("-|>", head_length=6, head_width=4),
        color=color, lw=lw, transform=ax.transAxes, clip_on=False))


def _txt(ax, x, y, s, fs=8, c=TXT, ha="center", **kw):
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va="center",
            fontsize=fs, color=c, fontfamily="sans-serif", **kw)


def _sub(ax, cx, cy, s, fs=7, c=DIM):
    """Subtitle below a box."""
    ax.text(cx, cy, s, transform=ax.transAxes, ha="center", va="top",
            fontsize=fs, color=c, fontfamily="monospace")


# ── Main ──

def generate_method(out_path="method.png"):
    em, seg = _load()
    bnd = _boundary(seg)
    H, W = em.shape

    fig = plt.figure(figsize=(20, 8.0), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)

    # ── Title ──
    ax.text(0.50, 0.96, "Vista2D  --  Method Overview",
            ha="center", va="center", fontsize=24, color=WHITE,
            fontfamily="monospace", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="#0077b6")])

    # ==================================================================
    # COL 1 — Input EM image (centred on R_MID)
    # ==================================================================
    img_w = 0.115
    img_h = 0.38
    img_ax = fig.add_axes([C_IMG, R_MID - img_h / 2, img_w, img_h])
    img_ax.imshow(em, cmap="gray", vmin=0, vmax=1, aspect="equal")
    img_ax.set_title("Input\n[1, H, W]", fontsize=8.5, color=TXT, pad=6,
                     fontfamily="monospace")
    img_ax.axis("off")

    # ==================================================================
    # COL 2 — Backbone  (centred on R_MID)
    # ==================================================================
    bb_cx = C_BB + BW / 2
    _box(ax, bb_cx, R_MID, BW, BH, "#1a3a5c", "SegResNet\nBackbone",
         fs=11, ec=BLUE)
    _sub(ax, bb_cx, R_MID - BH / 2 - 0.01, "encoder-decoder")

    _harrow(ax, C_IMG + img_w + GAP, R_MID, C_BB - GAP, color=CYAN)

    # ==================================================================
    # COL 3 — Feature map  (centred on R_MID)
    # ==================================================================
    ft_cx = C_FEAT + BW / 2
    _box(ax, ft_cx, R_MID, BW, BH, "#1a3a5c", "Features\n[48, H, W]",
         fs=10, ec=BLUE)

    _harrow(ax, C_BB + BW + GAP, R_MID, C_FEAT - GAP, color=CYAN)

    # ==================================================================
    # COL 4 — Three heads  (snapped to R_TOP / R_MID / R_BOT)
    # ==================================================================
    hd_cx = C_HEAD + HW / 2
    feat_right = C_FEAT + BW + GAP

    # Semantic head
    _box(ax, hd_cx, R_TOP, HW, HH, "#1b3326", "Semantic\nHead", fs=10, ec=GREEN)
    _sub(ax, hd_cx, R_TOP - HH / 2 - 0.008, "Conv3x3 > BN > ReLU > Conv1x1", fs=6)
    _manhattan(ax, feat_right, R_MID, C_HEAD - GAP, R_TOP, color=GREEN, vfirst=False)

    # Instance head
    _box(ax, hd_cx, R_MID, HW, HH, "#332b1a", "Instance\nHead", fs=10, ec=ORANGE)
    _sub(ax, hd_cx, R_MID - HH / 2 - 0.008, "Conv3x3 > BN > ReLU > Conv1x1", fs=6)
    _harrow(ax, feat_right, R_MID, C_HEAD - GAP, color=ORANGE)

    # Geometry head
    _box(ax, hd_cx, R_BOT, HW, HH, "#2a1a33", "Geometry\nHead", fs=10, ec=PURPLE)
    _sub(ax, hd_cx, R_BOT - HH / 2 - 0.008, "Conv3x3 > BN > ReLU > Conv1x1", fs=6)
    _manhattan(ax, feat_right, R_MID, C_HEAD - GAP, R_BOT, color=PURPLE, vfirst=False)

    # ==================================================================
    # COL 5 — Output patches  (aligned to same rows)
    # ==================================================================
    def _patch_axes(row_cy):
        return fig.add_axes([C_OUT, row_cy - PH / 2, PW, PH])

    # Semantic output
    sax = _patch_axes(R_TOP)
    sax.imshow((seg > 0).astype(np.float32), cmap="Greens", vmin=0, vmax=1,
               aspect="equal")
    sax.set_title("[16, H, W]", fontsize=7, color=GREEN, pad=2,
                  fontfamily="monospace")
    sax.axis("off")
    _harrow(ax, C_HEAD + HW + GAP, R_TOP, C_OUT - GAP, color=GREEN, lw=1.4)

    # Instance output
    iax = _patch_axes(R_MID)
    rng = np.random.RandomState(3)
    emb_vis = np.zeros((H, W, 3), dtype=np.float32)
    for uid in np.unique(seg):
        if uid == 0:
            continue
        emb_vis[seg == uid] = rng.rand(3) * 0.7 + 0.3
    iax.imshow(emb_vis, aspect="equal")
    iax.set_title("[16, H, W]", fontsize=7, color=ORANGE, pad=2,
                  fontfamily="monospace")
    iax.axis("off")
    _harrow(ax, C_HEAD + HW + GAP, R_MID, C_OUT - GAP, color=ORANGE, lw=1.4)

    # Geometry output
    gax = _patch_axes(R_BOT)
    yy, xx = np.mgrid[:H, :W]
    gv = np.stack([xx / W, yy / H, gaussian_filter(rng.rand(H, W), sigma=8)],
                  axis=-1).astype(np.float32)
    gax.imshow(gv, aspect="equal")
    gax.set_title("[16, H, W]", fontsize=7, color=PURPLE, pad=2,
                  fontfamily="monospace")
    gax.axis("off")
    _harrow(ax, C_HEAD + HW + GAP, R_BOT, C_OUT - GAP, color=PURPLE, lw=1.4)

    # ==================================================================
    # COL 6 — Loss boxes  (same rows)
    # ==================================================================
    ls_cx = C_LOSS + LW / 2

    _box(ax, ls_cx, R_TOP, LW, LH, "#1b3326", "L_sem\nCE Loss",
         fs=9, ec=GREEN, lw=1.2)
    _harrow(ax, C_OUT + PW + GAP, R_TOP, C_LOSS - GAP, color=GREEN, lw=1.2)

    _box(ax, ls_cx, R_MID, LW, LH, "#332b1a", "L_ins\nPull / Push",
         fs=9, ec=ORANGE, lw=1.2)
    _harrow(ax, C_OUT + PW + GAP, R_MID, C_LOSS - GAP, color=ORANGE, lw=1.2)

    _box(ax, ls_cx, R_BOT, LW, LH, "#2a1a33", "L_aff\nL1 Loss",
         fs=9, ec=PURPLE, lw=1.2)
    _harrow(ax, C_OUT + PW + GAP, R_BOT, C_LOSS - GAP, color=PURPLE, lw=1.2)

    # ==================================================================
    # COL 7 — Total loss  (centred at R_MID)
    # ==================================================================
    tot_cx = C_TOTAL + TW / 2
    _box(ax, tot_cx, R_MID, TW, TH, "#1c1c2e",
         "L_total\n= L_sem + L_aff\n   + L_ins",
         fs=10.5, ec=RED, lw=1.8)

    loss_right = C_LOSS + LW + GAP
    _manhattan(ax, loss_right, R_TOP, C_TOTAL - GAP, R_MID + TH / 2 * 0.5,
               color=GREEN, lw=1.2, vfirst=False)
    _harrow(ax, loss_right, R_MID, C_TOTAL - GAP, color=ORANGE, lw=1.2)
    _manhattan(ax, loss_right, R_BOT, C_TOTAL - GAP, R_MID - TH / 2 * 0.5,
               color=PURPLE, lw=1.2, vfirst=False)

    # ==================================================================
    # Bottom strip — detail annotations
    # ==================================================================
    # Boundary weight map
    bw_ax = fig.add_axes([0.015, 0.025, 0.085, 0.20])
    bw_ax.imshow(gaussian_filter(bnd, 0.5), cmap="inferno", aspect="equal")
    bw_ax.set_title("w_edge", fontsize=7.5, color=RED, pad=3,
                    fontfamily="monospace")
    bw_ax.axis("off")

    # GT instance labels
    gt_ax = fig.add_axes([0.115, 0.025, 0.085, 0.20])
    cmap = _icmap(128)
    masked = np.ma.masked_where(seg == 0, seg % 127 + 1)
    gt_ax.imshow(em * 0.3, cmap="gray", vmin=0, vmax=1, aspect="equal")
    gt_ax.imshow(masked, cmap=cmap, alpha=0.65, vmin=0, vmax=127,
                 aspect="equal", interpolation="nearest")
    gt_ax.set_title("GT labels", fontsize=7.5, color=CYAN, pad=3,
                    fontfamily="monospace")
    gt_ax.axis("off")

    # Geometry targets breakdown
    lx_txt = 0.28
    _txt(ax, lx_txt, 0.155, "Geometry targets (16 ch):", fs=8.5, c=PURPLE,
         fontweight="bold", ha="left")
    for i, line in enumerate([
        "ch  0-8    gt_diff   (local displacements, 9ch)",
        "ch  9-11   gt_grid   (normalised coords, 3ch)",
        "ch 12-15   gt_rgba   (auxiliary intensity, 4ch)",
    ]):
        _txt(ax, lx_txt, 0.12 - i * 0.033, line, fs=7, c=DIM, ha="left")

    # Instance loss breakdown
    rx_txt = 0.58
    _txt(ax, rx_txt, 0.155, "Instance loss detail:", fs=8.5, c=ORANGE,
         fontweight="bold", ha="left")
    for i, line in enumerate([
        "L_pull :  embed -> cluster centre   (margin dv = 0.5)",
        "L_push :  centres pushed apart       (margin dd = 1.5)",
        "L_reg  :  centre norm regularisation",
        "weighted by  w_edge  x  w_bone",
    ]):
        _txt(ax, rx_txt, 0.12 - i * 0.033, line, fs=7, c=DIM, ha="left")

    # ── Save ──
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved method diagram -> {out_path}")


if __name__ == "__main__":
    generate_method()

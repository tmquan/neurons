# Tensor Glyph Formation

This document explains how the **EDT structure tensor** is computed in
`neurons.losses.discriminative._compute_covariance` and visualised as
elliptical tensor glyphs in `notebooks/00_explore_utility.ipynb`.

---

## 1. Motivation

Instance segmentation losses benefit from per-pixel geometric targets that
capture the local shape of each instance.  A natural choice is the
**structure tensor** of the Euclidean distance transform (EDT): a symmetric
positive-semi-definite matrix at every foreground pixel that encodes both
the *direction* and *magnitude* of morphological variation.

When visualised as an ellipse (a "tensor glyph"), the final blended
structure tensor reveals local geometry at a glance:

| Region | Glyph shape | Interpretation |
|---|---|---|
| Near boundary | thin ellipse along boundary tangent | gradient is coherent and perpendicular to the boundary |
| Interior / medial axis | round circle | EDT-blended isotropy makes the interior direction-free |
| Elongated interior | mild ellipse aligned with the short axis | blending preserves residual shape elongation |

A learned linear projection $W_\text{cov} \in \mathbb{R}^{E \times S^2}$
is trained to regress this tensor from the embedding, teaching the network
to encode local shape.

---

## 2. Mathematical Background

### 2.1 Euclidean Distance Transform

Let $M_k \subset \mathbb{Z}^S$ be the set of pixels belonging to instance
$k$, where $S$ is the number of spatial dimensions (2 for images, 3 for
volumes).  The EDT assigns each interior pixel $\mathbf{p} \in M_k$ the
Euclidean distance to the nearest exterior pixel:

$$
D(\mathbf{p}) \;=\; \min_{\mathbf{q} \,\notin\, M_k} \lVert \mathbf{p} - \mathbf{q} \rVert_2
$$

Key properties of $D$:

- $D(\mathbf{p}) = 0$ for every boundary pixel (adjacent to
  $\mathbb{Z}^S \setminus M_k$).
- $D$ reaches its maximum at the **medial axis** (the skeleton ridge).
- The spatial gradient $\nabla D$ has unit magnitude almost everywhere
  inside $M_k$; it is undefined on the medial axis where multiple nearest
  boundary points exist.
- The gradient direction at $\mathbf{p}$ points away from the nearest
  boundary point.

We write $D_\max = \max_{\mathbf{p} \in M_k} D(\mathbf{p})$ for the
deepest interior distance of the instance.

### 2.2 Gaussian Derivative Kernels

The implementation computes smooth gradient estimates by convolving $D$
with Gaussian derivative kernels rather than using finite differences.

Let $G_\sigma : \mathbb{R}^S \to \mathbb{R}$ denote the isotropic
Gaussian of standard deviation $\sigma$:

$$
G_\sigma(\mathbf{x}) \;=\; \frac{1}{(2\pi\sigma^2)^{S/2}} \exp\!\Bigl(-\frac{\lVert\mathbf{x}\rVert^2}{2\sigma^2}\Bigr)
$$

The partial derivative of the smoothed EDT along the $i$-th coordinate
axis is:

$$
\partial_i \bigl(G_{\sigma_d} * D\bigr)(\mathbf{p})
\;=\; \bigl(\partial_i G_{\sigma_d}\bigr) * D\;(\mathbf{p})
$$

where $*$ is convolution and $\sigma_d$ is the **derivative scale**,
set to $\sigma_d = \max(1,\; \sigma / 3)$ with $\sigma$ being the
user-facing integration scale parameter.

This yields a smoothed gradient vector at each pixel:

$$
\mathbf{g}(\mathbf{p})
\;=\; \Bigl(\partial_1(G_{\sigma_d} * D),\;\ldots,\;\partial_S(G_{\sigma_d} * D)\Bigr)(\mathbf{p})
$$

In the code the axes are ordered $(x, y) = (\text{col}, \text{row})$ in
2-D and $(x, y, z) = (\text{col}, \text{row}, \text{depth})$ in 3-D, so
$g_1 = \partial D / \partial x_\text{col}$, etc.

After convolution the gradient is multiplied by the instance mask
indicator $\mathbf{1}_{M_k}$ to suppress artificial gradients at the mask
boundary (where $D$ drops discontinuously to zero):

$$
\tilde{g}_i(\mathbf{p}) \;=\; \mathbf{1}_{M_k}(\mathbf{p}) \;\cdot\; \partial_i(G_{\sigma_d} * D)(\mathbf{p})
$$

### 2.3 Raw Structure Tensor with Mask-Normalised Smoothing

The classical structure tensor is the Gaussian-weighted local average of
the gradient outer product.  Because the masked gradient
$\tilde{\mathbf{g}}$ is zero outside $M_k$, a naive Gaussian average would
be diluted by the surrounding zeros.  The mask-normalised form corrects
for this:

$$
S_{ij}^{\,\text{raw}}(\mathbf{p})
\;=\;
\frac
  {\bigl(G_\sigma * (\tilde{g}_i \,\tilde{g}_j)\bigr)(\mathbf{p})}
  {\bigl(G_\sigma * \mathbf{1}_{M_k}\bigr)(\mathbf{p})}
$$

The denominator $G_\sigma * \mathbf{1}_{M_k}$ is the local mask coverage
under the Gaussian kernel.  Dividing by it restricts the weighted average
to instance-interior pixels only, preventing directional bias from
asymmetric zero-padding.

In 2-D this gives a symmetric $2 \times 2$ matrix at each pixel:

$$
S^{\text{raw}}(\mathbf{p})
\;=\;
\begin{pmatrix}
  S_{xx}^{\text{raw}} & S_{xy}^{\text{raw}} \\[4pt]
  S_{xy}^{\text{raw}} & S_{yy}^{\text{raw}}
\end{pmatrix}
$$

where $x = \text{col}$ and $y = \text{row}$.  In 3-D the matrix is
$3 \times 3$ with an analogous $z = \text{depth}$ axis.

**Why normalisation matters.**  Without the denominator, a pixel near
the boundary of a thin instance has most of its Gaussian window outside
$M_k$.  The short-axis direction sees more exterior zeros than the
long-axis direction, producing a spurious anisotropy that reflects the
window's partial coverage rather than the local gradient field.

### 2.4 Eigendecomposition of the Raw Tensor

Since $S^{\text{raw}}$ is real symmetric and positive-semi-definite, its
eigendecomposition is:

$$
S^{\text{raw}} = V \,\Lambda\, V^\top
\quad\text{with}\quad
\Lambda = \text{diag}(\lambda_1,\;\ldots,\;\lambda_S),
\quad \lambda_1 \le \cdots \le \lambda_S
$$

where the columns of $V$ are orthonormal eigenvectors.

- $\lambda_S$ (largest eigenvalue): direction of strongest gradient
  coherence, typically perpendicular to the nearest boundary.
- $\lambda_1$ (smallest eigenvalue): direction of weakest gradient
  coherence, typically along the boundary tangent.

The **isotropy ratio** $r = \lambda_1 / \lambda_S \in [0, 1]$ measures
how circular the glyph is.

### 2.5 The Isotropy Problem with Fixed Integration Scale

For a pixel deep inside the instance (large $D$), the integration
Gaussian $G_\sigma$ with fixed $\sigma$ only "sees" nearby pixels whose
gradients all point roughly toward the same nearest boundary segment.
The raw structure tensor is therefore highly anisotropic even at the
medial axis of a round instance.

Scaling $\sigma$ to the instance size (e.g. $\sigma = D_\max$) does not
solve this: a single large $\sigma$ makes $S^{\text{raw}}$ spatially
uniform (the same global average everywhere), destroying both boundary
anisotropy and any spatial variation.

### 2.6 EDT-Blended Isotropy

The solution is a post-hoc per-pixel blend of $S^{\text{raw}}$ toward
an isotropic matrix, controlled by the normalised EDT depth.

Define the **depth weight**:

$$
w(\mathbf{p})
\;=\; \left(\frac{D(\mathbf{p})}{D_\max}\right)^{\!\alpha}
\qquad \alpha = 2
$$

where $D_\max = \max_{\mathbf{q} \in M_k} D(\mathbf{q})$.  The weight
satisfies $w = 0$ at the boundary and $w = 1$ at the deepest interior
point.  The squared exponent ($\alpha = 2$) keeps $w$ small near the
boundary, preserving local anisotropy there, while ramping quickly
toward 1 in the interior.

The **isotropic component** with the same energy (trace) is:

$$
S^{\text{iso}}(\mathbf{p})
\;=\; \frac{\text{tr}\bigl(S^{\text{raw}}(\mathbf{p})\bigr)}{S} \; I_S
\;=\; \frac{\lambda_1 + \cdots + \lambda_S}{S} \; I_S
$$

where $I_S$ is the $S \times S$ identity matrix.

The blended tensor is:

$$
\boxed{\;
S(\mathbf{p})
\;=\;
\bigl(1 - w(\mathbf{p})\bigr)\; S^{\text{raw}}(\mathbf{p})
\;+\;
w(\mathbf{p})\; S^{\text{iso}}(\mathbf{p})
\;}
$$

Expanding component-wise in 2-D:

$$
S_{ij}(\mathbf{p}) =
\begin{cases}
  (1 - w)\, S_{ij}^{\text{raw}} + w \,\dfrac{S_{xx}^{\text{raw}} + S_{yy}^{\text{raw}}}{2}
    & \text{if } i = j \\[8pt]
  (1 - w)\, S_{ij}^{\text{raw}}
    & \text{if } i \ne j
\end{cases}
$$

**Effect on eigenvalues.**  Writing the raw eigenvalues as
$\lambda_\text{min}, \lambda_\text{max}$ and their mean as
$\bar\lambda = (\lambda_\text{min} + \lambda_\text{max})/2$, the blended
eigenvalues are:

$$
\lambda_i' \;=\; (1 - w)\,\lambda_i \;+\; w\,\bar\lambda
$$

so the blended isotropy ratio becomes:

$$
r'
\;=\; \frac{\lambda_\text{min}'}{\lambda_\text{max}'}
\;=\; \frac{(1-w)\,\lambda_\text{min} + w\,\bar\lambda}
           {(1-w)\,\lambda_\text{max} + w\,\bar\lambda}
$$

At the boundary ($w = 0$): $r' = \lambda_\text{min}/\lambda_\text{max} = r$
(unchanged).

At the medial axis ($w = 1$): $r' = \bar\lambda / \bar\lambda = 1$
(perfectly isotropic).

The blend is monotone in $w$: as $w$ increases from 0 to 1, $r'$
increases from the raw ratio $r$ to 1.

### 2.7 Glyph Geometry from the Blended Tensor

Given the eigendecomposition of the blended tensor
$S = V \,\text{diag}(\lambda_1', \lambda_2') \, V^\top$ at pixel
$\mathbf{p}$, the glyph ellipse is parameterised by:

$$
\text{width} = 2\,R, \qquad
\text{height} = 2\,R \cdot r', \qquad
\text{angle} = \arctan\!\frac{v_y^{(1)}}{v_x^{(1)}}
$$

where $R$ is a fixed visual radius (`glyph_radius`),
$r' = \lambda_1' / \lambda_2'$ is the blended isotropy ratio, and
$\mathbf{v}^{(1)} = (v_x^{(1)}, v_y^{(1)})$ is the eigenvector of the
*smaller* eigenvalue $\lambda_1'$.

The matplotlib `Ellipse(angle=...)` parameter rotates the width axis
(the longer one, since $r' \le 1$) to this angle, so the glyph is
elongated along the **minor** eigenvector direction — the boundary
tangent.

---

## 3. Implementation Walkthrough

### 3.1 Per-Instance Loop

`_compute_covariance` in `discriminative.py` processes each foreground
instance independently.  For instance $k$:

```python
mask   = (labels == k)
dt     = scipy.ndimage.distance_transform_edt(mask)   # D(p)
mask_f = mask.astype(float)                            # 1_{M_k}
edt_max = dt.max()                                     # D_max
```

### 3.2 Gradient Masking

The EDT is identically zero outside $M_k$.  Convolving with a Gaussian
derivative kernel across the mask boundary produces an artificial spike
from the discontinuity $D \to 0$.  The gradient is therefore zeroed
outside the mask after convolution:

```python
g = gaussian_filter(dt, sigma=sigma_d, order=derivative_order)
g *= mask_f          # g_i(p) * 1_{M_k}(p)
```

### 3.3 Mask-Normalised Smoothing

The outer product $\tilde{g}_i \tilde{g}_j$ is smoothed by $G_\sigma$ and
divided by the smoothed mask to produce $S_{ij}^{\text{raw}}$:

```python
norm = max(gaussian_filter(mask_f, sigma=sigma), 1e-10)      # G_sigma * 1_M
S_ij = gaussian_filter(g_i * g_j, sigma=sigma) / norm        # Eq. in sec 2.3
```

### 3.4 EDT-Blended Isotropy

The depth weight $w(\mathbf{p}) = (D(\mathbf{p}) / D_\max)^2$ blends the
diagonal components toward the isotropic value $\bar{S} = \text{tr}(S^{\text{raw}}) / S$,
and shrinks the off-diagonal components toward zero:

```python
w   = (dt / edt_max) ** 2
iso = (S_xx + S_yy) / 2               # tr(S) / S  for S=2

S_xx = (1 - w) * S_xx + w * iso       # diagonal blended
S_yy = (1 - w) * S_yy + w * iso
S_xy = (1 - w) * S_xy                 # off-diag shrinks to 0
```

### 3.5 Storage Layout

The blended tensor is stored as a `[S*S, N]` tensor in row-major order of
the matrix indices, where $N = H \times W$ (flattened spatially in
row-major order) and $S$ is the number of spatial dimensions:

```
# 2-D (S=2): 4 components stored as indices 0..3
#   0 -> S_xx  (col, col)
#   1 -> S_xy  (col, row)
#   2 -> S_yx  (row, col)  [= S_xy by symmetry]
#   3 -> S_yy  (row, row)
#
# 3-D (S=3): 9 components stored as indices 0..8
#   0 -> S_xx   1 -> S_xy   2 -> S_xz
#   3 -> S_yx   4 -> S_yy   5 -> S_yz
#   6 -> S_zx   7 -> S_zy   8 -> S_zz
```

Background pixels ($\text{label} = 0$) are left as zero.

---

## 4. Visualisation (Notebook)

The notebook `00_explore_utility.ipynb` (cell 13) draws two side-by-side
panels.

### 4.1 Left Panel — Per-Instance Global Covariance

For each instance $k$ with pixel coordinates
$\{(x_n, y_n)\}_{n=1}^{N_k}$ where $x = \text{col}$, $y = \text{row}$,
the **sample covariance matrix** is:

$$
C_k
= \frac{1}{N_k - 1}
  \sum_{n=1}^{N_k}
  \begin{pmatrix} x_n - \bar{x} \\ y_n - \bar{y} \end{pmatrix}
  \begin{pmatrix} x_n - \bar{x} \\ y_n - \bar{y} \end{pmatrix}^{\!\top}
$$

with centroid $(\bar{x}, \bar{y})$.  This is a single $2\times 2$ matrix
per instance.  Its eigenvalues $\mu_1 \le \mu_2$ determine the ellipse
axes as $2\sqrt{\mu_1}$ and $2\sqrt{\mu_2}$, and the major eigenvector
gives the orientation angle.  The ellipse is drawn at the centroid and
captures the overall shape elongation of the instance.

### 4.2 Right Panel — Local EDT Structure Tensor Glyphs

On a regular subsampled grid with spacing `step` pixels, each foreground
sample gets a small ellipse parameterised by the blended structure tensor
$S(\mathbf{p})$ at that pixel (see section 2.7):

```python
T = st_2d[:, :, r, c]                        # 2x2 blended tensor
eigvals, eigvecs = np.linalg.eigh(T)          # eigendecomposition
abs_eig = np.abs(eigvals)
idx_max, idx_min = abs_eig.argmax(), abs_eig.argmin()
ratio = abs_eig[idx_min] / abs_eig[idx_max]  # r' = isotropy ratio
angle = atan2(eigvecs[1, idx_min],            # minor eigvec direction
              eigvecs[0, idx_min])
```

| Ellipse parameter | Value | Meaning |
|---|---|---|
| centre | $(c, r)$ on the subsampled grid | pixel location |
| width | $2 R$ (fixed `glyph_radius`) | visual major axis aligned to `angle` |
| height | $2 R \cdot r'$ | visual minor axis, scaled by isotropy ratio |
| angle | $\arctan(v_y^{(1)} / v_x^{(1)})$ | minor eigenvector = boundary tangent |

Near the boundary $r' \approx 0$ and the glyph is a thin line; at the
medial axis $r' \approx 1$ and the glyph is a circle.

---

## 5. Role in the Loss

`CentroidEmbeddingLoss` uses the blended structure tensor as a
regression target for a learned linear projection head.

The projection maps the $E$-dimensional embedding $\mathbf{e}_\mathbf{p}$
at pixel $\mathbf{p}$ to an $S^2$-dimensional prediction:

$$
\hat{\mathbf{s}}_\mathbf{p}
= W_\text{cov}\, \mathbf{e}_\mathbf{p},
\qquad W_\text{cov} \in \mathbb{R}^{S^2 \times E}
$$

The target $\mathbf{s}_\mathbf{p} \in \mathbb{R}^{S^2}$ is the
row-major flattening of $S(\mathbf{p})$.  The loss averages over all
foreground pixels:

$$
\mathcal{L}_\text{cov}
= \frac{1}{N_\text{fg}}
  \sum_{\mathbf{p}\, \in\, \text{fg}}
  \bigl\lVert \hat{\mathbf{s}}_\mathbf{p} - \mathbf{s}_\mathbf{p} \bigr\rVert^2
$$

and the total loss includes it weighted by $w_\text{cov}$:

$$
\mathcal{L}_\text{total}
= A\,\mathcal{L}_\text{var}
+ B\,\mathcal{L}_\text{dst}
+ R\,\mathcal{L}_\text{reg}
+ w_\text{dir}\,\mathcal{L}_\text{dir}
+ w_\text{cov}\,\mathcal{L}_\text{cov}
+ w_\text{raw}\,\mathcal{L}_\text{raw}
$$

By regressing the structure tensor, the network learns to encode local
morphology: boundary pixels map to anisotropic targets (encoding
orientation), while interior pixels map to isotropic targets (encoding
depth without a preferred direction).

---

## 6. Spatial Conventions

All spatial quantities use **(x, y) = (col, row)** ordering:

| Index | Axis | Image convention |
|---|---|---|
| 0 | $x$ = column | increases rightward |
| 1 | $y$ = row | increases downward |
| 2 (3-D only) | $z$ = depth | increases into the volume |

The coordinate grid (`_make_coord_grid`) stacks axes in reversed order
relative to the spatial dimensions tuple: for `spatial_shape = (H, W)`,
`coords[0]` is columns ($x$) and `coords[1]` is rows ($y$).

The structure tensor gradient indices follow the same convention:
$\tilde{g}_0 = \partial D / \partial x$ (column derivative),
$\tilde{g}_1 = \partial D / \partial y$ (row derivative).

The gradient loop uses `order[S - 1 - i] = 1`, which maps index $i = 0$
to the last spatial axis (columns in 2-D) and $i = 1$ to the first
(rows in 2-D), producing the $(x, y, \ldots)$ ordering.

---

## 7. Parameter Guide

| Parameter | Default | Set by | Effect |
|---|---|---|---|
| `sigma` | 5.0 | `_compute_covariance(sigma=...)` | Integration scale $\sigma$ for the structure tensor Gaussian; also determines $\sigma_d = \max(1, \sigma/3)$ |
| `sigma_d` | $\max(1, \sigma/3)$ | computed | Derivative kernel scale; kept small relative to $\sigma$ for sharp local gradients |
| $\alpha$ | 2 | hard-coded | Exponent in $w = (D/D_\max)^\alpha$; higher values sharpen the boundary-to-interior transition |
| `step` | 8 | notebook | Grid spacing for glyph subsampling (visualisation only) |
| `glyph_radius` | `step * 0.4` | notebook | Visual half-width $R$ of each glyph ellipse (visualisation only) |
| `w_cov` | 0.0 | `CentroidEmbeddingLoss(w_cov=...)` | Weight $w_\text{cov}$ of the structure-tensor regression loss |

---

## 8. Diagnostic Checklist

If tensor glyphs look wrong, check:

1. **Interior glyphs are not isotropic (round)?**
   The EDT-blended isotropy step (section 3.4) must be present.  Verify
   that $w = (D / D_\max)^2$ is computed and that diagonal components
   are blended toward $\text{tr}(S)/S$ while off-diagonal
   components are scaled by $(1 - w)$.

2. **All glyphs are round (no boundary anisotropy)?**
   The blending exponent $\alpha$ may be too low.  Increasing it from 2
   to 3 or 4 preserves more anisotropy near the boundary.  Also check
   that $\sigma_d$ is small; a large derivative scale blurs the gradient
   field.

3. **Glyphs leak across instance boundaries?**
   Verify that gradients are multiplied by $\mathbf{1}_{M_k}$ before
   forming the outer product, and that the Gaussian smoothing is divided
   by $G_\sigma * \mathbf{1}_{M_k}$.

4. **Eigenvalues contain negatives or NaN?**
   The blended tensor is PSD by construction (convex combination of PSD
   matrices).  Negative eigenvalues are numerical noise; the notebook
   takes `abs(eigvals)` before computing the ratio.

5. **Orientation looks flipped?**
   Matplotlib `Ellipse(angle=...)` rotates in data coordinates.  With an
   inverted $y$-axis (`set_ylim(H, 0)`), the visual rotation is
   reflected.  The angle is $\arctan(v_y / v_x)$ in $(x, y)$ =
   $(\text{col}, \text{row})$ coordinates.

6. **Tensor is spatially uniform across the instance?**
   This happens when the integration $\sigma$ is set to the instance size
   (e.g. $\sigma = D_\max$), making every pixel see the same global
   average.  The correct approach is a fixed small $\sigma$ combined with
   the EDT-blending post-processing (section 2.6 / 3.4).

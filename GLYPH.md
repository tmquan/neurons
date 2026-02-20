# Tensor Glyph Formation

This document explains how the **EDT structure tensor** is computed in
`neurons.losses.discriminative._compute_covariance` and visualised as
elliptical tensor glyphs in `notebooks/00_explore_utility.ipynb`.

---

## 1. Motivation

Instance segmentation losses benefit from per-pixel geometric targets that
capture the local shape of each instance.  A natural choice is the
**structure tensor** of the Euclidean distance transform (EDT): a
symmetric positive-semi-definite 2x2 (or 3x3) matrix at every pixel that
encodes both the *direction* and *magnitude* of morphological variation.

When visualised as an ellipse (a "tensor glyph"), the structure tensor
reveals two things at a glance:

| Region | Glyph shape | Interpretation |
|---|---|---|
| Near boundary | thin ellipse, elongated along boundary tangent | gradient is coherent and normal to the boundary |
| Interior / medial axis | round circle | gradients arrive from all directions and average out |
| Elongated interior | ellipse aligned with the short axis | nearest boundary is always on the same side |

A learned linear projection `W_cov [E, S*S]` is trained to predict this
tensor from the embedding, teaching the network to encode local shape.

---

## 2. Mathematical Background

### 2.1 Euclidean Distance Transform

For a binary instance mask \( M_k \) (pixels belonging to instance \( k \)),
the EDT assigns each interior pixel the Euclidean distance to the nearest
boundary pixel:

\[
  \mathrm{EDT}(\mathbf{p}) = \min_{\mathbf{q} \notin M_k} \|\mathbf{p} - \mathbf{q}\|_2
\]

Properties:
- Zero at the boundary, maximal at the **medial axis** (skeleton ridge).
- Gradient magnitude is 1 almost everywhere inside (except at the medial
  axis where it is undefined).
- The gradient direction points radially away from the nearest boundary.

### 2.2 Gaussian Derivatives of the EDT

Rather than finite differences, the implementation uses Gaussian derivative
kernels to obtain smooth gradient estimates.  For derivative scale
\( \sigma_d \):

\[
  \nabla_{\sigma_d}\!\mathrm{EDT}
  = \bigl(G_{\sigma_d}' * \mathrm{EDT}\bigr)
  = \Bigl(\frac{\partial}{\partial x}(G_{\sigma_d} * \mathrm{EDT}),\;
          \frac{\partial}{\partial y}(G_{\sigma_d} * \mathrm{EDT})\Bigr)
\]

where \( G_\sigma \) is a Gaussian with standard deviation \( \sigma \),
and \( * \) denotes convolution.  In the code `sigma_d = max(1.0, sigma / 3.0)`.

### 2.3 Structure Tensor

The structure tensor at pixel \( \mathbf{p} \) is the Gaussian-weighted
average of the gradient outer product over a neighbourhood:

\[
  S(\mathbf{p})
  = \frac
    {(G_{\sigma_\text{int}} * (\nabla\mathrm{EDT}\;\nabla\mathrm{EDT}^\top \cdot \mathbf{1}_M))(\mathbf{p})}
    {(G_{\sigma_\text{int}} * \mathbf{1}_M)(\mathbf{p})}
\]

Key elements:

| Symbol | Meaning |
|---|---|
| \( \nabla\mathrm{EDT} \) | Gaussian derivative of the per-instance EDT (masked to the instance interior) |
| \( \nabla\mathrm{EDT}\;\nabla\mathrm{EDT}^\top \) | rank-1 outer product at each pixel |
| \( G_{\sigma_\text{int}} \) | integration Gaussian that averages the outer product |
| \( \mathbf{1}_M \) | binary indicator of the instance mask |
| denominator | mask-normalised average (avoids leakage from zero-padded exterior) |

In 2-D, \( S \) is a 2x2 symmetric matrix:

\[
  S = \begin{pmatrix} S_{xx} & S_{xy} \\ S_{xy} & S_{yy} \end{pmatrix}
\]

where \( x \) = column direction and \( y \) = row direction.

### 2.4 Eigendecomposition and Glyph Shape

The eigendecomposition \( S = V \Lambda V^\top \) yields:

- \( \lambda_{\max} \) and its eigenvector: direction of strongest gradient
  coherence (normal to the boundary).
- \( \lambda_{\min} \) and its eigenvector: direction of weakest gradient
  coherence (along the boundary tangent).

The **isotropy ratio**:

\[
  r = \frac{\lambda_{\min}}{\lambda_{\max}} \in [0, 1]
\]

| \( r \) | Meaning |
|---|---|
| \(\approx 0\) | highly anisotropic (thin ellipse) |
| \(\approx 1\) | isotropic (circle) |

---

## 3. Implementation Walkthrough

### 3.1 Per-Instance Loop

`_compute_covariance` in `discriminative.py` processes each foreground
instance independently.  For instance \( k \):

```
mask   = (labels == k)              # binary mask
dt     = scipy.ndimage.distance_transform_edt(mask)
mask_f = mask.astype(float)
```

### 3.2 Adaptive Integration Scale

A fixed integration scale cannot work for all instance sizes.  If the
Gaussian window is much smaller than the medial-axis depth, the window at
the centre only sees gradients pointing in one direction (toward the
nearest boundary), producing falsely anisotropic tensors even for round
instances.

The implementation adapts per-instance:

```
sigma_int = max(sigma, dt.max())
```

Since `dt.max()` is the medial-axis depth (maximum distance to boundary),
setting `sigma_int >= dt.max()` ensures the Gaussian weight at the boundary
is at least `exp(-1/4) ~ 0.78` of the centre weight, giving boundary
contributions from **all** directions substantial influence at the medial
axis.

| Instance size | sigma_int | Effect |
|---|---|---|
| Small (max EDT < sigma) | sigma (unchanged) | standard smoothing |
| Large (max EDT >> sigma) | max EDT | window reaches boundaries from everywhere inside |

### 3.3 Gradient Masking

The EDT is zero outside the instance mask.  Taking Gaussian derivatives
across the mask boundary creates artificial gradients from the
discontinuity.  These are suppressed by zeroing the gradient outside the
mask before forming the outer product:

```
g = gaussian_filter(dt, sigma=sigma_d, order=derivative_order)
g *= mask_f          # zero outside instance
```

### 3.4 Mask-Normalised Smoothing

Naively smoothing the outer product with `gaussian_filter` averages in
zeros from outside the instance, directionally biasing the tensor (the
short-axis boundary is closer, so more zeros leak in from that direction).
Dividing by the smoothed mask corrects this:

```
norm = max(gaussian_filter(mask_f, sigma=sigma_int), 1e-10)

S_ij = gaussian_filter(g_i * g_j, sigma=sigma_int) / norm
```

This is equivalent to computing the weighted average restricted to
instance-interior pixels only.

### 3.5 Storage Layout

The structure tensor is stored as a `[S*S, N]` tensor in row-major order
of the matrix indices, where `N = H * W` (flattened spatially in row-major
order) and `S` is the number of spatial dimensions:

```
# 2-D: indices [0..3] correspond to
#   0 -> S_xx  (col-col)
#   1 -> S_xy  (col-row)
#   2 -> S_yx  (row-col)
#   3 -> S_yy  (row-row)
```

Background pixels are left as zero.

---

## 4. Visualisation (Notebook)

The notebook `00_explore_utility.ipynb` (cell 13) draws two panels:

### 4.1 Left Panel: Per-Instance Global Covariance

For each instance, the **sample covariance matrix** of all pixel
coordinates `(col, row)` is computed.  This is a single 2x2 matrix per
instance, drawn as an ellipse at the centroid.  Axes lengths are
`2 * sqrt(eigenvalue)`.  It captures the overall shape elongation of the
instance.

### 4.2 Right Panel: Local EDT Structure Tensor Glyphs

On a regular subsampled grid (`step = 8` pixels), each foreground pixel
gets a small ellipse whose shape is derived from the local 2x2 structure
tensor:

```python
T = st_2d[:, :, r, c]                   # 2x2 structure tensor
eigvals, eigvecs = np.linalg.eigh(T)     # eigendecomposition
ratio = min_eigval / max_eigval          # isotropy ratio
angle = atan2(minor_eigvec[1], minor_eigvec[0])
```

The glyph is drawn as:

| Parameter | Value | Meaning |
|---|---|---|
| centre | `(col, row)` on the subsampled grid | pixel location |
| width  | `2 * glyph_radius` (fixed) | major axis = minor eigenvector direction |
| height | `2 * glyph_radius * ratio` | minor axis, scaled by isotropy |
| angle  | direction of the minor eigenvector | elongation along boundary tangent |

**Convention**: the ellipse is elongated along the **minor** eigenvector
(direction of least gradient coherence), which corresponds to the boundary
tangent.  Near the medial axis the ratio approaches 1 and the glyph
becomes a circle.

---

## 5. Role in the Loss

`CentroidEmbeddingLoss` uses the structure tensor as a regression target
for a learned linear projection head:

```
proj_cov: nn.Linear(E, S*S)    # E = embedding dim, S = spatial dims
```

The loss is activated by setting `w_cov > 0`:

```
L_cov = (1 / N_fg) * sum_over_fg || W_cov @ e_p  -  S(p) ||^2
```

where `e_p` is the E-dimensional embedding at pixel `p` and `S(p)` is the
`S*S`-dimensional flattened structure tensor target.

This teaches the embedding to encode local morphological shape: the
network learns to distinguish boundary pixels (anisotropic target) from
interior pixels (isotropic target), and to predict the boundary
orientation where it exists.

---

## 6. Spatial Conventions

All spatial quantities use **(x, y) = (col, row)** ordering:

| Index | Axis | Image convention |
|---|---|---|
| 0 | x = column | increases rightward |
| 1 | y = row | increases downward |

The coordinate grid (`_make_coord_grid`) stores axes in reversed order
relative to spatial dimensions: for `spatial_shape = (H, W)`,
`coords[0]` is columns (x) and `coords[1]` is rows (y).

The structure tensor gradient indices follow the same convention:
`grads[0]` = `dEDT/dx` (column derivative), `grads[1]` = `dEDT/dy`
(row derivative).

---

## 7. Parameter Guide

| Parameter | Default | Set by | Effect |
|---|---|---|---|
| `sigma` | 5.0 | `_compute_covariance(sigma=...)` | Minimum integration scale; also controls `sigma_d = max(1, sigma/3)` for the derivative kernel |
| `sigma_int` | `max(sigma, max_edt)` | computed per-instance | Actual integration scale; adapts to instance size |
| `sigma_d` | `max(1.0, sigma / 3.0)` | computed once | Derivative kernel scale; kept small for sharp gradients |
| `step` | 8 | notebook | Grid spacing for glyph subsampling (visualisation only) |
| `glyph_radius` | `step * 0.4` | notebook | Half-width of each glyph ellipse (visualisation only) |
| `w_cov` | 0.0 | `CentroidEmbeddingLoss(w_cov=...)` | Weight of the structure-tensor regression loss |

---

## 8. Diagnostic Checklist

If tensor glyphs look wrong, check:

1. **Interior glyphs are not isotropic (round)?**
   `sigma_int` must be at least `dt.max()` per instance so the integration
   window reaches boundaries from the medial axis.

2. **All glyphs are round (no boundary anisotropy)?**
   `sigma_d` may be too large, blurring the gradient field.  It should be
   much smaller than `sigma_int`.

3. **Glyphs leak across instance boundaries?**
   Verify that gradients are multiplied by `mask_f` before forming the
   outer product, and that the Gaussian smoothing is divided by the
   smoothed mask.

4. **Eigenvalues contain negatives or NaN?**
   The structure tensor is PSD by construction; negative eigenvalues come
   from numerical noise.  The code takes `abs(eigvals)` before computing
   the ratio.

5. **Orientation looks flipped?**
   Matplotlib's `Ellipse(angle=...)` rotates in data coordinates.  With
   an inverted y-axis (`set_ylim(H, 0)`), the visual rotation is
   reflected.  The angle is `atan2(eigvec_y, eigvec_x)` in (col, row)
   coordinates.

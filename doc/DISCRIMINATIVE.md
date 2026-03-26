# Discriminative Instance Loss

A deep dive into `InstanceLoss` (`losses/instance.py`): the discriminative
embedding loss that trains per-pixel embeddings to cluster into distinct
instances.

## 1. Background

The discriminative loss for instance segmentation originates from
[De Brabandere et al., 2017](https://arxiv.org/abs/1708.02551). The core idea:
a network outputs an E-dimensional embedding vector for every pixel (or voxel).
During training, three complementary forces shape the embedding space:

| Force | Goal | Analogy |
| ----- | ---- | ------- |
| **Pull** | Embeddings of the *same* instance converge to their centroid | Gravity within a cluster |
| **Push** | Centroids of *different* instances repel each other | Electrostatic repulsion between clusters |
| **Norm** | Centroids stay near the origin | Spring anchoring the constellation |

At inference, a simple mean-shift or seed-based clustering in embedding space
recovers the instance segmentation.

## 2. Mathematical Formulation

Given a batch element with **K** foreground instances, let:

- $\mathbf{x}_i \in \mathbb{R}^E$ be the embedding at pixel $i$,
- $S_k$ be the set of pixels belonging to instance $k$,
- $N_k = |S_k|$,
- $\boldsymbol{\mu}_k = \frac{1}{\sum_{i \in S_k} w_i} \sum_{i \in S_k} w_i \, \mathbf{x}_i$
  be the **weighted centroid** of instance $k$, where $w_i$ is the per-pixel
  weight (see [Section 5](#5-per-pixel-weighting-boundary--skeleton)).

### 2.1 Pull Loss (Intra-Cluster Variance)

Each embedding is pulled toward its instance centroid with a hinge at
$\delta_v$:

$$
L_{\text{pull}} = \frac{1}{K} \sum_{k=1}^{K} \frac{1}{N_k}
  \sum_{i \in S_k} w_i \cdot
  \Big[\,\|\mathbf{x}_i - \boldsymbol{\mu}_k\| - \delta_v\,\Big]_+^{\,2}
$$

where $[\cdot]_+ = \max(\cdot, 0)$ is the hinge.

**Effect of $\delta_v$**: pixels within distance $\delta_v$ of their
centroid incur zero pull loss. This slack prevents the loss from wasting
capacity compressing already-tight clusters; training focuses on outliers
that lie beyond the margin. Typical value: **0.5**.

### 2.2 Push Loss (Inter-Cluster Separation)

Every pair of centroids is pushed apart with a hinge at $2\delta_d$:

$$
L_{\text{push}} = \frac{1}{\binom{K}{2}} \sum_{k_a < k_b}
  \Big[\,2\delta_d - \|\boldsymbol{\mu}_{k_a} - \boldsymbol{\mu}_{k_b}\|\,\Big]_+^{\,2}
$$

**Effect of $\delta_d$**: centroids separated by more than $2\delta_d$ incur
zero push loss. The factor of 2 comes from the symmetry of the margin — each
centroid "owns" a ball of radius $\delta_d$. Once the balls no longer overlap,
the push term turns off. Typical value: **1.5** (so centroids must be at least
3.0 apart).

### 2.3 Norm Loss (Centroid Regularisation)

$$
L_{\text{norm}} = \frac{1}{K} \sum_{k=1}^{K} \|\boldsymbol{\mu}_k\|
$$

Prevents the embedding space from drifting arbitrarily far from the origin,
which would make the push term trivially satisfiable (just place centroids
at infinity).

### 2.4 Total Instance Loss

$$
L_{\text{instance}} =
  \alpha \, L_{\text{pull}} +
  \beta  \, L_{\text{push}} +
  \gamma \, L_{\text{norm}}
$$

| Config key | Symbol | Default | Role |
| ---------- | ------ | ------- | ---- |
| `weight_pull` | $\alpha$ | 1.0 | Pull importance |
| `weight_push` | $\beta$  | 1.0 | Push importance |
| `weight_norm` | $\gamma$ | 0.001 | Norm importance (kept small) |

## 3. Implementation Walk-Through

### 3.1 Flattening with einops

The spatial dimensions (2D or 3D) are collapsed into a single pixel axis so
all downstream operations are shape-agnostic:

```python
emb_flat = rearrange(embed, "b e ... -> b e (...)")   # [B, E, N]
lbl_flat = rearrange(label, "b ... -> b (...)")        # [B, N]
wgt_flat = rearrange(w_edge * w_bone, "b ... -> b (...)") # [B, N]
```

The `...` wildcard matches `(H, W)` in 2D or `(D, H, W)` in 3D, and `(...)`
merges those axes into a flat pixel count `N`. This is the only
dimension-dependent line — everything after operates on `[B, E, N]`.

### 3.2 Weighted Centroid via Scatter

`_scatter_weighted_mean` computes the weighted centroid of each instance using
`scatter_add_`, avoiding explicit per-instance loops:

```python
# weighted_emb: [E, M]  (foreground pixels only)
# lbl_fg:       [M]     (zero-based instance indices)
c_sum = torch.zeros(E, K)
c_sum.scatter_add_(1, lbl_fg.expand(E, -1), weighted_emb)  # sum per instance
w_sum = torch.zeros(K)
w_sum.scatter_add_(0, lbl_fg, wgt_fg)                      # weight sum per instance
centers = (c_sum / w_sum).T  # [K, E]
```

This is equivalent to a weighted group-by-mean but runs entirely on GPU
without Python loops over K.

### 3.3 Pull Term — Per-Pixel Hinged Distance

```python
center_per_pixel = centers[inverse]       # [M, E] — broadcast centroid to each pixel
emb_fg = emb_b[:, fg].T                   # [M, E]

dist = ((emb_fg - center_per_pixel) ** 2).sum(dim=1).clamp(min=1e-12).sqrt()
pull_per_pixel = (dist - self.delta_v).clamp(min=0).pow(2)
```

1. L2 distance from each foreground pixel to its centroid.
2. Hinge: subtract $\delta_v$, clamp negatives to zero.
3. Square the residual (smooth gradient near the margin).

Pixel weights are multiplied in, then the loss is averaged per-instance
(via `scatter_add_` + `bincount`) and finally averaged over instances.

### 3.4 Push Term — Pairwise Centroid Repulsion with einops

```python
pw_diff = (rearrange(centers, "i e -> i 1 e") -
           rearrange(centers, "j e -> 1 j e"))        # [K, K, E]
pw = (pw_diff ** 2).sum(dim=2).clamp(min=1e-12).sqrt()  # [K, K]
triu = torch.triu_indices(K, K, offset=1)

loss_push += reduce(
    (2 * self.delta_d - pw[triu[0], triu[1]]).clamp(min=0).pow(2),
    "n -> ", "mean",
)
```

The two `rearrange` calls broadcast the centroid matrix into a pair-wise
difference tensor without an explicit double loop:

- `"i e -> i 1 e"` adds a broadcast dimension at position 1 (the "j" axis).
- `"j e -> 1 j e"` adds a broadcast dimension at position 0 (the "i" axis).
- Subtraction yields `[K, K, E]` where entry `[a, b, :]` is
  $\boldsymbol{\mu}_a - \boldsymbol{\mu}_b$.

Only the upper triangle is kept (`triu_indices` with `offset=1`) since the
matrix is antisymmetric and the diagonal is zero.

`reduce("n -> ", "mean")` (einops) computes the scalar mean of the upper-
triangle push penalties, equivalent to `.mean()` but explicit about
collapsing all axes.

### 3.5 Norm Term

```python
loss_norm += (centers ** 2).sum(dim=1).clamp(min=1e-12).sqrt().mean()
```

L2 norm of each centroid, averaged over K instances.

### 3.6 Skeleton Weight — Einops Normalisation

Inside `_skeleton_weight_torch`, the per-instance distance transform is
normalised to [0, 1] using einops:

```python
max_d = reduce(dt, "k c ... -> k c", "max")          # per-instance max
dt = dt / rearrange(max_d, "k c -> k c 1 1 1")       # broadcast division
```

The `reduce` collapses spatial dims to get the max EDT value per instance.
The `rearrange` adds trailing singleton dimensions to match the spatial
shape for broadcasting. The pattern `"k c -> k c 1 1 1"` is dynamically
constructed as `f"k c -> k c {' '.join(['1'] * spatial_dims)}"` to handle
both 2D and 3D.

### 3.7 Semantic-Class Partitioning

When `semantic_ids` is provided, the loss is computed **independently per
semantic class** and averaged:

```python
for cid in classes:
    out = self._loss_single(
        embed, label * (semantic_ids == cid).long(), weight_edge, weight_bone,
    )
```

This prevents the push term from wasting capacity separating instances that
already belong to different semantic classes (e.g., neurites vs. mitochondria).

## 4. Distance Parameters

### 4.1 δ_v — Pull Margin (`delta_v`)

| Value | Effect |
| ----- | ------ |
| 0.0 | No slack — every pixel is penalised for *any* distance from centroid |
| **0.5** (default) | Pixels within 0.5 of their centroid are "close enough" |
| > 1.0 | Very lax — clusters can be diffuse without penalty |

**Tuning guide**: set $\delta_v$ to roughly the acceptable intra-cluster
scatter. If downstream clustering uses a bandwidth of $b$, a good starting
point is $\delta_v \approx b/2$.

### 4.2 δ_d — Push Margin (`delta_d`)

| Value | Effect |
| ----- | ------ |
| Small (< 1.0) | Centroids only need to be > $2\delta_d$ apart — easy but fragile |
| **1.5** (default) | Centroids must be > 3.0 apart — comfortable margin for clustering |
| Large (> 3.0) | Very aggressive separation — can cause slow convergence |

**Relationship to $\delta_v$**: the embedding space is well-posed when
$\delta_d > \delta_v$. The gap $\delta_d - \delta_v$ is the "no-man's
land" between the pull and push radii where neither term applies — a
buffer zone that prevents the two forces from fighting.

### 4.3 Interaction Between δ_v and δ_d

```
                              ◄── 2δ_d ──►
             ┌───────────────────────────────────────────┐
             │                                           │
    ─────────┤    ●────δ_v────μ_a────δ_v────●            │
             │         pull zone a                       │
             │                                           │
             │              gap (no gradient)            │
             │                                           │
             │            ●────δ_v────μ_b────δ_v────●    │
             │                  pull zone b              │
             └───────────────────────────────────────────┘
                            push zone
```

Embeddings inside their pull radius (circle of radius $\delta_v$ around
each centroid) incur zero pull loss. Centroid pairs outside the push
diameter ($2\delta_d$) incur zero push loss. The configuration is stable
when each cluster fits within its $\delta_v$-ball and all balls are
separated by at least $2\delta_d$.

## 5. Per-Pixel Weighting: Boundary + Skeleton

Raw discriminative loss treats all foreground pixels equally. This is
problematic for connectomics where:

- **Touching instances** share thin boundaries that the model must get
  exactly right (a single misclassified boundary pixel can merge two neurons).
- **Elongated instances** have medial axes (skeletons) that carry topological
  information — breaking the skeleton splits the instance.

The implementation multiplies two independent weight maps element-wise:

```
w_total = w_boundary ⊙ w_skeleton
```

### 5.1 Boundary Weight (`weight_edge`)

| Config key | Default | Meaning |
| ---------- | ------- | ------- |
| `weight_edge` | 10.0 | Boundary pixels receive this weight; non-boundary pixels receive 1.0 |

Computed via **morphological gradient**: a pixel is on the boundary if the
local max-pool and min-pool of the label map disagree (i.e., two different
instance labels appear in the 3×3×3 neighbourhood). The GPU path uses
`find_boundaries(mode="inner", connectivity=1)` reimplemented in pure PyTorch
via `max_pool3d` / `max_pool2d`, avoiding CPU round-trips.

The weight map is:

$$
w_{\text{edge}}(i) = 1 + \mathbb{1}[\text{boundary}(i)] \cdot (\texttt{weight\_edge} - 1)
$$

Setting `weight_edge = 1.0` disables boundary weighting (the map is not
computed at all).

### 5.2 Skeleton Weight (`weight_bone`)

| Config key | Default | Meaning |
| ---------- | ------- | ------- |
| `weight_bone` | 10.0 | Medial-axis (skeleton) pixels receive this weight; instance periphery receives 1.0 |

The weight is proportional to the **normalised Euclidean Distance Transform
(EDT)**: for each instance, the EDT is computed (distance from each interior
pixel to the nearest boundary), then divided by the instance's maximum EDT
value. This maps to [0, 1], where 1.0 is at the medial axis.

**GPU path** (avoids scipy): iterative morphological erosion via
`-max_pool(-mask, 3, stride=1, padding=1)`, peeling one layer per iteration
and recording the layer index (an approximation to the L∞ distance
transform). Each instance's DT is normalised using einops `reduce` and
`rearrange` (see Section 3.6).

**CPU path**: scipy `distance_transform_edt` via `pmap` (parallelised across
instances and batch elements in a single pool).

The weight map is:

$$
w_{\text{bone}}(i) = 1 + \text{EDT}_{\text{norm}}(i) \cdot (\texttt{weight\_bone} - 1)
$$

Setting `weight_bone = 1.0` disables skeleton weighting.

### 5.3 Combined Effect

| Region | $w_{\text{edge}}$ | $w_{\text{bone}}$ | $w_{\text{total}}$ (defaults) |
| ------ | ------------------- | ------------------- | ------------------------------- |
| Interior, away from skeleton | 1.0 | ~1.0 | ~1.0 |
| Interior, on skeleton | 1.0 | 10.0 | 10.0 |
| Boundary, thin | 10.0 | ~1.0 | ~10.0 |
| Boundary at skeleton tip | 10.0 | high | up to 100.0 |

The multiplicative combination means skeleton-boundary intersections (branch
tips touching another instance) receive the strongest gradient signal.

## 6. Full Parameter Reference

| Parameter | Config key | Type | Default | Description |
| --------- | ---------- | ---- | ------- | ----------- |
| `spatial_dims` | (constructor) | int | 3 | 2 for 2D, 3 for 3D; selects pool function and pad tuple |
| `weight_pull` | `weight_pull` | float | 1.0 | Multiplier on $L_{\text{pull}}$ |
| `weight_push` | `weight_push` | float | 1.0 | Multiplier on $L_{\text{push}}$ |
| `weight_norm` | `weight_norm` | float | 0.001 | Multiplier on $L_{\text{norm}}$; kept small to avoid overwhelming pull/push |
| `weight_edge` | `weight_edge` | float | 10.0 | Boundary pixel weight; 1.0 = disabled |
| `weight_bone` | `weight_bone` | float | 10.0 | Skeleton pixel weight; 1.0 = disabled |
| `delta_v` | `delta_v` | float | 0.5 | Pull hinge margin; pixels closer than this to centroid are penalty-free |
| `delta_d` | `delta_d` | float | 1.5 | Half the push hinge margin; centroids farther than $2\delta_d$ are penalty-free |

## 7. Typical Configurations

### Connectomics (3D volumes, touching neurites)

```yaml
weight_pull: 1.0
weight_push: 1.0
weight_norm: 0.001
delta_v: 0.5
delta_d: 1.5
weight_edge: 2.0    # moderate boundary boost
weight_bone: 2.0    # moderate skeleton boost
```

### Aggressive boundary separation

```yaml
weight_edge: 10.0
weight_bone: 10.0
delta_d: 2.0         # demand wider centroid separation
```

### No pixel weighting (vanilla discriminative loss)

```yaml
weight_edge: 1.0
weight_bone: 1.0
```

## 8. Computational Notes

- **No Python loops over instances** for the centroid computation (scatter-based).
- **One Python loop over batch elements** is required because instances differ
  per sample.
- The push term is $O(K^2)$ in the number of instances; this is acceptable
  for connectomics patch sizes (typically K < 100).
- Boundary and skeleton weight maps are computed inside `@torch.no_grad()`
  and detached from the computation graph — they modulate gradients but do
  not receive gradients themselves.
- When both `weight_edge` and `weight_bone` are 1.0, the weight computation
  is skipped entirely (returns `None`), saving a full-volume allocation.

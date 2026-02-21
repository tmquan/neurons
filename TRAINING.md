# Training Modes

The training loop supports two modes that run **in the same step** on every batch.
When both are enabled the losses are averaged and a single backward pass updates all
weights (backbone, task heads, and point prompt encoder together).

```
training_step(batch)
│
├── targets = _prepare_targets(batch)
│       ├── semantic_labels   [B, *spatial]   (labels > 0).long()
│       ├── labels            [B, *spatial]   instance ids (0 = bg)
│       └── raw_image         [B, 1, *spatial] input image for L_raw
│
├── mode = "automatic"
│   └── predictions = model(images)                        # no prompts
│       └── loss_auto = criterion(predictions, targets)
│
├── mode = "proofread"
│   ├── sub_mode?
│   │   ├── fractionary  (labels contain ignore_index)
│   │   │   └── resolve labels → model(images, semantic_ids=...)
│   │   └── interactive  (fully annotated)
│   │       └── sample points → model(images, point_prompts=...)
│   └── loss_proof = criterion(predictions, targets)
│
└── total_loss = (loss_auto + loss_proof) / 2
```

Configure which modes run in `training.training_modes`:

```yaml
training:
  training_modes:
    - automatic          # always recommended
    - proofread          # adds the interactive/fractionary branch
  num_pos_points: 5      # positive prompt points per sample
  num_neg_points: 5      # negative prompt points per sample
  point_sample_mode: instance  # "class" or "instance"
```

---

## 1. Automatic Mode

The baseline mode.  The model sees only the raw EM image and must predict
everything from scratch — no hints, no prompts.

### Forward path

```
image [B, 1, D, H, W]
  │
  ▼
backbone (SegResNet / Vista3D)
  │
  ▼
feat [B, F, D, H, W]        (F = feature_size, e.g. 64)
  │
  ├──▶ head_semantic  → semantic  [B, C, D, H, W]   class logits
  ├──▶ head_instance  → instance  [B, E, D, H, W]   embedding vectors
  └──▶ head_geometry  → geometry  [B, G, D, H, W]   dir + cov + raw
```

### Loss

The criterion computes three terms on the predictions:

| Term | Head | Components | Description |
|------|------|------------|-------------|
| **L_sem** | semantic | CE + IoU + Dice | Per-voxel class loss |
| **L_ins** | instance | pull + push + norm | Discriminative clustering loss |
| **L_geom** | geometry | L_dir + L_cov + L_raw | Geometry regression loss |

```
loss = w_sem * L_sem  +  w_ins * L_ins  +  w_geom * L_geom
```

All targets are derived from the instance label map alone (the ground truth):

- **L_sem** — cross-entropy between predicted class logits and binary
  foreground / background labels (or multi-class `semantic_ids` when
  available).  Optional IoU and Dice soft losses are added.

- **L_ins** — the discriminative loss from De Brabandere et al. (2017):
  - *pull*: hinge-L2 from each voxel's embedding to its instance centroid.
    Weighted by boundary (`weight_edge`) and skeleton depth (`weight_bone`)
    maps, so the model pays more attention to boundaries and medial axes.
    Averaged over instances, then over batch.
  - *push*: pairwise margin on instance centroids — pushes different
    instances apart by at least `2 * delta_d`.
  - *norm*: L2 regularisation on centroid embeddings.

- **L_geom** — per-voxel regression on three channel groups:
  - *L_dir* (`S` channels): unit direction from each foreground voxel toward
    its instance centroid (or nearest skeleton point if `dir_target=skeleton`).
    Target computed on-the-fly.
  - *L_cov* (`S*S` channels): EDT structure tensor — the smoothed outer
    product of the distance-transform gradient, encoding local shape.
    Blended toward isotropy at the medial axis.  Target computed on-the-fly.
  - *L_raw* (`4` channels): RGBA reconstruction of the input image.
    Target = `[R, G, B, alpha]` where RGB copies the grayscale input and
    alpha = foreground mask.  Predicted through `sigmoid` so both prediction
    and target live in `[0, 1]`.

### When to use automatic alone

- Early experiments / debugging (simpler, one forward pass per step).
- When you only have fully-annotated data and do not need interactive
  inference later.

---

## 2. Proofread Mode

Proofread mode teaches the model to leverage **additional context** beyond
the image — either partial annotation or point prompts.  This is critical
for interactive segmentation at inference time: a human clicks on an object
and the model refines its prediction.

Proofread has two sub-modes, selected automatically per batch:

### 2a. Interactive sub-mode (fully annotated data)

Triggered when the labels do **not** contain any `ignore_index` values — the
patch is fully annotated.  Since we already have full ground truth, we
*simulate* an interactive session:

1. **Sample a target** — pick a random foreground instance (mode `"instance"`)
   or a random foreground class (mode `"class"`).

2. **Sample point prompts** from ground truth:
   - `num_pos_points` positive points sampled uniformly from the target mask.
   - `num_neg_points` negative points sampled from everywhere else (background
     + other instances).

3. **Encode prompts** — the `PointPromptEncoder` builds a sparse indicator
   volume with `num_classes + 3` channels:

   | Channel(s) | Content |
   |------------|---------|
   | 0 | `+1` at each positive point |
   | 1 | `+1` at each negative point |
   | 2 .. 2+C-1 | one-hot of target semantic class at all point locations |
   | -1 | binary instance indicator at positive points |

   A small Conv + GroupNorm + ReLU block projects this to `[B, F, *spatial]`.

4. **Residual injection** — the encoded prompt is **added** to the backbone
   features before the task heads:

   ```
   feat_proofread = backbone(image) + point_encoder(prompts)
   ```

   At initialization the encoder's conv weights are near-zero (`std=1e-4`),
   so the residual is negligible and the model starts as if in automatic mode.
   As training progresses the encoder learns to modulate features based on
   the user-provided points.

5. **Full loss** — the same three-term criterion (L_sem + L_ins + L_geom)
   is computed on the proofread predictions against the full targets.  The
   model is expected to improve *all* predictions given the extra context,
   not just the prompted instance.

### 2b. Fractionary sub-mode (partially annotated data)

Triggered when labels contain **both** valid foreground IDs and `ignore_index`
values in the same patch — meaning the annotator labelled some regions but
left others unknown.

1. **Resolve labels** (`_resolve_fractionary_labels`):
   - Semantic labels at unknown voxels → set to `ignore_index` (excluded
     from CE loss).
   - Instance labels at unknown voxels → set to 0 (treated as background
     for the discriminative loss).
   - Valid instance IDs are remapped to contiguous integers `1, 2, …`.
   - A `semantic_ids` tensor is built so the instance loss can run per-class.

2. **Forward with `semantic_ids`** — passed through the model so the
   prediction dict carries class information for per-class instance loss.

3. **Loss** — same criterion, but the CE loss automatically ignores
   unknown voxels (via `ignore_index`), and the instance loss only sees
   the remapped known foreground.

### When to use proofread

- You plan to deploy interactive segmentation (user clicks to refine).
- You have a mix of fully- and partially-annotated volumes.
- You want the model to learn prompt-conditioned behaviour alongside
  automatic segmentation.

---

## 3. How the Two Modes Combine

When both modes are enabled, every training step runs **two forward passes**
through the model:

```
                     ┌──────────────────┐
                     │   Same backbone  │
                     │   Same heads     │
  image ────────────▶│   Same weights   │
                     └──────────────────┘
                          │         │
              automatic   │         │  proofread
              (no prompts)│         │  (+ point encoder residual)
                          ▼         ▼
                     predictions₁  predictions₂
                          │         │
              criterion(p₁,tgt)  criterion(p₂,tgt)
                          │         │
                     loss_auto   loss_proof
                          │         │
                          ▼         ▼
                total = (loss_auto + loss_proof) / 2
                          │
                          ▼
                     backward()
```

Key properties:

- **Shared weights** — both passes update the same backbone and task heads.
  The point encoder is only exercised by proofread mode, so it only receives
  gradients from that branch.

- **Averaged loss** — the final scalar is the mean of both mode losses.
  Per-mode sub-losses are logged separately under `train/automatic/*` and
  `train/proofread/*` for TensorBoard inspection.

- **No interference at init** — the point encoder starts near-zero, so
  both modes produce nearly identical predictions at the beginning of
  training.  The proofread branch gradually diverges as the encoder learns.

---

## 4. Loss Components Reference

### Semantic loss

```
L_sem = w_ce * CrossEntropy  +  w_iou * (1 - SoftIoU)  +  w_dice * (1 - SoftDice)
```

### Instance loss

```
L_ins = w_pull * L_pull  +  w_push * L_push  +  w_norm * L_norm
```

- `L_pull`: per-instance weighted mean of `relu(||e_i - μ_k|| - δ_v)²`, where
  weights come from `weight_edge` (boundary boost) and `weight_bone` (skeleton
  depth boost).  Averaged over instances, then over batch.
- `L_push`: `mean( relu(2·δ_d - ||μ_i - μ_j||)² )` over all centroid pairs.
- `L_norm`: mean centroid L2 norm (regularisation).

### Geometry loss

```
L_geom = w_dir * L_dir  +  w_cov * L_cov  +  w_raw * L_raw
```

All terms use foreground-masked MSE averaged over foreground voxels and
channels.

- `L_dir`: target = unit offset toward instance centroid (or skeleton).
- `L_cov`: target = EDT structure tensor (smoothed gradient outer product).
- `L_raw`: target = `[img, img, img, fg_mask]` in `[0, 1]`.
  Prediction passed through `sigmoid` before MSE.

---

## 5. Configuration Quick Reference

```yaml
loss:
  # Top-level branch weights
  weight_semantic: 1.0
  weight_instance: 1.0
  weight_geometry: 1.0      # set 0.0 to disable geometry head

  # Semantic
  weight_ce: 1.0
  weight_iou: 1.0
  weight_dice: 1.0

  # Instance
  weight_pull: 1.0
  weight_push: 1.0
  weight_norm: 0.001
  delta_v: 0.5              # pull hinge margin
  delta_d: 1.5              # push margin (centroids pushed apart by 2·δ_d)
  weight_edge: 10.0         # boundary pixel weight multiplier
  weight_bone: 10.0         # skeleton pixel weight multiplier

  # Geometry
  weight_dir: 1.0
  weight_cov: 1.0
  weight_raw: 1.0
  dir_target: centroid       # "centroid" or "skeleton"

training:
  training_modes:
    - automatic
    - proofread
  num_pos_points: 5
  num_neg_points: 5
  point_sample_mode: instance  # "class" or "instance"
```

---

## 6. TensorBoard Logged Scalars

When both modes are active, the following keys are logged per step:

| Key pattern | Example | Description |
|-------------|---------|-------------|
| `train/loss` | — | Averaged total across modes (shown on progress bar) |
| `train/{mode}/loss` | `train/automatic/loss` | Total loss for one mode |
| `train/{mode}/loss_sem` | `train/proofread/loss_sem` | Semantic loss |
| `train/{mode}/loss_sem/ce` | — | CE component |
| `train/{mode}/loss_sem/iou` | — | IoU component |
| `train/{mode}/loss_sem/dice` | — | Dice component |
| `train/{mode}/loss_ins` | — | Instance loss |
| `train/{mode}/loss_ins/pull` | — | Pull component |
| `train/{mode}/loss_ins/push` | — | Push component |
| `train/{mode}/loss_ins/norm` | — | Norm component |
| `train/{mode}/loss_geom` | — | Geometry loss |
| `train/{mode}/loss_geom/dir` | — | Direction component |
| `train/{mode}/loss_geom/cov` | — | Covariance component |
| `train/{mode}/loss_geom/raw` | — | RGBA reconstruction component |

Compare `train/automatic/loss` vs `train/proofread/loss` to check whether
the proofread branch is training at a similar scale.  A large gap
early on usually means the point encoder is disrupting backbone features
(check GroupNorm initialisation) or that the prompted targets are
misaligned.

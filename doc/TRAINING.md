# Training Modes

The training loop supports two modes that run **in the same step** on every batch.
When both are enabled the losses are averaged and a single backward pass updates all
weights (backbone, task heads, and point prompt encoder together).

> **Note:** Proofread mode is currently implemented for the **Vista** family only
> (Vista3D / Vista2D).  The Cosmos modules (`BaseCosmosModule`) raise
> `NotImplementedError` if proofread is enabled.

All Lightning modules inherit from `BaseVistaModule` (Vista family) in
`modules/vista.py` or `BaseCosmosModule` (Cosmos family) in `modules/cosmos.py`,
which centralise the shared `training_step` / `validation_step` logic, target
preparation, and scheduler configuration.  Loss composition is handled by
`BaseCombinedLoss` in `losses/cosmos.py`.

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

  # Geometry (auxiliary — not used at inference)
  weight_dir: 1.0
  weight_cov: 0.0            # 0.0 recommended: disables expensive structure tensor
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

---

## 7. TensorBoard Image Visualizations

The `ImageLogger` callback logs visual grids at the end of each epoch:

| Panel | Content |
|---|---|
| `image` | EM input (grayscale) |
| `label` | Ground truth instance labels (random color per ID) |
| `semantic` | Predicted semantic class (argmax) |
| `instance_pca` | Instance embedding (PCA → RGB) |
| `instance_pred` | Clustered instances (mean-shift on **predicted** fg mask) |
| `geometry_dir_{centroid\|skeleton}` | Direction vectors as quiver arrows (orange) |
| `geometry_cov` | Structure tensor as ellipse glyphs (cyan) |
| `geometry_raw` | RGBA reconstruction (sigmoid output) |

For 3D volumes, the central Z-slice is displayed.

### Quick test command

Run a single epoch on one GPU to verify visualizations render correctly:

```bash
env CUDA_VISIBLE_DEVICES='0' PYTHONPATH=$(pwd) python scripts/train.py \
    --config-name snemi3d \
    training.max_epochs=1 \
    training.devices=1 \
    training.strategy=auto \
    training.limit_val_batches=2 \
    data.num_samples=16

tensorboard --logdir=outputs/
```

This runs ~8 training steps (16 samples / batch 2), triggers the epoch-end
callback, and produces all visualization panels. Open TensorBoard at
`http://localhost:6006` to inspect.

### Full training command

Multi-GPU training on 4 GPUs with default hyperparameters:

```bash
env CUDA_VISIBLE_DEVICES='0,1,2,3' PYTHONPATH=$(pwd) python scripts/train.py --config-name combine
```

---

## 8. Performance Tips

### Disable L_cov (structure tensor)

The covariance target computation is the single most expensive operation
per training step — it runs EDT + multiple gaussian_filters per instance
per batch element.  Set `weight_cov: 0.0` to skip it entirely:

```yaml
loss:
  weight_cov: 0.0    # eliminates ~5000 cupy ops/step
```

L_dir and L_raw remain active and provide useful auxiliary gradients at
negligible cost.

### GPU acceleration with cupy

Install [cupy](https://cupy.dev/) to accelerate EDT, gaussian_filter,
and connected-component labelling on GPU.  Data transfers between PyTorch
and cupy use DLPack zero-copy (no host round-trips).

The codebase automatically falls back to scipy when cupy is unavailable,
and to sequential scipy inside DataLoader workers (where CUDA contexts
are invalid after fork).

### Data pipeline tuning

- **`cache_rate: 0.0`** with `fork`-based workers: volumes are loaded
  once and shared via copy-on-write.  No need for MONAI caching.
- **`num_workers: 8`** is a reasonable default for 2 volumes with
  scipy-based transforms in each worker.
- **`persistent_workers: true`** (default) keeps worker processes alive
  between epochs, avoiding re-fork overhead.
- **`prefetch_factor: 2`** (default) ensures the next batch is ready
  while the current batch trains.

### Validation budget

`limit_val_batches` controls how many validation batches run per epoch.
Each val batch computes the full loss (including geometry targets) plus
SoftMeanShift clustering and four sklearn metrics.  For faster iteration,
reduce to 10--20:

```yaml
training:
  limit_val_batches: 10
```

### DDP strategy

When only `automatic` mode is active, the point encoder receives no
gradients.  The training script automatically sets
`find_unused_parameters=True` and `static_graph=False` for this case.
When both `automatic` and `proofread` modes are enabled, all parameters
are used and `static_graph=True` is set for better DDP performance.

---

## 9. Cosmos Freeze Strategy

When using CosmosPredict3D or CosmosTransfer3D with pretrained weights
from natural video generation, a phased freeze strategy produces better
results than training everything end-to-end from step 1.

### 9.1 The Domain Gap

Cosmos was pretrained on **natural RGB video** (diverse scenes, textures,
motion).  Connectomics EM data is fundamentally different:

| Property | Natural video | Connectomics EM |
|---|---|---|
| Channels | 3-ch RGB | 1-ch grayscale |
| Content | Diverse objects, scenes | Repetitive ultrastructure (membranes, vesicles) |
| Resolution | Isotropic frames | Anisotropic (Z is 5--10x coarser than XY) |
| Boundaries | Sparse object edges | Dense instance boundaries (thousands of touching neurons) |

The DiT backbone's low/mid-level features (edges, textures, spatial
relationships) transfer well across domains.  The high-level features and
all task-specific outputs must be learned from scratch.

### 9.2 Three-Phase Training

#### Phase 1: Frozen backbone -- train heads only

```yaml
model:
  freeze_vae_encoder: true       # VAE encoder (tokenizer) -- frozen
  freeze_dit_backbone: true      # DiT backbone -- frozen
  freeze_vae_decoder: false      # VAE decoder -- trainable

optimizer:
  lr: 1.0e-4
  scheduler:
    type: cosine
    T_max: 40

training:
  max_epochs: 40
```

**What trains**: input adapter, feature projector, VAE decoder, task heads,
point encoder (~5M parameters).

**Rationale**: The randomly initialized heads produce noisy gradients.
If these flow into the pretrained DiT from step 1, they can damage the
good pretrained features.  By freezing the backbone first, the heads
converge to reasonable outputs cheaply and quickly.

**Duration**: 20--40 epochs, until head losses plateau.

#### Phase 2: Unfreeze backbone with differential LR

Resume from the Phase 1 checkpoint:

```yaml
model:
  freeze_vae_encoder: true       # VAE encoder -- still frozen
  freeze_dit_backbone: false     # DiT backbone -- now trainable at lower LR
  freeze_vae_decoder: false      # VAE decoder -- trainable

optimizer:
  lr: 1.0e-4                 # head + decoder learning rate
  dit_backbone_lr: 1.0e-5    # 10x lower for pretrained DiT backbone
  scheduler:
    type: cosine_warmup       # warm up the backbone slowly
    warmup_epochs: 5
    T_max: 160

training:
  max_epochs: 160
```

**What trains**: everything except the VAE encoder.  The DiT backbone
receives gradients at 10x lower learning rate via the `dit_backbone_lr`
parameter (already supported by all Cosmos modules).

**Rationale**: Once the heads have converged, the backbone can be
fine-tuned to produce better features for EM data specifically.  The
lower LR prevents catastrophic forgetting of the pretrained
representations.  The warmup avoids large gradient shocks to the
pretrained weights in the first few epochs.

**Duration**: 100--160 epochs for full convergence.

**Resuming from Phase 1**: Point the training script at the Phase 1
checkpoint via Hydra or PyTorch Lightning's `ckpt_path` mechanism:

```bash
PYTHONPATH=$(pwd) python scripts/train.py \
    --config-name combine \
    model.freeze_dit_backbone=false \
    optimizer.dit_backbone_lr=1.0e-5 \
    optimizer.scheduler.type=cosine_warmup \
    training.max_epochs=160 \
    +ckpt_path=outputs/checkpoints/phase1-last.ckpt
```

#### Phase 3 (optional): Unfreeze VAE encoder

Only attempt this if Phase 2 plateaus and you suspect the VAE tokenizer's
latent space is too coarse for EM data.

```yaml
model:
  freeze_vae_encoder: false      # VAE encoder -- fine-tune the tokenizer
  freeze_dit_backbone: false
  freeze_vae_decoder: false

optimizer:
  lr: 1.0e-5
  dit_backbone_lr: 1.0e-6    # very conservative for both backbone and encoder
```

**Warning**: The VAE encoder defines the latent space that the DiT was
pretrained on.  Changing it shifts the input distribution the DiT
expects, which can destabilize training.  Use a very small learning rate
(1e-6) and monitor closely.

### 9.3 Component Freeze Reference

| Component | Phase 1 | Phase 2 | Phase 3 | Notes |
|---|---|---|---|---|
| VAE encoder | Frozen | Frozen | Trainable | Defines latent space; almost always frozen |
| DiT backbone | **Frozen** | **Trainable** (low LR) | Trainable | Main feature extractor |
| VAE decoder | Trainable | Trainable | Trainable | Adapts upsampling to EM |
| Input adapter | Trainable | Trainable | Trainable | Bridges 1-ch EM to 3-ch RGB |
| Feature projector | Trainable | Trainable | Trainable | Fuses DiT multi-layer features |
| Task heads | Trainable | Trainable | Trainable | Randomly initialized, EM-specific |
| Point encoder | Trainable | Trainable | Trainable | Starts near-zero, learned from scratch |

### 9.4 Config Flags

All freeze decisions are controlled by three config keys under `model:`:

```yaml
model:
  freeze_vae_encoder: true     # VAE encoder  (default true; int N = freeze for N epochs)
  freeze_dit_backbone: false   # DiT backbone (default false; int N = freeze for N epochs)
  freeze_vae_decoder: false    # VAE decoder  (default false; int N = freeze for N epochs)
```

These flags are identical across all four Cosmos models
(CosmosPredict2D, CosmosPredict3D, CosmosTransfer2D, CosmosTransfer3D).

The `dit_backbone_lr` differential learning rate is set under `optimizer:`:

```yaml
optimizer:
  lr: 1.0e-4               # default LR for heads, decoder, adapter, projector
  dit_backbone_lr: 1.0e-5  # separate LR for DiT backbone (omit to use same as lr)
```

### 9.5 Loading a Previous Checkpoint

Use `+ckpt_path=` to warm-start training from a previously saved checkpoint.
The script loads **model weights only** and starts a fresh optimizer, which
allows you to change the freeze schedule, optimizer settings, or model
config between runs without hitting state-dict mismatches.

```bash
# Full resume (optimizer, LR schedule, epoch, global_step) — same config
PYTHONPATH=$(pwd) python scripts/train.py \
    --config-name snemi3d \
    training.resume_from_checkpoint=outputs/checkpoints/last.ckpt
```

**Warm-start** (weights only, fresh optimizer) — use when changing phases, optimizer, or architecture:

```bash
PYTHONPATH=$(pwd) python scripts/train.py \
    --config-name snemi3d \
    +ckpt_path=outputs/checkpoints/last.ckpt
```

With `+ckpt_path=`, the script loads **model weights only**:

- Optimizer state, learning rate scheduler, and epoch counter are **not**
  restored — training begins from epoch 0 with a fresh optimizer.
- `strict=False` is used, so missing or unexpected keys are reported but
  do not cause failures.  This lets you load checkpoints even when the
  model architecture has changed (e.g. added/removed heads).

Do **not** set both `training.resume_from_checkpoint` and `+ckpt_path=` in the same run.

#### When to use warm-start vs full resume

| Scenario | Mechanism |
|---|---|
| Changing freeze config (e.g. unfreezing backbone for Phase 2) | `+ckpt_path=` |
| Changing optimizer or LR schedule | `+ckpt_path=` |
| Changing model architecture (added heads, different feature_size) | `+ckpt_path=` (with `strict=False` warnings) |
| Exact resume after crash or preemption (same config) | `training.resume_from_checkpoint=` |

#### Automatic checkpointing

The `ModelCheckpoint` callback saves `last.ckpt` every epoch and
`best-epoch=*.ckpt` based on validation loss.  Both are valid inputs
for `+ckpt_path=`.

#### PyTorch 2.6 compatibility

Checkpoints are loaded with `weights_only=False` because Lightning
checkpoints contain OmegaConf objects and other non-tensor globals.
This is safe for your own checkpoints.

### 9.6 Cosmos-Transfer vs Cosmos-Predict

Both model families share the same architecture and freeze interface.
The difference is in pretraining:

| | Predict2.5 | Transfer2.5 |
|---|---|---|
| Pretraining task | Unconditional video generation | Conditioned generation (edge/depth maps) |
| Domain gap to EM | Larger | Smaller (already learned structure-preserving features) |
| Recommendation | Phase 1 is more important (features are less aligned) | Can be more aggressive with Phase 2 (lower warmup) |

Transfer2.5 was pretrained with spatial conditioning signals (edge maps,
depth maps), giving it a stronger prior for boundary-sensitive dense
prediction.  This makes it a better starting point for connectomics tasks
that depend on membrane fidelity (neurite tracing, synapse detection).

# Neurons Architecture

A comprehensive guide to the three volumetric model families and the overall
codebase for newcomers.

## 1. Data Flow

```
Config (YAML)
    │
    ▼
get_datamodule()  ──►  DataModule.setup()
    │                        │
    │                        ├─► Dataset._prepare_data()   # load volumes
    │                        ├─► get_train_transforms()    # build pipeline
    │                        └─► Dataset(..., transform=...)
    │
    ▼
get_module()  ──►  Vista3DModule / CosmosPredict3DModule / CosmosTransfer3DModule
    │
    ▼
Trainer.fit(module, datamodule)
    │
    ├─► training_step(batch)
    │       ├─► _prepare_targets(batch)
    │       ├─► model(images)  →  semantic, instance, geometry
    │       └─► criterion(predictions, targets)
    │
    └─► validation_step(batch)
            ├─► model(images)
            ├─► criterion(...)
            └─► _eval_metrics(predictions, targets)
```

## 2. Key Modules


| Module                               | Role                                                               |
| ------------------------------------ | ------------------------------------------------------------------ |
| `datasets/base.py`                   | `CircuitDataset` base; `_fast_crop` for early patch extraction     |
| `datamodules/base.py`                | `CircuitDataModule`; setup, DataLoader creation                    |
| `datamodules/combine.py`             | `CombineDataModule`; multi-dataset training with union label space |
| `models/vista3d_model.py`            | SegResNet backbone + semantic/instance/geometry heads              |
| `models/cosmospredict3d_model.py`    | Cosmos-Predict2.5 DiT backbone + VAE decoder adapter               |
| `models/cosmostransfer3d_model.py`   | Cosmos-Transfer2.5 DiT backbone + VAE decoder adapter              |
| `modules/vista3d_module.py`          | Lightning module for Vista3D training                              |
| `modules/cosmospredict3d_module.py`  | Lightning module for CosmosPredict3D training                      |
| `modules/cosmostransfer3d_module.py` | Lightning module for CosmosTransfer3D training                     |
| `losses/vista3d_losses.py`           | Vista3DLoss: L_sem + L_ins + L_geom                                |
| `losses/cosmospredict3d_losses.py`   | CosmosPredict3DLoss: same + optional L_flow_consistency            |
| `losses/cosmostransfer3d_losses.py`  | CosmosTransfer3DLoss: same + optional L_flow_consistency           |
| `losses/semantic.py`                 | SemanticLoss: CE + IoU + Dice                                      |
| `losses/instance.py`                 | InstanceLoss: pull / push / norm with boundary+skeleton weighting  |
| `losses/geometry.py`                 | GeometryLoss: direction / covariance / raw reconstruction          |
| `metrics/semantic.py`                | Dice, IoU for semantic evaluation                                  |
| `metrics/instance.py`                | ARI, AMI, AXI, VOI, TED for instance evaluation                    |
| `inference/soft_clustering.py`       | SoftMeanShift for instance clustering at eval                      |
| `utils/point_sampling.py`            | `sample_point_prompts` for interactive/proofread training          |
| `models/point_prompt_encoder.py`     | Sparse point prompts → dense feature residual                      |


## 3. Training Modes

All three model families support the same two training modes:

- **automatic**: model sees only the image; predicts from scratch.
- **proofread**: model receives extra context. Sub-modes:
  - *interactive*: fully annotated GT; point prompts are sampled and encoded
  via `PointPromptEncoder` as a residual added to backbone features.
  - *fractionary*: partial annotation (mix of valid labels and `ignore_index`);
  unknown regions are masked, `semantic_ids` are forwarded.

Both modes can run per step; losses are averaged.

## 4. Tensor Shapes (3D)


| Stage                 | image           | label        | semantic        | instance        | geometry         |
| --------------------- | --------------- | ------------ | --------------- | --------------- | ---------------- |
| Batch from DataLoader | [B, 1, D, H, W] | [B, D, H, W] | --              | --              | --               |
| Model output          | --              | --           | [B, C, D, H, W] | [B, E, D, H, W] | [B, 16, D, H, W] |
| Targets               | --              | [B, D, H, W] | [B, D, H, W]    | --              | --               |


Geometry channels: `S + S*S + 4 = 3 + 9 + 4 = 16` (direction + covariance + RGBA).

---

## 5. Vista3D Architecture

### 5.1 Overview

Vista3D is the baseline architecture built on MONAI's SegResNet. It is a
lightweight, fully-convolutional 3D encoder-decoder designed specifically for
connectomics segmentation without any pretrained foundation model.

### 5.2 Block Diagram

```
Input [B, 1, D, H, W]
    │
    ▼
┌─────────────────────────┐
│  SegResNetDS2 Backbone   │  (or SegResNet fallback)
│  spatial_dims=3          │
│  init_filters=feature_size
│  blocks_down=(1,2,2,4,4) │
│  norm=instance           │
│  dsdepth=1               │
└────────────┬────────────┘
             │  feat [B, feature_size, D, H, W]
             │
     ┌───────┼───────┐
     │       │       │
     ▼       ▼       ▼
┌────────┐┌────────┐┌────────┐
│Semantic││Instance││Geometry│  Three parallel task heads
│  Head  ││  Head  ││  Head  │
│Conv3d→ ││Conv3d→ ││Conv3d→ │
│GN→ReLU ││GN→ReLU ││GN→ReLU │
│Conv3d  ││Conv3d  ││Conv3d  │
└───┬────┘└───┬────┘└───┬────┘
    │         │         │
    ▼         ▼         ▼
 [B,C,D,H,W] [B,E,D,H,W] [B,16,D,H,W]
  semantic     instance     geometry
```

### 5.3 Components


| Component     | Class                                            | Description                                                                                                                                                     |
| ------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Backbone      | `SegResNetDS2` / `SegResNet`                     | MONAI encoder-decoder. Tries `SegResNetDS2` first (deep-supervision variant from MONAI >= 1.3), falls back to standard `SegResNet`.                             |
| Semantic head | `nn.Sequential(Conv3d, GroupNorm, ReLU, Conv3d)` | Outputs `num_classes` channels (default 16).                                                                                                                    |
| Instance head | Same structure                                   | Outputs `emb_dim` channels (default 16) as per-voxel embeddings for discriminative clustering.                                                                  |
| Geometry head | Same structure                                   | Outputs `S + S*S + 4 = 16` channels: 3 direction, 9 covariance, 4 RGBA.                                                                                         |
| Point encoder | `PointPromptEncoder`                             | Builds a sparse indicator volume from point prompts and projects it through a small conv block. Added as a residual to backbone features before the task heads. |


### 5.4 Freeze Policy


| Component            | Default state              |
| -------------------- | -------------------------- |
| SegResNet backbone   | **Trainable** (end-to-end) |
| All three task heads | **Trainable**              |
| Point prompt encoder | **Trainable**              |


Vista3D has no pretrained weights from external sources -- everything is trained
from scratch on connectomics data.

### 5.5 Loss Function: `Vista3DLoss`

```
L_total = w_sem * L_sem + w_ins * L_ins + w_geom * L_geom

L_sem  = w_ce * CE + w_iou * (1 - SoftIoU) + w_dice * (1 - SoftDice)
L_ins  = w_pull * L_pull + w_push * L_push + w_norm * L_norm
L_geom = w_dir * L_dir + w_cov * L_cov + w_raw * L_raw
```

Instance loss uses boundary (morphological gradient) and skeleton (EDT-based)
pixel weighting so the model focuses on touching boundaries and medial axes.

### 5.6 When to Use Vista3D

- Moderate-sized datasets (SNEMI3D, CREMI3D) where training from scratch is
feasible.
- When you need the lightest memory footprint (no foundation-model overhead).
- As a performance baseline before trying Cosmos backbones.

---

## 6. Cosmos-Predict2.5 3D Architecture

### 6.1 Overview

CosmosPredict3D adapts NVIDIA's Cosmos-Predict2.5 Diffusion Transformer (DiT)
as a feature extractor for volumetric segmentation. Cosmos-Predict2.5 is
natively a video generation model, so the EM volume depth axis maps naturally
to the video temporal axis:

```
EM volume  [B, C, D, H, W]  <-->  video  [B, C, T, H, W]
```

Available in 2B and 14B parameter variants, loaded from HuggingFace Hub.

### 6.2 Block Diagram

```
Input [B, 1, D, H, W]
    │
    ▼
┌───────────────-───┐
│  InputAdapter3D   │  Conv3d(1→3, k=1)  -- adapt EM to RGB
└────────┬───────-──┘
         │ [B, 3, D, H, W]
         ▼
┌──────────────-────┐
│  VAE Encoder      │  ALWAYS FROZEN, eval mode
│  (from HuggingFace│  Compresses spatial by 8x, temporal by 8x
│   or fallback     │  Output: [B, 16, D/8, H/8, W/8]
│   conv downsample)│
└────────┬─────-────┘
         │ latent [B, C_lat, D_lat, H_lat, W_lat]
         ▼
┌─────────────────-─┐
│  DiT Backbone     │  28 layers (2B) or 40 layers (14B)
│  _StandaloneDiT3D │  Volumetric patchify → self-attention blocks
│  (or diffusers    │  Features extracted at quartile layers:
│   transformer)    │    {n//4, n//2, 3n//4, n-1}
└────────┬───────-──┘
         │ intermediate features: List[[B, N, hidden_dim]]
         ▼
┌───────────────────┐
│ FeatureProjector3D│  Concat multi-layer features → Conv3d projection
│                   │  Output: [B, feature_size, D_lat, H_lat, W_lat]
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ DecoderAdapter3D  │  Two paths depending on pretrained decoder availability:
│                   │
│  WITH pretrained VAE decoder:
│  │ to_latent     │  Conv3d: feature_size → latent_channels (64→16)
│  │ decoder_body  │  Pretrained VAE decoder (frozen body, trainable last block)
│  │ task heads    │  semantic / instance / geometry
│                   │
│  WITHOUT pretrained decoder (standalone fallback):
│  │ decoder_body  │  ProgressiveUpsampler3D(feature_size → feature_size)
│  │ task heads    │  Heads read directly from feature_size channels
│  │               │  (no to_latent bottleneck — skipped entirely)
└───────────────────┘
```

### 6.3 Components


| Component         | Class                                            | Description                                                                                                                     |
| ----------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| Input adapter     | `_InputAdapter3D`                                | `Conv3d(in_ch, 3, k=1)` to map single-channel EM to 3-ch RGB expected by Cosmos.                                                |
| VAE encoder       | From `AutoencoderKLCosmos`                       | Compresses `[B,3,D,H,W]` to latent `[B,16,D/8,H/8,W/8]`. Always frozen.                                                         |
| DiT backbone      | `_StandaloneDiT3D` or `CosmosTransformer3DModel` | N layers of self-attention blocks with adaptive layer norm. Processes 3D patch tokens.                                          |
| Feature projector | `_FeatureProjector3D`                            | Reshapes DiT token sequences to spatial grids, fuses 4 layers via concat, and projects `hidden_dim*4 → feature_size`. |
| Decoder adapter   | `_DecoderAdapter3D`                              | With pretrained VAE: projects `feature_size → latent_channels`, runs frozen decoder, then task heads. Without pretrained VAE: upsamples directly from `feature_size` (no bottleneck). |
| Point encoder     | `PointPromptEncoder`                             | Same as Vista3D -- sparse point prompts added as a feature residual.                                                            |


### 6.4 Why FeatureProjector + DecoderAdapter (Not One Module)

The FeatureProjector and DecoderAdapter serve distinct, non-overlapping roles:

- **FeatureProjector** handles the DiT-specific translation: reshaping flat
  token sequences `[B, N, hidden_dim]` into 3D spatial grids, fusing
  multi-layer features (4 layers concatenated), and compressing from
  `hidden_dim * 4` (e.g. 8192) down to `feature_size` (e.g. 64).

- **DecoderAdapter** handles spatial upsampling back to input resolution.
  When a pretrained VAE decoder is available, it first projects
  `feature_size → latent_channels` via `to_latent` to match the decoder's
  expected input, then runs the (partially frozen) pretrained decoder.
  When no pretrained decoder is available, the `to_latent` projection is
  skipped entirely and the fallback `ProgressiveUpsampler` works directly
  from `feature_size`, avoiding a wasteful channel bottleneck.

The point prompt encoder residual is added between these two stages,
operating in `feature_size`-dimensional space at latent resolution.

### 6.5 Weight Loading Strategy

Three loading strategies are tried in order:

1. **diffusers** -- `CosmosTransformer3DModel.from_pretrained()` + `AutoencoderKLCosmos`
2. **cosmos_predict2 package** -- `CosmosPredict2Pipeline.from_pretrained()`
3. **Raw checkpoint** -- `snapshot_download()` + `load_state_dict(strict=False)`
4. **Standalone fallback** -- Random init with matching architecture shape

### 6.6 Freeze Policy (Default)


| Component                 | Default state     | Notes                                                  |
| ------------------------- | ----------------- | ------------------------------------------------------ |
| Input adapter             | **Trainable**     | Learns EM → RGB mapping                                |
| VAE encoder               | **Always frozen** | `requires_grad_(False)` + `.eval()`                    |
| DiT backbone              | **Trainable**     | `freeze_dit_backbone=False` by default                 |
| Feature projector         | **Trainable**     | Randomly initialized                                   |
| `to_latent` projection    | **Trainable**     | Only exists when pretrained VAE decoder is loaded       |
| VAE decoder body          | **Mostly frozen** | All params frozen except last up-block and output norm |
| VAE decoder last up-block | **Trainable**     | Fine-tuned for domain adaptation                       |
| VAE decoder output norm   | **Trainable**     | Fine-tuned for domain adaptation                       |
| Fallback upsampler        | **Trainable**     | Used when no VAE decoder; upsamples from `feature_size` directly |
| Task heads (sem/ins/geom) | **Trainable**     | Randomly initialized                                   |
| Point encoder             | **Trainable**     | Initialized near-zero for no-op at start               |


### 6.7 Loss Function: `CosmosPredict3DLoss`

Same three-component structure as Vista3DLoss, plus an optional fourth term:

```
L_total = w_sem * L_sem + w_ins * L_ins + w_geom * L_geom + w_fc * L_fc

L_fc = mean( ||normalize(feat_a) - normalize(feat_b)||^2 )
```

The **feature-consistency loss** (`L_fc`) penalizes L2 distance between
DiT features computed on two augmented views of the same input, encouraging
augmentation-invariant backbone representations.

### 6.8 Optimizer Enhancements

CosmosPredict3DModule supports **differential learning rates**:

```yaml
optimizer:
  lr: 1.0e-4           # head learning rate
  backbone_lr: 1.0e-5  # lower LR for pretrained DiT
```

Also supports `cosine_warmup` scheduler (linear warmup then cosine decay),
in addition to the plain `cosine` scheduler from Vista3D.

### 6.9 When to Use CosmosPredict3D

- When you want to leverage pretrained video generation features for EM data.
- Large-scale datasets (MICRONS, combined SNEMI3D+MICRONS) where the pretrained
backbone provides a strong feature prior.
- The 2B variant fits on a single GPU (approximately 12 GB); the 14B variant requires
multi-GPU or CPU offloading.

---

## 7. Cosmos-Transfer2.5 3D Architecture

### 7.1 Overview

CosmosTransfer3D adapts NVIDIA's Cosmos-Transfer2.5 DiT for volumetric
segmentation. Transfer2.5 is a *conditioned* video generation model (edge/depth
control), making it architecturally suited for dense prediction tasks where
the model must respect spatial structure.

The architecture is structurally identical to CosmosPredict3D but loads weights
from the Transfer2.5 model family:


|                  | Predict2.5                          | Transfer2.5                                    |
| ---------------- | ----------------------------------- | ---------------------------------------------- |
| HuggingFace repo | `nvidia/Cosmos-Predict2.5-{2B,14B}` | `nvidia/Cosmos-Transfer2.5-{2B,14B}`           |
| Default revision | `diffusers/base/post-trained`       | `diffusers/general`                            |
| Pretraining task | Unconditional video generation      | Conditioned video generation (edge/depth maps) |
| Architecture     | DiT + VAE                           | DiT + VAE (same structure)                     |


### 7.2 Block Diagram

Identical to CosmosPredict3D (Section 6.2). The only differences are:

1. Weights are loaded from `nvidia/Cosmos-Transfer2.5-`* repos.
2. The default HuggingFace revision is `diffusers/general` (vs `diffusers/base/post-trained`).

### 7.3 Components

Same component table as CosmosPredict3D (Section 6.3). All class names are
prefixed with `CosmosTransfer` instead of `CosmosPredict`.

### 7.4 Freeze Policy (Default)

Identical to CosmosPredict3D (Section 6.5):


| Component              | Default state                                   |
| ---------------------- | ----------------------------------------------- |
| Input adapter          | **Trainable**                                   |
| VAE encoder            | **Always frozen**                               |
| DiT backbone           | **Trainable** (`freeze_dit_backbone=False`)     |
| Feature projector      | **Trainable**                                   |
| `to_latent` projection | **Trainable** (only when pretrained VAE loaded)  |
| VAE decoder body       | **Mostly frozen** (except last up-block + norm) |
| Fallback upsampler     | **Trainable** (when no VAE; works from `feature_size` directly) |
| Task heads             | **Trainable**                                   |
| Point encoder          | **Trainable**                                   |


### 7.5 Loss Function: `CosmosTransfer3DLoss`

Identical structure to `CosmosPredict3DLoss` (Section 6.6), including the
optional feature-consistency term.

### 7.6 When to Use CosmosTransfer3D

- Transfer2.5 was pretrained with spatial conditioning (edge maps, depth maps),
giving it a stronger prior for structure-preserving dense prediction compared
to the unconditional Predict2.5.
- Recommended when the EM segmentation task heavily depends on boundary fidelity
(e.g., dense neurite tracing, synapse detection near membranes).
- Same memory footprint and training configuration as CosmosPredict3D.

---

## 8. Architecture Comparison

### 8.1 Feature Extraction Pipeline

```
              Vista3D                CosmosPredict3D / CosmosTransfer3D
              ───────                ──────────────────────────────────
Backbone      SegResNetDS2           Cosmos DiT (2B or 14B params)
              (from scratch)         (pretrained on video data)

Encoding      Direct convolution     VAE encoder → latent (8x compress)

Features      Single feature map     Multi-layer extraction at
              from encoder-decoder   {n/4, n/2, 3n/4, n-1} + projection

Decoding      Direct from backbone   With pretrained VAE: to_latent →
                                       frozen decoder (last block trainable)
                                     Without: ProgressiveUpsampler
                                       directly from feature_size (no bottleneck)

Task heads    Conv3d + GN + ReLU     Same structure
              + Conv3d

Point prompts Sparse → Conv3d        Same mechanism
              residual
```

### 8.2 Parameter Counts (Approximate)


| Model                     | Backbone | Heads + Adapter | Total |
| ------------------------- | -------- | --------------- | ----- |
| Vista3D (feature_size=48) | ~2M      | ~0.5M           | ~2.5M |
| Vista3D (feature_size=64) | ~4M      | ~0.8M           | ~4.8M |
| CosmosPredict3D-2B        | ~2B      | ~5M             | ~2B   |
| CosmosPredict3D-14B       | ~14B     | ~5M             | ~14B  |
| CosmosTransfer3D-2B       | ~2B      | ~5M             | ~2B   |
| CosmosTransfer3D-14B      | ~14B     | ~5M             | ~14B  |


### 8.3 Memory Footprint


| Model            | VRAM (2B, bf16) | VRAM (14B, bf16) |
| ---------------- | --------------- | ---------------- |
| Vista3D          | ~2 GB           | N/A              |
| CosmosPredict3D  | ~12 GB          | ~48 GB           |
| CosmosTransfer3D | ~12 GB          | ~48 GB           |


### 8.4 Training Configuration Matrix


| Feature                         | Vista3D           | CosmosPredict3D         | CosmosTransfer3D        |
| ------------------------------- | ----------------- | ----------------------- | ----------------------- |
| `freeze_dit_backbone` default   | N/A (no pretrain) | `False`                 | `False`                 |
| Differential LR (`backbone_lr`) | No                | Yes                     | Yes                     |
| `cosine_warmup` scheduler       | No                | Yes                     | Yes                     |
| Feature-consistency loss        | No                | Optional                | Optional                |
| Compatibility check             | No                | `compatibility_check()` | `compatibility_check()` |
| VAE encoder                     | None              | Always frozen           | Always frozen           |
| VAE decoder reuse               | None              | Yes (partially frozen)  | Yes (partially frozen)  |


---

## 9. Three-Stage Loss Architecture

All three model families share the same three-stage loss decomposition.

### Stage 1: Semantic Loss (`SemanticLoss`)

Per-voxel classification. Supports `sigmoid` (multi-label) and `softmax`
(mutually exclusive) modes.

```
L_sem = w_ce * CrossEntropy + w_iou * (1 - SoftIoU) + w_dice * (1 - SoftDice)
```

The `active_classes` parameter restricts gradients to the first N channels,
allowing the model to output 16 classes but only train on the 2 that have
labels today.

### Stage 2: Instance Loss (`InstanceLoss`)

Discriminative embedding loss with boundary/skeleton weighting.

```
L_ins = w_pull * L_pull + w_push * L_push + w_norm * L_norm

L_pull: embeddings pulled toward weighted cluster centroid (hinge margin δ_v)
L_push: cluster centroids pushed apart (hinge margin 2*δ_d)
L_norm: L2 regularization on centroid norms
```

Weight maps boost gradients at:

- **Boundaries**: morphological gradient via max_pool (dilate != erode)
- **Medial axes**: normalized EDT gives higher weight to skeleton voxels

### Stage 3: Geometry Loss (`GeometryLoss`)

Regression on three channel groups (foreground-only):


| Group      | Channels      | Target                                  | Loss                 |
| ---------- | ------------- | --------------------------------------- | -------------------- |
| Direction  | First S (=3)  | Unit vector toward instance centroid    | smooth_l1 / l1 / mse |
| Covariance | Next S*S (=9) | EDT-based structure tensor per instance | l1 / mse             |
| Raw (RGBA) | Last 4        | RGB reconstruction + foreground alpha   | l1                   |


---

## 10. Transform Pipeline (SNEMI3D, patch_mode)

**With `_fast_crop` (patch_size set, slice_mode=False):**

- Dataset: `_fast_crop` → crop in `__getitem_`_ before transforms.
- Transforms: no EnsureChannelFirstd / SpatialPadd / RandSpatialCropd;
only `_label_post_crop` + augmentations.

**Without `_fast_crop`:**

- Transforms: EnsureChannelFirstd → SpatialPadd → RandSpatialCropd →
Labeld (connected-component relabel) → RandFlip → RandRotate90 →
Directiond + Covarianced → intensity augmentations → ToTensord.

---

## 11. Reading Order (Suggested)

### For Vista3D:

1. `scripts/train.py` → `get_datamodule`, `get_module`
2. `datamodules/base.py` → `setup`, DataLoader
3. `datamodules/snemi3d.py` → `get_train_transforms`, `_get_dataset_kwargs`
4. `datasets/base.py` → `_fast_crop`, `CircuitDataset`
5. `models/vista3d_model.py` → backbone + three heads
6. `modules/vista3d_module.py` → `training_step`, `_prepare_targets`
7. `losses/vista3d_losses.py` → Vista3DLoss branches

### For Cosmos models:

1. `models/cosmospredict3d_model.py` or `models/cosmostransfer3d_model.py`
  → `_build_backbone`, weight loading, `_extract_features`, `_DecoderAdapter3D`
2. `modules/cosmospredict3d_module.py` or `modules/cosmostransfer3d_module.py`
  → `training_step`, `configure_optimizers` (differential LR)
3. `losses/cosmospredict3d_losses.py` → feature-consistency loss
4. `configs/snemi3d_microns.yaml` → example combined config with model selection


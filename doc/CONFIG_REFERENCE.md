# Configuration Reference

Reference for Hydra YAML configuration used by `scripts/train.py`.
All paths are relative to the project root.

---

## 1. Config Hierarchy

Configs in `configs/` compose via Hydra `defaults`:

```
configs/
├── default.yaml          # Base: data, model, optimizer, loss, training, callbacks
├── snemi3d.yaml         # SNEMI3D single-dataset
├── snemi3d_microns.yaml # Combined SNEMI3D + MICRONS (CosmosTransfer3D)
├── cremi3d.yaml
├── microns.yaml
├── combine.yaml
└── profiler.yaml
```

Use `--config-name <name>` to select a config. Override any key via CLI:
`data.batch_size=8`, `optimizer.lr=5e-4`, etc.

---

## 2. Data (`data`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `dataset` | str | `snemi3d` | `snemi3d`, `cremi3d`, `microns`, `combine` |
| `data_root` | str | `data` | Default root for volumes without explicit `root` |
| `batch_size` | int | 4 | Per-GPU batch size |
| `num_workers` | int | 16 | DataLoader workers per process |
| `cache_rate` | float | 0.5 | Fraction to cache (0.0 for LazyVolDataset) |
| `pin_memory` | bool | true | Pin memory for faster GPU transfer |
| `slice_mode` | bool | true | 2D slices vs 3D patches |
| `num_samples` | int | null | Virtual epoch length (null = dataset default) |
| `patch_size` | [D,H,W] | null | 3D crop size; enables LazyVolDataset when set |
| `overcrop_factor` | float | 1.0 | Train crop = patch_size × factor, then zoom-out |
| `train_volumes` | list | null | `[{vol, seg, root?}, ...]` |
| `val_volumes` | list | null | Defaults to train_volumes |
| `test_volumes` | list | null | Defaults to train_volumes |

**Volume entry format** (per volume):

```yaml
- vol: AC4_inputs           # Input volume filename (no extension)
  seg: AC4_labels           # Label volume filename
  root: data/SNEMI3D        # Optional; overrides data_root
```

Supported formats: HDF5, TIFF, NRRD, NIfTI (auto-detected by extension).

---

## 3. Model (`model`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | str | `vista3d` | `vista3d`, `vista2d`, `cosmospredict3d`, `cosmostransfer3d` |
| `in_channels` | int | 1 | Input channels |
| `num_classes` | int | 16 | Semantic head output channels |
| `emb_dim` | int | 16 | Instance embedding dimension |
| `feature_size` | int | 64 | Backbone feature channels |
| `dropout` | float | 0.2 | Dropout rate |

**Cosmos-specific** (CosmosPredict3D, CosmosTransfer3D):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `variant` | str | `"2B"` | `"2B"` or `"14B"` |
| `dtype` | str | `"bf16"` | `"bf16"` or `"fp32"` |
| `freeze_vae_encoder` | bool/int | true | true = always frozen; int N = freeze N epochs |
| `freeze_dit_backbone` | bool/int | false | DiT backbone freeze schedule |
| `freeze_vae_decoder` | bool/int | false | VAE decoder freeze schedule |
| `gradient_checkpointing` | bool | false | Enable to save memory |
| `compile` | bool | false | `torch.compile()` (experimental) |

---

## 4. Optimizer (`optimizer`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | str | `adamw` | Optimizer type |
| `lr` | float | 1e-3 | Learning rate |
| `weight_decay` | float | 1e-4 | Weight decay |
| `dit_backbone_lr` | float | null | Separate LR for DiT backbone (Cosmos only) |

**Scheduler** (`optimizer.scheduler`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | str | `cosine` | `cosine`, `cosine_warmup` |
| `T_max` | int | 100 | Max epochs for cosine |
| `eta_min` | float | 1e-6 | Min LR |
| `warmup_epochs` | int | 5 | Warmup epochs (cosine_warmup only) |

---

## 5. Loss (`loss`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `semantic_mode` | str | `sigmoid` | `sigmoid` (multi-label) or `softmax` |
| `active_classes` | int | null | Train only first N semantic channels |
| `class_weights` | list | null | Per-class CE weights |
| `label_smoothing` | float | 0.0 | CE label smoothing |
| `weight_semantic` | float | 1.0 | Branch weight |
| `weight_ce` | float | 1.0 | Cross-entropy |
| `weight_iou` | float | 1.0 | Soft IoU |
| `weight_dice` | float | 1.0 | Soft Dice |
| `weight_instance` | float | 1.0 | Instance branch |
| `weight_pull` | float | 1.0 | Discriminative pull |
| `weight_push` | float | 1.0 | Discriminative push |
| `weight_norm` | float | 0.001 | Centroid norm regularisation |
| `delta_v` | float | 0.5 | Pull hinge margin |
| `delta_d` | float | 1.5 | Push margin (2×δ_d between centroids) |
| `weight_edge` | float | 10.0 | Boundary pixel weight |
| `weight_bone` | float | 10.0 | Skeleton pixel weight |
| `weight_geometry` | float | 0.0 | Geometry branch (0 = disabled) |
| `weight_dir` | float | 1.0 | Direction sub-loss |
| `weight_cov` | float | 0.0 | Covariance (0 recommended for speed) |
| `weight_raw` | float | 1.0 | RGBA reconstruction |
| `loss_dir` | str | `smooth_l1` | `l1`, `mse`, `smooth_l1` |
| `loss_cov` | str | `mse` | Same options |
| `loss_raw` | str | `l1` | Same options |
| `ignore_index` | int | -100 | Ignored label value |

---

## 6. Training (`training`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_epochs` | int | 100 | Training epochs |
| `accelerator` | str | `auto` | `cpu`, `gpu`, `tpu`, `auto` |
| `devices` | int/str | `auto` | GPU count or `auto` |
| `strategy` | str | `ddp` | `ddp`, `fsdp`, `auto` |
| `precision` | str | `bf16-mixed` | `32-true`, `16-mixed`, `bf16-mixed` |
| `gradient_clip_val` | float | 1.0 | Gradient clipping |
| `accumulate_grad_batches` | int | 1 | Gradient accumulation |
| `limit_val_batches` | int | 10 | Val batches per epoch |
| `num_sanity_val_steps` | int | 2 | Sanity check steps |
| `log_every_n_steps` | int | 50 | Logging interval |
| `benchmark` | bool | true | cudnn.benchmark |
| `boundary_in_semantic` | bool | true | Boundary in semantic targets only |
| `training_modes` | list | `[automatic]` | `automatic`, `proofread` |

---

## 7. snemi3d_microns Quick Reference

The `snemi3d_microns` config trains CosmosTransfer3D on combined SNEMI3D + MICRONS:

```yaml
# Key overrides from default
data:
  dataset: snemi3d
  data_root: data/SNEMI3D
  batch_size: 4
  num_workers: 16
  cache_rate: 0.0
  slice_mode: false
  num_samples: 16000
  patch_size: [48, 256, 256]
  overcrop_factor: 1.25
  train_volumes: [...]  # SNEMI3D AC4 + 4 MICRONS crops
  val_volumes: [...]     # SNEMI3D AC3 + 1 MICRONS crop

model:
  type: cosmostransfer3d
  variant: "2B"
  freeze_vae_encoder: true
  freeze_dit_backbone: false
  freeze_vae_decoder: false

optimizer:
  lr: 2.0e-4
  dit_backbone_lr: 2.0e-5
  scheduler:
    type: cosine_warmup
    warmup_epochs: 5
    T_max: 100
```

**Run:**

```bash
python scripts/train.py --config-name snemi3d_microns
python scripts/train.py --config-name snemi3d_microns data.batch_size=8 +ckpt_path=outputs/checkpoints/last.ckpt
```

# Neurons Architecture

A high-level guide to the codebase for newcomers.

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
get_module()  ──►  Vista3DModule
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

| Module | Role |
|--------|------|
| `datasets/base.py` | `CircuitDataset` base; `_fast_crop` for early patch extraction |
| `datamodules/base.py` | `CircuitDataModule`; setup, DataLoader creation |
| `datamodules/snemi3d.py` | SNEMI3D-specific transforms and volume loading |
| `models/vista3d_model.py` | SegResNet backbone + semantic/instance/geometry heads |
| `modules/vista3d_module.py` | Lightning module; `_prepare_targets`, training/val steps |
| `losses/vista3d_losses.py` | Vista3DLoss: L_sem, L_ins, L_geom |
| `metrics/semantic.py` | Dice, IoU for semantic evaluation |
| `metrics/instance.py` | ARI, AMI, VOI, TED for instance evaluation |
| `inference/soft_clustering.py` | SoftMeanShift for instance clustering at eval |

## 3. Training Modes

- **automatic**: model sees only image; predicts from scratch.
- **proofread**: model receives extra context (point prompts or fractionary labels).

Both can run per step; losses are averaged.

## 4. Tensor Shapes (3D)

| Stage | image | label | semantic | instance |
|-------|-------|-------|----------|----------|
| Batch from DataLoader | [B, 1, D, H, W] | [B, D, H, W] | — | — |
| Model output | — | — | [B, C, D, H, W] | [B, E, D, H, W] |
| Targets | — | [B, D, H, W] | [B, D, H, W] | — |

`_prepare_targets` squeezes labels when needed; Vista3DModule handles `rearrange` for 4D vs 5D inputs.

## 5. Transform Pipeline (SNEMI3D, patch_mode)

**With `_fast_crop` (patch_size set, slice_mode=False):**

- Dataset: `_fast_crop` → crop in `__getitem__` before transforms.
- Transforms: no EnsureChannelFirstd / SpatialPadd / RandSpatialCropd; only `_label_post_crop` + augmentations.

**Without `_fast_crop`:**

- Transforms: EnsureChannelFirstd → SpatialPadd → RandSpatialCropd → `_label_post_crop` → augmentations → ToTensord.

## 6. Reading Order (Suggested)

1. `scripts/train.py` → `get_datamodule`, `get_module`
2. `datamodules/base.py` → `setup`, DataLoader
3. `datamodules/snemi3d.py` → `get_train_transforms`, `_get_dataset_kwargs`
4. `datasets/base.py` → `_fast_crop`, `CircuitDataset`
5. `modules/vista3d_module.py` → `training_step`, `_prepare_targets`
6. `losses/vista3d_losses.py` → Vista3DLoss branches

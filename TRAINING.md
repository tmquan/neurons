# Training Guide

## Prerequisites

```bash
conda create -n neurons python=3.12
conda activate neurons

pip install -e ".[dev]"
```

Verify the installation:

```bash
pytest tests/ -v
```

## Download Data

Download scripts default to `data/<dataset>` if `--output` is omitted:

```bash
python scripts/download_snemi3d.py
python scripts/download_cremi3d.py
python scripts/download_mitoem2.py
```

Expected directory structure:

```
data/
├── snemi3d/
│   ├── AC3_inputs.h5
│   └── AC3_labels.h5
├── cremi3d/
│   ├── sampleA.h5
│   ├── sampleB.h5
│   └── sampleC.h5
└── mitoem2/
    ├── human/
    └── rat/
```

## Quick Start

### Single-dataset training

SNEMI3D 2D slices (Vista2D):

```bash
python scripts/train.py --config-name snemi2d
```

SNEMI3D 3D patches (Vista3D):

```bash
python scripts/train.py --config-name snemi3d
```

CREMI3D 3D patches (Vista3D):

```bash
python scripts/train.py --config-name cremi3d
```

### Multi-dataset training

Combined SNEMI3D + CREMI3D with unified label space:

```bash
python scripts/train.py --config-name combine
```

### Foundation model training

All datasets (SNEMI3D + CREMI3D + MitoEM2) with per-class instance loss,
CE + Dice semantic loss, and bf16 mixed precision:

```bash
python scripts/train.py --config-name foundation
```

### Fast development run

Runs one training batch and one validation batch to verify the pipeline:

```bash
python scripts/train.py --config-name foundation training.fast_dev_run=true
```

## Configuration

All training is configured via YAML files in `configs/`. Every config inherits
from `configs/default.yaml` and overrides only what it needs.

| Config | Dataset | Model | Precision | Epochs |
|---|---|---|---|---|
| `default.yaml` | SNEMI3D (2D) | Vista3D | fp32 | 100 |
| `snemi2d.yaml` | SNEMI3D (2D) | Vista2D | fp32 | 100 |
| `snemi3d.yaml` | SNEMI3D (3D) | Vista3D | fp16 | 100 |
| `cremi3d.yaml` | CREMI3D (3D) | Vista3D | fp16 | 200 |
| `combine.yaml` | SNEMI3D + CREMI3D | Vista3D | fp16 | 300 |
| `foundation.yaml` | All datasets | Vista3D | bf16 | 500 |

### Override parameters via CLI

Hydra allows overriding any config value from the command line:

```bash
python scripts/train.py --config-name foundation \
    data.batch_size=4 \
    optimizer.lr=1e-3

python scripts/train.py --config-name foundation \
    logger=tensorboard

python scripts/train.py --config-name foundation \
    training.max_epochs=10
```

## Architecture

The Vista model has two parallel output heads:

- **Semantic head** -- per-voxel class logits (CE + Dice loss)
- **Instance head** -- per-voxel embeddings (pull/push discriminative loss)

When `class_ids` are available in the batch (provided automatically by
`CombineDataModule`), the instance loss is computed per semantic class --
neurons only cluster with neurons, mitochondria only with mitochondria.

```yaml
model:
  type: vista3d       # or vista2d for slice-based training
  encoder_name: vista3d
  feature_size: 48    # backbone width
  num_classes: 16     # semantic classes (headroom for future additions)
  emb_dim: 16         # instance embedding dimension
```

For 2D slice-based training, set `model.type: vista2d` and `data.slice_mode: true`.

## Loss Configuration

### Semantic branch

```yaml
loss:
  ce_weight: 0.5        # cross-entropy weight
  dice_weight: 0.5       # soft Dice weight (0 to disable)
  class_weights: [...]   # per-class CE weights (length must equal num_classes)
  ignore_index: -100     # label to ignore
```

### Instance branch

```yaml
loss:
  weight_pull: 1.0       # pull embeddings toward cluster center
  weight_push: 1.0       # push different cluster centers apart
  weight_norm: 0.001     # regularize embedding norms
  delta_v: 0.5           # pull margin
  delta_d: 1.5           # push margin
  weight_edge: 10.0      # extra weight on boundary voxels
  weight_bone: 10.0      # extra weight on skeleton (medial axis) voxels
```

## Multi-GPU / Distributed Training

```bash
# Single GPU (default)
python scripts/train.py --config-name foundation \
    training.devices=1

# Multi-GPU DDP
python scripts/train.py --config-name foundation \
    training.devices=4 \
    training.strategy=ddp

# DeepSpeed ZeRO Stage 2
python scripts/train.py --config-name foundation \
    training.devices=4 \
    training.strategy=deepspeed_stage_2

# FSDP
python scripts/train.py --config-name foundation \
    training.devices=4 \
    training.strategy=fsdp
```

## Mixed Precision

```bash
# FP32
python scripts/train.py --config-name foundation training.precision=32-true

# FP16 mixed precision
python scripts/train.py --config-name foundation training.precision=16-mixed

# BF16 mixed precision (recommended for A100/H100)
python scripts/train.py --config-name foundation training.precision=bf16-mixed
```

## Monitoring

### TensorBoard

```bash
python scripts/train.py --config-name snemi3d logger=tensorboard
tensorboard --logdir logs/
```

### Weights and Biases

```bash
wandb login
python scripts/train.py --config-name foundation logger=wandb
```

### Logged metrics

| Metric | Description |
|---|---|
| `train/loss` | Total training loss |
| `train/loss_sem` | Semantic branch loss |
| `train/loss_ce` | Cross-entropy component |
| `train/loss_dice` | Dice component |
| `train/loss_ins` | Instance branch loss |
| `val/loss` | Total validation loss |
| `val/loss_sem` | Semantic validation loss |
| `val/loss_ins` | Instance validation loss |
| `val/accuracy` | Semantic argmax accuracy |

## Checkpoints

Checkpoints are saved automatically:

```yaml
callbacks:
  checkpoint:
    dirpath: checkpoints
    save_top_k: 3         # keep top 3 by val/loss
    save_last: true       # always keep last epoch
    monitor: val/loss
```

Resume training from a checkpoint by passing `ckpt_path` to the trainer.
Add this to `train.py` or load manually:

```python
from neurons.modules import Vista3DModule
module = Vista3DModule.load_from_checkpoint("checkpoints/last.ckpt")
```

## Inference

After training, run sliding window inference on a full volume:

```bash
python scripts/infer_vista3d.py \
    --checkpoint checkpoints/last.ckpt \
    --input data/snemi3d/AC3_inputs.h5 \
    --output output/AC3_segmentation.h5 \
    --patch-size 64 128 128 \
    --aggregation gaussian
```

Segment only a specific class (e.g., neurons = class 1):

```bash
python scripts/infer_vista3d.py \
    --checkpoint checkpoints/last.ckpt \
    --input data/volume.h5 \
    --output output/neurons_only.h5 \
    --class-id 1
```

Output semantic probabilities instead of instance labels:

```bash
python scripts/infer_vista3d.py \
    --checkpoint checkpoints/last.ckpt \
    --input data/volume.h5 \
    --output output/probs.h5 \
    --output-probs
```

All inference options: `python scripts/infer_vista3d.py --help`

## Testing

Run the full test suite:

```bash
pytest tests/ -v
```

Run a specific test file:

```bash
pytest tests/test_losses.py -v
pytest tests/test_modules.py -v
```

## Troubleshooting

**Out of memory** -- Reduce `data.batch_size` or `model.feature_size`, or use
`training.accumulate_grad_batches` to simulate larger batches.

**Slow data loading** -- Increase `data.num_workers` or set `data.cache_rate: 1.0`
to cache the entire dataset in RAM.

**NaN loss** -- Lower the learning rate, verify `training.gradient_clip_val` is
set (default: 1.0), and check that `loss.class_weights` has exactly
`model.num_classes` entries.

**Hydra working directory** -- Hydra changes the working directory by default.
Use absolute paths for `data.data_root` or add `hydra.run.dir=.` to stay in the
project root.

**Vista3D backbone unavailable** -- If MONAI's Vista3D is not installed, the
model automatically falls back to SegResNet. Install a recent MONAI version
(`pip install monai>=1.5`) to use the Vista3D backbone.

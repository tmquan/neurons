# Neurons

<p align="center">
  <img src="teaser.png" alt="Neurons — from electron microscopy to boundary detection to instance segmentation" width="100%">
</p>

A modular, extensible PyTorch Lightning-based infrastructure for connectomics research.

## Overview

**Neurons** provides a production-ready codebase for training segmentation models on electron microscopy (EM) data. It supports multiple dataset types, model architectures, and training paradigms out of the box, while remaining flexible enough for custom extensions.

## Features

- **Multi-Dataset Support** — SNEMI3D, CREMI3D, MICrONS, MitoEM2, Neurite, and combined multi-dataset training with unified label space
- **Vista Architecture** — Vista3D and Vista2D with semantic + instance dual heads
- **Cosmos Foundation Models** — CosmosPredict3D and CosmosTransfer3D (DiT + VAE from NVIDIA)
- **Model Zoo** — Vista3D backbone via MONAI (SegResNet fallback)
- **Geometric Instance Losses** -- Centroid and skeleton discriminative losses with learned projection heads for direction, structure tensor, and image reconstruction
- **Evaluation Metrics** -- ARI, AMI, AXI, VOI, TED (instance); Dice, IoU (semantic)
- **Hydra Configuration** -- YAML-based config with CLI overrides, no code changes needed
- **Experiment Tracking** -- Weights & Biases and TensorBoard integration
- **EM-Specific Augmentations** -- Elastic deformation, missing sections, imaging defects
- **Multi-Format I/O** -- HDF5, TIFF, NRRD, NIfTI with automatic format detection
- **Lazy Volume Loading** -- `LazyVolDataset` reads patches on-demand from disk, keeping RAM usage constant regardless of volume count/size

## Installation

```bash
git clone <repo-url> neurons
cd neurons
pip install -e ".[dev]"
```

### Optional GPU acceleration (recommended)

The clustering (cuML HDBSCAN) and transform (cupy EDT, Gaussian,
connected components) paths run ~4–8× faster with RAPIDS installed.
Use the extras that match your CUDA toolkit, pointed at NVIDIA's
wheel index:

```bash
# CUDA 13 stack (torch 2.10+/cu130, B100/B200/B300, H200, etc.)
pip install -e ".[gpu-cu13]" --extra-index-url https://pypi.nvidia.com

# CUDA 12 stack (torch 2.1-2.9, A100, H100, L40, RTX 40xx, etc.)
pip install -e ".[gpu-cu12]" --extra-index-url https://pypi.nvidia.com
```

If RAPIDS is not installed, everything still works — the clusterer
transparently falls back to scikit-learn (HDBSCAN / MeanShift) and
transforms fall back to scipy/skimage.

### Dependencies

Core: PyTorch, PyTorch Lightning, MONAI, einops, Hydra, h5py, tifffile, pynrrd, scipy

Optional GPU: cupy, cuml (see `[gpu-cu13]` / `[gpu-cu12]` extras)

## Directory Structure

```
neurons/
├── neurons/
│   ├── datasets/       # Dataset classes: SNEMI3D, CREMI3D, MICRONS, MitoEM2; LazyVolDataset (lazy.py)
│   ├── datamodules/    # Lightning DataModules + CombineDataModule
│   ├── models/         # Model wrappers: Vista3D, Vista2D, CosmosPredict3D, CosmosTransfer3D
│   ├── modules/        # Lightning modules: BaseVistaModule, BaseCosmosModule + per-model subclasses
│   ├── losses/         # BaseCombinedLoss + semantic, instance, geometry sub-losses
│   ├── metrics/        # Instance (ARI, AMI, VOI, TED) and semantic (Dice, IoU)
│   ├── preprocessors/  # Format handlers: TIFF, HDF5, NRRD, NIfTI
│   ├── transforms/     # EM-specific augmentations
│   └── utils/          # I/O helpers and label utilities
├── configs/            # Hydra YAML configuration files
├── scripts/            # Training entry points and dataset download scripts
├── notebooks/          # Exploratory Jupyter notebooks
└── tests/              # Unit test suite
```

## Loss Functions

### Vista3DLoss / Vista2DLoss

The main training losses compose three branches:

| Branch | Head | Loss components |
|--------|------|-----------------|
| **Semantic** | `head_semantic` | CE + soft IoU + soft Dice |
| **Instance** | `head_instance` | Pull/push/norm discriminative (boundary + skeleton weighted) |
| **Geometry** | `head_geometry` | L_dir + L_cov + L_raw (auxiliary, not used at inference) |

The geometry head is purely an auxiliary training signal that enriches
backbone gradients.  Set `weight_cov: 0.0` to disable the expensive
structure tensor computation (recommended for large-scale training).
L_dir (centroid offsets) and L_raw (image reconstruction) are cheap
and provide useful regularisation.

### CentroidEmbeddingLoss

Classic De Brabandere et al. (2017) discriminative loss.
Pull same-instance embeddings together, push different-instance centroids
apart, regularise norms.

### SkeletonEmbeddingLoss

Geometry-aware variant operating on predicted offset fields. Four
differentiable terms: L2 pull to nearest skeleton point, pairwise push on
instance centres, cosine boundary penalty (DT gradient alignment), and
skeleton benefit (differentiable DT sampling via `grid_sample`).

## Quick Start

### 1. Explore your data

```bash
jupyter notebook notebooks/01_explore_snemi3d.ipynb
```

### 2. Train a segmentation model

```bash
# CosmosTransfer3D on SNEMI3D
python scripts/train.py --config-name snemi3d

# CosmosTransfer3D on combined SNEMI3D + Neurite11 + MICrONS
python scripts/train.py --config-name combine
```

### 3. Override parameters via CLI

```bash
python scripts/train.py --config-name combine \
    data.batch_size=8 \
    training.max_epochs=200 \
    optimizer.lr=5e-4
```

### 4. Train with combined datasets

```bash
python scripts/train.py --config-name combine   # SNEMI3D + Neurite11 + MICrONS
```

### 5. Resume from a previous checkpoint

**Full resume** (same config — restores optimizer, LR schedule, epoch, step):

```bash
python scripts/train.py --config-name snemi3d \
    training.resume_from_checkpoint=outputs/checkpoints/last.ckpt
```

**Warm-start** (weights only — use when changing freeze/LR/architecture):

```bash
python scripts/train.py --config-name snemi3d \
    +ckpt_path=outputs/checkpoints/last.ckpt
```

See [TRAINING.md](doc/TRAINING.md#95-loading-a-previous-checkpoint).

### 6. Fast development run

```bash
python scripts/train.py training.fast_dev_run=true
```

### 7. Visualize volumes

```bash
# SNEMI3D (AC4 training volume, resolution 6×6×30 nm)
python -m neurons.visualizer \
    --raw data/SNEMI3D/AC4_inputs.h5 \
    --seg data/SNEMI3D/AC4_labels.h5 \
    --spacing 30,6,6

# CREMI3D (sample A, resolution 4×4×40 nm)
python -m neurons.visualizer \
    --raw data/CREMI3D/sample_A.h5:volumes/raw \
    --seg data/CREMI3D/sample_A.h5:volumes/labels/neuron_ids \
    --spacing 40,4,4

# MICrONS (resolution 8×8×40 nm)
python -m neurons.visualizer \
    --raw data/MICRONS/volume.h5 \
    --seg data/MICRONS/segmentation.h5 \
    --spacing 40,8,8
```

Opens a web viewer at `http://localhost:8899` with 4-panel layout (axial, coronal, sagittal, 3D Gaussian splats). Add `--no-browser` to skip auto-opening.

### 8. Profile training

```bash
python scripts/train.py --config-name snemi3d training.profiler=simple
```

## Configuration

All behavior is driven by YAML configs in `configs/`. See [doc/CONFIG_REFERENCE.md](doc/CONFIG_REFERENCE.md) for full parameter documentation.

| Config | Description |
|--------|-------------|
| `default.yaml` | Base configuration with all defaults |
| `snemi2d.yaml` | SNEMI3D 2D slice segmentation (Vista2D) |
| `snemi3d.yaml` | SNEMI3D 3D volumetric segmentation (CosmosTransfer3D) |
| `cremi3d.yaml` | CREMI3D multi-class segmentation |
| `microns.yaml` | MICrONS large-scale connectomics |
| `combine.yaml` | Combined SNEMI3D + Neurite11 + MICrONS training |

## Training

Vista3D (default) and Vista2D modules jointly train semantic and instance heads.

```yaml
model:
  type: vista3d          # or vista2d
  num_classes: 16
  emb_dim: 16
loss:
  weight_ce: 1.0
  weight_dice: 1.0
  weight_iou: 1.0
  weight_pull: 1.0
  weight_push: 1.0
  delta_v: 0.5
  delta_d: 1.5
  weight_geometry: 1.0   # auxiliary geometry head (0.0 to disable)
  weight_cov: 0.0        # disable expensive structure tensor (recommended)
```

## GPU Acceleration

Install the `gpu-cu13` (or `gpu-cu12`) extras to pull
[cupy](https://cupy.dev/), [cuML](https://docs.rapids.ai/api/cuml/),
and — via `cucim` if also installed — GPU kernels for several expensive
operations.  The codebase probes each dependency at runtime and falls
back cleanly when missing.

| Operation | CPU fallback | GPU |
|-----------|-------------|-------------|
| Instance clustering (val + inference) | `sklearn.cluster.HDBSCAN` / `hdbscan` package | `cuml.cluster.HDBSCAN` |
| Distance transform (EDT) | `scipy.ndimage.distance_transform_edt` | `cucim.core.operations.morphology.distance_transform_edt` |
| Gaussian filter | `scipy.ndimage.gaussian_filter` | `cupyx.scipy.ndimage.gaussian_filter` |
| Connected components | `scipy.ndimage.label` | `cupyx.scipy.ndimage.label` |
| Boundary detection | `skimage.segmentation.find_boundaries` | `cucim.skimage.segmentation.find_boundaries` |

All GPU-accelerated functions live in
`neurons/transforms/edt.py`, `neurons/transforms/find_boundaries.py`,
and `neurons/inference/clusterer.py` with automatic CPU fallback.

Note: RAPIDS dropped `cuml.cluster.MeanShift` in cuML 23.x, so the
`MeanShiftClusterer` always runs on scikit-learn (CPU).  Use
`HDBSCANClusterer` for a GPU-accelerated alternative.

DataLoader workers (forked processes) automatically fall back to the CPU
path since CUDA contexts do not survive `fork()`.

## Running Tests

```bash
pytest tests/ -v
```

## License

See LICENSE file.

# Neurons

<p align="center">
  <img src="teaser.png" alt="Neurons — from electron microscopy to boundary detection to instance segmentation" width="100%">
</p>

A modular, extensible PyTorch Lightning-based infrastructure for connectomics research.

## Overview

**Neurons** provides a production-ready codebase for training segmentation models on electron microscopy (EM) data. It supports multiple dataset types, model architectures, and training paradigms out of the box, while remaining flexible enough for custom extensions.

## Features

- **Multi-Dataset Support** -- SNEMI3D, CREMI3D, MICRONS, and combined multi-dataset training
- **Vista Architecture** -- Vista3D and Vista2D with semantic + instance dual heads
- **Model Zoo** -- SegResNet and Vista3D wrappers via MONAI
- **Connectomics Losses** -- Discriminative, Boundary, and Weighted Boundary losses
- **Evaluation Metrics** -- Adjusted Rand Index (ARI), Adjusted Mutual Information (AMI), Dice, IoU
- **Hydra Configuration** -- YAML-based config with CLI overrides, no code changes needed
- **Experiment Tracking** -- Weights & Biases and TensorBoard integration
- **EM-Specific Augmentations** -- Elastic deformation, missing sections, imaging defects
- **Multi-Format I/O** -- HDF5, TIFF, NRRD with automatic format detection

## Installation

```bash
# Clone the repository
git clone <repo-url> neurons
cd neurons

# Install in development mode
pip install -e ".[dev]"
```

### Dependencies

Core: PyTorch, PyTorch Lightning, MONAI, einops, Hydra, h5py, tifffile, pynrrd

## Directory Structure

```
neurons/
├── neurons/
│   ├── datasets/       # Dataset classes: BASE, SNEMI3D, CREMI3D, MICRONS
│   ├── datamodules/    # Lightning DataModules + COMBINE datamodule
│   ├── modules/        # Lightning modules: semantic_seg, instance_seg, vista3d, vista2d
│   ├── models/         # Model wrappers: Base, Vista3D, SegResNet
│   ├── losses/         # Loss functions: discriminative, boundary, weighted_boundary
│   ├── preprocessors/  # Data loaders: TIFF, HDF5, NRRD
│   ├── transforms/     # EM-specific augmentation pipelines
│   └── utils/          # I/O utilities: find_path, load_volume, save_volume
├── configs/            # Hydra YAML configuration files
├── scripts/            # Training entry points
├── notebooks/          # EDA Jupyter notebooks
└── tests/              # Unit test suite
```

## Quick Start

### 1. Explore your data

```bash
jupyter notebook notebooks/01_explore_snemi3d.ipynb
```

### 2. Train a semantic segmentation model

```bash
python scripts/train.py --config-name snemi3d
```

### 3. Override parameters via CLI

```bash
python scripts/train.py --config-name snemi3d \
    data.batch_size=8 \
    training.max_epochs=200 \
    optimizer.lr=5e-4
```

### 4. Train with combined datasets

```bash
python scripts/train.py --config-name combine
```

### 5. Fast development run

```bash
python scripts/train.py training.fast_dev_run=true
```

## Configuration

All behavior is driven by YAML configs in `configs/`:

| Config | Description |
|--------|-------------|
| `default.yaml` | Base configuration with all defaults |
| `snemi3d.yaml` | SNEMI3D neuron segmentation |
| `cremi3d.yaml` | CREMI3D multi-class segmentation |
| `microns.yaml` | MICRONS large-scale connectomics |
| `combine.yaml` | Multi-dataset Vista3D training |

## Training

Vista3D (default) and Vista2D modules jointly train semantic and instance heads.

```yaml
model:
  type: vista3d          # or vista2d
  num_classes: 16
  emb_dim: 16
loss:
  ce_weight: 0.5
  dice_weight: 0.5
  weight_pull: 1.0
  weight_push: 1.0
  delta_v: 0.5
  delta_d: 1.5
```

## Running Tests

```bash
pytest tests/ -v
```

## License

See LICENSE file.

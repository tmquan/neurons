# Datasets

This document describes the four connectomics datasets supported by the **neurons** codebase, their label semantics, and how they are unified through the combined datamodule.

---

## 1. SNEMI3D

| Property | Value |
|---|---|
| **Tissue** | Mouse somatosensory cortex |
| **Modality** | Serial-section TEM (ssTEM) |
| **Resolution** | 6 x 6 x 30 nm (anisotropic) |
| **Volumes** | AC3 (test), AC4 (train) |
| **Slices** | 100 per volume, 1024 x 1024 px |
| **Labels** | Neuron instance segmentation |
| **Source** | [SNEMI3D Grand Challenge](https://snemi3d.grand-challenge.org/) |
| **Reference** | Kasthuri et al. (2015) Cell 162(3) |

**Label scheme** (2 classes):

| Value | Class |
|---|---|
| 0 | Background |
| > 0 | Neuron instance ID |

**Download:**
```bash
python scripts/download_snemi3d.py --output data/snemi3d
```

---

## 2. CREMI3D

| Property | Value |
|---|---|
| **Tissue** | Drosophila melanogaster brain |
| **Modality** | Serial-section TEM (ssTEM) |
| **Resolution** | 4 x 4 x 40 nm (anisotropic) |
| **Volumes** | Sample A, B (train+val), C (test) |
| **Slices** | 125 per volume, 1250 x 1250 px |
| **Labels** | Neurons, synaptic clefts, (mitochondria) |
| **Source** | [CREMI Challenge](https://cremi.org/) |

**Label scheme** (4 classes via offset encoding):

| Value Range | Class |
|---|---|
| 0 | Background |
| 1 -- 999,999 | Neuron instance ID |
| 1,000,000 -- 1,999,999 | Synaptic cleft instance ID |
| >= 2,000,000 | Mitochondria instance ID |

The offset encoding avoids ID collisions when neuron, cleft, and mitochondria annotations are stored in a single label volume.

**Download:**
```bash
python scripts/download_cremi3d.py --output data/cremi3d
```

---

## 3. MICrONS (minnie65)

| Property | Value |
|---|---|
| **Tissue** | Mouse visual cortex (layer 2/3 -- 5) |
| **Modality** | Serial-section EM (ssEM) |
| **Resolution** | 8 x 8 x 40 nm (mip 0, anisotropic) |
| **Full volume** | 212,992 x 180,224 x 13,088 voxels (~500 TB) |
| **Our crop** | 1,024 x 1,024 x 1,024 voxels at origin (140000, 100000, 20000) |
| **Labels** | Dense neuron segmentation (proofread) |
| **Source** | [MICrONS Explorer](https://www.microns-explorer.org/) |
| **Reference** | MICrONS Consortium (2021) bioRxiv |

**Segmentation versions:**

| Version | Date | Status |
|---|---|---|
| v117 | June 2021 | First proofread release |
| v343 | February 2022 | Updated proofreading |
| v943 | January 2024 | Updated proofreading |
| **v1300** | **January 2025** | **Latest (default)** |

**Label scheme** (2 classes):

| Value | Class |
|---|---|
| 0 | Background |
| > 0 | Neuron segment ID (uint64) |

**Download:**
```bash
# Default: v1300 segmentation, 128^3 crop
python scripts/download_microns.py

# 1024^3 crop
python scripts/download_microns.py --size 1024 1024 1024

# Multiple versions
python scripts/download_microns.py --seg-version 117 1300
```

---

## 4. MitoEM2

| Property | Value |
|---|---|
| **Tissue** | 8 cell types (see table below) |
| **Modality** | FIB-SEM, ssSEM, SBF-SEM |
| **Format** | NIfTI (.nii.gz), nnU-Net convention |
| **Labels** | Background, mitochondria interior, mitochondria boundary |
| **Source** | [MitoEM Grand Challenge](https://mitoem.grand-challenge.org/) |
| **Reference** | Wei et al. (2020) MICCAI |

**Sub-datasets:**

| Dataset | Cell Type | Modality | Resolution (nm) | Train | Test |
|---|---|---|---|---|---|
| Dataset001_ME2-Beta | Beta cells | FIB-SEM | 16 x 16 x 16 | 4 | 3 |
| Dataset002_ME2-Jurkat | Jurkat cells | FIB-SEM | 16 x 16 x 16 | 2 | 1 |
| Dataset003_ME2-Macro | Macrophages | FIB-SEM | 16 x 16 x 16 | 1 | 1 |
| Dataset004_ME2-Mossy | Mossy fibers | ssSEM | 8 x 8 x 30 | 3 | 2 |
| Dataset005_ME2-Podo | Podocytes | FIB-SEM | 16 x 16 x 16 | 2 | 1 |
| Dataset006_ME2-Pyra | Pyramidal neurons | ssSEM | 8 x 8 x 30 | 17 | 1 |
| Dataset007_ME2-Sperm | Sperm cells | FIB-SEM | 16 x 16 x 16 | 2 | 1 |
| Dataset008_ME2-Stem | Stem cells | SBF-SEM | 8 x 8 x 30 | 2 | 1 |

**Label scheme** (3 classes):

| Value | Class | Meaning |
|---|---|---|
| 0 | Background | Cytoplasm, membranes, other organelles |
| 1 | Mitochondria | Interior of mitochondria (matrix + cristae) |
| 2 | Boundary | Outer membrane separating adjacent mitochondria |

The **boundary class** serves as a separator between touching mitochondria. At inference, individual mitochondria instances are recovered by running connected components on the mitochondria mask (label == 1) after removing boundary pixels. Without this separator, adjacent mitochondria would merge into a single component.

**Download:**
```bash
python scripts/download_mitoem2.py --output data/mitoem2 --link /scratch/MitoEM2
```

---

## Union Label Map

When training on multiple datasets simultaneously via `CombineDataModule`, all native labels are mapped to a shared 5-class semantic space:

| Union ID | Class Name | Source Datasets |
|---|---|---|
| 0 | background | All datasets (native label 0) |
| 1 | neuron | SNEMI3D (fg), CREMI3D (< 1M), MICrONS (fg) |
| 2 | cleft | CREMI3D (1M -- 2M) |
| 3 | mitochondria | CREMI3D (>= 2M), MitoEM2 (native 1) |
| 4 | mito_boundary | MitoEM2 (native 2) |

**Ignoring classes:** If you don't need certain classes (e.g., `mito_boundary`), pass `ignore_classes={"mito_boundary"}` to `CombineDataModule` and those pixels revert to background (0).

```python
from neurons.datamodules import CombineDataModule, SNEMI3DDataModule, MitoEM2DataModule

combine = CombineDataModule(
    datamodules={
        "snemi3d": (SNEMI3DDataModule(data_root="data/snemi3d"), 1.0),
        "mitoem2": (MitoEM2DataModule(data_root="data/mitoem2"), 2.0),
    },
    ignore_classes={"mito_boundary"},  # optional
)
```

---

## Data Location

| Dataset | Default Path | Scratch Path |
|---|---|---|
| SNEMI3D | `data/snemi3d/` | `/scratch/SNEMI3D/` |
| CREMI3D | `data/cremi3d/` | `/scratch/CREMI3D/` |
| MICrONS | `data/MICRONS/` | `/scratch/MICRONS/` |
| MitoEM2 | `data/mitoem2/` | `/scratch/MitoEM2/` |

## EDA Notebooks

| Notebook | Dataset |
|---|---|
| `notebooks/01_explore_snemi3d.ipynb` | SNEMI3D (AC3 + AC4) |
| `notebooks/02_explore_cremi3d.ipynb` | CREMI3D (samples A, B, C) |
| `notebooks/03_explore_microns.ipynb` | MICrONS minnie65 (mip 0) |
| `notebooks/04_explore_mitoem2.ipynb` | MitoEM2 (all 8 sub-datasets) |

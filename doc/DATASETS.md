# Datasets

This document describes the six connectomics datasets supported by the **neurons** codebase, their label semantics, and how they are unified through the combined datamodule.

---

## 1. SNEMI3D

| Property | Value |
|---|---|
| **Tissue** | Mouse somatosensory cortex |
| **Modality** | Serial-section TEM (ssTEM) |
| **Resolution** | 6 x 6 x 30 nm (anisotropic) |
| **Volumes** | AC4 (train, EM + labels), AC3 (test, EM only — labels never released) |
| **Slices** | AC4: 100, AC3: 100 (1024 x 1024 px) |
| **Labels** | Neuron instance segmentation |
| **Source** | [SNEMI3D Grand Challenge](https://snemi3d.grand-challenge.org/) |
| **Reference** | Kasthuri et al. (2015) Cell 162(3) |

**Label scheme** (2 classes):

| Value | Class |
|---|---|
| 0 | Background |
| > 0 | Neuron instance ID |

AC3/AC4 GCS coordinates (verified, OCP scale 1 = GCS mip0):
  AC4: x=[4400,5424], y=[5440,6464], z=[1099,1199] — outside ground_truth cylinder.
  AC3/AC4 segmentations were separate OCP annotation tokens (now offline).
  Labels only available via snemi.zip from rhoana/Zenodo.

**Download:**
```bash
# AC3 EM + AC4 EM/labels from snemi.zip
python scripts/download_snemi3d.py --source snemi
```

### 1b. Neurite11 (SNEMI3D extended)

Neurite11 (Kasthuri et al. 2015, "kasthuri11") is the **parent volume**
of SNEMI3D — AC3 and AC4 are tiny 1024×1024×100 crops from it. The full
volume is much larger (~10752×13312×1850) and has dense annotation
(`ground_truth`) at exactly 6×6×30 nm, the same resolution as SNEMI3D.

| Property | Value |
|---|---|
| **Tissue** | Mouse somatosensory cortex (same as SNEMI3D) |
| **Resolution** | 6 x 6 x 30 nm (mip1, identical to SNEMI3D) |
| **Full volume** | 10752 × 13312 × 1850 voxels (mip0) |
| **Annotated cylinder** | ~5000 × 2900 × 300 voxels (X=3000–8000, Y=7200–10100, Z=950–1250) |
| **Train crop** | 1 × 5000×2900×300 (full annotated cylinder) |
| **Labels** | Neuron instances (`ground_truth`), synapses, vesicles, mito |
| **Source** | [neuroglancer-public-data](gs://neuroglancer-public-data/kasthuri2011/) |

**Download:**
```bash
# Probe volume info first
python scripts/download_snemi3d.py --probe

# Download (saved alongside AC3/AC4 in data/SNEMI3D/)
python scripts/download_snemi3d.py --source neurite11
```

---

## 2. CREMI3D

| Property | Value |
|---|---|
| **Tissue** | Drosophila melanogaster brain |
| **Modality** | Serial-section TEM (ssTEM) |
| **Resolution** | 4 x 4 x 40 nm (anisotropic) |
| **Train/Val** | Samples A, B, C (with ground truth) |
| **Test** | Samples A+, B+, C+ (padded, disjoint) |
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
python scripts/download_cremi3d.py --output data/CREMI3D
```

---

## 3. MICrONS (minnie65)

| Property | Value |
|---|---|
| **Tissue** | Mouse visual cortex (layer 2/3 -- 5) |
| **Modality** | Serial-section EM (ssEM) |
| **Resolution** | 8 x 8 x 40 nm (mip 0, anisotropic) |
| **Full volume** | ~175,104 x 108,544 x 21,056 voxels (~117 TB EM) |
| **Train crops** | 4 × 4096×4096×800 at disjoint XY positions across dense-tissue region |
| **Test crops** | 1 × 4096×4096×800 at center-left, upper-Y (disjoint from train) |
| **Labels** | Dense neuron segmentation (proofread, ~200K cells, ~120K neurons) |
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
# All 5 splits (4 train + 1 test, 4096×4096×800 each)
python scripts/download_microns.py --split --seg-version 1300

# Custom single crop
python scripts/download_microns.py --size 4096 4096 800 --seg-version 1300

# Multiple versions
python scripts/download_microns.py --split --seg-version 117 1300
```

---

## 4. Neurite (Kasthuri14 s1colEM)

| Property | Value |
|---|---|
| **Tissue** | Mouse somatosensory cortex (S1), layer 4 |
| **Modality** | Serial-section EM (sEM) |
| **Resolution** | 2 x 2 x 10 nm (mip1, anisotropic) |
| **Full volume** | 24576 x 16384 x 254 voxels (mip1) |
| **Train crops** | 3 × 4096×4096×254 at Y=4096 + 1 × at Y=8192 (volume center) |
| **Test crops** | 1 × 4096×4096×254 centered in top row (x=12288, Y=4096) |
| **Labels** | Neuron instance segmentation |
| **Source** | [Open Neurodata](https://open-neurodata.s3.amazonaws.com/kasthuri/kasthuri14s1colEM) |
| **Reference** | Kasthuri et al. (2015) Cell 162(3) |

**Label scheme** (2 classes):

| Value | Class |
|---|---|
| 0 | Background |
| > 0 | Neuron instance ID |

**Download:**

No automated download script is available yet. Image and segmentation
volumes can be fetched manually via `cloud-volume` from the
[Open Neurodata](https://open-neurodata.s3.amazonaws.com/kasthuri/kasthuri14s1colEM)
S3 bucket (image at mip1, segmentation at mip0 then downsampled 2× in XY).

---

## 5. MitoEM2

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
python scripts/download_mitoem2.py --output data/MitoEM2 --link /scratch/MitoEM2
```

---

## Union Label Map

When training on multiple datasets simultaneously via `CombineDataModule`, all native labels are mapped to a shared 5-class semantic space:

| Union ID | Class Name | Source Datasets |
|---|---|---|
| 0 | background | All datasets (native label 0) |
| 1 | neuron | SNEMI3D (fg), CREMI3D (< 1M), MICrONS (fg), Neurite (fg) |
| 2 | cleft | CREMI3D (1M -- 2M) |
| 3 | mitochondria | CREMI3D (>= 2M), MitoEM2 (native 1) |
| 4 | mito_boundary | MitoEM2 (native 2) |

**Ignoring classes:** If you don't need certain classes (e.g., `mito_boundary`), pass `ignore_classes={"mito_boundary"}` to `CombineDataModule` and those pixels revert to background (0).

```python
from neurons.datamodules import CombineDataModule, SNEMI3DDataModule, MitoEM2DataModule

combine = CombineDataModule(
    datamodules={
        "snemi3d": (SNEMI3DDataModule(data_root="data/SNEMI3D"), 1.0),
        "mitoem2": (MitoEM2DataModule(data_root="data/MitoEM2"), 2.0),
    },
    ignore_classes={"mito_boundary"},  # optional
)
```

---

## Data Location

| Dataset | Path |
|---|---|
| SNEMI3D | `data/SNEMI3D/` |
| CREMI3D | `data/CREMI3D/` |
| MICrONS | `data/MICRONS/` |
| Neurite | `data/NEURITE/` |
| MitoEM2 | `data/MitoEM2/` |

## Memory-Efficient Volume Loading

Two complementary strategies keep memory usage manageable:

### 1. `LazyVolDataset` — on-demand patch I/O (`datasets/lazy.py`)

`LazyVolDataset` stores only file paths and volume shapes (~bytes per
volume).  Each `__getitem__` reads a random `patch_size` crop directly from
disk using HDF5 chunked reads or TIFF memory-mapped reads — no full-volume
caching at all.

```
┌──────────────────────────────────────────────────────────┐
│  _discover_volumes()  scans files, stores metadata only  │
│  __getitem__(i)       reads patch from disk on demand    │
│  Memory: O(num_volumes × metadata) ≈ negligible          │
│  vs. CacheDataset: O(num_volumes × volume_size) ≈ GBs   │
└──────────────────────────────────────────────────────────┘
```

This is the recommended path for large-scale or DDP training where each
rank would otherwise duplicate full volumes in RAM.

### 2. `CircuitDataset` virtual-length scheme (`datasets/base.py`)

All datasets inherit from `CircuitDataset` (which wraps MONAI's `CacheDataset`).
When operating in 3D volume mode (`slice_mode=False`), samples are random crops
from the same underlying volume.  `CircuitDataset` uses a **virtual length**
scheme to decouple epoch length from the number of unique cached entries:

```
┌──────────────────────────────────────────────────────────┐
│  _prepare_data()  returns only UNIQUE entries:           │
│    slice_mode=True  → one entry per Z slice              │
│    slice_mode=False → one entry per volume               │
│                                                          │
│  _virtual_len  controls __len__() for the DataLoader     │
│  __getitem__(i) maps  i % real_len  to a cached entry    │
└──────────────────────────────────────────────────────────┘
```

| Mode | Unique entries cached | `__len__()` reports |
|---|---|---|
| `slice_mode=True`, default `num_samples` | N_slices | N_slices |
| `slice_mode=True`, custom `num_samples` | N_slices | `num_samples` |
| `slice_mode=False`, any `num_samples` | **1** | `num_samples` (or N_slices) |

For a 100-slice volume with `num_samples=100` in 3D mode, only **1 copy** is
cached instead of 100.  The DataLoader still iterates 100 steps per epoch, and
random transforms (e.g. `RandCropByPosNegLabel`) produce different crops each
time.

This applies uniformly to all six datasets:

| Dataset | 3D volume mode | Slice mode with `num_samples` |
|---|---|---|
| SNEMI3D (incl. Neurite11) | 1 entry, virtual len | N_slices entries, virtual len |
| CREMI3D | 1 entry, virtual len | N/A (always 3D) |
| MICrONS | 1 entry, virtual len | N_slices entries, virtual len |
| Neurite (Kasthuri14) | 1 entry, virtual len | N_slices entries, virtual len |
| MitoEM2 | per-volume entries, virtual len | per-slice entries, virtual len |

---

## EDA Notebooks

| Notebook | Dataset |
|---|---|
| `notebooks/01_explore_snemi3d.ipynb` | SNEMI3D (AC3 + AC4) |
| `notebooks/02_explore_cremi3d.ipynb` | CREMI3D (samples A, B, C) |
| `notebooks/03_explore_microns.ipynb` | MICrONS minnie65 (mip 0) |
| `notebooks/04_explore_mitoem2.ipynb` | MitoEM2 (all 8 sub-datasets) |
| `notebooks/05_explore_neurite.ipynb` | Neurite (Kasthuri14 s1colEM, mip1) |

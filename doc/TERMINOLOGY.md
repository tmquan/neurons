# Terminology

A glossary of key terms used throughout the **neurons** codebase.
Terms are grouped thematically; see the cross-references for where each
concept appears in code and documentation.

---

## Segmentation Tasks

**Semantic segmentation** -- Classify every voxel into a category
(background, neuron, cleft, mitochondria).  Produces a class label map.
See `SemanticLoss` in `losses/semantic.py`.

**Instance segmentation** -- Assign a unique integer ID to each
individual object (e.g. each neuron gets a different ID).
See `InstanceLoss` in `losses/instance.py`.

**Foreground / background** -- Foreground = any voxel with instance
label > 0.  Background = label 0.  Many losses and metrics ignore
background voxels.

---

## Architecture

**Backbone** -- The shared encoder that maps the input image to a feature
volume. Vista uses SegResNet/SegResNetDS2 from MONAI; Cosmos uses a
pretrained DiT (Diffusion Transformer) from NVIDIA. All task heads read
from the same backbone features.

**Task head** -- A lightweight decoder (Conv-Norm-ReLU-Conv) that
projects backbone features to a specific output space.  Three heads:
`head_semantic` (C classes), `head_instance` (E embedding dims),
`head_geometry` (S + S*S + 4 channels).

**Embedding** -- A learned per-voxel vector (E dimensions) produced by
`head_instance`.  Same-instance voxels should cluster together in
embedding space.  See `emb_dim` in config.

**Point prompt encoder** -- A small Conv module that encodes user-provided
point prompts (positive/negative clicks) into a residual that is added
to the backbone features.  Used only in proofread mode.
See `PointPromptEncoder` in `models/point_prompt_encoder.py`.

---

## Training Modes

**Automatic mode** -- The model sees only the raw EM image and must
predict everything without any hints.  This is the default mode.
See `training_modes: [automatic]` in config.

**Proofread mode** -- The model receives additional context beyond the
image: either partial annotations (fractionary) or simulated user
clicks (interactive).

**Fractionary sub-mode** -- A proofread variant triggered when the label
volume contains `ignore_index` values (partially annotated).  Unknown
voxels are masked out of the loss.

**Interactive sub-mode** -- A proofread variant triggered when labels are
fully annotated.  Random point prompts are sampled from ground truth to
simulate a human annotator clicking on objects.

---

## Loss Components

**Discriminative loss** -- The instance embedding loss from De Brabandere
et al. (2017).  Composed of three terms: pull, push, and norm.

**Pull loss (L_pull / variance)** -- Hinge-L2 that attracts each voxel's
embedding toward its instance centroid.  Parameterised by `delta_v`.
See `InstanceLoss._loss_single`.

**Push loss (L_push / distance)** -- Pairwise margin that repels
different instance centroids apart.  Parameterised by `delta_d`.

**Norm loss (L_norm / regularisation)** -- L2 penalty on centroid
embeddings to keep them near the origin.

**Boundary weight (`weight_edge`)** -- Multiplicative boost applied to
boundary voxels (detected via morphological gradient: max_pool != min_pool).
Forces the model to pay extra attention to instance boundaries.

**Skeleton weight (`weight_bone`)** -- Multiplicative boost based on the
normalised Euclidean distance transform (EDT).  Voxels near the medial
axis receive higher weight, encouraging the model to reconstruct the
interior structure of each instance.

---

## Geometry Head

**L_dir (direction)** -- Per-voxel unit vector pointing toward the
instance centroid (or nearest skeleton point when `dir_target=skeleton`).
Magnitude encodes relative distance: boundary voxels produce long
vectors, centre voxels produce short ones.

**L_cov (covariance / structure tensor)** -- Per-voxel 3x3 symmetric
matrix derived from the smoothed outer product of EDT gradients.
Encodes local shape (elongation, orientation).  Computationally expensive;
set `weight_cov: 0.0` to disable.

**L_raw (reconstruction)** -- RGBA reconstruction of the input image
through a sigmoid.  Target = [img, img, img, fg_mask].  Acts as a
decoder-style regulariser that prevents feature collapse.

---

## Geometric Primitives

**EDT (Euclidean distance transform)** -- For each foreground voxel,
the Euclidean distance to the nearest background voxel.  Computed per
instance via `cupyx.scipy.ndimage.distance_transform_edt` (GPU) or
`scipy.ndimage.distance_transform_edt` (CPU).

**Skeleton (medial axis)** -- A topology-preserving thinning of each
instance mask to a 1-pixel-wide centreline.  Computed via the Menten
et al. (ICCV 2023) iterative boundary-peeling algorithm (pure PyTorch
convolutions).  See `losses/skeletonize.py`.

**Structure tensor** -- The smoothed outer product of the EDT gradient
field.  Encodes local boundary orientation and instance thickness.
Visualised as elliptical tensor glyphs.  See `doc/GLYPH.md`.

**Centroid** -- The spatial mean of all voxel coordinates belonging to
an instance.  Used as the default target for the direction head
(`dir_target: centroid`).

---

## Metrics

**ARI (Adjusted Rand Index)** -- Measures agreement between predicted
and ground-truth instance segmentations, adjusted for chance.
Range: 0 (random) to 1 (perfect).

**AMI (Adjusted Mutual Information)** -- Information-theoretic agreement
metric, also adjusted for chance.

**VOI (Variation of Information)** -- Decomposed into split error
(over-segmentation) and merge error (under-segmentation).
Lower is better; 0 = perfect.

**TED (Tolerant Edit Distance)** -- Minimum number of split + merge
operations to transform the prediction into the ground truth.

**Dice / IoU** -- Per-class overlap metrics for semantic segmentation.

---

## Data Pipeline

**Volume** -- A 3D array of EM image data, shape [D, H, W].

**Patch** -- A random 3D crop from a volume, e.g. [48, 256, 256].
Configured via `patch_size` in YAML.

**Relabel after crop** -- After cropping, one instance may be split
into disconnected fragments.  `RelabelAfterCropd` runs connected-component
labelling (via `cupyx.scipy.ndimage.label` on GPU or `scipy.ndimage.label`
on CPU) to assign unique IDs to each fragment.

**Find boundaries** -- `FindBoundariesd` detects and erases
inter-instance boundary pixels (sets them to background 0), teaching
the model to separate touching instances.

---

## GPU Acceleration

**cupy** -- NumPy-compatible GPU array library.  Used for EDT,
gaussian_filter, and connected-component labelling on GPU.

**DLPack** -- A zero-copy tensor exchange protocol.  `torch_to_cupy()`
and `cupy_to_torch()` in `utils/gpu_ndimage.py` convert between
PyTorch CUDA tensors and cupy arrays without any host transfer.

**pmap** -- Persistent fork-based process pool in `utils/parallel.py`.
Used as CPU fallback when cupy is unavailable (e.g. in DataLoader
worker processes where CUDA contexts are invalid after fork).

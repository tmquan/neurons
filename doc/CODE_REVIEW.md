# Neurons Repository — Code Review

A readable, pedagogical review focused on einops, logic correctness, and code quality.

---

## 1. Einops Usage

### 1.1 Current Usage (Good)

Einops is used consistently in:
- `modules/vista3d_module.py`: `_EXPAND_PATTERN`, `_SQUEEZE_PATTERN` for batch/channel handling
- `metrics/semantic.py`: `rearrange(pred, "... -> (...)")` for flattening
- `utils/labels.py`: `rearrange(embedding, "e d h w -> (d h w) e")` in clustering
- `losses/`: reduce, rearrange for spatial reductions
- `inference/`: clustering label reshaping

### 1.2 Changes Made

| File | Before | After |
|------|--------|-------|
| `datasets/base.py` | `arr.unsqueeze(0)`, `crop.unsqueeze(0)` | `rearrange(arr, "... -> 1 ...")` |
| `utils/labels.py` | `labels_full.view(spatial_shape)` | `rearrange(labels_full, "(d h w) -> d h w", d=..., h=..., w=...)` |

### 1.3 Further Opportunities

- **`losses/skeletonize.py`**: kernel tensors `view(1, 1, 3, 3, 3)` → `rearrange(..., "d h w -> 1 1 d h w")`
- **`losses/discriminative.py`**: `.view(spatial_shape)` → einops with explicit spatial dims
- **`inference/soft_clustering.py`**: `.view(B, *spatial_shape)` → `rearrange(..., "b (d h w) -> b d h w", ...)`

---

## 2. Logic Correctness

### 2.1 `find_boundaries` — Shape Handling

**File:** `neurons/utils/labels.py`

- **2D labels `[1, H, W]`**: `has_channel = (dim==4 and shape[0]==1)` is False → treated as 3D with D=1. The roll and pool paths both handle this; no crash.
- **`instance_only`**: When True, marks only instance-background boundaries (erased); keeps instance-instance boundaries.

### 2.2 Transform Pipeline

- **Fast crop path**: Skips EnsureChannelFirstd, SpatialPadd, RandSpatialCropd (data already cropped).
- **Dataset normalization**: SNEMI3D/MICRONS normalize per volume in `_prepare_data`; no ScaleIntensityd in transforms when using fast crop. Documented implicitly by dataset-specific logic.

### 2.3 Semantic Metrics

- **`num_classes`**: Default 2; Vista3D uses `predictions["semantic"].shape[1]` (e.g. 16). Docstrings updated to note passing model's num_classes.
- **Background (class 0)**: Included in Dice/IoU; standard for multi-class semantic eval.

### 2.4 Data Flow

- Dataset → `image` [1, D, H, W], `label` [D, H, W]
- Collate → batch [B, 1, D, H, W], [B, D, H, W]
- `_prepare_targets` maps `labels > 0` to semantic when `semantic_ids` missing; CombineDataModule adds `semantic_ids` via CreateClassIds.

---

## 3. Code Quality

### 3.1 Strengths

- Clear separation: datasets, datamodules, models, losses, metrics, inference
- Solid docstrings in core modules
- Type hints in public APIs
- Consistent transform ordering across datamodules

### 3.2 Naming

- `_SQUEEZE_PATTERN`, `_EXPAND_PATTERN` in modules — clear
- `_prepare_targets` vs `_prepare_data` — dataset vs module, distinct roles

### 3.3 Documentation Gaps (Addressed)

- `doc/ARCHITECTURE.md` added for data flow and reading order
- Semantic metrics `num_classes` doc improved
- `find_boundaries` `instance_only` documented

---

## 4. Potential Edge Cases

### 4.1 Numerical Stability

- Dice/IoU use `eps=1e-7`; fine for float32
- Instance loss: `n = max(n_valid, 1)` avoids division by zero when no instances

### 4.2 `relabel_sequential`

- Preserves negative labels (`ignore_index`); correct for proofread mode

### 4.3 Contingency Table (Instance Metrics)

- Labels remapped to contiguous 0..K-1 to avoid huge sparse matrices with large instance IDs (e.g. CREMI offsets)
- Fallback when `n_true * n_pred > 500M` elements

### 4.4 `_label_to_rgb` (TensorBoard)

- Uses `torch.unique` + inverse for contiguous indexing; no overflow for large instance IDs

---

## 5. Summary of Applied Changes

| Change | File(s) |
|--------|---------|
| Einops for unsqueeze in _fast_crop | `datasets/base.py` |
| Einops for view in cluster_embeddings | `utils/labels.py` |
| Docstring for num_classes in semantic metrics | `metrics/semantic.py` |
| ARCHITECTURE.md | `doc/ARCHITECTURE.md` |
| CODE_REVIEW.md | `doc/CODE_REVIEW.md` |

---

## 6. Recommended Next Steps (Optional)

- Add tests for `RandFindBoundariesd` (2D and 3D, instance_only True/False)
- Document `ElasticDeformationd` as per-slice 2D for 3D input
- Consider einops in `skeletonize.py` and `discriminative.py` for consistency

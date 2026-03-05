"""Connected-component relabeling transforms using GPU-accelerated ndimage.

Replaces skimage-based relabeling with ``cupyx.scipy.ndimage.label``
(falling back to ``scipy.ndimage.label`` on CPU-only machines) via the
``neurons.utils.gpu_ndimage`` wrapper.

Standalone helpers
------------------
- ``relabel_sequential``            — map foreground labels to 1..N.
- ``relabel_connected_components``  — per-instance connected-component split.
- ``relabel_after_crop``            — convenience alias for the above.

MONAI dictionary transform
--------------------------
- ``Labeld`` / ``RelabelAfterCropd`` — apply the above inside a MONAI pipeline.
"""

from typing import Dict, Optional

import numpy as np
import torch
from einops import rearrange
from monai.config import KeysCollection
from monai.transforms import MapTransform

from neurons.utils.gpu_ndimage import (
    label as _gpu_label,
    generate_binary_structure as _gpu_gen_struct,
)

_CONN_TO_STRUCT: Dict[int, int] = {
    4: 1, 8: 2,          # 2-D
    6: 1, 18: 2, 26: 3,  # 3-D
}

_DEFAULT_CONN: Dict[int, int] = {2: 4, 3: 6}


# ── Standalone helpers ────────────────────────────────────────────────

def relabel_sequential(
    labels: torch.Tensor,
    start_label: int = 1,
) -> torch.Tensor:
    """Map foreground labels to consecutive integers starting at *start_label*.

    Background (``0``) is preserved.  Negative values pass through unchanged.
    """
    device, dtype = labels.device, labels.dtype
    labels_np = labels.cpu().numpy()

    neg_mask = labels_np < 0
    safe_np = labels_np.copy()
    safe_np[neg_mask] = 0

    max_val = safe_np.max()
    if max_val == 0:
        return labels.clone()

    unique_fg = np.unique(safe_np)
    unique_fg = unique_fg[unique_fg > 0]

    lut = np.zeros(int(max_val) + 1, dtype=labels_np.dtype)
    for i, val in enumerate(unique_fg):
        lut[val] = start_label + i

    relabeled_np = lut[safe_np]
    relabeled_np[neg_mask] = labels_np[neg_mask]

    return torch.from_numpy(relabeled_np).to(device=device, dtype=dtype)


def relabel_connected_components(
    labels: torch.Tensor,
    spatial_dims: int = 3,
    connectivity: Optional[int] = None,
) -> torch.Tensor:
    """Relabel by finding per-instance connected components.

    Uses ``cupyx.scipy.ndimage.label`` (GPU) with scipy CPU fallback.

    Two pixels are in the same component when they are neighbours **and**
    share the same original value.

    Args:
        labels: ``[*spatial]`` or ``[batch, *spatial]`` integer labels.
        spatial_dims: ``2`` or ``3``.
        connectivity: Scipy-style (default ``4`` for 2-D, ``6`` for 3-D).

    Returns:
        Relabelled tensor, same shape and dtype as *labels*.
    """
    if connectivity is None:
        connectivity = _DEFAULT_CONN[spatial_dims]

    if labels.dim() == spatial_dims + 1:
        return torch.stack([
            relabel_connected_components(labels[b], spatial_dims, connectivity)
            for b in range(labels.shape[0])
        ])

    device, dtype = labels.device, labels.dtype
    labels_np = labels.cpu().numpy()

    rank = spatial_dims
    struct_conn = _CONN_TO_STRUCT.get(connectivity, connectivity)
    struct = _gpu_gen_struct(rank, struct_conn)

    unique_ids = np.unique(labels_np)
    unique_ids = unique_ids[unique_ids > 0]

    result = np.zeros_like(labels_np)
    next_label = 1

    for uid in unique_ids:
        mask = (labels_np == uid).astype(np.int32)
        labeled, num_features = _gpu_label(mask, structure=struct)
        if num_features == 0:
            continue
        lut = np.zeros(num_features + 1, dtype=result.dtype)
        for i in range(1, num_features + 1):
            lut[i] = next_label
            next_label += 1
        result[mask.astype(bool)] = lut[labeled[mask.astype(bool)]]

    return torch.from_numpy(result).to(device=device, dtype=dtype)


def relabel_connected_components_3d(
    labels: torch.Tensor, connectivity: int = 6,
) -> torch.Tensor:
    """Relabel 3-D labels by connected components."""
    return relabel_connected_components(labels, spatial_dims=3, connectivity=connectivity)


def relabel_connected_components_2d(
    labels: torch.Tensor, connectivity: int = 4,
) -> torch.Tensor:
    """Relabel 2-D labels by connected components."""
    return relabel_connected_components(labels, spatial_dims=2, connectivity=connectivity)


def relabel_after_crop(
    labels: torch.Tensor,
    spatial_dims: int = 3,
    connectivity: Optional[int] = None,
) -> torch.Tensor:
    """Relabel instance labels after cropping.

    After cropping, instances may be split into disconnected fragments.
    This function assigns a unique ID to each connected component.
    """
    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
    return relabel_connected_components(labels, spatial_dims, connectivity)


# ── MONAI dictionary transform ───────────────────────────────────────

class Labeld(MapTransform):
    """Relabel instance labels via connected components after spatial cropping.

    After ``RandSpatialCropd``, a single instance can be split into
    disconnected fragments.  This transform assigns a unique ID to each
    connected component and renumbers them sequentially.

    Uses ``cupyx.scipy.ndimage.label`` for GPU acceleration.

    Deterministic — always applied.

    Args:
        keys: Keys of label tensors to transform.
        spatial_dims: 2 or 3 (default 3).
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__(keys)
        self.spatial_dims = spatial_dims

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            lbl = d[key]
            ndim = lbl.ndim if isinstance(lbl, np.ndarray) else lbl.dim()
            has_channel = ndim == self.spatial_dims + 1
            if has_channel:
                lbl = lbl[0] if isinstance(lbl, np.ndarray) else rearrange(lbl, "1 ... -> ...")
            if isinstance(lbl, np.ndarray):
                lbl = torch.from_numpy(np.ascontiguousarray(lbl).astype(np.int64))
            lbl = relabel_sequential(
                relabel_after_crop(lbl, spatial_dims=self.spatial_dims),
            )
            if has_channel:
                lbl = rearrange(lbl, "... -> 1 ...")
            d[key] = lbl
        return d


RelabelAfterCropd = Labeld

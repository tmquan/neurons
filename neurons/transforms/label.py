"""Connected-component relabeling transforms using GPU-accelerated ndimage.

Replaces skimage-based relabeling with ``cupyx.scipy.ndimage.label``
(falling back to ``scipy.ndimage.label`` on CPU-only machines) via the
``neurons.transforms.edt`` module.

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

from neurons.transforms.edt import (
    distance_transform_edt as _edt,
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
    Runs a single connected-component pass on the full foreground mask,
    then splits components that span different original IDs.

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

    fg = (labels_np > 0).astype(np.int32)
    if not fg.any():
        return torch.zeros_like(labels)

    cc_labeled, _ = _gpu_label(fg, structure=struct)

    pair_ids = labels_np.ravel().astype(np.int64) * (int(cc_labeled.max()) + 1) + cc_labeled.ravel().astype(np.int64)
    fg_mask = labels_np.ravel() > 0

    _, inverse = np.unique(pair_ids[fg_mask], return_inverse=True)

    result = np.zeros(labels_np.size, dtype=labels_np.dtype)
    result[fg_mask] = inverse + 1
    result = result.reshape(labels_np.shape)

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


class InstanceWeightsd(MapTransform):
    """Precompute boundary + skeleton weights in the data pipeline.

    Moves the expensive per-instance EDT / morphological erosion from the
    GPU training loop into CPU data workers, where it runs in parallel.

    Writes ``weight_edge`` and ``weight_bone`` into the sample dict.

    Args:
        keys: Keys of label tensors to compute weights from.
        spatial_dims: 2 or 3 (default 3).
        weight_edge: Boundary weight multiplier (1.0 = disabled).
        weight_bone: Skeleton weight multiplier (1.0 = disabled).
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_dims: int = 3,
        weight_edge: float = 10.0,
        weight_bone: float = 10.0,
    ) -> None:
        super().__init__(keys)
        self.spatial_dims = spatial_dims
        self.weight_edge = weight_edge
        self.weight_bone = weight_bone

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            lbl = d[key]
            if isinstance(lbl, torch.Tensor):
                lbl_np = lbl.cpu().numpy()
            else:
                lbl_np = np.asarray(lbl)

            has_channel = lbl_np.ndim == self.spatial_dims + 1
            if has_channel:
                lbl_np = lbl_np[0]

            if self.weight_edge > 1.0:
                import torch.nn.functional as F
                lbl_t = torch.from_numpy(lbl_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                _pool = F.max_pool3d if self.spatial_dims == 3 else F.max_pool2d
                _pad = (1, 1, 1, 1, 1, 1) if self.spatial_dims == 3 else (1, 1, 1, 1)
                padded = F.pad(lbl_t, _pad, mode="replicate")
                dilated = _pool(padded, 3, stride=1, padding=0)
                eroded = _pool(-padded, 3, stride=1, padding=0).neg_()
                boundary = (dilated != eroded).squeeze(0).squeeze(0).float()
                d["weight_edge"] = 1.0 + boundary * (self.weight_edge - 1.0)

            if self.weight_bone > 1.0:
                weight = np.ones_like(lbl_np, dtype=np.float32)
                unique_ids = np.unique(lbl_np)
                unique_ids = unique_ids[unique_ids > 0]
                for uid in unique_ids:
                    mask = lbl_np == uid
                    dt = _edt(mask).astype(np.float32)
                    dt_max = dt.max()
                    if dt_max > 0:
                        dt /= dt_max
                    weight[mask] = 1.0 + dt[mask] * (self.weight_bone - 1.0)
                d["weight_bone"] = torch.from_numpy(weight)

        return d

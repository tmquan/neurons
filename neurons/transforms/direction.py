"""Direction field transform using GPU-accelerated center-of-mass.

For each foreground pixel, computes a unit vector pointing toward the
center of mass of its instance.  This is a common regression target for
instance segmentation models ("offset" or "direction" head).

Uses ``cupyx.scipy.ndimage.center_of_mass`` (GPU) with scipy CPU fallback
via ``neurons.utils.gpu_ndimage``.

MONAI dictionary transform
--------------------------
- ``Directiond`` — compute per-pixel direction vectors from instance labels.
"""

from typing import Dict

import numpy as np
import torch
from einops import rearrange
from monai.config import KeysCollection
from monai.transforms import MapTransform

from neurons.utils.gpu_ndimage import center_of_mass as _center_of_mass


def _to_numpy_labels(labels) -> np.ndarray:
    """Convert labels to a contiguous int64 numpy array.

    Accepts ``torch.Tensor``, MONAI ``MetaTensor``, or ``np.ndarray``.
    """
    if isinstance(labels, np.ndarray):
        return np.ascontiguousarray(labels).astype(np.int64, copy=False)
    return labels.detach().cpu().numpy().astype(np.int64, copy=False)


def compute_direction_field(labels, normalize: bool = True) -> torch.Tensor:
    """Compute per-pixel direction vectors toward each instance's center of mass.

    Works identically for 2-D and 3-D inputs.

    Args:
        labels: ``[*spatial]`` integer instance labels (0 = background).
            Accepts ``torch.Tensor``, MONAI ``MetaTensor``, or ``np.ndarray``.
        normalize: If True, return unit vectors; otherwise raw offsets.

    Returns:
        ``[S, *spatial]`` direction field (``torch.float32``) where
        ``S`` = number of spatial dims.  Background pixels are zero.
    """
    labels_np = _to_numpy_labels(labels)
    spatial_shape = labels_np.shape
    ndim = len(spatial_shape)

    unique_ids = np.unique(labels_np)
    unique_ids = unique_ids[unique_ids > 0]

    direction = np.zeros((ndim, *spatial_shape), dtype=np.float32)

    if len(unique_ids) == 0:
        return torch.from_numpy(direction)

    ones = np.ones(spatial_shape, dtype=np.float32)
    index = unique_ids.tolist()
    centers = _center_of_mass(ones, labels=labels_np, index=index)
    if not isinstance(centers, list):
        centers = [centers]

    coords = np.indices(spatial_shape, dtype=np.float32)

    for uid, center in zip(unique_ids, centers):
        mask = labels_np == uid
        for d in range(ndim):
            direction[d][mask] = center[d] - coords[d][mask]

    if normalize:
        mag = np.sqrt(
            np.sum(direction ** 2, axis=0, keepdims=True),
        ).clip(min=1e-8)
        direction /= mag
        bg = labels_np == 0
        direction[:, bg] = 0.0

    return torch.from_numpy(direction)


class Directiond(MapTransform):
    """Compute per-pixel direction vectors from instance label maps.

    For each key, reads the instance label tensor and writes the
    direction field to ``{key}_direction`` (configurable via *suffix*).

    Uses ``cupyx.scipy.ndimage.center_of_mass`` for GPU acceleration.

    Args:
        keys: Keys of instance-label tensors.
        spatial_dims: 2 or 3 (default 3).
        suffix: Suffix appended to each key for the output direction field.
        normalize: If True (default), output unit direction vectors.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_dims: int = 3,
        suffix: str = "_direction",
        normalize: bool = True,
    ) -> None:
        super().__init__(keys)
        self.spatial_dims = spatial_dims
        self.suffix = suffix
        self.normalize = normalize

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            lbl = d[key]
            if isinstance(lbl, np.ndarray):
                if lbl.ndim == self.spatial_dims + 1:
                    lbl = lbl[0]
            else:
                if lbl.dim() == self.spatial_dims + 1:
                    lbl = rearrange(lbl, "1 ... -> ...")
            d[f"{key}{self.suffix}"] = compute_direction_field(
                lbl, normalize=self.normalize,
            )
        return d

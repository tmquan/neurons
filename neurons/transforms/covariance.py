"""Spatial covariance transform using GPU-accelerated center-of-mass.

For each instance, computes the spatial covariance matrix (encoding shape
and orientation) using the center of mass as the origin.  The per-pixel
output stores the upper-triangle covariance parameters of the instance
that pixel belongs to, providing a dense regression target for predicting
instance shape.

Uses ``cupyx.scipy.ndimage.center_of_mass`` (GPU) with scipy CPU fallback
via ``neurons.utils.gpu_ndimage``.

MONAI dictionary transform
--------------------------
- ``Covarianced`` — compute per-pixel covariance features from instance labels.
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


def compute_covariance_field(labels, normalized: bool = True) -> torch.Tensor:
    """Compute per-pixel spatial covariance features from instance labels.

    Works identically for 2-D and 3-D inputs.  For a ``D``-dimensional
    volume the upper triangle of each ``D x D`` covariance matrix is
    stored, giving ``D * (D + 1) / 2`` channels.

    Args:
        labels: ``[*spatial]`` integer instance labels (0 = background).
            Accepts ``torch.Tensor``, MONAI ``MetaTensor``, or ``np.ndarray``.
        normalized: If True, normalize covariance by the instance's
            bounding-box diagonal squared so values are scale-invariant.

    Returns:
        ``[C, *spatial]`` covariance field (``torch.float32``) where
        ``C = D*(D+1)//2``.  Background pixels are zero.
        Channel ordering for 3-D: ``[var_z, cov_zy, cov_zx, var_y, cov_yx, var_x]``.
    """
    labels_np = _to_numpy_labels(labels)
    spatial_shape = labels_np.shape
    ndim = len(spatial_shape)
    n_upper = ndim * (ndim + 1) // 2

    unique_ids = np.unique(labels_np)
    unique_ids = unique_ids[unique_ids > 0]

    cov_field = np.zeros((n_upper, *spatial_shape), dtype=np.float32)

    if len(unique_ids) == 0:
        return torch.from_numpy(cov_field)

    ones = np.ones(spatial_shape, dtype=np.float32)
    index = unique_ids.tolist()
    centers = _center_of_mass(ones, labels=labels_np, index=index)
    if not isinstance(centers, list):
        centers = [centers]

    coords = np.indices(spatial_shape, dtype=np.float32)

    for uid, center in zip(unique_ids, centers):
        mask = labels_np == uid
        n_pixels = int(mask.sum())
        if n_pixels < 2:
            continue

        deltas = np.empty((ndim, n_pixels), dtype=np.float32)
        for d in range(ndim):
            deltas[d] = coords[d][mask] - center[d]

        cov_matrix = (deltas @ deltas.T) / n_pixels

        if normalized:
            bbox_diag_sq = sum(
                (deltas[d].max() - deltas[d].min()) ** 2 for d in range(ndim)
            )
            cov_matrix /= max(bbox_diag_sq, 1.0)

        ch = 0
        for i in range(ndim):
            for j in range(i, ndim):
                cov_field[ch][mask] = cov_matrix[i, j]
                ch += 1

    return torch.from_numpy(cov_field)


class Covarianced(MapTransform):
    """Compute per-pixel spatial covariance features from instance labels.

    For each key, reads the instance label tensor and writes the
    covariance field to ``{key}_covariance`` (configurable via *suffix*).

    Uses ``cupyx.scipy.ndimage.center_of_mass`` for GPU acceleration.

    Args:
        keys: Keys of instance-label tensors.
        spatial_dims: 2 or 3 (default 3).
        suffix: Suffix appended to each key for the output covariance field.
        normalized: If True (default), covariances are scale-invariant.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_dims: int = 3,
        suffix: str = "_covariance",
        normalized: bool = True,
    ) -> None:
        super().__init__(keys)
        self.spatial_dims = spatial_dims
        self.suffix = suffix
        self.normalized = normalized

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
            d[f"{key}{self.suffix}"] = compute_covariance_field(
                lbl, normalized=self.normalized,
            )
        return d

"""Shared helpers for neurons transforms."""

import functools

import numpy as np


def _to_numpy_labels(labels) -> np.ndarray:
    """Convert labels to a contiguous int64 numpy array.

    Accepts ``torch.Tensor``, MONAI ``MetaTensor``, or ``np.ndarray``.
    """
    if isinstance(labels, np.ndarray):
        return np.ascontiguousarray(labels).astype(np.int64, copy=False)
    return labels.detach().cpu().numpy().astype(np.int64, copy=False)


@functools.lru_cache(maxsize=8)
def _cached_coordinate_grid(spatial_shape: tuple) -> np.ndarray:
    """Return a coordinate grid for *spatial_shape*, cached across calls.

    The result is read-only (``ndarray.flags.writeable == False``) so that
    callers cannot accidentally mutate the cached array.
    """
    grid = np.indices(spatial_shape, dtype=np.float32)
    grid.flags.writeable = False
    return grid

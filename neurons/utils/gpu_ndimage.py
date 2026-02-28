"""GPU-accelerated ndimage operations via cupy, with scipy CPU fallback.

Provides ``distance_transform_edt`` and ``gaussian_filter`` that operate
on cupy arrays when a CUDA device is available, falling back to scipy
for CPU-only environments.

All public functions accept **numpy arrays** and return **numpy arrays**
so they are drop-in replacements for their scipy counterparts.  Internally
they move data to/from the GPU only when cupy is available.

Usage::

    from neurons.utils.gpu_ndimage import distance_transform_edt, gaussian_filter

    dt = distance_transform_edt(mask)           # same API as scipy
    g  = gaussian_filter(volume, sigma=2.0)
"""

from typing import List, Optional, Sequence, Union

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as _cp_edt
    from cupyx.scipy.ndimage import gaussian_filter as _cp_gaussian
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


def is_available() -> bool:
    """Return True if cupy is importable and a CUDA device is visible."""
    if not _HAS_CUPY:
        return False
    try:
        cp.cuda.Device(0).compute_capability
        return True
    except cp.cuda.runtime.CUDARuntimeError:
        return False


_CHECKED: Optional[bool] = None


def _use_gpu() -> bool:
    global _CHECKED
    if _CHECKED is None:
        _CHECKED = is_available()
    return _CHECKED


def distance_transform_edt(
    mask: np.ndarray,
    sampling: Optional[Sequence[float]] = None,
    return_distances: bool = True,
    return_indices: bool = False,
) -> np.ndarray:
    """Euclidean distance transform -- GPU-accelerated when possible.

    Same signature as ``scipy.ndimage.distance_transform_edt``.
    """
    if _use_gpu() and not return_indices:
        mask_gpu = cp.asarray(mask)
        kwargs = {}
        if sampling is not None:
            kwargs["sampling"] = sampling
        dt_gpu = _cp_edt(mask_gpu, **kwargs)
        return cp.asnumpy(dt_gpu)

    from scipy.ndimage import distance_transform_edt as _sp_edt
    return _sp_edt(mask, sampling=sampling,
                   return_distances=return_distances,
                   return_indices=return_indices)


def gaussian_filter(
    input: np.ndarray,
    sigma: Union[float, Sequence[float]],
    order: Union[int, Sequence[int]] = 0,
    mode: str = "reflect",
    truncate: float = 4.0,
) -> np.ndarray:
    """Gaussian filter -- GPU-accelerated when possible.

    Same signature subset as ``scipy.ndimage.gaussian_filter``.
    """
    if _use_gpu():
        inp_gpu = cp.asarray(input)
        out_gpu = _cp_gaussian(inp_gpu, sigma=sigma, order=order,
                               mode=mode, truncate=truncate)
        return cp.asnumpy(out_gpu)

    from scipy.ndimage import gaussian_filter as _sp_gaussian
    return _sp_gaussian(input, sigma=sigma, order=order,
                        mode=mode, truncate=truncate)

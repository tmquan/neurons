"""GPU-accelerated ndimage operations via cucim/cupy, with scipy CPU fallback.

Provides drop-in replacements for common scipy.ndimage functions:

- ``distance_transform_edt`` — via ``cucim.core.operations.morphology``
- ``gaussian_filter``        — via ``cupyx.scipy.ndimage``
- ``label``                  — via ``cupyx.scipy.ndimage``
- ``generate_binary_structure`` — via ``cupyx.scipy.ndimage``
- ``center_of_mass``         — via ``cupyx.scipy.ndimage``

Falls back to scipy on CPU-only machines or in forked DataLoader workers
where CUDA contexts are invalid.

All public functions accept **numpy arrays** and return **numpy arrays**.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cupy as cp
    from cucim.core.operations.morphology import (
        distance_transform_edt as _cucim_edt,
    )
    from cupyx.scipy.ndimage import (
        center_of_mass as _cp_center_of_mass,
        gaussian_filter as _cp_gaussian,
        generate_binary_structure as _cp_gen_struct,
        label as _cp_label,
    )

    _HAS_CUCIM = True
except ImportError:
    _HAS_CUCIM = False

_CHECKED: Optional[bool] = None
_MAIN_PID: Optional[int] = None


def is_available() -> bool:
    """Return True if cucim/cupy is importable and a CUDA device is visible."""
    if not _HAS_CUCIM:
        return False
    try:
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


def _use_gpu() -> bool:
    """Return True only when cucim/cupy is usable in the *current* process.

    CUDA contexts do not survive ``fork()``, so forked DataLoader workers
    must not attempt GPU ops even if the main process can.
    """
    global _CHECKED, _MAIN_PID

    current_pid = os.getpid()

    if _CHECKED is None:
        _CHECKED = is_available()
        if _CHECKED:
            _MAIN_PID = current_pid
        return _CHECKED

    if not _CHECKED:
        return False

    if _MAIN_PID is not None and current_pid != _MAIN_PID:
        return False

    return True


# ======================================================================
# numpy-in / numpy-out wrappers  (drop-in scipy replacements)
# ======================================================================


def distance_transform_edt(
    mask: np.ndarray,
    sampling: Optional[Sequence[float]] = None,
    return_distances: bool = True,
    return_indices: bool = False,
) -> np.ndarray:
    """Euclidean distance transform via cucim (GPU) or scipy (CPU)."""
    if _use_gpu() and not return_indices:
        mask_gpu = cp.asarray(mask)
        dt_gpu = _cucim_edt(mask_gpu, sampling=sampling)
        return cp.asnumpy(dt_gpu)

    from scipy.ndimage import distance_transform_edt as _sp_edt

    return _sp_edt(
        mask,
        sampling=sampling,
        return_distances=return_distances,
        return_indices=return_indices,
    )


def gaussian_filter(
    input: np.ndarray,
    sigma: Union[float, Sequence[float]],
    order: Union[int, Sequence[int]] = 0,
    mode: str = "reflect",
    truncate: float = 4.0,
) -> np.ndarray:
    """Gaussian filter via cupy (GPU) or scipy (CPU)."""
    if _use_gpu():
        out_gpu = _cp_gaussian(
            cp.asarray(input), sigma=sigma, order=order,
            mode=mode, truncate=truncate,
        )
        return cp.asnumpy(out_gpu)

    from scipy.ndimage import gaussian_filter as _sp_gaussian

    return _sp_gaussian(
        input, sigma=sigma, order=order, mode=mode, truncate=truncate,
    )


def label(
    input: np.ndarray,
    structure: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """Connected-component labelling via cupy (GPU) or scipy (CPU)."""
    if _use_gpu():
        struct_gpu = cp.asarray(structure) if structure is not None else None
        lbl_gpu, n = _cp_label(cp.asarray(input), structure=struct_gpu)
        return cp.asnumpy(lbl_gpu), int(n)

    from scipy.ndimage import label as _sp_label

    return _sp_label(input, structure=structure)


def generate_binary_structure(rank: int, connectivity: int) -> np.ndarray:
    """Generate structuring element via cupy (GPU) or scipy (CPU)."""
    if _use_gpu():
        return cp.asnumpy(_cp_gen_struct(rank, connectivity))

    from scipy.ndimage import generate_binary_structure as _sp_gen

    return _sp_gen(rank, connectivity)


def center_of_mass(
    input: np.ndarray,
    labels: Optional[np.ndarray] = None,
    index: Optional[Union[int, Sequence[int]]] = None,
) -> Union[Tuple, List[Tuple]]:
    """Centre-of-mass via cupy (GPU) or scipy (CPU)."""
    if _use_gpu():
        lbl_gpu = cp.asarray(labels) if labels is not None else None
        return _cp_center_of_mass(
            cp.asarray(input), labels=lbl_gpu, index=index,
        )

    from scipy.ndimage import center_of_mass as _sp_com

    return _sp_com(input, labels=labels, index=index)

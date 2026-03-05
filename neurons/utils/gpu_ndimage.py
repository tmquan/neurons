"""GPU-accelerated ndimage operations via cupy, with scipy CPU fallback.

Provides ``distance_transform_edt``, ``gaussian_filter``, ``label``,
``generate_binary_structure``, and ``center_of_mass`` that operate on
cupy arrays when a CUDA device is available, falling back to scipy
for CPU-only environments.

All public functions accept **numpy arrays** and return **numpy arrays**
so they are drop-in replacements for their scipy counterparts.  Internally
they move data to/from the GPU only when cupy is available.

For hot-loop code that processes many instances, use the ``cupy_*``
prefixed helpers which accept and return **cupy arrays** directly,
eliminating per-call host<->device transfers.

Usage::

    from neurons.utils.gpu_ndimage import distance_transform_edt, gaussian_filter

    dt = distance_transform_edt(mask)           # same API as scipy
    g  = gaussian_filter(volume, sigma=2.0)
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as _cp_edt
    from cupyx.scipy.ndimage import gaussian_filter as _cp_gaussian
    from cupyx.scipy.ndimage import label as _cp_label
    from cupyx.scipy.ndimage import generate_binary_structure as _cp_gen_struct
    from cupyx.scipy.ndimage import center_of_mass as _cp_center_of_mass
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


def is_available() -> bool:
    """Return True if cupy is importable and a CUDA device is visible.

    Catches all exceptions (not just ``CUDARuntimeError``) because
    forked DataLoader workers can raise ``RuntimeError`` or ``OSError``
    from a corrupted CUDA context inherited across ``fork()``.
    """
    if not _HAS_CUPY:
        return False
    try:
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


_CHECKED: Optional[bool] = None
_MAIN_PID: Optional[int] = None


def _use_gpu() -> bool:
    """Return True only when cupy is usable in the *current* process.

    CUDA contexts do not survive ``fork()``, so forked DataLoader workers
    must not attempt cupy even if the main process can.  We record the PID
    of the first successful check and return False whenever the current PID
    differs (i.e. we are in a forked child).
    """
    import os
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
    """Euclidean distance transform -- GPU-accelerated when possible."""
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
    """Gaussian filter -- GPU-accelerated when possible."""
    if _use_gpu():
        inp_gpu = cp.asarray(input)
        out_gpu = _cp_gaussian(inp_gpu, sigma=sigma, order=order,
                               mode=mode, truncate=truncate)
        return cp.asnumpy(out_gpu)

    from scipy.ndimage import gaussian_filter as _sp_gaussian
    return _sp_gaussian(input, sigma=sigma, order=order,
                        mode=mode, truncate=truncate)


def label(
    input: np.ndarray,
    structure: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """Connected-component labelling -- GPU-accelerated when possible.

    Same signature as ``scipy.ndimage.label``.
    """
    if _use_gpu():
        inp_gpu = cp.asarray(input)
        struct_gpu = cp.asarray(structure) if structure is not None else None
        lbl_gpu, n = _cp_label(inp_gpu, structure=struct_gpu)
        return cp.asnumpy(lbl_gpu), int(n)

    from scipy.ndimage import label as _sp_label
    return _sp_label(input, structure=structure)


def generate_binary_structure(rank: int, connectivity: int) -> np.ndarray:
    """Generate structuring element -- GPU-accelerated when possible."""
    if _use_gpu():
        return cp.asnumpy(_cp_gen_struct(rank, connectivity))

    from scipy.ndimage import generate_binary_structure as _sp_gen
    return _sp_gen(rank, connectivity)


def center_of_mass(
    input: np.ndarray,
    labels: Optional[np.ndarray] = None,
    index: Optional[Union[int, Sequence[int]]] = None,
) -> Union[Tuple, List[Tuple]]:
    """Centre-of-mass -- GPU-accelerated when possible."""
    if _use_gpu():
        inp_gpu = cp.asarray(input)
        lbl_gpu = cp.asarray(labels) if labels is not None else None
        return _cp_center_of_mass(inp_gpu, labels=lbl_gpu, index=index)

    from scipy.ndimage import center_of_mass as _sp_com
    return _sp_com(input, labels=labels, index=index)


# ======================================================================
# cupy-native helpers  (cupy-in / cupy-out -- no host transfers)
# ======================================================================

def cupy_edt(mask_cp):
    """EDT on a cupy array, returns cupy array. No host transfer."""
    return _cp_edt(mask_cp)


def cupy_gaussian(input_cp, sigma, order=0, mode="reflect", truncate=4.0):
    """Gaussian filter on cupy array, returns cupy array."""
    return _cp_gaussian(input_cp, sigma=sigma, order=order,
                        mode=mode, truncate=truncate)


def cupy_label(input_cp, structure_cp=None):
    """Connected-component labelling on cupy array, returns (cupy, int)."""
    return _cp_label(input_cp, structure=structure_cp)


def cupy_gen_struct(rank: int, connectivity: int):
    """Generate structuring element as cupy array."""
    return _cp_gen_struct(rank, connectivity)


def cupy_center_of_mass(input_cp, labels_cp=None, index=None):
    """Centre-of-mass on cupy arrays."""
    return _cp_center_of_mass(input_cp, labels=labels_cp, index=index)


# ======================================================================
# DLPack zero-copy conversion  (torch CUDA ↔ cupy, no host transfer)
# ======================================================================

def torch_to_cupy(t):
    """Convert a CUDA torch.Tensor → cupy.ndarray via DLPack (zero-copy).

    The tensor must be contiguous and on a CUDA device.
    Falls back to cpu().numpy() → cp.asarray() for CPU tensors.
    """
    if t.is_cuda:
        return cp.from_dlpack(t.detach())
    return cp.asarray(t.cpu().numpy())


def cupy_to_torch(a, device=None):
    """Convert a cupy.ndarray → torch.Tensor via DLPack (zero-copy).

    If *device* is given the tensor is moved there; otherwise it stays
    on the same CUDA device as the cupy array.
    """
    import torch
    t = torch.from_dlpack(a)
    if device is not None and t.device != device:
        t = t.to(device)
    return t

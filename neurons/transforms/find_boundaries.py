"""Boundary detection with cucim GPU acceleration and skimage fallback."""

from typing import Dict, Optional

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable

from neurons.transforms.edt import _use_gpu


def find_boundaries(
    label,
    mode: str = "inner",
    connectivity: int = 1,
    **kwargs,
):
    """Find boundaries between labeled regions.

    Accepts ``numpy.ndarray`` or CUDA ``torch.Tensor``.  When a CUDA
    tensor is passed and cucim is available, the entire operation stays
    on the GPU via DLPack zero-copy — no CPU roundtrip.

    Args:
        label: Integer label array ``[*spatial]`` (numpy or torch).
        mode: Boundary mode (``'inner'``, ``'outer'``, ``'thick'``).
        connectivity: Neighbourhood connectivity.  ``1`` = face-adjacent
            only (6-connected in 3D, thinnest boundaries).  Higher values
            include edge/corner neighbours (up to 26-connected in 3D).

    Returns:
        Boolean boundary mask, same type and device as *label*.
    """
    is_tensor = isinstance(label, torch.Tensor)

    if is_tensor and label.is_cuda and _use_gpu():
        try:
            import cupy as cp
            from cucim.skimage.segmentation import (
                find_boundaries as _cucim_fb,
            )
            cp_label = cp.from_dlpack(label)
            cp_bnd = _cucim_fb(cp_label, mode=mode,
                               connectivity=connectivity, **kwargs)
            return torch.from_dlpack(cp_bnd)
        except Exception:
            pass

    if is_tensor:
        label_np = label.detach().cpu().numpy()
    else:
        label_np = label

    if _use_gpu():
        try:
            import cupy as cp
            from cucim.skimage.segmentation import (
                find_boundaries as _cucim_fb,
            )
            return cp.asnumpy(_cucim_fb(cp.asarray(label_np), mode=mode,
                                        connectivity=connectivity, **kwargs))
        except Exception:
            pass

    from skimage.segmentation import find_boundaries as _skimage_fb
    return _skimage_fb(label_np, mode=mode, connectivity=connectivity, **kwargs)


def boundary_mask_batch(
    labels: torch.Tensor,
    mode: str = "inner",
    connectivity: int = 1,
) -> torch.Tensor:
    """Batch boundary mask using thinnest connectivity (6-connected in 3D).

    Args:
        labels: Instance labels [B, *spatial].
        mode: Boundary mode (``'inner'``, ``'outer'``, ``'thick'``).
        connectivity: 1 = face-adjacent only (thinnest).

    Returns:
        Boolean mask [B, *spatial], True at boundary voxels.
    """
    parts = []
    for b in range(labels.shape[0]):
        bnd = find_boundaries(labels[b], mode=mode, connectivity=connectivity)
        if isinstance(bnd, np.ndarray):
            parts.append(torch.from_numpy(bnd).to(labels.device))
        else:
            parts.append(bnd)
    return torch.stack(parts)


class FindBoundariesd(MapTransform, Randomizable):
    """Zero out boundary voxels in instance labels (label × (1 − boundary)).

    Uses thinnest boundary (connectivity=1, 6-connected in 3D) so boundary
    voxels are multiplied out: label[boundary] = 0.

    Wraps :func:`find_boundaries` as a MONAI dictionary transform.
    Expects input labels in ``[C, *spatial]`` format (post
    ``EnsureChannelFirstd``).  Each channel is processed independently.

    Args:
        keys: Keys of instance label maps to process.
        mode: Boundary mode (``'inner'``, ``'outer'``, ``'thick'``).
        connectivity: 1 = face-adjacent only (thinnest boundaries).
        prob: Probability of applying the transform per sample.
    """

    def __init__(
        self,
        keys: KeysCollection,
        mode: str = "inner",
        connectivity: int = 1,
        prob: float = 1.0,
    ) -> None:
        super().__init__(keys)
        self.mode = mode
        self.connectivity = connectivity
        self.prob = prob
        self._do_transform = True

    def randomize(self, data: Optional[Dict] = None) -> None:  # type: ignore[override]
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data: Dict) -> Dict:
        self.randomize(data)

        if not self._do_transform:
            return data

        d = dict(data)

        for key in self.key_iterator(d):
            arr = d[key]
            is_tensor = isinstance(arr, torch.Tensor)

            if is_tensor:
                device = arr.device
                label_np = arr.cpu().numpy().copy()
            else:
                label_np = np.array(arr, copy=True)

            if label_np.ndim > 1:
                for c in range(label_np.shape[0]):
                    boundaries = find_boundaries(
                        label_np[c], mode=self.mode, connectivity=self.connectivity
                    )
                    label_np[c][boundaries] = 0
            else:
                boundaries = find_boundaries(
                    label_np, mode=self.mode, connectivity=self.connectivity
                )
                label_np[boundaries] = 0

            if is_tensor:
                label_np = torch.from_numpy(label_np).to(device)

            d[key] = label_np

        return d

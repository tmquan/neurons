"""Boundary detection with cucim GPU acceleration and skimage fallback."""

from typing import Dict, Optional

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable

from neurons.transforms.edt import _use_gpu


def find_boundaries(
    label: np.ndarray,
    mode: str = "inner",
    **kwargs,
) -> np.ndarray:
    """Find boundaries between labeled regions.

    Uses cucim on GPU when available, falls back to skimage.

    Args:
        label: Integer label array ``[*spatial]``.
        mode: Boundary mode (``'inner'``, ``'outer'``, ``'thick'``).

    Returns:
        Boolean boundary mask, same shape as *label*.
    """
    if _use_gpu():
        try:
            import cupy as cp
            from cucim.skimage.segmentation import (
                find_boundaries as _cucim_fb,
            )
            return cp.asnumpy(_cucim_fb(cp.asarray(label), mode=mode, **kwargs))
        except Exception:
            pass
    from skimage.segmentation import find_boundaries as _skimage_fb
    return _skimage_fb(label, mode=mode, **kwargs)


class FindBoundariesd(MapTransform, Randomizable):
    """Zero out boundary voxels in instance labels.

    Wraps :func:`find_boundaries` as a MONAI dictionary transform.
    Expects input labels in ``[C, *spatial]`` format (post
    ``EnsureChannelFirstd``).  Each channel is processed independently.

    Args:
        keys: Keys of instance label maps to process.
        mode: Boundary mode (``'inner'``, ``'outer'``, ``'thick'``).
        prob: Probability of applying the transform per sample.
    """

    def __init__(
        self,
        keys: KeysCollection,
        mode: str = "inner",
        prob: float = 1.0,
    ) -> None:
        super().__init__(keys)
        self.mode = mode
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
                    boundaries = find_boundaries(label_np[c], mode=self.mode)
                    label_np[c][boundaries] = 0
            else:
                boundaries = find_boundaries(label_np, mode=self.mode)
                label_np[boundaries] = 0

            if is_tensor:
                label_np = torch.from_numpy(label_np).to(device)

            d[key] = label_np

        return d

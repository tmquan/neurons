"""Boundary detection via cucim (GPU) with skimage CPU fallback.

Standalone helper
-----------------
- ``find_boundaries`` — detect boundaries in a label image.

MONAI dictionary transform
--------------------------
- ``FindBoundariesd`` — set boundary voxels to 0 in the label map.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable

from neurons.transforms.edt import _use_gpu

try:
    import cupy as cp
    from cucim.skimage.segmentation import (
        find_boundaries as _cucim_find_boundaries,
    )

    _HAS_CUCIM = True
except ImportError:
    _HAS_CUCIM = False


def find_boundaries(
    label_img: np.ndarray,
    mode: str = "inner",
) -> np.ndarray:
    """Find boundaries between labelled regions.

    Uses ``cucim.skimage.segmentation.find_boundaries`` (GPU) when
    available, falls back to ``skimage.segmentation.find_boundaries``.

    Args:
        label_img: Integer array of instance labels.
        mode: ``"inner"``, ``"outer"``, or ``"thick"``.

    Returns:
        Boolean array of the same shape, True at boundaries.
    """
    if _HAS_CUCIM and _use_gpu():
        result = _cucim_find_boundaries(cp.asarray(label_img), mode=mode)
        return cp.asnumpy(result)

    from skimage.segmentation import find_boundaries as _sk_find_boundaries

    return _sk_find_boundaries(label_img, mode=mode)


class FindBoundariesd(MapTransform, Randomizable):
    """Set boundary voxels to background (0) in label maps.

    Optionally applied stochastically (controlled by *prob*).
    Uses cucim for GPU acceleration with skimage CPU fallback.

    Args:
        keys: Keys of label tensors to transform.
        mode: Boundary mode (``"inner"``, ``"outer"``, ``"thick"``).
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

    def randomize(self, data=None) -> None:  # type: ignore[override]
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data: Dict) -> Dict:
        self.randomize(data)
        if not self._do_transform:
            return data

        d = dict(data)
        for key in self.key_iterator(d):
            lbl = d[key]
            if isinstance(lbl, np.ndarray):
                lbl = lbl.copy()
                lbl[find_boundaries(lbl, mode=self.mode)] = 0
                d[key] = lbl
            else:
                lbl_np = lbl.cpu().numpy().copy()
                lbl_np[find_boundaries(lbl_np, mode=self.mode)] = 0
                import torch
                d[key] = torch.from_numpy(lbl_np).to(lbl.device)
        return d

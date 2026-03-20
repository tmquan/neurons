"""Connected-component relabeling for instance segmentation labels.

After random spatial cropping, a single instance can become split into
disconnected parts that share the same label value.  ``Labeld`` runs
connected-component analysis (value-based connectivity) to assign each
disconnected part a unique label, then renumbers sequentially.

GPU path uses ``cucim.skimage.measure.label``; CPU fallback uses
``skimage.measure.label``.  Both treat two pixels as connected only
when they are neighbours **and** share the same value, which is the
correct behaviour for instance segmentation maps.
"""

from typing import Dict

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform

from neurons.transforms.edt import _use_gpu


def _cc_label(label_np: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Value-based connected-component labeling (cucim / skimage).

    connectivity=1: face-adjacent only (6-connected in 3D, thinnest).
    """
    if _use_gpu():
        try:
            import cupy as cp
            from cucim.skimage.measure import label as _cucim_label
            result = _cucim_label(cp.asarray(label_np), background=0, connectivity=connectivity)
            return cp.asnumpy(result)
        except Exception:
            pass
    from skimage.measure import label as _skimage_label
    return _skimage_label(label_np, background=0, connectivity=connectivity)


class Labeld(MapTransform):
    """Connected-component relabeling for instance labels.

    After cropping, instances that were previously connected may become
    disconnected.  This transform splits them via connected-component
    analysis and renumbers labels sequentially.

    Uses ``cucim.skimage.measure.label`` (GPU) or
    ``skimage.measure.label`` (CPU) which treats two pixels as connected
    only when they are neighbours **and** share the same value.

    Args:
        keys: Keys of instance label maps.
        spatial_dims: Number of spatial dimensions (2 or 3).
        connectivity: 1 = face-adjacent only (6-connected in 3D, thinnest).
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_dims: int = 3,
        connectivity: int = 1,
    ) -> None:
        super().__init__(keys)
        self.spatial_dims = spatial_dims
        self.connectivity = connectivity

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)

        for key in self.key_iterator(d):
            arr = d[key]
            is_tensor = isinstance(arr, torch.Tensor)

            if is_tensor:
                label_np = arr.detach().cpu().numpy()
            else:
                label_np = np.asarray(arr)

            had_channel = label_np.ndim > self.spatial_dims
            while label_np.ndim > self.spatial_dims:
                label_np = label_np[0]

            relabeled = _cc_label(label_np.astype(np.int64), connectivity=self.connectivity)

            if had_channel:
                relabeled = relabeled[np.newaxis]

            if is_tensor:
                result = torch.from_numpy(relabeled.astype(np.int64)).to(arr.device)
                if hasattr(arr, "meta"):
                    from monai.data import MetaTensor
                    result = MetaTensor(result, meta=arr.meta, applied_operations=arr.applied_operations)
                d[key] = result
            else:
                d[key] = relabeled

        return d

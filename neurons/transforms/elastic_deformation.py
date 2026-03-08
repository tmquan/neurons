"""Elastic deformation transform for connectomics/EM data."""

from typing import Dict, Optional

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable

from neurons.utils.gpu_ndimage import gaussian_filter


class ElasticDeformationd(MapTransform, Randomizable):
    """Apply elastic deformation to simulate tissue deformation artifacts.

    Common in serial section EM where slices can be warped during
    sample preparation.  Uses cupy-accelerated gaussian_filter when
    available, falls back to scipy.

    Args:
        keys: Keys of data to transform.
        sigma: Gaussian filter sigma for smoothing displacement field.
        alpha: Scaling factor for displacement magnitude.
        prob: Probability of applying transform.
    """

    def __init__(
        self,
        keys: KeysCollection,
        sigma: float = 10.0,
        alpha: float = 100.0,
        prob: float = 0.5,
    ) -> None:
        super().__init__(keys)
        self.sigma = sigma
        self.alpha = alpha
        self.prob = prob
        self._do_transform = True

    def randomize(self, data: Optional[Dict] = None) -> None:  # type: ignore[override]
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data: Dict) -> Dict:
        self.randomize(data)

        if not self._do_transform:
            return data

        d = dict(data)

        ref_key = self.keys[0]
        shape = d[ref_key].shape[-2:]  # H, W

        dx = gaussian_filter(
            self.R.random(shape).astype(np.float32) * 2 - 1,
            self.sigma,
        ) * self.alpha
        dy = gaussian_filter(
            self.R.random(shape).astype(np.float32) * 2 - 1,
            self.sigma,
        ) * self.alpha

        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        map_y = y + dy
        map_x = x + dx
        np.clip(map_y, 0, shape[0] - 1, out=map_y)
        np.clip(map_x, 0, shape[1] - 1, out=map_x)
        indices = (map_y.astype(np.int32), map_x.astype(np.int32))

        for key in self.key_iterator(d):
            arr = d[key]
            is_tensor = isinstance(arr, torch.Tensor)
            device = arr.device if is_tensor else None

            if is_tensor:
                arr = arr.cpu().numpy()

            if arr.ndim == 4:
                result = np.zeros_like(arr)
                for c in range(arr.shape[0]):
                    for z in range(arr.shape[1]):
                        result[c, z] = arr[c, z][indices]
            elif arr.ndim == 3:
                result = np.zeros_like(arr)
                for c in range(arr.shape[0]):
                    result[c] = arr[c][indices]
            else:
                result = arr[indices]

            if is_tensor:
                result = torch.from_numpy(result).to(device)

            d[key] = result

        return d

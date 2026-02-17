"""
Domain-specific transforms for connectomics/EM data.

These transforms simulate common artifacts and variations found in
electron microscopy imaging.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from monai.transforms import MapTransform, Randomizable


class ElasticDeformation(MapTransform, Randomizable):
    """
    Apply elastic deformation to simulate tissue deformation artifacts.

    Common in serial section EM where slices can be warped during
    sample preparation.

    Args:
        keys: Keys of data to transform.
        sigma: Gaussian filter sigma for smoothing displacement field.
        alpha: Scaling factor for displacement magnitude.
        prob: Probability of applying transform.
    """

    def __init__(
        self,
        keys: Tuple[str, ...],
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

        from scipy.ndimage import gaussian_filter

        dx = gaussian_filter(
            (self.R.random(shape) * 2 - 1),
            self.sigma,
        ) * self.alpha
        dy = gaussian_filter(
            (self.R.random(shape) * 2 - 1),
            self.sigma,
        ) * self.alpha

        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        indices = (
            np.clip(y + dy, 0, shape[0] - 1).astype(np.int32),
            np.clip(x + dx, 0, shape[1] - 1).astype(np.int32),
        )

        for key in self.keys:
            if key in d:
                arr = d[key]
                is_tensor = isinstance(arr, torch.Tensor)

                if is_tensor:
                    arr = arr.cpu().numpy()

                if arr.ndim == 3:
                    result = np.zeros_like(arr)
                    for c in range(arr.shape[0]):
                        result[c] = arr[c][indices]
                else:
                    result = arr[indices]

                if is_tensor:
                    result = torch.from_numpy(result)

                d[key] = result

        return d


class MissingSection(MapTransform, Randomizable):
    """
    Simulate missing sections in serial EM data.

    Replaces random slices with interpolated values or zeros to simulate
    missing sections that can occur during sample preparation.

    Args:
        keys: Keys of data to transform.
        prob: Probability of applying transform.
        fill_mode: How to fill missing section ('interpolate', 'zero', 'copy').
    """

    def __init__(
        self,
        keys: Tuple[str, ...],
        prob: float = 0.1,
        fill_mode: str = "interpolate",
    ) -> None:
        super().__init__(keys)
        self.prob = prob
        self.fill_mode = fill_mode
        self._do_transform = True
        self._missing_idx: Optional[int] = None

    def randomize(self, data: Optional[Dict] = None) -> None:  # type: ignore[override]
        self._do_transform = self.R.random() < self.prob

        if self._do_transform and data is not None:
            ref_key = self.keys[0]
            if ref_key in data:
                depth = data[ref_key].shape[0] if data[ref_key].ndim >= 3 else 1
                if depth > 2:
                    self._missing_idx = int(self.R.randint(1, depth - 1))

    def __call__(self, data: Dict) -> Dict:
        self.randomize(data)

        if not self._do_transform or self._missing_idx is None:
            return data

        d = dict(data)

        for key in self.keys:
            if key in d and d[key].ndim >= 3:
                arr = d[key]
                idx = self._missing_idx

                if self.fill_mode == "interpolate":
                    arr[idx] = (arr[idx - 1] + arr[idx + 1]) / 2
                elif self.fill_mode == "zero":
                    arr[idx] = 0
                elif self.fill_mode == "copy":
                    arr[idx] = arr[idx - 1]

                d[key] = arr

        return d


class Defects(MapTransform, Randomizable):
    """
    Simulate imaging defects common in EM data.

    Includes:
    - Line artifacts (charging)
    - Intensity variations

    Args:
        keys: Keys of data to transform (typically just 'image').
        prob: Probability of applying any defect.
        line_prob: Probability of line artifacts.
        intensity_prob: Probability of intensity shift.
    """

    def __init__(
        self,
        keys: Tuple[str, ...],
        prob: float = 0.3,
        line_prob: float = 0.5,
        intensity_prob: float = 0.5,
    ) -> None:
        super().__init__(keys)
        self.prob = prob
        self.line_prob = line_prob
        self.intensity_prob = intensity_prob
        self._do_transform = True
        self._defect_type: Optional[str] = None

    def randomize(self, data: Optional[Dict] = None) -> None:  # type: ignore[override]
        self._do_transform = self.R.random() < self.prob

        if self._do_transform:
            r = self.R.random()
            if r < self.line_prob:
                self._defect_type = "line"
            elif r < self.line_prob + self.intensity_prob:
                self._defect_type = "intensity"
            else:
                self._defect_type = None
                self._do_transform = False

    def __call__(self, data: Dict) -> Dict:
        self.randomize(data)

        if not self._do_transform:
            return data

        d = dict(data)

        for key in self.keys:
            if key in d:
                arr = d[key]
                is_tensor = isinstance(arr, torch.Tensor)

                if is_tensor:
                    device = arr.device
                    arr = arr.cpu().numpy()

                if self._defect_type == "line":
                    shape = arr.shape[-2:]

                    if self.R.random() < 0.5:
                        y = int(self.R.randint(0, shape[0]))
                        thickness = int(self.R.randint(1, 5))
                        intensity = float(self.R.uniform(0.5, 1.5))

                        if arr.ndim == 3:
                            arr[:, y : y + thickness, :] *= intensity
                        else:
                            arr[y : y + thickness, :] *= intensity
                    else:
                        x = int(self.R.randint(0, shape[1]))
                        thickness = int(self.R.randint(1, 5))
                        intensity = float(self.R.uniform(0.5, 1.5))

                        if arr.ndim == 3:
                            arr[:, :, x : x + thickness] *= intensity
                        else:
                            arr[:, x : x + thickness] *= intensity

                elif self._defect_type == "intensity":
                    shift = float(self.R.uniform(-0.2, 0.2))
                    scale = float(self.R.uniform(0.8, 1.2))
                    arr = arr * scale + shift

                arr = np.clip(arr, 0, 1)

                if is_tensor:
                    arr = torch.from_numpy(arr).to(device)

                d[key] = arr

        return d

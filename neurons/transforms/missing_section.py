"""Missing section simulation transform for serial EM data."""

from typing import Dict, Optional

import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable


class MissingSectiond(MapTransform, Randomizable):
    """Simulate missing sections in serial EM data.

    Replaces random slices with interpolated values or zeros to simulate
    missing sections that can occur during sample preparation.

    Only applies the fill to image keys.  For label keys the missing
    slice is zeroed out (treated as unlabelled) to avoid producing
    fractional instance IDs from interpolation.

    Args:
        keys: Keys of data to transform.
        prob: Probability of applying transform.
        fill_mode: How to fill missing section ('interpolate', 'zero', 'copy').
        label_keys: Keys that are labels (will be zeroed instead of interpolated).
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        fill_mode: str = "interpolate",
        label_keys: tuple = ("label",),
    ) -> None:
        super().__init__(keys)
        self.prob = prob
        self.fill_mode = fill_mode
        self.label_keys = set(label_keys)
        self._do_transform = True
        self._missing_idx: Optional[int] = None

    def randomize(self, data: Optional[Dict] = None) -> None:  # type: ignore[override]
        self._do_transform = self.R.random() < self.prob
        self._missing_idx = None

        if self._do_transform and data is not None:
            ref_key = self.keys[0]
            if ref_key in data:
                arr = data[ref_key]
                if arr.ndim == 4:
                    depth = arr.shape[1]
                elif arr.ndim == 3:
                    depth = arr.shape[0]
                else:
                    depth = 1
                if depth > 2:
                    self._missing_idx = int(self.R.randint(1, depth - 1))

    def __call__(self, data: Dict) -> Dict:
        self.randomize(data)

        if not self._do_transform or self._missing_idx is None:
            return data

        d = dict(data)

        for key in self.key_iterator(d):
            arr = d[key]
            if arr.ndim < 3:
                continue

            idx = self._missing_idx
            is_label = key in self.label_keys

            if arr.ndim == 4:
                if is_label or self.fill_mode == "zero":
                    arr[:, idx] = 0
                elif self.fill_mode == "interpolate":
                    arr[:, idx] = (arr[:, idx - 1] + arr[:, idx + 1]) / 2
                elif self.fill_mode == "copy":
                    arr[:, idx] = arr[:, idx - 1]
            else:
                if is_label or self.fill_mode == "zero":
                    arr[idx] = 0
                elif self.fill_mode == "interpolate":
                    arr[idx] = (arr[idx - 1] + arr[idx + 1]) / 2
                elif self.fill_mode == "copy":
                    arr[idx] = arr[idx - 1]

            d[key] = arr

        return d

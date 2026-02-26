"""Missing section simulation transform for serial EM data."""

from typing import Dict, Optional, Tuple

from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable


class MissingSectiond(MapTransform, Randomizable):
    """Simulate missing sections in serial EM data.

    Replaces random slices with interpolated values or zeros to simulate
    missing sections that can occur during sample preparation.

    Args:
        keys: Keys of data to transform.
        prob: Probability of applying transform.
        fill_mode: How to fill missing section ('interpolate', 'zero', 'copy').
    """

    def __init__(
        self,
        keys: KeysCollection,
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

        for key in self.key_iterator(d):
            if d[key].ndim >= 3:
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

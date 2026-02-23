"""EM imaging defect simulation transform."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable


class Defectsd(MapTransform, Randomizable):
    """Simulate imaging defects common in EM data.

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
        keys: KeysCollection,
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

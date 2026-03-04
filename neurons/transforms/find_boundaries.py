"""Random boundary erosion transform using find_boundaries."""

from typing import Dict, Optional

from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable

from neurons.utils.labels import find_boundaries


class RandFindBoundariesd(MapTransform, Randomizable):
    """Randomly find and erase instance boundaries to background (0).

    Uses ``find_boundaries`` (torch reimplementation of skimage's
    ``find_boundaries``) to detect boundary pixels, then sets them to 0.

    Follows MONAI's ``Randomizable`` protocol: ``randomize()`` flips a
    coin per sample; the erosion is skipped when the coin says no.

    Args:
        keys: Keys of label tensors to transform.
        prob: Probability of applying per sample (default 0.5).
        connectivity: 1 = face-adjacent (thin, default),
            ``ndim`` = include corners (thick).
        mode: ``"inner"`` (default), ``"thick"``, or ``"outer"``.
            See ``find_boundaries`` for details.
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.5,
        connectivity: int = 1,
        mode: str = "inner",
        prob_key: str = "_find_boundaries",
    ) -> None:
        super().__init__(keys)
        self.prob = prob
        self.connectivity = connectivity
        self.mode = mode
        self.prob_key = prob_key
        self._do_transform = True

    def randomize(self, data: Optional[Dict] = None) -> None:  # type: ignore[override]
        prob = self.prob
        if data is not None and self.prob_key in data:
            prob = float(data[self.prob_key])
        self._do_transform = self.R.random() < prob

    def __call__(self, data: Dict) -> Dict:
        self.randomize(data)
        if not self._do_transform:
            return data

        d = dict(data)
        for key in self.key_iterator(d):
            lbl = d[key]
            boundary = find_boundaries(lbl, connectivity=self.connectivity, mode=self.mode)
            out = lbl.clone()
            out[boundary] = 0
            d[key] = out
        return d

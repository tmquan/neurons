"""Random neuron boundary erosion transform."""

from typing import Dict, Optional

from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable

from neurons.utils.labels import erode_neuron_boundaries


class RandErodeNeuronBoundariesd(MapTransform, Randomizable):
    """Randomly erode neuron instance boundaries to background (0).

    A pixel is on a boundary if any neighbor (3x3 kernel) has a different
    non-zero label.  When applied, boundary pixels become 0, creating a
    1-pixel gap between adjacent instances.

    Follows MONAI's ``Randomizable`` protocol: ``randomize()`` flips a
    coin per sample; the erosion is skipped when the coin says no.

    Args:
        keys: Keys of label tensors to transform.
        prob: Probability of applying erosion per sample (default 0.5).
        spatial_dims: 2 or 3 (default 3).
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.5,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__(keys)
        self.prob = prob
        self.spatial_dims = spatial_dims
        self._do_transform = True

    def randomize(self, data: Optional[Dict] = None) -> None:  # type: ignore[override]
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data: Dict) -> Dict:
        self.randomize(data)
        if not self._do_transform:
            return data

        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = erode_neuron_boundaries(d[key], spatial_dims=self.spatial_dims)
        return d

"""Post-crop connected-component relabeling transform."""

from typing import Dict

from einops import rearrange
from monai.config import KeysCollection
from monai.transforms import MapTransform

from neurons.utils.labels import relabel_after_crop


class RelabelAfterCropd(MapTransform):
    """Relabel instance labels via connected components after spatial cropping.

    After ``RandSpatialCropd``, a single instance can be split into
    disconnected fragments.  This transform assigns a unique ID to each
    connected component so the discriminative loss treats them as
    separate instances.

    Deterministic — always applied.

    Args:
        keys: Keys of label tensors to transform.
        spatial_dims: 2 or 3 (default 3).
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__(keys)
        self.spatial_dims = spatial_dims

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for key in self.key_iterator(d):
            lbl = d[key]
            has_channel = lbl.dim() == self.spatial_dims + 1
            if has_channel:
                lbl = rearrange(lbl, "1 ... -> ...")
            lbl = relabel_after_crop(lbl, spatial_dims=self.spatial_dims)
            if has_channel:
                lbl = rearrange(lbl, "... -> 1 ...")
            d[key] = lbl
        return d

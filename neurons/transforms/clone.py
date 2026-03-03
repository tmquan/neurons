"""
Clone transform — detach crop tensors from shared memory before augmentation.
"""

import torch
from monai.transforms import MapTransform


class Cloned(MapTransform):
    """Clone tensors so that later in-place ops don't corrupt shared memory."""

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if isinstance(d[key], torch.Tensor):
                d[key] = d[key].clone()
        return d

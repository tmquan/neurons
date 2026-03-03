"""
Training callbacks for connectomics segmentation.

Includes:
- ImageLogger: TensorBoard image/slice logger for epoch-end visualisation
"""

from neurons.callbacks.tensorboard import ImageLogger

__all__ = [
    "ImageLogger",
]

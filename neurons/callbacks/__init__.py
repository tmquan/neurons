"""
Training callbacks for connectomics segmentation.

Includes:
- ImageLogger: TensorBoard image/slice logger for epoch-end visualisation
"""

from neurons.callbacks.memory import CudaEmptyCacheCallback
from neurons.callbacks.tensorboard import ImageLogger

__all__ = [
    "CudaEmptyCacheCallback",
    "ImageLogger",
]

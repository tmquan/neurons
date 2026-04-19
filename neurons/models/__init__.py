"""
Model architectures for connectomics segmentation.

All model wrappers share a three-head structure (semantic, instance,
geometry) attached via :class:`VistaTaskHead3D`.
"""

from neurons.models.base import BaseModel
from neurons.models.vista3d_model import Vista3DWrapper
from neurons.models.vista2d_model import Vista2DWrapper
from neurons.models.cosmospredict3d_model import CosmosPredict3DWrapper
from neurons.models.cosmostransfer3d_model import CosmosTransfer3DWrapper

__all__ = [
    "BaseModel",
    "Vista3DWrapper",
    "Vista2DWrapper",
    "CosmosPredict3DWrapper",
    "CosmosTransfer3DWrapper",
]

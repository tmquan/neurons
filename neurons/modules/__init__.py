"""
PyTorch Lightning modules for connectomics segmentation training.

All modules share a common automatic-mode forward + loss + metric loop
defined in :class:`BaseVistaModule` (Vista2D / Vista3D) and
:class:`BaseCosmosModule` (Cosmos-Predict / Cosmos-Transfer 3D).
"""

from neurons.modules.vista import BaseVistaModule
from neurons.modules.cosmos import BaseCosmosModule
from neurons.modules.vista3d_module import Vista3DModule
from neurons.modules.vista2d_module import Vista2DModule
from neurons.modules.cosmospredict3d_module import CosmosPredict3DModule
from neurons.modules.cosmostransfer3d_module import CosmosTransfer3DModule

__all__ = [
    "BaseVistaModule",
    "BaseCosmosModule",
    "Vista3DModule",
    "Vista2DModule",
    "CosmosPredict3DModule",
    "CosmosTransfer3DModule",
]

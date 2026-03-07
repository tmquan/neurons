"""
PyTorch Lightning modules for connectomics training tasks.

Includes:
- Vista3DModule: 3D Vista with semantic + instance + geometry heads
- Vista2DModule: 2D Vista with semantic + instance + geometry heads
- CosmosPredict2DModule: 2D Cosmos-Predict2.5 DiT segmentation module
- CosmosPredict3DModule: 3D Cosmos-Predict2.5 DiT volumetric segmentation module
- CosmosTransfer2DModule: 2D Cosmos-Transfer2.5 DiT segmentation module
- CosmosTransfer3DModule: 3D Cosmos-Transfer2.5 DiT volumetric segmentation module
"""

from neurons.modules.vista3d_module import Vista3DModule
from neurons.modules.vista2d_module import Vista2DModule
from neurons.modules.cosmospredict2d_module import CosmosPredict2DModule
from neurons.modules.cosmospredict3d_module import CosmosPredict3DModule
from neurons.modules.cosmostransfer2d_module import CosmosTransfer2DModule
from neurons.modules.cosmostransfer3d_module import CosmosTransfer3DModule

__all__ = [
    "Vista3DModule",
    "Vista2DModule",
    "CosmosPredict2DModule",
    "CosmosPredict3DModule",
    "CosmosTransfer2DModule",
    "CosmosTransfer3DModule",
]

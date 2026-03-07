"""
Model architectures for connectomics segmentation.

Includes:
- BaseModel: Abstract base class for all models
- Vista3DWrapper: 3D Vista architecture (semantic + instance + geometry)
- Vista2DWrapper: 2D Vista architecture (semantic + instance + geometry)
- SegResNetWrapper: MONAI SegResNet fallback with customizable heads
- CosmosPredict2DWrapper: 2D Cosmos-Predict2.5 DiT backbone for segmentation
- CosmosPredict3DWrapper: 3D Cosmos-Predict2.5 DiT backbone for volumetric segmentation
- CosmosTransfer2DWrapper: 2D Cosmos-Transfer2.5 DiT backbone for segmentation
- CosmosTransfer3DWrapper: 3D Cosmos-Transfer2.5 DiT backbone for volumetric segmentation
"""

from neurons.models.base import BaseModel
from neurons.models.segresnet import SegResNetWrapper
from neurons.models.vista3d_model import Vista3DWrapper
from neurons.models.vista2d_model import Vista2DWrapper
from neurons.models.cosmospredict2d_model import CosmosPredict2DWrapper
from neurons.models.cosmospredict3d_model import CosmosPredict3DWrapper
from neurons.models.cosmostransfer2d_model import CosmosTransfer2DWrapper
from neurons.models.cosmostransfer3d_model import CosmosTransfer3DWrapper

__all__ = [
    "BaseModel",
    "SegResNetWrapper",
    "Vista3DWrapper",
    "Vista2DWrapper",
    "CosmosPredict2DWrapper",
    "CosmosPredict3DWrapper",
    "CosmosTransfer2DWrapper",
    "CosmosTransfer3DWrapper",
]

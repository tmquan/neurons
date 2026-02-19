"""
PyTorch Lightning modules for connectomics training tasks.

Includes:
- Vista3DModule: 3D Vista/GAPE with semantic + instance + geometry heads
- Vista2DModule: 2D Vista/GAPE with semantic + instance + geometry heads
"""

from neurons.modules.vista3d_module import Vista3DModule
from neurons.modules.vista2d_module import Vista2DModule

__all__ = [
    "Vista3DModule",
    "Vista2DModule",
]

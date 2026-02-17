"""
Dataset classes for connectomics research.

All datasets inherit from CircuitDataset and implement:
- paper: Reference/citation metadata
- resolution: Voxel/spatial resolution specification
- labels: List of segmentation class labels
- data_files: Dictionary with volume and segmentation paths/arrays
"""

from neurons.datasets.base import CircuitDataset
from neurons.datasets.snemi3d import SNEMI3DDataset
from neurons.datasets.cremi3d import CREMI3DDataset
from neurons.datasets.microns import MICRONSDataset
from neurons.datasets.mitoem2 import MitoEM2Dataset

__all__ = [
    "CircuitDataset",
    "SNEMI3DDataset",
    "CREMI3DDataset",
    "MICRONSDataset",
    "MitoEM2Dataset",
]

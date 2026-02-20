"""
Neurons: A modular, extensible PyTorch Lightning-based infrastructure for connectomics research.

This library provides:
- MONAI-compatible dataset classes with standardized interfaces
- Preprocessor classes for handling multiple data formats (TIFF, HDF5, NRRD, NIfTI)
- Distributed training scaffolding with PyTorch Lightning
- Pre-built models including Vista3D backbone for 2D and 3D segmentation
- Connectomics-specific loss functions and evaluation metrics
"""

import warnings
warnings.filterwarnings(
    "ignore",
    message="The cuda.cudart module is deprecated",
    category=FutureWarning,
)

__version__ = "0.1.0"

from neurons.datasets import (
    CircuitDataset,
    SNEMI3DDataset,
    CREMI3DDataset,
    MICRONSDataset,
    MitoEM2Dataset,
)
from neurons.preprocessors import (
    BasePreprocessor,
    TIFFPreprocessor,
    HDF5Preprocessor,
    NRRDPreprocessor,
    NFTYPreprocessor,
)
from neurons.datamodules import (
    CircuitDataModule,
    SNEMI3DDataModule,
    CREMI3DDataModule,
    MICRONSDataModule,
    MitoEM2DataModule,
    CombineDataModule,
    UNION_LABEL_MAP,
    UNION_LABEL_NAMES,
    NUM_UNION_CLASSES,
)

__all__ = [
    # Datasets
    "CircuitDataset",
    "SNEMI3DDataset",
    "CREMI3DDataset",
    "MICRONSDataset",
    "MitoEM2Dataset",
    # Preprocessors
    "BasePreprocessor",
    "TIFFPreprocessor",
    "HDF5Preprocessor",
    "NRRDPreprocessor",
    "NFTYPreprocessor",
    # DataModules
    "CircuitDataModule",
    "SNEMI3DDataModule",
    "CREMI3DDataModule",
    "MICRONSDataModule",
    "MitoEM2DataModule",
    "CombineDataModule",
    # Union label mapping
    "UNION_LABEL_MAP",
    "UNION_LABEL_NAMES",
    "NUM_UNION_CLASSES",
]

"""
Neurons: a PyTorch Lightning infrastructure for connectomics research.

Provides:
- MONAI-compatible dataset classes with a standardised interface
- Preprocessors for common data formats (TIFF, HDF5, NRRD, NIfTI)
- Pre-built models (Vista 2D/3D, Cosmos-Predict 3D, Cosmos-Transfer 3D)
  with a shared three-head architecture
- Connectomics-specific loss functions, metrics and clusterers
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
    NeuriteDataModule,
)

__all__ = [
    "CircuitDataset",
    "SNEMI3DDataset",
    "CREMI3DDataset",
    "MICRONSDataset",
    "MitoEM2Dataset",
    "BasePreprocessor",
    "TIFFPreprocessor",
    "HDF5Preprocessor",
    "NRRDPreprocessor",
    "NFTYPreprocessor",
    "CircuitDataModule",
    "SNEMI3DDataModule",
    "CREMI3DDataModule",
    "MICRONSDataModule",
    "MitoEM2DataModule",
    "NeuriteDataModule",
]

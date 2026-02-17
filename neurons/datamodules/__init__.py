"""
PyTorch Lightning DataModules for connectomics datasets.

DataModules handle train/val/test splitting, data loading configuration,
and transform pipelines.
"""

from neurons.datamodules.base import CircuitDataModule
from neurons.datamodules.snemi3d import SNEMI3DDataModule
from neurons.datamodules.cremi3d import CREMI3DDataModule
from neurons.datamodules.microns import MICRONSDataModule
from neurons.datamodules.mitoem2 import MitoEM2DataModule
from neurons.datamodules.combine import (
    CombineDataModule,
    UNION_LABEL_MAP,
    UNION_LABEL_NAMES,
    NUM_UNION_CLASSES,
)

__all__ = [
    "CircuitDataModule",
    "SNEMI3DDataModule",
    "CREMI3DDataModule",
    "MICRONSDataModule",
    "MitoEM2DataModule",
    "CombineDataModule",
    "UNION_LABEL_MAP",
    "UNION_LABEL_NAMES",
    "NUM_UNION_CLASSES",
]

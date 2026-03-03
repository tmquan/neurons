"""
Base DataModule for connectomics datasets.
"""

from abc import ABC
from typing import Dict, List, Optional, Type

import torch
import pytorch_lightning as pl
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    Resized,
    ScaleIntensityd,
    ToTensord,
)

from neurons.datasets.base import CircuitDataset


class CircuitDataModule(pl.LightningDataModule, ABC):
    """
    Base PyTorch Lightning DataModule for connectomics datasets.

    Args:
        data_root: Path to the data directory.
        batch_size: Batch size for training and validation.
        num_workers: Number of DataLoader worker processes per rank.
        cache_rate: Fraction of data to cache in memory (default: 0.0).
        pin_memory: Whether to pin memory for faster GPU transfer.
        image_size: Optional image size for resizing.
        train_volumes: Volume list for training (dataset-specific format).
        val_volumes: Volume list for validation (defaults to train_volumes).
        test_volumes: Volume list for testing (defaults to train_volumes).
    """

    dataset_class: Type[CircuitDataset] = CircuitDataset  # type: ignore[type-abstract]

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 0,
        val_num_workers: Optional[int] = None,
        test_num_workers: Optional[int] = None,
        cache_rate: float = 0.0,
        pin_memory: bool = True,
        image_size: Optional[tuple] = None,
        train_volumes: Optional[List[Dict[str, str]]] = None,
        val_volumes: Optional[List[Dict[str, str]]] = None,
        test_volumes: Optional[List[Dict[str, str]]] = None,
        cache_num_workers: Optional[int] = None,
        prefetch_factor: int = 4,
        persistent_workers: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_num_workers = num_workers if val_num_workers is None else val_num_workers
        self.test_num_workers = num_workers if test_num_workers is None else test_num_workers
        self.cache_rate = cache_rate
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.train_volumes = train_volumes
        self.val_volumes = val_volumes if val_volumes is not None else train_volumes
        self.test_volumes = test_volumes if test_volumes is not None else train_volumes
        self.cache_num_workers = num_workers if cache_num_workers is None else cache_num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = (self.num_workers > 0) if persistent_workers is None else persistent_workers

        self.train_dataset: Optional[CircuitDataset] = None
        self.val_dataset: Optional[CircuitDataset] = None
        self.test_dataset: Optional[CircuitDataset] = None

    def _get_dataset_kwargs(self) -> dict:
        """Override in subclasses to provide dataset-specific arguments."""
        return {}

    def setup(self, stage: Optional[str] = None) -> None:
        extra = self._get_dataset_kwargs()

        if stage in ("fit", None):
            self.train_dataset = self.dataset_class(
                root_dir=self.data_root,
                volumes=self.train_volumes,
                cache_rate=self.cache_rate,
                num_workers=self.cache_num_workers,
                transform=self.get_train_transforms(),
                **extra,
            )
            self.val_dataset = self.dataset_class(
                root_dir=self.data_root,
                volumes=self.val_volumes,
                cache_rate=0.0,
                num_workers=self.cache_num_workers,
                transform=self.get_val_transforms(),
                **extra,
            )

        if stage in ("test", None):
            self.test_dataset = self.dataset_class(
                root_dir=self.data_root,
                volumes=self.test_volumes,
                cache_rate=0.0,
                num_workers=self.cache_num_workers,
                transform=self.get_val_transforms(),
                **extra,
            )

    # ------------------------------------------------------------------
    # Transforms (override in subclasses for dataset-specific pipelines)
    # ------------------------------------------------------------------

    def get_train_transforms(self) -> Compose:
        transforms = [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ]

        if self.image_size is not None:
            transforms.append(
                Resized(keys=["image", "label"], spatial_size=self.image_size, mode=["bilinear", "nearest"])
            )

        transforms.extend([
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.8, 1.2)),
            ToTensord(keys=["image", "label"]),
        ])

        return Compose(transforms)

    def get_val_transforms(self) -> Compose:
        transforms = [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ]

        if self.image_size is not None:
            transforms.append(
                Resized(keys=["image", "label"], spatial_size=self.image_size, mode=["bilinear", "nearest"])
            )

        transforms.append(ToTensord(keys=["image", "label"]))
        return Compose(transforms)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        prefetch = self.prefetch_factor if self.num_workers > 0 else None
        return torch.utils.data.DataLoader(
            self.train_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=prefetch,
            drop_last=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        prefetch = self.prefetch_factor if self.val_num_workers > 0 else None
        return torch.utils.data.DataLoader(
            self.val_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.val_num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.val_num_workers > 0,
            prefetch_factor=prefetch,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        prefetch = self.prefetch_factor if self.test_num_workers > 0 else None
        return torch.utils.data.DataLoader(
            self.test_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.test_num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.test_num_workers > 0,
            prefetch_factor=prefetch,
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return self.test_dataloader()

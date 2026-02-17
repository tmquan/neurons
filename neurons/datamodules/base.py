"""
Base DataModule for connectomics datasets.
"""

from abc import ABC
from typing import Optional, Type

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

    Provides common functionality for:
    - Data loading and splitting
    - Transform application
    - DataLoader creation with distributed training support

    Subclasses should set ``dataset_class`` to the appropriate
 subclass.

    Args:
        data_root: Path to the data directory.
        batch_size: Batch size for training and validation.
        num_workers: Number of worker processes for data loading.
        train_val_split: Fraction for validation (default: 0.2).
        cache_rate: Fraction of data to cache in memory (default: 0.5).
        pin_memory: Whether to pin memory for faster GPU transfer.
        image_size: Optional image size for resizing (H, W) or (D, H, W).
        persistent_workers: Keep workers alive between epochs.
    """

    dataset_class: Type[CircuitDataset] = CircuitDataset  # type: ignore[type-abstract]

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        train_val_split: float = 0.2,
        cache_rate: float = 0.5,
        pin_memory: bool = True,
        image_size: Optional[tuple] = None,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.cache_rate = cache_rate
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.persistent_workers = persistent_workers and num_workers > 0

        self.train_dataset: Optional[CircuitDataset] = None
        self.val_dataset: Optional[CircuitDataset] = None
        self.test_dataset: Optional[CircuitDataset] = None

    def _get_dataset_kwargs(self) -> dict:
        """
        Get additional kwargs for dataset initialization.

        Override in subclasses to provide dataset-specific arguments.

        Returns:
            Dictionary of additional keyword arguments.
        """
        return {}

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets for each stage.

        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict').
        """
        extra_kwargs = self._get_dataset_kwargs()

        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_class(
                root_dir=self.data_root,
                split="train",
                train_val_split=self.train_val_split,
                cache_rate=self.cache_rate,
                transform=self.get_train_transforms(),
                **extra_kwargs,
            )

            self.val_dataset = self.dataset_class(
                root_dir=self.data_root,
                split="valid",
                train_val_split=self.train_val_split,
                cache_rate=1.0,
                transform=self.get_val_transforms(),
                **extra_kwargs,
            )

        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_class(
                root_dir=self.data_root,
                split="test",
                cache_rate=0.0,
                transform=self.get_val_transforms(),
                **extra_kwargs,
            )

    def get_train_transforms(self) -> Compose:
        """
        Get training transforms with augmentation.

        Override in subclasses for dataset-specific transforms.

        Returns:
            MONAI Compose transform pipeline.
        """
        transforms = [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ]

        if self.image_size is not None:
            transforms.append(
                Resized(
                    keys=["image", "label"],
                    spatial_size=self.image_size,
                    mode=["bilinear", "nearest"],
                )
            )

        transforms.extend(
            [
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
                RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
                RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.8, 1.2)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        return Compose(transforms)

    def get_val_transforms(self) -> Compose:
        """
        Get validation/test transforms without augmentation.

        Returns:
            MONAI Compose transform pipeline.
        """
        transforms = [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ]

        if self.image_size is not None:
            transforms.append(
                Resized(
                    keys=["image", "label"],
                    spatial_size=self.image_size,
                    mode=["bilinear", "nearest"],
                )
            )

        transforms.append(ToTensord(keys=["image", "label"]))

        return Compose(transforms)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Create training DataLoader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Create validation DataLoader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Create test DataLoader."""
        return torch.utils.data.DataLoader(
            self.test_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Create prediction DataLoader (same as test)."""
        return self.test_dataloader()

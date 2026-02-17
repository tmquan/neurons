"""
MitoEM2 DataModule for PyTorch Lightning.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from monai.transforms import (
    CastToTyped,
    Compose,
    EnsureChannelFirstd,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandSpatialCropd,
    Resized,
    ScaleIntensityd,
    SpatialPadd,
    ToTensord,
)

from neurons.datamodules.base import CircuitDataModule
from neurons.datasets.mitoem2 import MitoEM2Dataset


class MitoEM2DataModule(CircuitDataModule):
    """
    PyTorch Lightning DataModule for MitoEM2 dataset.

    Args:
        data_root: Path to MitoEM2 root directory.
        batch_size: Batch size (default: 4).
        num_workers: Data loading workers (default: 4).
        dataset_name: Specific dataset to load (e.g. 'Dataset001_ME2-Beta').
            None loads all datasets found in data_root.
        slice_mode: Return 2D slices if True (default: True).
        patch_size: Random crop size (H, W) or (D, H, W) for training.

    Example:
        >>> dm = MitoEM2DataModule(
        ...     data_root="data/mitoem2",
        ...     dataset_name="Dataset001_ME2-Beta",
        ...     batch_size=8,
        ... )
        >>> dm.setup("fit")
        >>> for batch in dm.train_dataloader():
        ...     images = batch["image"]  # [B, 1, H, W]
        ...     labels = batch["label"]  # [B, 1, H, W]  (0=bg, 1=mito, 2=boundary)
    """

    dataset_class = MitoEM2Dataset

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        train_val_split: float = 0.2,
        cache_rate: float = 0.5,
        pin_memory: bool = True,
        image_size: Optional[Tuple[int, ...]] = None,
        patch_size: Optional[Union[Tuple[int, ...], List[int]]] = None,
        dataset_name: Optional[str] = None,
        slice_mode: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        self.dataset_name = dataset_name
        self.slice_mode = slice_mode
        self.patch_size = tuple(patch_size) if patch_size is not None else None
        super().__init__(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            train_val_split=train_val_split,
            cache_rate=cache_rate,
            pin_memory=pin_memory,
            image_size=image_size,
            persistent_workers=persistent_workers,
        )

    def _get_dataset_kwargs(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "slice_mode": self.slice_mode,
        }

    def get_train_transforms(self) -> Compose:
        """Training transforms for MitoEM2 (3-class semantic segmentation)."""
        keys = ["image", "label"]
        transforms = [
            CastToTyped(keys=["image"], dtype=np.float32),
            CastToTyped(keys=["label"], dtype=np.int64),
            EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ]

        if self.patch_size is not None:
            transforms.extend([
                SpatialPadd(keys=keys, spatial_size=self.patch_size),
                RandSpatialCropd(keys=keys, roi_size=self.patch_size, random_size=False),
            ])
        elif self.image_size is not None:
            transforms.append(
                Resized(keys=keys, spatial_size=self.image_size, mode=["bilinear", "nearest"]),
            )

        transforms.extend([
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1)),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
            ToTensord(keys=keys),
        ])

        return Compose(transforms)

    def get_val_transforms(self) -> Compose:
        """Validation transforms for MitoEM2."""
        keys = ["image", "label"]
        transforms = [
            CastToTyped(keys=["image"], dtype=np.float32),
            CastToTyped(keys=["label"], dtype=np.int64),
            EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ]

        if self.patch_size is not None:
            transforms.extend([
                SpatialPadd(keys=keys, spatial_size=self.patch_size),
                RandSpatialCropd(keys=keys, roi_size=self.patch_size, random_size=False),
            ])
        elif self.image_size is not None:
            transforms.append(
                Resized(keys=keys, spatial_size=self.image_size, mode=["bilinear", "nearest"]),
            )

        transforms.append(ToTensord(keys=keys))

        return Compose(transforms)

"""
CREMI3D DataModule for PyTorch Lightning.
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

from neurons.datamodules import CircuitDataModule
from neurons.datasets import CREMI3DDataset


class CREMI3DDataModule(CircuitDataModule):
    """
    PyTorch Lightning DataModule for CREMI3D dataset.

    Wraps CREMI3DDataset with appropriate transforms for training
    neuron/synapse segmentation models.

    Args:
        data_root: Path to CREMI3D data directory.
        batch_size: Batch size (default: 4).
        num_workers: Data loading workers (default: 4).
        volumes: CREMI volumes to use (default: ["A", "B"]).
        include_clefts: Include cleft annotations (default: True).
        include_mito: Include mitochondria annotations (default: False).
        patch_size: Random crop size (D, H, W) for training.
    """

    dataset_class = CREMI3DDataset

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
        volumes: Optional[List[str]] = None,
        include_clefts: bool = True,
        include_mito: bool = False,
        num_samples: Optional[int] = None,
        persistent_workers: bool = True,
    ) -> None:
        self.volumes = volumes if volumes is not None else ["A", "B"]
        self.include_clefts = include_clefts
        self.include_mito = include_mito
        self.num_samples = num_samples
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
        kwargs = {
            "volumes": self.volumes,
            "include_clefts": self.include_clefts,
            "include_mito": self.include_mito,
        }
        if self.num_samples is not None:
            kwargs["num_samples"] = self.num_samples
        return kwargs

    def get_train_transforms(self) -> Compose:
        """Get training transforms optimized for CREMI3D."""
        keys = ["image", "label"]
        transforms = [
            CastToTyped(keys=["image"], dtype=np.float32),
            CastToTyped(keys=["label"], dtype=np.int64),
            EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ]

        if self.patch_size is not None:
            transforms.extend(
                [
                    SpatialPadd(keys=keys, spatial_size=self.patch_size),
                    RandSpatialCropd(keys=keys, roi_size=self.patch_size, random_size=False),
                ]
            )
        elif self.image_size is not None:
            transforms.append(
                Resized(
                    keys=keys,
                    spatial_size=self.image_size,
                    mode=["bilinear", "nearest"],
                )
            )

        transforms.extend(
            [
                RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                RandRotate90d(keys=keys, prob=0.5, spatial_axes=(1, 2)),
                RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
                RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
                ToTensord(keys=keys),
            ]
        )

        return Compose(transforms)

    def get_val_transforms(self) -> Compose:
        """Get validation transforms with consistent cropping."""
        keys = ["image", "label"]
        transforms = [
            CastToTyped(keys=["image"], dtype=np.float32),
            CastToTyped(keys=["label"], dtype=np.int64),
            EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ]

        if self.patch_size is not None:
            transforms.extend(
                [
                    SpatialPadd(keys=keys, spatial_size=self.patch_size),
                    RandSpatialCropd(keys=keys, roi_size=self.patch_size, random_size=False),
                ]
            )
        elif self.image_size is not None:
            transforms.append(
                Resized(
                    keys=keys,
                    spatial_size=self.image_size,
                    mode=["bilinear", "nearest"],
                )
            )

        transforms.append(ToTensord(keys=keys))

        return Compose(transforms)

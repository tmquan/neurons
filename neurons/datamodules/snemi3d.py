"""
SNEMI3D DataModule for PyTorch Lightning.
"""

from typing import List, Optional, Tuple, Union

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandSpatialCropd,
    Resized,
    SpatialPadd,
    ToTensord,
)

from neurons.datamodules import CircuitDataModule
from neurons.datasets import SNEMI3DDataset


class SNEMI3DDataModule(CircuitDataModule):
    """
    PyTorch Lightning DataModule for SNEMI3D dataset.

    Wraps SNEMI3DDataset with appropriate transforms and data loading
    configuration for training neuron segmentation models.

    Args:
        data_root: Path to SNEMI3D data directory.
        batch_size: Batch size (default: 4).
        num_workers: Data loading workers (default: 4).
        train_val_split: Validation fraction (default: 0.2).
        cache_rate: Cache fraction (default: 0.5).
        pin_memory: Pin memory for GPU transfer (default: True).
        image_size: Optional resize dimensions (D, H, W).
        patch_size: Random crop size (D, H, W) for training.
        slice_mode: Return 2D slices if True (default: False for 3D).
        num_samples: Number of samples per epoch.  Controls how many
            random crops/augmentations are drawn.  Defaults to the
            number of Z slices in the volume when ``None``.

    Example:
        >>> dm = SNEMI3DDataModule(
        ...     data_root="/path/to/snemi3d",
        ...     batch_size=8,
        ...     patch_size=(32, 128, 128),
        ...     num_samples=500,
        ... )
        >>> dm.setup("fit")
        >>> for batch in dm.train_dataloader():
        ...     images = batch["image"]  # [B, 1, D, H, W]
        ...     labels = batch["label"]  # [B, 1, D, H, W]
    """

    dataset_class = SNEMI3DDataset

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
        slice_mode: bool = False,
        num_samples: Optional[int] = None,
        persistent_workers: bool = True,
    ) -> None:
        self.slice_mode = slice_mode
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
        kwargs = {"slice_mode": self.slice_mode}
        if self.num_samples is not None:
            kwargs["num_samples"] = self.num_samples
        return kwargs

    def get_train_transforms(self) -> Compose:
        """
        Get training transforms optimized for SNEMI3D grayscale EM images.

        Returns:
            MONAI Compose transform pipeline with EM-specific augmentations.
        """
        keys = ["image", "label"]
        transforms = [
            EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
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

        rot_axes = (0, 1) if self.slice_mode else (1, 2)
        transforms.extend(
            [
                RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                RandRotate90d(keys=keys, prob=0.5, spatial_axes=rot_axes),
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
            EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
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

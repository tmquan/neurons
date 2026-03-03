"""
MitoEM2 DataModule for PyTorch Lightning.
"""

from typing import Dict, List, Optional, Tuple, Union

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandSpatialCropd,
    Resized,
    ToTensord,
)

from neurons.datamodules.base import CircuitDataModule
from neurons.datasets.mitoem2 import MitoEM2Dataset
from neurons.transforms import RelabelAfterCropd, RandFindBoundariesd


class MitoEM2DataModule(CircuitDataModule):
    """
    PyTorch Lightning DataModule for MitoEM2 dataset.

    Args:
        train_volumes: e.g. ``[{"subdataset": "Dataset001_ME2-Beta"}]``
        test_volumes: e.g. ``[{"subdataset": "Dataset001_ME2-Beta", "img_dir": "imagesTs", "lbl_dir": "labelsTs"}]``
        slice_mode: Return 2D slices if True (default: True).
        find_boundaries: Probability of zeroing boundary pixels (0.0=off).
    """

    dataset_class = MitoEM2Dataset

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        val_num_workers: Optional[int] = None,
        test_num_workers: Optional[int] = None,
        cache_rate: float = 0.5,
        cache_num_workers: Optional[int] = None,
        pin_memory: bool = True,
        image_size: Optional[Tuple[int, ...]] = None,
        patch_size: Optional[Union[Tuple[int, ...], List[int]]] = None,
        slice_mode: bool = True,
        num_samples: Optional[int] = None,
        find_boundaries: float = 0.0,
        relabel_after_crop: bool = True,
        train_volumes: Optional[List[Dict[str, str]]] = None,
        val_volumes: Optional[List[Dict[str, str]]] = None,
        test_volumes: Optional[List[Dict[str, str]]] = None,
        prefetch_factor: int = 4,
        persistent_workers: Optional[bool] = None,
    ) -> None:
        self.slice_mode = slice_mode
        self.num_samples = num_samples
        self.find_boundaries = find_boundaries
        self.relabel_after_crop = relabel_after_crop
        self.patch_size = tuple(patch_size) if patch_size is not None else None
        super().__init__(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            val_num_workers=val_num_workers,
            test_num_workers=test_num_workers,
            cache_rate=cache_rate,
            cache_num_workers=cache_num_workers,
            pin_memory=pin_memory,
            image_size=image_size,
            train_volumes=train_volumes,
            val_volumes=val_volumes,
            test_volumes=test_volumes,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    def _get_dataset_kwargs(self) -> dict:
        kwargs: dict = {"slice_mode": self.slice_mode}
        if self.num_samples is not None:
            kwargs["num_samples"] = self.num_samples
        return kwargs

    def _label_post_crop(self, spatial_dims: int) -> list:
        steps = []
        if self.relabel_after_crop:
            steps.append(RelabelAfterCropd(keys=["label"], spatial_dims=spatial_dims))
        if self.find_boundaries > 0:
            steps.append(RandFindBoundariesd(
                keys=["label"], prob=self.find_boundaries,
            ))
        return steps

    def get_train_transforms(self) -> Compose:
        keys = ["image", "label"]
        spatial_dims = 2 if self.slice_mode else 3
        transforms: list = []

        if self.patch_size is not None:
            transforms.extend([
                EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
                RandSpatialCropd(keys=keys, roi_size=self.patch_size, random_size=False),
                *self._label_post_crop(spatial_dims),
            ])
        else:
            transforms.append(EnsureChannelFirstd(keys=keys, channel_dim="no_channel"))
            if self.image_size is not None:
                transforms.append(Resized(keys=keys, spatial_size=self.image_size, mode=["bilinear", "nearest"]))

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
        keys = ["image", "label"]
        spatial_dims = 2 if self.slice_mode else 3
        transforms: list = []

        if self.patch_size is not None:
            transforms.extend([
                EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
                RandSpatialCropd(keys=keys, roi_size=self.patch_size, random_size=False),
                *self._label_post_crop(spatial_dims),
            ])
        else:
            transforms.append(EnsureChannelFirstd(keys=keys, channel_dim="no_channel"))
            if self.image_size is not None:
                transforms.append(Resized(keys=keys, spatial_size=self.image_size, mode=["bilinear", "nearest"]))

        transforms.append(ToTensord(keys=keys))
        return Compose(transforms)

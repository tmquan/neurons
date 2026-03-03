"""
CREMI3D DataModule for PyTorch Lightning.
"""

from typing import Dict, List, Optional, Tuple, Union

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    Resized,
    ToTensord,
)

from neurons.datamodules import CircuitDataModule
from neurons.datasets import CREMI3DDataset
from neurons.transforms import RelabelAfterCropd, RandFindBoundariesd


class CREMI3DDataModule(CircuitDataModule):
    """
    PyTorch Lightning DataModule for CREMI3D dataset.

    Args:
        train_volumes: e.g. ``[{"vol": "A"}, {"vol": "B"}, {"vol": "C"}]``
        test_volumes: e.g. ``[{"vol": "A+"}, {"vol": "B+"}, {"vol": "C+"}]``
        include_clefts: Include cleft annotations (default: True).
        include_mito: Include mitochondria annotations (default: False).
        find_boundaries: Probability of zeroing boundary pixels (0.0=off).
    """

    dataset_class = CREMI3DDataset

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
        include_clefts: bool = True,
        include_mito: bool = False,
        num_samples: Optional[int] = None,
        find_boundaries: float = 0.0,
        relabel_after_crop: bool = True,
        train_volumes: Optional[List[Dict[str, str]]] = None,
        val_volumes: Optional[List[Dict[str, str]]] = None,
        test_volumes: Optional[List[Dict[str, str]]] = None,
        prefetch_factor: int = 4,
        persistent_workers: Optional[bool] = None,
    ) -> None:
        self.include_clefts = include_clefts
        self.include_mito = include_mito
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
        kwargs: dict = {
            "include_clefts": self.include_clefts,
            "include_mito": self.include_mito,
        }
        if self.num_samples is not None:
            kwargs["num_samples"] = self.num_samples
        if self.patch_size is not None:
            kwargs["patch_size"] = self.patch_size
        return kwargs

    def _label_post_crop(self) -> list:
        steps = []
        if self.relabel_after_crop:
            steps.append(RelabelAfterCropd(keys=["label"], spatial_dims=3))
        if self.find_boundaries > 0:
            steps.append(RandFindBoundariesd(
                keys=["label"], prob=self.find_boundaries,
            ))
        return steps

    def get_train_transforms(self) -> Compose:
        keys = ["image", "label"]
        transforms: list = []

        if self.patch_size is not None:
            # CREMI3D always uses _fast_crop when patch_size is set (no slice_mode)
            # Skip EnsureChannelFirstd - _fast_crop already outputs (1, D, H, W)
            transforms.extend(self._label_post_crop())
        else:
            transforms.append(EnsureChannelFirstd(keys=keys, channel_dim="no_channel"))
            if self.image_size is not None:
                transforms.append(Resized(keys=keys, spatial_size=self.image_size, mode=["bilinear", "nearest"]))

        transforms.extend([
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandRotate90d(keys=keys, prob=0.5, spatial_axes=(1, 2)),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
            ToTensord(keys=keys),
        ])

        return Compose(transforms)

    def get_val_transforms(self) -> Compose:
        keys = ["image", "label"]
        transforms: list = []

        if self.patch_size is not None:
            # CREMI3D always uses _fast_crop when patch_size is set
            transforms.extend(self._label_post_crop())
        else:
            transforms.append(EnsureChannelFirstd(keys=keys, channel_dim="no_channel"))
            if self.image_size is not None:
                transforms.append(Resized(keys=keys, spatial_size=self.image_size, mode=["bilinear", "nearest"]))

        transforms.append(ToTensord(keys=keys))
        return Compose(transforms)

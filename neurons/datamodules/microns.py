"""
MICRONS DataModule for PyTorch Lightning.
"""

from typing import Optional, Tuple

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
from neurons.datasets import MICRONSDataset
from neurons.transforms import RelabelAfterCropd, RandErodeNeuronBoundariesd


class MICRONSDataModule(CircuitDataModule):
    """
    PyTorch Lightning DataModule for MICRONS dataset.

    Args:
        data_root: Path to MICRONS data directory.
        batch_size: Batch size (default: 4).
        volume_file: Base name of volume file (default: 'volume').
        segmentation_file: Base name of segmentation file (default: 'segmentation').
        include_synapses: Load synapse annotations (default: False).
        include_mitochondria: Load mitochondria labels (default: False).
        slice_mode: Return 2D slices if True (default: True).
        patch_size: 3D patch size (z, y, x) if not None.
        erode_boundaries: Probability of eroding neuron boundaries (0.0=off).
    """

    dataset_class = MICRONSDataset

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        train_val_split: float = 0.2,
        cache_rate: float = 0.5,
        pin_memory: bool = True,
        image_size: Optional[tuple] = None,
        volume_file: str = "volume",
        segmentation_file: str = "segmentation",
        include_synapses: bool = False,
        include_mitochondria: bool = False,
        slice_mode: bool = True,
        patch_size: Optional[Tuple[int, int, int]] = None,
        num_samples: Optional[int] = None,
        erode_boundaries: float = 0.0,
        persistent_workers: bool = True,
    ) -> None:
        self.volume_file = volume_file
        self.segmentation_file = segmentation_file
        self.include_synapses = include_synapses
        self.include_mitochondria = include_mitochondria
        self.slice_mode = slice_mode
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.erode_boundaries = erode_boundaries
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
            "volume_file": self.volume_file,
            "segmentation_file": self.segmentation_file,
            "include_synapses": self.include_synapses,
            "include_mitochondria": self.include_mitochondria,
            "slice_mode": self.slice_mode,
            "patch_size": self.patch_size,
        }
        if self.num_samples is not None:
            kwargs["num_samples"] = self.num_samples
        return kwargs

    def _label_post_crop(self, spatial_dims: int) -> list:
        steps = [RelabelAfterCropd(keys=["label"], spatial_dims=spatial_dims)]
        if self.erode_boundaries > 0:
            steps.append(RandErodeNeuronBoundariesd(
                keys=["label"], prob=self.erode_boundaries, spatial_dims=spatial_dims,
            ))
        return steps

    def get_train_transforms(self) -> Compose:
        keys = ["image", "label"]
        spatial_dims = 2 if self.slice_mode else 3
        transforms = [EnsureChannelFirstd(keys=keys, channel_dim="no_channel")]

        if self.patch_size is not None:
            transforms.extend([
                SpatialPadd(keys=keys, spatial_size=self.patch_size),
                RandSpatialCropd(keys=keys, roi_size=self.patch_size, random_size=False),
                *self._label_post_crop(spatial_dims),
            ])
        elif self.image_size is not None:
            transforms.append(Resized(keys=keys, spatial_size=self.image_size, mode=["bilinear", "nearest"]))

        rot_axes = (0, 1) if self.slice_mode else (1, 2)
        transforms.extend([
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandRotate90d(keys=keys, prob=0.5, spatial_axes=rot_axes),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
            ToTensord(keys=keys),
        ])

        return Compose(transforms)

    def get_val_transforms(self) -> Compose:
        keys = ["image", "label"]
        spatial_dims = 2 if self.slice_mode else 3
        transforms = [EnsureChannelFirstd(keys=keys, channel_dim="no_channel")]

        if self.patch_size is not None:
            transforms.extend([
                SpatialPadd(keys=keys, spatial_size=self.patch_size),
                RandSpatialCropd(keys=keys, roi_size=self.patch_size, random_size=False),
                *self._label_post_crop(spatial_dims),
            ])
        elif self.image_size is not None:
            transforms.append(Resized(keys=keys, spatial_size=self.image_size, mode=["bilinear", "nearest"]))

        transforms.append(ToTensord(keys=keys))
        return Compose(transforms)

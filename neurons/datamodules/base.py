"""
Base DataModule for connectomics datasets.
"""

from abc import ABC
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import pytorch_lightning as pl
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandSpatialCropd,
    Resized,
    SpatialPadd,
)

from neurons.datasets.base import CircuitDataset
from neurons.transforms import Labeld, Directiond, Covarianced


class CircuitDataModule(pl.LightningDataModule, ABC):
    """
    Base PyTorch Lightning DataModule for connectomics datasets.

    Subclasses set ``dataset_class``, override ``_get_dataset_kwargs``,
    and optionally override the three label-target hooks
    (``_instance_transforms``, ``_semantic_transforms``,
    ``_geometry_transforms``) or ``_get_spatial_dims`` for the
    appropriate dimensionality.

    Pipeline order (train)::

        EnsureChannelFirst → [Pad + Crop + instance_transforms]
        → spatial augmentations → geometry_transforms
        → semantic_transforms → intensity augmentations → EnsureType

    Args:
        data_root: Path to the data directory.
        batch_size: Batch size for training and validation.
        num_workers: Number of worker processes for data loading.
        cache_rate: Fraction of data to cache in memory (default: 0.5).
        pin_memory: Whether to pin memory for faster GPU transfer.
        image_size: Optional image size for resizing.
        patch_size: Spatial crop size (enables crop pipeline when set).
        train_volumes: Volume list for training (dataset-specific format).
        val_volumes: Volume list for validation (defaults to train_volumes).
        test_volumes: Volume list for testing (defaults to train_volumes).
        persistent_workers: Keep workers alive between epochs.
    """

    dataset_class: Type[CircuitDataset] = CircuitDataset  # type: ignore[type-abstract]

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        cache_rate: float = 0.5,
        pin_memory: bool = True,
        image_size: Optional[tuple] = None,
        patch_size: Optional[Union[Tuple[int, ...], List[int]]] = None,
        train_volumes: Optional[List[Dict[str, str]]] = None,
        val_volumes: Optional[List[Dict[str, str]]] = None,
        test_volumes: Optional[List[Dict[str, str]]] = None,
        persistent_workers: bool = True,
        compute_geometry: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.patch_size = tuple(patch_size) if patch_size is not None else None
        self.train_volumes = train_volumes
        self.val_volumes = val_volumes if val_volumes is not None else train_volumes
        self.test_volumes = test_volumes if test_volumes is not None else train_volumes
        self.persistent_workers = persistent_workers and num_workers > 0
        self.compute_geometry = compute_geometry

        self.train_dataset: Optional[CircuitDataset] = None
        self.val_dataset: Optional[CircuitDataset] = None
        self.test_dataset: Optional[CircuitDataset] = None

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _get_dataset_kwargs(self) -> dict:
        """Override in subclasses to provide dataset-specific arguments."""
        return {}

    def _get_spatial_dims(self) -> int:
        """Return the number of spatial dimensions for the current config.

        Override in subclasses that support ``slice_mode`` or other
        dimension-switching logic.  Default is 3 (volumetric).
        """
        return 3

    # ------------------------------------------------------------------
    # Label-target transform hooks  (override to customise)
    # ------------------------------------------------------------------

    def _original_transforms(self, spatial_dims: int) -> list:
        """Spatial augmentations applied to both image and label.

        Flips and rotations.  Override to customise augmentation strategy.
        """
        io_keys = ["image", "label"]
        rot_axes = (0, 1) if spatial_dims == 2 else (1, 2)
        return [
            RandFlipd(keys=io_keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=io_keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=io_keys, prob=0.5, spatial_axis=2 if spatial_dims == 3 else 1),
            RandRotate90d(keys=io_keys, prob=0.5, spatial_axes=rot_axes),
        ]

    def _semantic_transforms(self, spatial_dims: int) -> list:
        """Image intensity augmentations and semantic-level label transforms.

        Runs after spatial augmentations.  Override to add semantic
        targets (e.g. boundary maps, class maps).
        """
        return [
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.1, gamma=(0.7, 1.3)),
        ]

    def _instance_transforms(self, spatial_dims: int) -> list:
        """Post-crop connected-component relabeling.

        Splits instances that became disconnected after cropping and
        renumbers labels sequentially.  Runs immediately after crop.
        """
        return [Labeld(keys=["label"], spatial_dims=spatial_dims)]

    def _geometry_transforms(self, spatial_dims: int) -> list:
        """Direction and covariance targets for the geometry loss head.

        Runs after spatial augmentations so targets are consistent with
        the augmented label layout.  Skipped when ``compute_geometry=False``.
        """
        if not self.compute_geometry:
            return []
        return [
            Directiond(keys=["label"], spatial_dims=spatial_dims),
            Covarianced(keys=["label"], spatial_dims=spatial_dims),
        ]

    # ------------------------------------------------------------------
    # Pipeline assembly
    # ------------------------------------------------------------------

    def _output_keys(self) -> list:
        """All keys that must pass through ``EnsureTyped``."""
        keys = ["image", "label"]
        if self.compute_geometry:
            keys.extend(["label_direction", "label_covariance"])
        return keys

    def _train_output_keys(self) -> list:
        """Output keys for train pipeline (includes precomputed weights)."""
        return self._output_keys() + ["weight_edge", "weight_bone"]

    def get_train_transforms(self) -> Compose:
        io_keys = ["image", "label"]
        sd = self._get_spatial_dims()

        transforms: list = [
            EnsureChannelFirstd(keys=io_keys, channel_dim="no_channel"),
        ]

        if self.patch_size is not None:
            transforms.extend([
                SpatialPadd(keys=io_keys, spatial_size=self.patch_size),
                RandSpatialCropd(keys=io_keys, roi_size=self.patch_size, random_size=False),
            ])
        elif self.image_size is not None:
            transforms.append(
                Resized(keys=io_keys, spatial_size=self.image_size, mode=["bilinear", "nearest"]),
            )

        transforms.extend([
            *self._original_transforms(sd),
            *self._semantic_transforms(sd),
            *self._instance_transforms(sd),
            *self._geometry_transforms(sd),
            EnsureTyped(keys=self._output_keys()),
        ])

        return Compose(transforms)

    def get_val_transforms(self) -> Compose:
        io_keys = ["image", "label"]
        sd = self._get_spatial_dims()

        transforms: list = [
            EnsureChannelFirstd(keys=io_keys, channel_dim="no_channel"),
        ]

        if self.patch_size is not None:
            transforms.extend([
                SpatialPadd(keys=io_keys, spatial_size=self.patch_size),
                CenterSpatialCropd(keys=io_keys, roi_size=self.patch_size),
            ])
        elif self.image_size is not None:
            transforms.append(
                Resized(keys=io_keys, spatial_size=self.image_size, mode=["bilinear", "nearest"]),
            )

        transforms.extend([
            *self._semantic_transforms(sd),
            *self._instance_transforms(sd),
            *self._geometry_transforms(sd),
            EnsureTyped(keys=self._output_keys()),
        ])
        return Compose(transforms)

    # ------------------------------------------------------------------
    # Dataset / DataLoader wiring
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        extra = self._get_dataset_kwargs()

        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_class(
                root_dir=self.data_root,
                volumes=self.train_volumes,
                cache_rate=self.cache_rate,
                transform=self.get_train_transforms(),
                **extra,
            )
            self.val_dataset = self.dataset_class(
                root_dir=self.data_root,
                volumes=self.val_volumes,
                cache_rate=1.0,
                transform=self.get_val_transforms(),
                **extra,
            )

        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_class(
                root_dir=self.data_root,
                volumes=self.test_volumes,
                cache_rate=0.0,
                transform=self.get_val_transforms(),
                **extra,
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=4 if self.num_workers > 0 else None,
            multiprocessing_context="forkserver" if self.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=4 if self.num_workers > 0 else None,
            multiprocessing_context="forkserver" if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=4 if self.num_workers > 0 else None,
            multiprocessing_context="forkserver" if self.num_workers > 0 else None,
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return self.test_dataloader()

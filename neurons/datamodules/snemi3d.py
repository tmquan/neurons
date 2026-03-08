"""
SNEMI3D DataModule for PyTorch Lightning.

Uses :class:`LazyVolDataset` for 3D patch mode (reads patches from
disk on demand — constant memory regardless of volume count) and
the legacy :class:`SNEMI3DDataset` for 2D slice mode.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from neurons.datamodules.base import CircuitDataModule
from neurons.datasets import SNEMI3DDataset

logger = logging.getLogger(__name__)


class SNEMI3DDataModule(CircuitDataModule):
    """
    PyTorch Lightning DataModule for SNEMI3D dataset.

    In 3D patch mode (``slice_mode=False`` with ``patch_size`` set), uses
    :class:`LazyVolDataset` which reads only the requested patch from
    disk per sample — keeping system RAM usage constant regardless of
    how many volumes are configured.

    In 2D slice mode (``slice_mode=True``), falls back to the legacy
    :class:`SNEMI3DDataset` which loads volumes into memory.

    Args:
        data_root: Path to SNEMI3D data directory.
        train_volumes: e.g. ``[{"vol": "AC4_inputs", "seg": "AC4_labels"}]``
        val_volumes: defaults to train_volumes.
        test_volumes: e.g. ``[{"vol": "AC3_inputs", "seg": "AC3_labels"}]``
        slice_mode: Return 2D slices if True (default: False for 3D).
        num_samples: Number of samples per epoch.
    """

    dataset_class = SNEMI3DDataset

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        cache_rate: float = 0.5,
        pin_memory: bool = True,
        image_size: Optional[Tuple[int, ...]] = None,
        patch_size: Optional[Union[Tuple[int, ...], List[int]]] = None,
        slice_mode: bool = False,
        num_samples: Optional[int] = None,
        train_volumes: Optional[List[Dict[str, str]]] = None,
        val_volumes: Optional[List[Dict[str, str]]] = None,
        test_volumes: Optional[List[Dict[str, str]]] = None,
        persistent_workers: bool = True,
    ) -> None:
        self.slice_mode = slice_mode
        self.num_samples = num_samples
        super().__init__(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            cache_rate=cache_rate,
            pin_memory=pin_memory,
            image_size=image_size,
            patch_size=patch_size,
            train_volumes=train_volumes,
            val_volumes=val_volumes,
            test_volumes=test_volumes,
            persistent_workers=persistent_workers,
        )

    @property
    def _use_lazy(self) -> bool:
        return not self.slice_mode and self.patch_size is not None

    def _get_dataset_kwargs(self) -> dict:
        kwargs: dict = {"slice_mode": self.slice_mode}
        if self.num_samples is not None:
            kwargs["num_samples"] = self.num_samples
        return kwargs

    def _get_spatial_dims(self) -> int:
        return 2 if self.slice_mode else 3

    def setup(self, stage: Optional[str] = None) -> None:
        if not self._use_lazy:
            return super().setup(stage)

        from neurons.datasets.lazy import LazyVolDataset

        num_samples = self.num_samples or 16000
        patch_size = self.patch_size

        if stage == "fit" or stage is None:
            train_vols = self.train_volumes or [{"vol": "AC4_inputs", "seg": "AC4_labels"}]
            self.train_dataset = LazyVolDataset(
                root_dir=self.data_root,
                volumes=train_vols,
                patch_size=patch_size,
                transform=self.get_train_transforms(),
                num_samples=num_samples,
            )
            val_vols = self.val_volumes or train_vols
            self.val_dataset = LazyVolDataset(
                root_dir=self.data_root,
                volumes=val_vols,
                patch_size=patch_size,
                transform=self.get_val_transforms(),
                num_samples=min(num_samples // 10, 500),
            )

        if stage == "test" or stage is None:
            test_vols = self.test_volumes or self.train_volumes or [{"vol": "AC4_inputs", "seg": "AC4_labels"}]
            self.test_dataset = LazyVolDataset(
                root_dir=self.data_root,
                volumes=test_vols,
                patch_size=patch_size,
                transform=self.get_val_transforms(),
                num_samples=min(num_samples // 10, 500),
            )

        logger.info(
            "SNEMI3DDataModule: using LazyVolumeDataset (on-demand disk reads, "
            "~0 MB base RAM per rank)"
        )

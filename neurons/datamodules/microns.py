"""
MICRONS DataModule for PyTorch Lightning.

Uses :class:`LazyVolDataset` for 3D patch mode and the legacy
:class:`MICRONSDataset` for 2D slice mode.
"""

import logging
from typing import Dict, List, Optional, Tuple

from neurons.datamodules.base import CircuitDataModule
from neurons.datasets import MICRONSDataset

logger = logging.getLogger(__name__)


class MICRONSDataModule(CircuitDataModule):
    """
    PyTorch Lightning DataModule for MICRONS dataset.

    In 3D patch mode (``slice_mode=False`` with ``patch_size``), uses
    :class:`LazyVolDataset` for on-demand disk reads.

    Args:
        train_volumes: e.g. ``[{"vol": "train_volume", "seg": "train_seg"}]``
        test_volumes: e.g. ``[{"vol": "test_volume", "seg": "test_seg"}]``
        slice_mode: Return 2D slices if True (default: True).
    """

    dataset_class = MICRONSDataset

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        cache_rate: float = 0.5,
        pin_memory: bool = True,
        image_size: Optional[tuple] = None,
        slice_mode: bool = True,
        patch_size: Optional[Tuple[int, int, int]] = None,
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
        kwargs: dict = {
            "slice_mode": self.slice_mode,
            "patch_size": self.patch_size,
        }
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
            train_vols = self.train_volumes or []
            if train_vols:
                self.train_dataset = LazyVolDataset(
                    root_dir=self.data_root,
                    volumes=train_vols,
                    patch_size=patch_size,
                    transform=self.get_train_transforms(),
                    num_samples=num_samples,
                )
            val_vols = self.val_volumes or train_vols
            if val_vols:
                self.val_dataset = LazyVolDataset(
                    root_dir=self.data_root,
                    volumes=val_vols,
                    patch_size=patch_size,
                    transform=self.get_val_transforms(),
                    num_samples=min(num_samples // 10, 500),
                )

        if stage == "test" or stage is None:
            test_vols = self.test_volumes or self.train_volumes or []
            if test_vols:
                self.test_dataset = LazyVolDataset(
                    root_dir=self.data_root,
                    volumes=test_vols,
                    patch_size=patch_size,
                    transform=self.get_val_transforms(),
                    num_samples=min(num_samples // 10, 500),
                )

        logger.info("MICRONSDataModule: using LazyVolDataset (~0 MB base RAM per rank)")

"""
CREMI3D DataModule for PyTorch Lightning.

Uses :class:`LazyVolDataset` for 3D patch mode and the legacy
:class:`CREMI3DDataset` for full-volume access.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

from neurons.datamodules.base import CircuitDataModule
from neurons.datasets import CREMI3DDataset

logger = logging.getLogger(__name__)


class CREMI3DDataModule(CircuitDataModule):
    """
    PyTorch Lightning DataModule for CREMI3D dataset.

    In 3D patch mode (``patch_size`` set), uses :class:`LazyVolDataset`
    for on-demand disk reads with constant RAM usage.

    Args:
        train_volumes: e.g. ``[{"vol": "A"}, {"vol": "B"}, {"vol": "C"}]``
        test_volumes: e.g. ``[{"vol": "A+"}, {"vol": "B+"}, {"vol": "C+"}]``
        include_clefts: Include cleft annotations (default: True).
        include_mito: Include mitochondria annotations (default: False).
    """

    dataset_class = CREMI3DDataset

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        cache_rate: float = 0.5,
        pin_memory: bool = True,
        image_size: Optional[Tuple[int, ...]] = None,
        patch_size: Optional[Union[Tuple[int, ...], List[int]]] = None,
        include_clefts: bool = True,
        include_mito: bool = False,
        num_samples: Optional[int] = None,
        train_volumes: Optional[List[Dict[str, str]]] = None,
        val_volumes: Optional[List[Dict[str, str]]] = None,
        test_volumes: Optional[List[Dict[str, str]]] = None,
        persistent_workers: bool = True,
        overcrop_factor: float = 1.0,
    ) -> None:
        self.include_clefts = include_clefts
        self.include_mito = include_mito
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
            overcrop_factor=overcrop_factor,
        )

    @property
    def _use_lazy(self) -> bool:
        return self.patch_size is not None

    def _get_dataset_kwargs(self) -> dict:
        kwargs: dict = {
            "include_clefts": self.include_clefts,
            "include_mito": self.include_mito,
        }
        if self.num_samples is not None:
            kwargs["num_samples"] = self.num_samples
        return kwargs

    def _get_spatial_dims(self) -> int:
        return 3

    def setup(self, stage: Optional[str] = None) -> None:
        if not self._use_lazy:
            return super().setup(stage)

        from neurons.datasets.lazy import LazyVolDataset

        num_samples = self.num_samples or 16000
        train_patch = self.overcrop_size or self.patch_size

        if stage == "fit" or stage is None:
            train_vols = self.train_volumes or [{"vol": "A"}, {"vol": "B"}, {"vol": "C"}]
            self.train_dataset = LazyVolDataset(
                root_dir=self.data_root,
                volumes=train_vols,
                patch_size=train_patch,
                transform=self.get_train_transforms(),
                num_samples=num_samples,
            )
            val_vols = self.val_volumes or train_vols
            self.val_dataset = LazyVolDataset(
                root_dir=self.data_root,
                volumes=val_vols,
                patch_size=self.patch_size,
                transform=self.get_val_transforms(),
                num_samples=num_samples,
            )

        if stage == "test" or stage is None:
            test_vols = self.test_volumes or self.train_volumes or []
            if test_vols:
                self.test_dataset = LazyVolDataset(
                    root_dir=self.data_root,
                    volumes=test_vols,
                    patch_size=self.patch_size,
                    transform=self.get_val_transforms(),
                    num_samples=num_samples,
                )

        logger.info("CREMI3DDataModule: using LazyVolDataset (~0 MB base RAM per rank)")

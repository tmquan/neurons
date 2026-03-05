"""
CREMI3D DataModule for PyTorch Lightning.
"""

from typing import Dict, List, Optional, Tuple, Union

from neurons.datamodules import CircuitDataModule
from neurons.datasets import CREMI3DDataset


class CREMI3DDataModule(CircuitDataModule):
    """
    PyTorch Lightning DataModule for CREMI3D dataset.

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
        )

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

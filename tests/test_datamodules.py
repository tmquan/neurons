"""
Tests for DataModule classes and helper wrappers.

Covers:
- CircuitDataModule (base): hyperparameters, transforms, dataloaders
- SNEMI3DDataModule: dataset_class binding, kwargs forwarding
- CREMI3DDataModule: dataset_class binding, kwargs forwarding
- MICRONSDataModule: dataset_class binding, kwargs forwarding
- MitoEM2DataModule: dataset_class binding, kwargs forwarding
"""

from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from neurons.datasets.base import CircuitDataset
from neurons.datamodules.base import CircuitDataModule
from neurons.datamodules.snemi3d import SNEMI3DDataModule
from neurons.datamodules.cremi3d import CREMI3DDataModule
from neurons.datamodules.microns import MICRONSDataModule


# ---------------------------------------------------------------------------
# Helpers: minimal concrete implementations for testing
# ---------------------------------------------------------------------------

class _DummyDataset(CircuitDataset):
    """Minimal concrete dataset that yields synthetic samples."""

    def __init__(
        self,
        root_dir: str = ".",
        volumes: Any = None,
        transform: Any = None,
        cache_rate: float = 0.0,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        self.volumes = volumes
        self._transform = transform
        self._data = [
            {"image": np.random.rand(32, 32).astype(np.float32),
             "label": np.random.randint(0, 5, (32, 32)).astype(np.int64)}
            for _ in range(8)
        ]

    # --- abstract property stubs ---
    @property
    def paper(self) -> str:
        return "Dummy"

    @property
    def resolution(self) -> Dict[str, float]:
        return {"x": 1.0, "y": 1.0, "z": 1.0}

    @property
    def labels(self) -> List[str]:
        return ["bg", "fg"]

    @property
    def data_files(self) -> Dict[str, Union[str, np.ndarray]]:
        return {"vol": "v.h5", "seg": "s.h5"}

    def _prepare_data(self) -> List[Dict[str, Any]]:
        return self._data

    # --- Dataset protocol ---
    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._data[idx]
        if self._transform is not None:
            sample = self._transform(sample)
        return sample


class _DummyDataModule(CircuitDataModule):
    """Concrete datamodule wired to _DummyDataset."""

    dataset_class = _DummyDataset  # type: ignore[assignment]

    def _get_spatial_dims(self) -> int:
        return 2


# ---------------------------------------------------------------------------
# Tests: CircuitDataModule (base)
# ---------------------------------------------------------------------------

class TestCircuitDataModule:
    """Tests for the base CircuitDataModule."""

    def test_hyperparameters_stored(self) -> None:
        dm = _DummyDataModule(data_root="/tmp", batch_size=8, num_workers=2)
        assert dm.batch_size == 8
        assert dm.num_workers == 2
        assert dm.data_root == "/tmp"

    def test_persistent_workers_disabled_when_zero(self) -> None:
        dm = _DummyDataModule(data_root=".", num_workers=0, persistent_workers=True)
        assert dm.persistent_workers is False

    def test_setup_creates_datasets(self) -> None:
        dm = _DummyDataModule(data_root=".", batch_size=2, num_workers=0)
        dm.setup("fit")
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

    def test_train_dataloader_returns_loader(self) -> None:
        dm = _DummyDataModule(data_root=".", batch_size=2, num_workers=0)
        dm.setup("fit")
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert "image" in batch
        assert "label" in batch
        assert batch["image"].shape[0] == 2

    def test_val_dataloader_returns_loader(self) -> None:
        dm = _DummyDataModule(data_root=".", batch_size=2, num_workers=0)
        dm.setup("fit")
        loader = dm.val_dataloader()
        batch = next(iter(loader))
        assert "image" in batch

    def test_get_train_transforms_returns_compose(self) -> None:
        from monai.transforms import Compose

        dm = _DummyDataModule(data_root=".")
        t = dm.get_train_transforms()
        assert isinstance(t, Compose)

    def test_get_val_transforms_returns_compose(self) -> None:
        from monai.transforms import Compose

        dm = _DummyDataModule(data_root=".")
        t = dm.get_val_transforms()
        assert isinstance(t, Compose)


# ---------------------------------------------------------------------------
# Tests: concrete datamodule class bindings
# ---------------------------------------------------------------------------

class TestSNEMI3DDataModule:
    """Tests for SNEMI3DDataModule."""

    def test_dataset_class_set(self) -> None:
        from neurons.datasets import SNEMI3DDataset
        assert SNEMI3DDataModule.dataset_class is SNEMI3DDataset

    def test_kwargs_forwarded(self) -> None:
        dm = SNEMI3DDataModule(data_root=".", slice_mode=True, num_workers=0)
        assert dm._get_dataset_kwargs() == {"slice_mode": True}

    def test_patch_size_stored(self) -> None:
        dm = SNEMI3DDataModule(data_root=".", patch_size=[32, 128, 128])
        assert dm.patch_size == (32, 128, 128)


class TestCREMI3DDataModule:
    """Tests for CREMI3DDataModule."""

    def test_dataset_class_set(self) -> None:
        from neurons.datasets import CREMI3DDataset
        assert CREMI3DDataModule.dataset_class is CREMI3DDataset

    def test_kwargs_forwarded(self) -> None:
        dm = CREMI3DDataModule(
            data_root=".", include_clefts=False, include_mito=True,
            train_volumes=[{"vol": "A"}],
        )
        kw = dm._get_dataset_kwargs()
        assert kw["include_clefts"] is False
        assert kw["include_mito"] is True


class TestMICRONSDataModule:
    """Tests for MICRONSDataModule."""

    def test_dataset_class_set(self) -> None:
        from neurons.datasets import MICRONSDataset
        assert MICRONSDataModule.dataset_class is MICRONSDataset

    def test_kwargs_forwarded(self) -> None:
        dm = MICRONSDataModule(
            data_root=".",
            slice_mode=False,
            patch_size=(16, 64, 64),
            train_volumes=[{"vol": "train_volume", "seg": "train_seg"}],
        )
        kw = dm._get_dataset_kwargs()
        assert kw["slice_mode"] is False
        assert kw["patch_size"] == (16, 64, 64)


# ---------------------------------------------------------------------------
# Tests: MitoEM2DataModule
# ---------------------------------------------------------------------------

class TestMitoEM2DataModule:
    """Tests for MitoEM2DataModule."""

    def test_dataset_class_set(self) -> None:
        from neurons.datasets.mitoem2 import MitoEM2Dataset
        from neurons.datamodules.mitoem2 import MitoEM2DataModule

        assert MitoEM2DataModule.dataset_class is MitoEM2Dataset

    def test_kwargs_forwarded(self) -> None:
        from neurons.datamodules.mitoem2 import MitoEM2DataModule

        dm = MitoEM2DataModule(
            data_root=".",
            slice_mode=False,
            train_volumes=[{"subdataset": "Dataset001_ME2-Beta"}],
        )
        kw = dm._get_dataset_kwargs()
        assert kw["slice_mode"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

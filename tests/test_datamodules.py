"""
Tests for DataModule classes and helper wrappers.

Covers:
- CircuitDataModule (base): hyperparameters, transforms, dataloaders
- SNEMI3DDataModule: dataset_class binding, kwargs forwarding
- CREMI3DDataModule: dataset_class binding, kwargs forwarding
- MICRONSDataModule: dataset_class binding, kwargs forwarding
- ExpandedDataset: virtual expansion
- CreateClassIds: class-id mapping for snemi3d / cremi3d / generic
- CombineDataModule: construction and setup wiring
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
from neurons.datamodules.combine import (
    CombineDataModule,
    CreateClassIds,
    ExpandedDataset,
)


# ---------------------------------------------------------------------------
# Helpers: minimal concrete implementations for testing
# ---------------------------------------------------------------------------

class _DummyDataset(CircuitDataset):
    """Minimal concrete dataset that yields synthetic samples."""

    def __init__(
        self,
        root_dir: str = ".",
        split: str = "train",
        transform: Any = None,
        cache_rate: float = 0.0,
        train_val_split: float = 0.2,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        # bypass parent __init__ which checks for root_dir existence
        self._split = split
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
            data_root=".", volumes=["A"], include_clefts=False, include_mito=True,
        )
        kw = dm._get_dataset_kwargs()
        assert kw["volumes"] == ["A"]
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
            volume_file="vol",
            segmentation_file="seg",
            include_synapses=True,
            slice_mode=False,
            patch_size=(16, 64, 64),
        )
        kw = dm._get_dataset_kwargs()
        assert kw["volume_file"] == "vol"
        assert kw["segmentation_file"] == "seg"
        assert kw["include_synapses"] is True
        assert kw["slice_mode"] is False
        assert kw["patch_size"] == (16, 64, 64)


# ---------------------------------------------------------------------------
# Tests: ExpandedDataset
# ---------------------------------------------------------------------------

class _SimpleDataset(Dataset):
    """Trivial list-backed dataset."""

    def __init__(self, n: int = 5) -> None:
        self.data = list(range(n))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> int:
        return self.data[idx]


class TestExpandedDataset:
    """Tests for ExpandedDataset wrapper."""

    def test_length_multiplied(self) -> None:
        base = _SimpleDataset(n=3)
        expanded = ExpandedDataset(base, expansion_factor=10)
        assert len(expanded) == 30

    def test_items_cycle(self) -> None:
        base = _SimpleDataset(n=3)
        expanded = ExpandedDataset(base, expansion_factor=4)
        assert expanded[0] == 0
        assert expanded[3] == 0  # wraps around
        assert expanded[4] == 1

    def test_expansion_factor_one(self) -> None:
        base = _SimpleDataset(n=5)
        expanded = ExpandedDataset(base, expansion_factor=1)
        assert len(expanded) == 5
        assert expanded[4] == 4


# ---------------------------------------------------------------------------
# Tests: CreateClassIds
# ---------------------------------------------------------------------------

class _LabelDataset(Dataset):
    """Dataset that returns dict samples with 'label'."""

    def __init__(self, labels: List[np.ndarray]) -> None:
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"image": np.zeros((4, 4)), "label": self.labels[idx]}


class TestCreateClassIds:
    """Tests for CreateClassIds wrapper."""

    def test_snemi3d_foreground_mapped_to_1(self) -> None:
        lbl = np.array([[0, 0, 5], [10, 0, 3]], dtype=np.int64)
        ds = _LabelDataset([lbl])
        wrapped = CreateClassIds(ds, dataset_type="snemi3d")
        sample = wrapped[0]

        assert "class_ids" in sample
        cids = sample["class_ids"]
        assert cids[0, 0] == 0  # background stays 0
        assert cids[0, 2] == 1  # foreground -> 1
        assert cids[1, 0] == 1

    def test_cremi3d_neuron_cleft_mito(self) -> None:
        lbl = np.array([0, 500, 1_000_001, 2_000_001], dtype=np.int64)
        ds = _LabelDataset([lbl])
        wrapped = CreateClassIds(ds, dataset_type="cremi3d")
        cids = wrapped[0]["class_ids"]

        assert cids[0] == 0  # background
        assert cids[1] == 1  # neuron (< 1M)
        assert cids[2] == 2  # cleft  (>= 1M, < 2M)
        assert cids[3] == 3  # mito   (>= 2M)

    def test_generic_uses_default_class(self) -> None:
        lbl = np.array([0, 1, 2], dtype=np.int64)
        ds = _LabelDataset([lbl])
        wrapped = CreateClassIds(ds, dataset_type="custom", default_class=7)
        cids = wrapped[0]["class_ids"]

        assert cids[0] == 0
        assert cids[1] == 7
        assert cids[2] == 7

    def test_tensor_input_returns_tensor(self) -> None:
        lbl = torch.tensor([0, 5, 10], dtype=torch.long)
        ds = _LabelDataset([lbl])
        wrapped = CreateClassIds(ds, dataset_type="snemi3d")
        cids = wrapped[0]["class_ids"]

        assert isinstance(cids, torch.Tensor)

    def test_mitoem2_mito_and_boundary(self) -> None:
        lbl = np.array([0, 1, 2, 1, 0], dtype=np.int64)
        ds = _LabelDataset([lbl])
        wrapped = CreateClassIds(ds, dataset_type="mitoem2")
        cids = wrapped[0]["class_ids"]

        assert cids[0] == 0  # background
        assert cids[1] == 3  # mitochondria (union label 3)
        assert cids[2] == 4  # mito_boundary (union label 4)
        assert cids[3] == 3  # mitochondria
        assert cids[4] == 0  # background

    def test_ignore_classes_drops_to_background(self) -> None:
        lbl = np.array([0, 1, 2, 1, 0], dtype=np.int64)
        ds = _LabelDataset([lbl])
        wrapped = CreateClassIds(
            ds, dataset_type="mitoem2", ignore_classes={"mito_boundary"},
        )
        cids = wrapped[0]["class_ids"]

        assert cids[0] == 0  # background
        assert cids[1] == 3  # mitochondria kept
        assert cids[2] == 0  # mito_boundary -> background (ignored)
        assert cids[3] == 3  # mitochondria kept
        assert cids[4] == 0  # background

    def test_ignore_multiple_classes(self) -> None:
        lbl = np.array([0, 500, 1_000_001, 2_000_001], dtype=np.int64)
        ds = _LabelDataset([lbl])
        wrapped = CreateClassIds(
            ds, dataset_type="cremi3d", ignore_classes={"cleft", "mitochondria"},
        )
        cids = wrapped[0]["class_ids"]

        assert cids[0] == 0  # background
        assert cids[1] == 1  # neuron kept
        assert cids[2] == 0  # cleft -> background (ignored)
        assert cids[3] == 0  # mito -> background (ignored)

    def test_microns_foreground_mapped_to_neuron(self) -> None:
        lbl = np.array([0, 42, 0, 99], dtype=np.int64)
        ds = _LabelDataset([lbl])
        wrapped = CreateClassIds(ds, dataset_type="microns")
        cids = wrapped[0]["class_ids"]

        assert cids[0] == 0  # background
        assert cids[1] == 1  # neuron
        assert cids[2] == 0  # background
        assert cids[3] == 1  # neuron

    def test_dataset_type_added(self) -> None:
        ds = _LabelDataset([np.zeros(4, dtype=np.int64)])
        wrapped = CreateClassIds(ds, dataset_type="cremi3d")
        assert wrapped[0]["dataset_type"] == "cremi3d"


# ---------------------------------------------------------------------------
# Tests: CombineDataModule + union label map
# ---------------------------------------------------------------------------

class TestCombineDataModule:
    """Tests for CombineDataModule."""

    def test_batch_size_from_arg(self) -> None:
        dm = CombineDataModule(batch_size=16)
        assert dm.batch_size == 16

    def test_batch_size_defaults_to_4(self) -> None:
        dm = CombineDataModule()
        assert dm.batch_size == 4

    def test_no_dataloaders_before_setup(self) -> None:
        dm = CombineDataModule()
        assert dm.train_dataset is None
        assert dm.val_dataset is None

    def test_train_dataloader_none_without_data(self) -> None:
        dm = CombineDataModule()
        dm.setup("fit")
        assert dm.train_dataloader() is None

    def test_weighted_sampling_flag(self) -> None:
        dm = CombineDataModule(use_weighted_sampling=False)
        assert dm.use_weighted_sampling is False

    def test_union_label_map_accessible(self) -> None:
        from neurons.datamodules.combine import UNION_LABEL_MAP, UNION_LABEL_NAMES, NUM_UNION_CLASSES

        assert NUM_UNION_CLASSES == 5
        assert UNION_LABEL_MAP["background"] == 0
        assert UNION_LABEL_MAP["neuron"] == 1
        assert UNION_LABEL_MAP["cleft"] == 2
        assert UNION_LABEL_MAP["mitochondria"] == 3
        assert UNION_LABEL_MAP["mito_boundary"] == 4
        assert len(UNION_LABEL_NAMES) == 5

    def test_datamodules_dict_api(self) -> None:
        dm = CombineDataModule(datamodules={"snemi3d": (None, 1.0)})
        assert "snemi3d" in dm._dm_entries


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
            dataset_name="Dataset001_ME2-Beta",
            slice_mode=False,
        )
        kw = dm._get_dataset_kwargs()
        assert kw["dataset_name"] == "Dataset001_ME2-Beta"
        assert kw["slice_mode"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

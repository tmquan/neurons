"""
Combined DataModule for training on multiple connectomics datasets.

Provides a unified semantic label space across all available datasets
via a configurable ``UnionLabelMap``.

Union semantic classes (default):
    0 = background
    1 = neuron
    2 = cleft (synapse)
    3 = mitochondria
    4 = mito_boundary

Each source dataset maps its native labels to this shared space through
the ``CreateClassIds`` wrapper.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# Union label mapping
# ---------------------------------------------------------------------------

UNION_LABEL_MAP: Dict[str, int] = {
    "background":     0,
    "neuron":         1,
    "cleft":          2,
    "mitochondria":   3,
    "mito_boundary":  4,
}

UNION_LABEL_NAMES: List[str] = [
    "background",
    "neuron",
    "cleft",
    "mitochondria",
    "mito_boundary",
]

NUM_UNION_CLASSES: int = len(UNION_LABEL_NAMES)


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------

class ExpandedDataset(Dataset):
    """
    Wrapper that virtually expands a dataset by repeating samples.

    Useful when you have few samples (e.g., one volume per dataset)
    but want many iterations per epoch.  Random transforms ensure
    each access returns different patches/augmentations.

    Args:
        dataset: Base dataset to wrap.
        expansion_factor: How many times to virtually repeat the dataset.
    """

    def __init__(self, dataset: Dataset, expansion_factor: int = 100) -> None:
        self.dataset = dataset
        self.expansion_factor = expansion_factor
        self._base_len = len(dataset)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return self._base_len * self.expansion_factor

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[idx % self._base_len]


class CreateClassIds(Dataset):
    """
    Wrapper that maps each dataset's native labels to the union label space.

    For every sample it adds two keys:
    - ``class_ids``: per-pixel union semantic class (int64, same spatial shape)
    - ``dataset_type``: string identifying the source dataset

    Per-dataset mapping rules
    -------------------------
    **snemi3d**
        foreground (label > 0) -> 1 (neuron)

    **cremi3d**
        Instance IDs are offset-encoded:
            0                           -> 0 (background)
            1 .. CLEFT_ID_OFFSET-1      -> 1 (neuron)
            CLEFT_ID_OFFSET .. MITO-1   -> 2 (cleft)
            >= MITO_ID_OFFSET           -> 3 (mitochondria)

    **microns**
        foreground (label > 0) -> 1 (neuron)

    **mitoem2**
        Native labels are already semantic:
            0 -> 0 (background)
            1 -> 3 (mitochondria)
            2 -> 4 (mito_boundary)

    **generic / unknown**
        foreground -> ``default_class``

    Args:
        dataset: Base dataset to wrap.
        dataset_type: One of 'snemi3d', 'cremi3d', 'microns', 'mitoem2'.
        default_class: Fallback class for unknown dataset types.
        ignore_classes: Optional set of union class names to ignore.
            Pixels that would be mapped to an ignored class are set to 0
            (background) instead.  Example: ``{"mito_boundary"}``
    """

    CLEFT_ID_OFFSET: int = 1_000_000
    MITO_ID_OFFSET: int = 2_000_000

    def __init__(
        self,
        dataset: Dataset,
        dataset_type: str = "snemi3d",
        default_class: int = 1,
        ignore_classes: Optional[set] = None,
    ) -> None:
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.default_class = default_class
        self._ignore_ids: set = set()
        if ignore_classes:
            for name in ignore_classes:
                if name in UNION_LABEL_MAP:
                    self._ignore_ids.add(UNION_LABEL_MAP[name])

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        label = sample["label"]

        is_tensor = isinstance(label, torch.Tensor)
        label_np = label.numpy() if is_tensor else np.asarray(label)

        class_ids = np.zeros_like(label_np, dtype=np.int64)

        dt = self.dataset_type

        if dt == "snemi3d" or dt == "microns":
            class_ids[label_np > 0] = UNION_LABEL_MAP["neuron"]

        elif dt == "cremi3d":
            neuron_mask = (label_np > 0) & (label_np < self.CLEFT_ID_OFFSET)
            class_ids[neuron_mask] = UNION_LABEL_MAP["neuron"]

            cleft_mask = (label_np >= self.CLEFT_ID_OFFSET) & (label_np < self.MITO_ID_OFFSET)
            class_ids[cleft_mask] = UNION_LABEL_MAP["cleft"]

            mito_mask = label_np >= self.MITO_ID_OFFSET
            class_ids[mito_mask] = UNION_LABEL_MAP["mitochondria"]

        elif dt == "mitoem2":
            class_ids[label_np == 1] = UNION_LABEL_MAP["mitochondria"]
            class_ids[label_np == 2] = UNION_LABEL_MAP["mito_boundary"]

        else:
            class_ids[label_np > 0] = self.default_class

        # Zero-out ignored classes -> background
        if self._ignore_ids:
            for cid in self._ignore_ids:
                class_ids[class_ids == cid] = 0

        sample = dict(sample)
        sample["semantic_ids"] = torch.from_numpy(class_ids) if is_tensor else class_ids
        sample["dataset_type"] = self.dataset_type
        return sample


# ---------------------------------------------------------------------------
# CombineDataModule
# ---------------------------------------------------------------------------

class CombineDataModule(pl.LightningDataModule):
    """
    Lightning DataModule that combines any subset of available datasets
    into a single training pipeline with a unified semantic label space.

    Supported source datamodules (all optional):
    - SNEMI3DDataModule
    - CREMI3DDataModule
    - MICRONSDataModule
    - MitoEM2DataModule

    Each source dataset's labels are mapped to the union label space
    (see ``UNION_LABEL_MAP``) via ``CreateClassIds``.

    Args:
        datamodules: Dict mapping dataset_type -> (datamodule, weight).
            E.g. ``{"snemi3d": (snemi_dm, 1.0), "mitoem2": (mito_dm, 2.0)}``.
        batch_size: Override batch size.
        num_workers: Override num_workers.
        use_weighted_sampling: Balance sampling across datasets.
        train_expansion_factor: Virtual expansion for training.
        val_expansion_factor: Virtual expansion for validation.

    Example:
        >>> from neurons.datamodules import SNEMI3DDataModule, MitoEM2DataModule
        >>> snemi = SNEMI3DDataModule(data_root="data/snemi3d", batch_size=4)
        >>> mito  = MitoEM2DataModule(data_root="data/mitoem2", batch_size=4)
        >>> combine = CombineDataModule(
        ...     datamodules={
        ...         "snemi3d": (snemi, 1.0),
        ...         "mitoem2": (mito,  2.0),
        ...     },
        ...     batch_size=4,
        ... )
        >>> combine.setup("fit")
        >>> batch = next(iter(combine.train_dataloader()))
        >>> batch["class_ids"]  # union semantic labels
    """

    # Expose the shared mapping so callers can inspect it
    union_label_map = UNION_LABEL_MAP
    union_label_names = UNION_LABEL_NAMES
    num_union_classes = NUM_UNION_CLASSES

    def __init__(
        self,
        datamodules: Optional[Dict[str, tuple]] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        use_weighted_sampling: bool = True,
        train_expansion_factor: int = 100,
        val_expansion_factor: int = 10,
        ignore_classes: Optional[set] = None,
        # Legacy kwargs for backward compat
        snemi3d_datamodule: Optional[Any] = None,
        cremi3d_datamodule: Optional[Any] = None,
        snemi3d_weight: float = 1.0,
        cremi3d_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.ignore_classes = ignore_classes

        # Build canonical dict from either new or legacy API
        self._dm_entries: Dict[str, tuple] = {}
        if datamodules is not None:
            self._dm_entries = dict(datamodules)
        else:
            if snemi3d_datamodule is not None:
                self._dm_entries["snemi3d"] = (snemi3d_datamodule, snemi3d_weight)
            if cremi3d_datamodule is not None:
                self._dm_entries["cremi3d"] = (cremi3d_datamodule, cremi3d_weight)

        # Resolve batch_size / num_workers
        first_dm = next((dm for dm, _ in self._dm_entries.values()), None)
        self.batch_size = batch_size or (getattr(first_dm, "batch_size", 4) if first_dm else 4)
        self.num_workers = num_workers or (getattr(first_dm, "num_workers", 4) if first_dm else 4)

        self.pin_memory = pin_memory
        self.use_weighted_sampling = use_weighted_sampling
        self.train_expansion_factor = train_expansion_factor
        self.val_expansion_factor = val_expansion_factor

        self.train_dataset: Optional[ConcatDataset] = None
        self.val_dataset: Optional[ConcatDataset] = None
        self.train_sampler: Optional[WeightedRandomSampler] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup combined datasets with expansion for sufficient batch sampling."""
        train_datasets: List[Dataset] = []
        train_weights: List[float] = []
        val_datasets: List[Dataset] = []

        for dtype, (dm, weight) in self._dm_entries.items():
            dm.setup(stage)

            if getattr(dm, "train_dataset", None) is not None:
                wrapped = CreateClassIds(
                    dm.train_dataset, dataset_type=dtype,
                    ignore_classes=self.ignore_classes,
                )
                expanded = ExpandedDataset(wrapped, self.train_expansion_factor)
                train_datasets.append(expanded)
                train_weights.extend([weight] * len(expanded))

            if getattr(dm, "val_dataset", None) is not None:
                wrapped = CreateClassIds(
                    dm.val_dataset, dataset_type=dtype,
                    ignore_classes=self.ignore_classes,
                )
                expanded = ExpandedDataset(wrapped, self.val_expansion_factor)
                val_datasets.append(expanded)

        if train_datasets:
            self.train_dataset = ConcatDataset(train_datasets)
            if self.use_weighted_sampling:
                self.train_sampler = WeightedRandomSampler(
                    weights=train_weights,
                    num_samples=len(train_weights),
                    replacement=True,
                )

        if val_datasets:
            self.val_dataset = ConcatDataset(val_datasets)

    def train_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        """Create training DataLoader."""
        if self.train_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler if self.use_weighted_sampling else None,
            shuffle=not self.use_weighted_sampling,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        """Create validation DataLoader."""
        if self.val_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

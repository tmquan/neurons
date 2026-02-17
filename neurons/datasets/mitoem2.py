"""
MitoEM2 Dataset for mitochondria segmentation.

MitoEM2 provides 8 EM datasets from different cell types with
three-class labels: background (0), mitochondria (1), boundary (2).

Data format: NIfTI (.nii.gz) in nnU-Net directory convention.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from neurons.datasets.base import CircuitDataset
from neurons.preprocessors.nfty import NFTYPreprocessor


class MitoEM2Dataset(CircuitDataset):
    """
    MitoEM2 Dataset for mitochondria instance/semantic segmentation.

    Expected directory structure (nnU-Net convention):
        root_dir/
        +-- Dataset001_ME2-Beta/
        |   +-- dataset.json
        |   +-- imagesTr/  (training EM volumes, .nii.gz)
        |   +-- labelsTr/  (training labels, .nii.gz)
        |   +-- imagesTs/  (test EM volumes)
        |   +-- labelsTs/  (test labels)
        +-- Dataset006_ME2-Pyra/
        +-- ...

    Labels:
        0 = background
        1 = mitochondria
        2 = boundary

    Args:
        root_dir: Path to the MitoEM2 root (parent of Dataset* dirs).
        split: Data split ('train', 'valid', 'test').
        transform: Optional MONAI transforms to apply.
        cache_rate: Fraction of data to cache in memory.
        train_val_split: Fraction for validation split.
        dataset_name: Specific dataset to load, e.g. 'Dataset001_ME2-Beta'.
            If None, loads all datasets found in root_dir.
        slice_mode: If True, return individual 2D slices (default: True).

    Example:
        >>> ds = MitoEM2Dataset(
        ...     root_dir="data/mitoem2",
        ...     dataset_name="Dataset001_ME2-Beta",
        ...     split="train",
        ... )
        >>> sample = ds[0]
        >>> print(sample["image"].shape, sample["label"].shape)
    """

    _paper = (
        "Wei, D., et al. (2020). MitoEM Dataset: Large-scale 3D Mitochondria "
        "Instance Segmentation from EM Images. MICCAI 2020."
    )
    _labels_list: List[str] = ["background", "mitochondria", "boundary"]

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        train_val_split: float = 0.2,
        num_workers: int = 0,
        dataset_name: Optional[str] = None,
        slice_mode: bool = True,
    ) -> None:
        self.dataset_name = dataset_name
        self.slice_mode = slice_mode
        self._nfty = NFTYPreprocessor()

        super().__init__(
            root_dir=root_dir,
            split=split,
            transform=transform,
            cache_rate=cache_rate,
            train_val_split=train_val_split,
            num_workers=num_workers,
        )

    @property
    def paper(self) -> str:
        return self._paper

    @property
    def resolution(self) -> Dict[str, float]:
        # Read from dataset.json if available, else default
        ds_dirs = self._get_dataset_dirs()
        if ds_dirs:
            json_path = ds_dirs[0] / "dataset.json"
            if json_path.exists():
                with open(json_path) as f:
                    meta = json.load(f)
                sp = meta.get("spacing", [8, 8, 8])
                return {"x": float(sp[0]), "y": float(sp[1]), "z": float(sp[2])}
        return {"x": 8.0, "y": 8.0, "z": 8.0}

    @property
    def labels(self) -> List[str]:
        return self._labels_list.copy()

    @property
    def data_files(self) -> Dict[str, Union[str, np.ndarray]]:
        return {"vol": "imagesTr/*.nii.gz", "seg": "labelsTr/*.nii.gz"}

    def _get_dataset_dirs(self) -> List[Path]:
        """Return list of dataset directories to load."""
        if self.dataset_name is not None:
            ds_dir = self.root_dir / self.dataset_name
            return [ds_dir] if ds_dir.exists() else []
        return sorted(
            d for d in self.root_dir.iterdir()
            if d.is_dir() and d.name.startswith("Dataset")
        )

    def _prepare_data(self) -> List[Dict[str, Any]]:
        """Prepare data dictionaries from NIfTI volumes."""
        data_list: List[Dict[str, Any]] = []

        ds_dirs = self._get_dataset_dirs()

        for ds_dir in ds_dirs:
            if self.split in ("train", "valid"):
                img_dir = ds_dir / "imagesTr"
                lbl_dir = ds_dir / "labelsTr"
            else:
                img_dir = ds_dir / "imagesTs"
                lbl_dir = ds_dir / "labelsTs"

            if not img_dir.exists():
                continue

            img_files = sorted(img_dir.glob("*.nii.gz"))
            lbl_files = sorted(lbl_dir.glob("*.nii.gz")) if lbl_dir.exists() else []

            # Build pairs
            pairs: List[Tuple[Path, Optional[Path]]] = []
            for img_f in img_files:
                # Match label: image is *_0000.nii.gz, label is *.nii.gz (no channel suffix)
                stem = img_f.name.replace("_0000.nii.gz", ".nii.gz")
                lbl_f = lbl_dir / stem if (lbl_dir / stem).exists() else None
                pairs.append((img_f, lbl_f))

            # Train/valid split
            n_total = len(pairs)
            n_train = int(n_total * (1.0 - self.train_val_split))

            if self.split == "train":
                pairs = pairs[:n_train]
            elif self.split == "valid":
                pairs = pairs[n_train:]

            for vol_idx, (img_path, lbl_path) in enumerate(pairs):
                image = self._nfty.load(str(img_path))
                label = self._nfty.load(str(lbl_path)) if lbl_path is not None else None

                if self.slice_mode and image.ndim == 3:
                    z_dim = image.shape[2] if image.shape[2] < image.shape[0] else image.shape[0]
                    use_last_axis = image.shape[2] < image.shape[0]

                    for z in range(z_dim):
                        sl_img = image[:, :, z] if use_last_axis else image[z]
                        entry: Dict[str, Any] = {
                            "image": sl_img.astype(np.float32),
                            "dataset": ds_dir.name,
                            "volume_idx": vol_idx,
                            "slice_idx": z,
                            "idx": len(data_list),
                        }
                        if label is not None:
                            sl_lbl = label[:, :, z] if use_last_axis else label[z]
                            entry["label"] = sl_lbl.astype(np.int64)
                        data_list.append(entry)
                else:
                    entry = {
                        "image": image.astype(np.float32),
                        "dataset": ds_dir.name,
                        "volume_idx": vol_idx,
                        "idx": len(data_list),
                    }
                    if label is not None:
                        entry["label"] = label.astype(np.int64)
                    data_list.append(entry)

        return data_list

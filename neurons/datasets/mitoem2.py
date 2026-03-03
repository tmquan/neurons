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
import torch

from neurons.datasets.base import CircuitDataset
from neurons.preprocessors.nfty import NFTYPreprocessor


class MitoEM2Dataset(CircuitDataset):
    """
    MitoEM2 Dataset for mitochondria instance/semantic segmentation.

    Volume format:
        ``[{"subdataset": "Dataset001_ME2-Beta", "img_dir": "imagesTr", "lbl_dir": "labelsTr"}]``

    When ``img_dir``/``lbl_dir`` are omitted, defaults to ``imagesTr``/``labelsTr``.
    When ``volumes`` is None, loads all ``Dataset*`` dirs under ``root_dir``.

    Args:
        root_dir: Path to the MitoEM2 root (parent of Dataset* dirs).
        volumes: List of volume dicts with ``subdataset`` key.
        transform: Optional MONAI transforms to apply.
        cache_rate: Fraction of data to cache in memory.
        slice_mode: If True, return individual 2D slices (default: True).
        slice_axis: Axis to slice along in slice_mode (0=first, -1=last).
            Default 0 (standard NIfTI Z-axis).
        num_samples: Number of samples per epoch.
    """

    _paper = (
        "Wei, D., et al. (2020). MitoEM Dataset: Large-scale 3D Mitochondria "
        "Instance Segmentation from EM Images. MICCAI 2020."
    )
    _labels_list: List[str] = ["background", "mitochondria", "boundary"]

    def __init__(
        self,
        root_dir: str,
        volumes: Optional[List[Dict[str, str]]] = None,
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        num_workers: int = 0,
        slice_mode: bool = True,
        slice_axis: int = 0,
        num_samples: Optional[int] = None,
    ) -> None:
        self.slice_mode = slice_mode
        self.slice_axis = slice_axis
        self._num_samples = num_samples
        self._nfty = NFTYPreprocessor()

        super().__init__(
            root_dir=root_dir,
            volumes=volumes,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    @property
    def paper(self) -> str:
        return self._paper

    @property
    def resolution(self) -> Dict[str, float]:
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

    def _default_volumes(self) -> List[Dict[str, str]]:
        """Auto-discover all Dataset* dirs, using imagesTr/labelsTr."""
        result = []
        for d in sorted(self.root_dir.iterdir()):
            if d.is_dir() and d.name.startswith("Dataset"):
                result.append({"subdataset": d.name, "img_dir": "imagesTr", "lbl_dir": "labelsTr"})
        return result

    def _get_dataset_dirs(self) -> List[Path]:
        """Return list of dataset directories from volumes list."""
        vol_list = self._get_volume_list()
        dirs = []
        for v in vol_list:
            sub = v.get("subdataset", "")
            d = self.root_dir / sub if sub else self.root_dir
            if d.exists():
                dirs.append(d)
        return dirs

    def _prepare_data(self) -> List[Dict[str, Any]]:
        data_list: List[Dict[str, Any]] = []

        for vol_spec in self._get_volume_list():
            sub = vol_spec.get("subdataset", "")
            ds_dir = self.root_dir / sub if sub else self.root_dir
            img_dir_name = vol_spec.get("img_dir", "imagesTr")
            lbl_dir_name = vol_spec.get("lbl_dir", "labelsTr")

            img_dir = ds_dir / img_dir_name
            lbl_dir = ds_dir / lbl_dir_name

            if not img_dir.exists():
                continue

            img_files = sorted(img_dir.glob("*.nii.gz"))
            lbl_files = sorted(lbl_dir.glob("*.nii.gz")) if lbl_dir.exists() else []

            pairs: List[Tuple[Path, Optional[Path]]] = []
            for img_f in img_files:
                stem = img_f.name.replace("_0000.nii.gz", ".nii.gz")
                lbl_f = lbl_dir / stem if (lbl_dir / stem).exists() else None
                pairs.append((img_f, lbl_f))

            for vol_idx, (img_path, lbl_path) in enumerate(pairs):
                image = self._nfty.load(str(img_path)).astype(np.float32)
                vmin, vmax = float(image.min()), float(image.max())
                if vmax > vmin:
                    image = (image - vmin) / (vmax - vmin)
                label = self._nfty.load(str(lbl_path)) if lbl_path is not None else None
                if label is not None:
                    label = label.astype(np.int64)

                if self.slice_mode and image.ndim == 3:
                    ax = self.slice_axis if self.slice_axis >= 0 else image.ndim + self.slice_axis
                    z_dim = image.shape[ax]

                    for z in range(z_dim):
                        sl_img = np.take(image, z, axis=ax)
                        entry: Dict[str, Any] = {
                            "image": self._to_shared(sl_img),
                            "dataset": ds_dir.name,
                            "volume_idx": vol_idx,
                            "slice_idx": z,
                            "idx": len(data_list),
                        }
                        if label is not None:
                            sl_lbl = np.take(label, z, axis=ax)
                            entry["label"] = self._to_shared(sl_lbl)
                        data_list.append(entry)
                else:
                    entry: Dict[str, Any] = {
                        "image": self._to_shared(image),
                        "dataset": ds_dir.name,
                        "volume_idx": vol_idx,
                        "idx": len(data_list),
                    }
                    if label is not None:
                        entry["label"] = self._to_shared(label)
                    data_list.append(entry)

        if self._num_samples is not None and len(data_list) > 0:
            self._virtual_len = self._num_samples

        return data_list

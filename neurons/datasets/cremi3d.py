"""
CREMI3D Dataset for connectomics instance segmentation.

CREMI (Circuit Reconstruction from Electron Microscopy Images) Challenge:
- 3 volumes: A, B, C (training A, B have labels; C is test)
- Resolution: 4nm x 4nm x 40nm (anisotropic)
- Annotations: neurons, synaptic clefts, (optionally mitochondria)
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from neurons.datasets.base import CircuitDataset


class CREMI3DDataset(CircuitDataset):
    """
    CREMI3D dataset for neuron and synapse segmentation.

    Expected directory structure:
        data_root/
        +-- sample_A.h5 (or sample_A/)
        |   +-- volumes/raw (EM image)
        |   +-- volumes/labels/neuron_ids
        |   +-- volumes/labels/clefts (optional)
        +-- sample_B.h5
        +-- sample_C.h5 (test, no labels)

    Attributes:
        volumes: List of volume names to load ["A", "B", "C"].
        include_clefts: Whether to include synaptic cleft annotations.
        include_mito: Whether to include mitochondria annotations.
    """

    NEURON_ID_OFFSET: int = 0
    CLEFT_ID_OFFSET: int = 1_000_000
    MITO_ID_OFFSET: int = 2_000_000

    NO_DATA_MARKER: int = int(np.iinfo(np.uint64).max)

    CLASS_BACKGROUND: int = 0
    CLASS_NEURON: int = 1
    CLASS_CLEFT: int = 2
    CLASS_MITO: int = 3

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        train_val_split: float = 0.2,
        num_workers: int = 0,
        volumes: Optional[List[str]] = None,
        include_clefts: bool = True,
        include_mito: bool = False,
        num_samples: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.volumes = volumes if volumes is not None else ["A", "B"]
        self.include_clefts = include_clefts
        self.include_mito = include_mito
        self._num_samples = num_samples
        self._image_data: Optional[np.ndarray] = None
        self._label_data: Optional[np.ndarray] = None

        super().__init__(
            root_dir=str(root_dir),
            split=split,
            transform=transform,
            cache_rate=cache_rate,
            train_val_split=train_val_split,
            num_workers=num_workers,
        )

    @property
    def paper(self) -> str:
        return "CREMI Challenge - https://cremi.org/"

    @property
    def resolution(self) -> Dict[str, float]:
        return {"x": 4.0, "y": 4.0, "z": 40.0}

    @property
    def labels(self) -> List[str]:
        return ["background", "neuron", "cleft", "mito"]

    @property
    def data_files(self) -> Dict[str, Union[str, np.ndarray]]:
        if self._image_data is not None and self._label_data is not None:
            return {"vol": self._image_data, "seg": self._label_data}
        return {
            "vol": "sample_*.h5/volumes/raw",
            "seg": "sample_*.h5/volumes/labels/neuron_ids",
        }

    def _prepare_data(self) -> List[Dict[str, Any]]:
        """Prepare list of data dictionaries for each sample."""
        image, label = self._load_data()
        self._image_data = image
        self._label_data = label

        total_slices = image.shape[0]
        volumes_str = "_".join(self.volumes)

        if self.split == "test":
            start_idx = int(total_slices * (1 - self.train_val_split))
            image = image[start_idx:]
            label = label[start_idx:]
            volume_name = f"CREMI_{volumes_str}_test"
        elif self.split == "valid":
            val_start = int(total_slices * (1 - self.train_val_split))
            image = image[val_start:]
            label = label[val_start:]
            volume_name = f"CREMI_{volumes_str}_valid"
        else:
            train_end = int(total_slices * (1 - self.train_val_split))
            image = image[:train_end]
            label = label[:train_end]
            volume_name = f"CREMI_{volumes_str}_train"

        data_dict = {
            "image": image,
            "label": label,
            "volume": volume_name,
            "idx": 0,
        }
        self._virtual_len = self._num_samples if self._num_samples is not None else image.shape[0]
        return [data_dict]

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and merge CREMI volumes."""
        all_images: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        for vol_name in self.volumes:
            image, label = self._load_volume(vol_name)
            if image is not None:
                all_images.append(image)
                all_labels.append(label)

        if not all_images:
            raise ValueError(f"No data found in {self.root_dir}")

        images = np.concatenate(all_images, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        return images, labels

    def _load_volume(
        self,
        vol_name: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load a single CREMI volume."""
        possible_paths = [
            self.root_dir / f"sample_{vol_name}_20160501.hdf",
            self.root_dir / f"sample_{vol_name}+_20160601.hdf",
            self.root_dir / f"sample_{vol_name}_padded_20160501.hdf",
            self.root_dir / f"sample_{vol_name}.h5",
            self.root_dir / f"sample_{vol_name}.hdf5",
            self.root_dir / f"sample_{vol_name}.hdf",
            self.root_dir / vol_name / "sample.h5",
        ]

        h5_path: Optional[Path] = None
        for path in possible_paths:
            if path.exists():
                h5_path = path
                break

        if h5_path is None:
            return self._load_volume_separate_files(vol_name)

        try:
            import h5py

            with h5py.File(h5_path, "r") as f:
                if "volumes/raw" in f:
                    image = f["volumes/raw"][:]
                elif "raw" in f:
                    image = f["raw"][:]
                else:
                    return None, None

                label = np.zeros_like(image, dtype=np.int64)

                if "volumes/labels/neuron_ids" in f:
                    neuron_ids = f["volumes/labels/neuron_ids"][:]
                    label[neuron_ids > 0] = neuron_ids[neuron_ids > 0] + self.NEURON_ID_OFFSET
                elif "neuron_ids" in f:
                    neuron_ids = f["neuron_ids"][:]
                    label[neuron_ids > 0] = neuron_ids[neuron_ids > 0] + self.NEURON_ID_OFFSET

                if self.include_clefts:
                    for cleft_key in ("volumes/labels/clefts", "clefts"):
                        if cleft_key in f:
                            cleft_ids = f[cleft_key][:]
                            valid = (cleft_ids > 0) & (cleft_ids < self.NO_DATA_MARKER)
                            label[valid] = cleft_ids[valid] + self.CLEFT_ID_OFFSET
                            break

                if self.include_mito:
                    for mito_key in ("volumes/labels/mitochondria", "mitochondria"):
                        if mito_key in f:
                            mito_ids = f[mito_key][:]
                            valid = (mito_ids > 0) & (mito_ids < self.NO_DATA_MARKER)
                            label[valid] = mito_ids[valid] + self.MITO_ID_OFFSET
                            break

                return image.astype(np.float32), label

        except Exception:
            return None, None

    def _load_volume_separate_files(
        self,
        vol_name: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load volume from separate files."""
        import h5py

        def load_h5(path: Path) -> Optional[np.ndarray]:
            with h5py.File(path, "r") as f:
                def find_dataset(group: Any) -> Optional[np.ndarray]:
                    for key in group.keys():
                        if isinstance(group[key], h5py.Dataset):
                            return group[key][:]
                        elif isinstance(group[key], h5py.Group):
                            result = find_dataset(group[key])
                            if result is not None:
                                return result
                    return None
                return find_dataset(f)

        raw_paths = [
            self.root_dir / f"sample_{vol_name}_raw.h5",
            self.root_dir / f"{vol_name}_raw.h5",
            self.root_dir / f"{vol_name}_image.h5",
        ]

        image: Optional[np.ndarray] = None
        for path in raw_paths:
            if path.exists():
                image = load_h5(path)
                break

        if image is None:
            return None, None

        label = np.zeros_like(image, dtype=np.int64)

        neuron_paths = [
            self.root_dir / f"sample_{vol_name}_neuron_ids.h5",
            self.root_dir / f"{vol_name}_neuron_ids.h5",
            self.root_dir / f"{vol_name}_labels.h5",
        ]

        for path in neuron_paths:
            if path.exists():
                neuron_ids = load_h5(path)
                if neuron_ids is not None:
                    label[neuron_ids > 0] = neuron_ids[neuron_ids > 0] + self.NEURON_ID_OFFSET
                break

        if self.include_clefts:
            cleft_paths = [
                self.root_dir / f"sample_{vol_name}_clefts.h5",
                self.root_dir / f"{vol_name}_clefts.h5",
            ]
            for path in cleft_paths:
                if path.exists():
                    cleft_ids = load_h5(path)
                    if cleft_ids is not None:
                        valid = (cleft_ids > 0) & (cleft_ids < self.NO_DATA_MARKER)
                        label[valid] = cleft_ids[valid] + self.CLEFT_ID_OFFSET
                    break

        return image.astype(np.float32), label

    @staticmethod
    def instance_id_to_class(instance_id: int) -> int:
        """Map instance ID to semantic class ID."""
        if instance_id == 0:
            return CREMI3DDataset.CLASS_BACKGROUND
        elif instance_id < CREMI3DDataset.CLEFT_ID_OFFSET:
            return CREMI3DDataset.CLASS_NEURON
        elif instance_id < CREMI3DDataset.MITO_ID_OFFSET:
            return CREMI3DDataset.CLASS_CLEFT
        else:
            return CREMI3DDataset.CLASS_MITO

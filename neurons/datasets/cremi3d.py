"""
CREMI3D Dataset for connectomics instance segmentation.

CREMI (Circuit Reconstruction from Electron Microscopy Images) Challenge:
- 3 volumes: A, B, C (training A, B have labels; C is test)
- Resolution: 4nm x 4nm x 40nm (anisotropic)
- Annotations: neurons, synaptic clefts, (optionally mitochondria)
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from neurons.datasets.base import CircuitDataset

logger = logging.getLogger(__name__)


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
        volumes: Optional[List[Dict[str, str]]] = None,
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        num_workers: int = 0,
        include_clefts: bool = True,
        include_mito: bool = False,
        num_samples: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.include_clefts = include_clefts
        self.include_mito = include_mito
        self._num_samples = num_samples
        super().__init__(
            root_dir=str(root_dir),
            volumes=volumes,
            transform=transform,
            cache_rate=cache_rate,
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

    def _default_volumes(self) -> List[Dict[str, str]]:
        return [{"vol": "A"}, {"vol": "B"}]

    @property
    def data_files(self) -> Dict[str, Union[str, np.ndarray]]:
        vols = self._get_volume_list()
        if vols:
            return {"vol": vols[0]["vol"], "seg": f"sample_{vols[0]['vol']}_*.hdf"}
        return {
            "vol": "sample_*.hdf/volumes/raw",
            "seg": "sample_*.hdf/volumes/labels/neuron_ids",
        }

    def _prepare_data(self) -> List[Dict[str, Any]]:
        """Load each volume in the list as a separate data entry."""
        data_list: List[Dict[str, Any]] = []
        total_slices = 0

        for vol_spec in self._get_volume_list():
            vol_letter = vol_spec["vol"]
            image, label = self._load_volume(vol_letter)
            if image is None:
                continue

            image = image.astype(np.float32)
            vmin, vmax = float(image.min()), float(image.max())
            if vmax > vmin:
                image = (image - vmin) / (vmax - vmin)
            label = label.astype(np.int64)

            data_dict = {
                "image": image,
                "label": label,
                "volume": f"CREMI_{vol_letter}",
                "idx": len(data_list),
            }
            data_list.append(data_dict)
            total_slices += image.shape[0]

        self._virtual_len = self._num_samples if self._num_samples is not None else total_slices
        return data_list

    def _load_volume(
        self,
        vol_name: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load a single CREMI volume."""
        is_padded = vol_name.endswith("+")
        letter = vol_name.rstrip("+")

        if is_padded:
            possible_paths = [
                self.root_dir / f"sample_{letter}+_20160601.hdf",
                self.root_dir / f"sample_{letter}_padded_20160601.hdf",
                self.root_dir / f"sample_{vol_name}.h5",
                self.root_dir / f"sample_{vol_name}.hdf5",
                self.root_dir / f"sample_{vol_name}.hdf",
                self.root_dir / vol_name / "sample.h5",
            ]
        else:
            possible_paths = [
                self.root_dir / f"sample_{letter}_20160501.hdf",
                self.root_dir / f"sample_{letter}.h5",
                self.root_dir / f"sample_{letter}.hdf5",
                self.root_dir / f"sample_{letter}.hdf",
                self.root_dir / letter / "sample.h5",
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

        except Exception as exc:
            logger.warning("Failed to load volume: %s", exc)
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

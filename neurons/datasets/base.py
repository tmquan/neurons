"""
Base dataset class for connectomics research.

Provides abstract interface that all connectomics datasets must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from monai.data import CacheDataset
from monai.transforms import Randomizable


class CircuitDataset(CacheDataset, Randomizable, ABC):
    """
    Abstract base class for connectomics datasets.

    All connectomics datasets must implement the following properties:
    - paper: Reference or citation metadata (string)
    - resolution: Voxel/spatial resolution specification (dict)
    - labels: List of segmentation class labels (list)
    - data_files: Dictionary with 'vol' and 'seg' keys for data paths/arrays

    This class inherits from MONAI's CacheDataset for efficient caching
    and Randomizable for reproducible random augmentations.

    Args:
        root_dir: Root directory containing the dataset files.
        split: Data split ('train', 'valid', 'test').
        transform: Optional transforms to apply to each sample.
        cache_rate: Fraction of data to cache in memory (0.0 to 1.0).
        train_val_split: Fraction of data to use for validation when
            splitting train data.
        num_workers: Number of workers for data loading.

    Example:
        >>> class MyDataset():
        ...     @property
        ...     def paper(self) -> str:
        ...         return "My Dataset (2024)"
        ...
        ...     @property
        ...     def resolution(self) -> Dict[str, float]:
        ...         return {"x": 4.0, "y": 4.0, "z": 40.0}
        ...
        ...     @property
        ...     def labels(self) -> List[str]:
        ...         return ["background", "neuron"]
        ...
        ...     @property
        ...     def data_files(self) -> Dict[str, Union[str, np.ndarray]]:
        ...         return {"vol": "volume.h5", "seg": "segmentation.h5"}
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        train_val_split: float = 0.2,
        num_workers: int = 0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.train_val_split = train_val_split

        if self.split not in ["train", "valid", "test"]:
            raise ValueError(
                f"split must be 'train', 'valid', or 'test', got '{split}'"
            )

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset root directory not found: {self.root_dir}\n"
                f"Please ensure the data is downloaded and extracted to this location."
            )

        data_dicts = self._prepare_data()

        if len(data_dicts) == 0:
            raise ValueError(
                f"No data found for split '{self.split}' in {self.root_dir}.\n"
                f"Expected data files: {self.data_files}"
            )

        super().__init__(
            data=data_dicts,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    @property
    @abstractmethod
    def paper(self) -> str:
        """
        Reference or citation metadata for the dataset.

        Returns:
            String containing paper reference, DOI, or citation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def resolution(self) -> Dict[str, float]:
        """
        Voxel/spatial resolution specification in nanometers.

        Returns:
            Dictionary with 'x', 'y', 'z' keys and resolution values in nm.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def labels(self) -> List[str]:
        """
        List of segmentation class labels.

        Returns:
            List of label names, where index corresponds to label value.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def data_files(self) -> Dict[str, Union[str, np.ndarray]]:
        """
        Dictionary specifying volume and segmentation data sources.

        Must contain keys:
        - 'vol': Path to volume data file or numpy array
        - 'seg': Path to segmentation mask file or numpy array

        Returns:
            Dictionary with 'vol' and 'seg' keys.
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_data(self) -> List[Dict[str, Any]]:
        """
        Prepare list of data dictionaries for each sample.

        This method should:
        1. Load or reference the volume and segmentation data
        2. Split data according to self.split and self.train_val_split
        3. Return list of dictionaries with at least 'image' and 'label' keys

        Returns:
            List of dictionaries, each containing sample data.
        """
        raise NotImplementedError

    def get_resolution_tuple(self) -> Tuple[float, float, float]:
        """
        Get resolution as (z, y, x) tuple for compatibility with numpy/torch.

        Returns:
            Tuple of (z, y, x) resolution values in nm.
        """
        res = self.resolution
        return (res["z"], res["y"], res["x"])

    def get_anisotropy_factor(self) -> float:
        """
        Calculate anisotropy factor (z_res / xy_res).

        Useful for determining if 3D augmentations should be applied
        differently along z-axis.

        Returns:
            Ratio of z-resolution to xy-resolution.
        """
        res = self.resolution
        xy_res = (res["x"] + res["y"]) / 2.0
        return res["z"] / xy_res

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  root_dir={self.root_dir},\n"
            f"  split='{self.split}',\n"
            f"  paper='{self.paper}',\n"
            f"  resolution={self.resolution},\n"
            f"  labels={self.labels},\n"
            f"  num_samples={len(self)}\n"
            f")"
        )

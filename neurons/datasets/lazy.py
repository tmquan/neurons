"""
Lazy-loading dataset for 3D volumetric connectomics data.

Reads only the requested patch from disk on each __getitem__ call,
keeping system memory usage constant regardless of volume count/size.
Designed for DDP training where each rank would otherwise load full
volumes, exhausting system RAM.

Supports HDF5 (chunked reads) and TIFF (memory-mapped reads).
File handles are cached per worker to avoid the overhead of opening
and closing HDF5 files on every sample.
"""

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_thread_local = threading.local()


class _VolumeHandle:
    """Lightweight metadata handle for a single volume — no data in RAM."""

    __slots__ = ("image_path", "label_path", "image_key", "label_key",
                 "shape", "name", "find_boundaries")

    def __init__(
        self,
        image_path: Path,
        label_path: Optional[Path],
        shape: Tuple[int, ...],
        name: str,
        image_key: Optional[str] = None,
        label_key: Optional[str] = None,
        find_boundaries: float = 0.0,
    ) -> None:
        self.image_path = image_path
        self.label_path = label_path
        self.image_key = image_key
        self.label_key = label_key
        self.shape = shape
        self.name = name
        self.find_boundaries = find_boundaries


def _resolve_hdf5_key(path: Path) -> Optional[str]:
    """Find the first dataset key in an HDF5 file without loading data."""
    import h5py
    with h5py.File(str(path), "r") as f:
        for k in ("main", "data", "raw", "volume", "image", "label"):
            if k in f:
                return k
        for k in f.keys():
            if isinstance(f[k], h5py.Dataset):
                return k
    return None


def _get_shape(path: Path, key: Optional[str] = None) -> Tuple[int, ...]:
    """Read volume shape from file metadata without loading data."""
    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        import h5py
        if key is None:
            key = _resolve_hdf5_key(path)
        with h5py.File(str(path), "r") as f:
            return tuple(f[key].shape)
    elif suffix in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(str(path)) as tif:
            series = tif.series[0]
            return tuple(series.shape)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _get_h5_handle(path: str):
    """Return a cached HDF5 file handle for the current worker thread.

    Handles are stored in thread-local storage so each DataLoader worker
    keeps its own set of open files.  This avoids the overhead of
    opening and parsing the HDF5 superblock on every sample.
    """
    import h5py
    cache = getattr(_thread_local, "h5_cache", None)
    if cache is None:
        cache = {}
        _thread_local.h5_cache = cache
    handle = cache.get(path)
    if handle is None:
        handle = h5py.File(path, "r", swmr=True)
        cache[path] = handle
    return handle


def _read_patch(
    path: Path,
    slices: Tuple[slice, ...],
    key: Optional[str] = None,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """Read a spatial patch from disk without loading the full volume."""
    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        if key is None:
            key = _resolve_hdf5_key(path)
        f = _get_h5_handle(str(path))
        data = f[key][slices]
        if dtype is not None:
            data = data.astype(dtype)
        return data
    elif suffix in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.memmap(str(path))
        data = np.array(arr[slices])
        if dtype is not None:
            data = data.astype(dtype)
        return data
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _find_file(search_dir: Path, base_name: str) -> Optional[Path]:
    """Find a volume file by base name with common extensions."""
    for ext in (".h5", ".hdf5", ".tif", ".tiff"):
        candidate = search_dir / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    for d in search_dir.iterdir():
        if d.is_dir() and d.name == base_name:
            for ext in (".h5", ".hdf5", ".tif", ".tiff"):
                for f in d.rglob(f"*{ext}"):
                    return f
    return None


class LazyVolDataset(Dataset):
    """Memory-efficient dataset that reads 3D patches on-demand from disk.

    Instead of loading entire volumes into RAM, this dataset stores only
    file paths and volume shapes (~bytes per volume). Each ``__getitem__``
    reads a random patch directly from the HDF5/TIFF file using sliced
    I/O, materializing only the patch_size crop in memory.

    Memory usage: O(num_volumes × metadata) ≈ negligible
    vs. CacheDataset: O(num_volumes × volume_size) ≈ GBs

    Args:
        root_dir: Root directory containing data files.
        volumes: List of ``{vol, seg}`` dicts specifying volume names.
        patch_size: Spatial crop size ``(D, H, W)`` for random patches.
        transform: MONAI transform pipeline to apply to each patch.
        num_samples: Virtual epoch length (random patches per epoch).
        normalize: Whether to normalize images to [0, 1] using per-volume
            min/max (pre-computed from metadata, not full load).
    """

    def __init__(
        self,
        root_dir: str,
        volumes: List[Dict[str, str]],
        patch_size: Tuple[int, int, int],
        transform: Optional[Callable] = None,
        num_samples: int = 16000,
        normalize: bool = True,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.transform = transform
        self.deterministic = deterministic
        self.num_samples = num_samples
        self.normalize = normalize

        self._handles: List[_VolumeHandle] = []
        self._cum_voxels: List[int] = []
        self._norm_params: Dict[str, Tuple[float, float]] = {}

        self._discover_volumes(volumes)

        if not self._handles:
            raise ValueError(f"No volumes found in {root_dir}")

        total = sum(np.prod(h.shape) for h in self._handles)
        logger.info(
            "LazyVolDataset: %d volumes, %s total voxels, "
            "patch_size=%s, num_samples=%d, ~%.1f MB metadata",
            len(self._handles), f"{total:,}", patch_size, num_samples,
            len(self._handles) * 0.001,
        )

    def _discover_volumes(self, volumes: List[Dict[str, str]]) -> None:
        """Scan volume files and store metadata only — no data loaded."""
        cumulative = 0
        for vol_spec in volumes:
            vol_root = Path(vol_spec.get("root", str(self.root_dir)))
            vol_name = vol_spec["vol"]

            img_path = _find_file(vol_root, vol_name)
            if img_path is None:
                logger.warning("Volume image not found: %s in %s", vol_name, vol_root)
                continue

            img_key = _resolve_hdf5_key(img_path) if img_path.suffix.lower() in (".h5", ".hdf5") else None
            shape = _get_shape(img_path, img_key)

            seg_name = vol_spec.get("seg")
            label_path = None
            label_key = None
            if seg_name:
                label_path = _find_file(vol_root, seg_name)
                if label_path is not None and label_path.suffix.lower() in (".h5", ".hdf5"):
                    label_key = _resolve_hdf5_key(label_path)

            handle = _VolumeHandle(
                image_path=img_path,
                label_path=label_path,
                shape=shape,
                name=vol_name,
                image_key=img_key,
                label_key=label_key,
                find_boundaries=float(vol_spec.get("find_boundaries", 0)),
            )
            self._handles.append(handle)
            cumulative += int(np.prod(shape))
            self._cum_voxels.append(cumulative)

        if self.normalize and self._handles:
            self._compute_norm_params()

    def _compute_norm_params(self) -> None:
        """Compute per-volume min/max by sampling a few slices — not full load."""
        for h in self._handles:
            spatial = self._spatial_shape(h)
            z_dim = spatial[0]
            sample_indices = [0, z_dim // 4, z_dim // 2, 3 * z_dim // 4, z_dim - 1]
            sample_indices = sorted(set(max(0, min(i, z_dim - 1)) for i in sample_indices))

            has_channel = len(h.shape) > len(spatial)
            vmin, vmax = float("inf"), float("-inf")
            for zi in sample_indices:
                sl = (slice(zi, zi + 1),) + tuple(slice(None) for _ in spatial[1:])
                if has_channel:
                    sl = (slice(None),) + sl
                patch = _read_patch(h.image_path, sl, h.image_key, dtype=np.float32)
                vmin = min(vmin, float(patch.min()))
                vmax = max(vmax, float(patch.max()))

            self._norm_params[h.name] = (vmin, vmax)
            logger.debug("Norm params for %s: min=%.4f, max=%.4f", h.name, vmin, vmax)

    def __len__(self) -> int:
        return self.num_samples

    def _pick_volume(self, index: int) -> _VolumeHandle:
        """Select a volume, weighted by total voxel count."""
        rng = np.random.RandomState(index)
        total = self._cum_voxels[-1]
        r = rng.randint(0, total)
        for i, cum in enumerate(self._cum_voxels):
            if r < cum:
                return self._handles[i]
        return self._handles[-1]

    def _random_patch_slices(
        self, shape: Tuple[int, ...], rng: np.random.RandomState,
    ) -> Tuple[slice, ...]:
        """Generate random crop slices that fit within the volume."""
        slices = []
        for dim_size, patch_dim in zip(shape, self.patch_size):
            max_start = max(0, dim_size - patch_dim)
            start = rng.randint(0, max_start + 1)
            slices.append(slice(start, start + patch_dim))
        return tuple(slices)

    def _spatial_shape(self, handle: _VolumeHandle) -> Tuple[int, ...]:
        """Strip leading channel dim if present, returning only spatial dims."""
        shape = handle.shape
        if len(shape) == len(self.patch_size) + 1:
            return shape[1:]
        return shape

    def __getitem__(self, index: int) -> Dict[str, Any]:
        seed = index if self.deterministic else index + int(torch.randint(0, 2**31, (1,)).item())
        rng = np.random.RandomState(seed)
        handle = self._pick_volume(index)

        spatial = self._spatial_shape(handle)
        crop_slices = self._random_patch_slices(spatial, rng)
        if len(handle.shape) > len(spatial):
            crop_slices = (slice(None),) + crop_slices

        image = _read_patch(
            handle.image_path, crop_slices, handle.image_key, dtype=np.float32,
        )

        if self.normalize and handle.name in self._norm_params:
            vmin, vmax = self._norm_params[handle.name]
            if vmax > vmin:
                image = (image - vmin) / (vmax - vmin)

        sample: Dict[str, Any] = {
            "image": image,
            "volume": handle.name,
        }

        if handle.label_path is not None:
            label = _read_patch(
                handle.label_path, crop_slices, handle.label_key, dtype=np.int64,
            )
            if handle.find_boundaries > 0 and rng.random() < handle.find_boundaries:
                from neurons.transforms.find_boundaries import find_boundaries
                label[find_boundaries(label, mode="inner")] = 0
            sample["label"] = label

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

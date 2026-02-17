#!/usr/bin/env python
"""
Download a representative sub-volume of the MICrONS minnie65 dataset.

Downloads:
- EM imagery (proofread minnie65)
- Static segmentation (multiple versions available)

Available segmentation versions:
  117  -- June 11, 2021   (first proofread release)
  343  -- February 22, 2022
  943  -- January 22, 2024
  1300 -- January 13, 2025  (latest, DEFAULT)

Uses cloud-volume to fetch from AWS / Google Cloud public buckets.

Usage:
    # Default: EM + seg v1300, 128^3 crop
    python scripts/download_microns.py

    # Custom size and version
    python scripts/download_microns.py --size 1024 1024 1024 --seg-version 1300

    # Download multiple segmentation versions
    python scripts/download_microns.py --seg-version 117 943 1300

    # All four versions
    python scripts/download_microns.py --seg-version all
"""

import argparse
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Cloud paths
# ---------------------------------------------------------------------------
EM_PATH = (
    "precomputed://https://bossdb-open-data.s3.amazonaws.com"
    "/iarpa_microns/minnie/minnie65/em"
)

SEG_VERSIONS: Dict[int, str] = {
    117: (
        "precomputed://https://bossdb-open-data.s3.amazonaws.com"
        "/iarpa_microns/minnie/minnie65/seg"
    ),
    343: (
        "precomputed://https://storage.googleapis.com"
        "/iarpa_microns/minnie/minnie65/seg_m343/"
    ),
    943: (
        "precomputed://https://storage.googleapis.com"
        "/iarpa_microns/minnie/minnie65/seg_m943/"
    ),
    1300: (
        "precomputed://https://storage.googleapis.com"
        "/iarpa_microns/minnie/minnie65/seg_m1300/"
    ),
}

DEFAULT_SEG_VERSION = 1300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def download_subvolume(
    cloud_path: str,
    bbox_start: tuple,
    bbox_size: tuple,
    mip: int = 0,
) -> np.ndarray:
    """
    Download a sub-volume from a cloud-volume precomputed source.

    Args:
        cloud_path: Precomputed cloud path (s3 or gs).
        bbox_start: (x, y, z) start coordinates.
        bbox_size: (x, y, z) size of the crop.
        mip: Resolution level (0 = full resolution).

    Returns:
        Numpy array of shape (Z, Y, X).
    """
    from cloudvolume import CloudVolume

    vol = CloudVolume(cloud_path, mip=mip, use_https=True, fill_missing=True,
                      bounded=False)

    x0, y0, z0 = bbox_start
    sx, sy, sz = bbox_size
    data = vol[x0 : x0 + sx, y0 : y0 + sy, z0 : z0 + sz]

    # cloud-volume returns (X, Y, Z, C) -- transpose to (Z, Y, X)
    arr = np.squeeze(data)
    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 1, 0))
    return arr


def save_h5(arr: np.ndarray, path: Path) -> None:
    """Save array to gzip-compressed HDF5."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("main", data=arr, compression="gzip")


def make_name(prefix: str, mip: int, crop_size: int, suffix: str = "") -> str:
    """Build standardised file name."""
    base = f"minnie65_mip{mip}_crop_{crop_size}"
    if suffix:
        base = f"{base}_{suffix}"
    return f"{base}_{prefix}.h5"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MICrONS minnie65 subvolume (EM + segmentation)",
    )
    parser.add_argument(
        "--output", type=str, default="/scratch/MICRONS",
        help="Output directory (default: /scratch/MICRONS)",
    )
    parser.add_argument(
        "--size", type=int, nargs=3, default=[128, 128, 128],
        help="Crop size in X Y Z (default: 128 128 128)",
    )
    parser.add_argument(
        "--start", type=int, nargs=3, default=[140000, 100000, 20000],
        help="Start coordinates in X Y Z (default: 140000 100000 20000)",
    )
    parser.add_argument(
        "--mip", type=int, default=0,
        help="Resolution level, 0 = full res (default: 0)",
    )
    parser.add_argument(
        "--seg-version", type=str, nargs="+", default=[str(DEFAULT_SEG_VERSION)],
        help=(
            "Segmentation version(s) to download. "
            "Options: 117, 343, 943, 1300, all. "
            f"Default: {DEFAULT_SEG_VERSION}"
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox_start = tuple(args.start)
    bbox_size = tuple(args.size)
    crop_size = bbox_size[0]  # used in file naming
    mip = args.mip

    # Resolve seg versions
    if "all" in args.seg_version:
        versions: List[int] = sorted(SEG_VERSIONS.keys())
    else:
        versions = sorted(int(v) for v in args.seg_version)

    for v in versions:
        if v not in SEG_VERSIONS:
            print(f"ERROR: Unknown seg version {v}. Available: {list(SEG_VERSIONS.keys())}")
            return

    print("=" * 60)
    print("MICrONS minnie65 Download")
    print("=" * 60)
    print(f"  Output      : {out_dir}")
    print(f"  Crop start  : {bbox_start}")
    print(f"  Crop size   : {bbox_size}")
    print(f"  Mip level   : {mip}")
    print(f"  Seg versions: {versions}")
    print()

    # -- EM imagery --
    em_file = out_dir / make_name("volume", mip, crop_size)
    if em_file.exists():
        print(f"EM imagery: SKIP (already exists: {em_file.name})")
    else:
        print(f"Downloading EM imagery ...")
        print(f"  source: {EM_PATH}")
        em_vol = download_subvolume(EM_PATH, bbox_start, bbox_size, mip=mip)
        print(f"  shape : {em_vol.shape}  dtype={em_vol.dtype}")
        save_h5(em_vol, em_file)
        print(f"  saved : {em_file}")
    print()

    # -- Segmentation versions --
    for ver in versions:
        seg_cloud = SEG_VERSIONS[ver]
        suffix = f"v{ver}"
        seg_file = out_dir / make_name("segmentation", mip, crop_size, suffix)

        if seg_file.exists():
            print(f"Seg v{ver}: SKIP (already exists: {seg_file.name})")
            continue

        print(f"Downloading segmentation v{ver} ...")
        print(f"  source: {seg_cloud}")
        seg_vol = download_subvolume(seg_cloud, bbox_start, bbox_size, mip=mip)
        print(f"  shape : {seg_vol.shape}  dtype={seg_vol.dtype}")
        n_ids = len(np.unique(seg_vol))
        print(f"  unique: {n_ids} segment IDs")
        save_h5(seg_vol, seg_file)
        print(f"  saved : {seg_file}")
        print()

    # -- Summary --
    print("=" * 60)
    print("Download complete!")
    print(f"  Output directory: {out_dir}")
    print()
    print("  Files:")
    for f in sorted(out_dir.glob(f"minnie65_mip{mip}_crop_{crop_size}*.h5")):
        size_mb = f.stat().st_size / 1e6
        print(f"    {f.name}  ({size_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()

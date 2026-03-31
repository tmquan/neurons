#!/usr/bin/env python
"""
Download representative sub-volumes of the MICrONS minnie65 dataset.

Downloads:
- EM imagery (proofread minnie65)
- Static segmentation (multiple versions available)

MICrONS minnie65 dataset information
-------------------------------------
Tissue:     Mouse primary visual cortex (V1), layers 2/3 – 5
Physical:   ~1.4 mm × 0.87 mm × 0.84 mm (~1 mm³)
Cells:      ~200,000 total cells, ~120,000 neurons
Synapses:   >523 million detected

EM imagery (mip0):
  Resolution:   8 × 8 × 40 nm (anisotropic, XYZ)
  Volume size:  ~175,104 × 108,544 × 21,056 voxels (XYZ, approximate)
  Data size:    ~117 TB (precomputed format)
  Mip levels:   [8,8,40], [16,16,40], [32,32,40], [64,64,40],
                [128,128,80], [256,256,160], [512,512,320],
                [1024,1024,640], [2048,2048,1280] nm

Segmentation versions:
  v117  -- June 11, 2021   (first proofread, ~12 TB)
  v343  -- February 22, 2022
  v943  -- January 22, 2024
  v1300 -- January 13, 2025  (latest, DEFAULT)

Crop size estimates (mip0, uint8 EM + uint64 seg, uncompressed):
  128³  =   2 MB EM +  16 MB seg  =   18 MB total
  256³  =  16 MB EM + 128 MB seg  =  144 MB total
  512³  = 128 MB EM +   1 GB seg  =  1.1 GB total
  1024³ =   1 GB EM +   8 GB seg  =    9 GB total
  2048³ =   8 GB EM +  64 GB seg  =   72 GB total
  4096³ =  64 GB EM + 512 GB seg  =  576 GB total

Pre-defined splits (12 total: 10 train + 2 test, all 1024x1024x2048):
  Arranged in 5-col x 2-row XY grid (train) + 2 gap positions (test).
  Z varies across crops for cortical-layer diversity (Z=16500..24500).
  All disjoint, >=26K margin from seg X/Y edges, >=1300 from Z edges.

  train01: ( 53000,  67000, 16500)    train06: ( 53000, 122000, 21000)
  train02: ( 87000,  67000, 17000)    train07: ( 87000, 122000, 21500)
  train03: (122000,  67000, 18000)    train08: (122000, 122000, 22500)
  train04: (156000,  67000, 19000)    train09: (156000, 122000, 23500)
  train05: (190000, 122000, 24500)
  test01:  ( 70000,  95000, 20000)    test02:  (139000,  95000, 21000)

File naming encodes coordinates:
  minnie65_mip0_1024x1024x2048_x53000_y67000_z16500_volume.h5
  minnie65_mip0_1024x1024x2048_x53000_y67000_z16500_v1300_segmentation.h5

Uses cloud-volume to fetch from AWS / Google Cloud public buckets.

Usage:
    # Default: EM + seg v1300, 128^3 crop
    python scripts/download_microns.py

    # Custom size and version
    python scripts/download_microns.py --size 1024 1024 2048 --seg-version 1300

    # Download all 11 pre-defined splits (9 train + 2 test)
    python scripts/download_microns.py --split

    # All four segmentation versions
    python scripts/download_microns.py --seg-version all
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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

# Pre-defined splits in disjoint regions of the minnie65 volume.
# Coordinates are (X, Y, Z) in mip0 voxels.  All 1024 x 1024 x 2048.
#
# 9 train crops + 2 test crops, all 1024x1024x2048, disjoint.
# Z varies across crops for cortical-layer diversity (16500..24500).
SPLITS: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "train01": {"start": (53000, 67000, 16500), "size": (1024, 1024, 2048)},
    "train02": {"start": (87000, 67000, 17000), "size": (1024, 1024, 2048)},
    "train03": {"start": (122000, 67000, 18000), "size": (1024, 1024, 2048)},
    "train04": {"start": (156000, 67000, 19000), "size": (1024, 1024, 2048)},
    "train05": {"start": (190000, 122000, 24500), "size": (1024, 1024, 2048)},
    "train06": {"start": (53000, 122000, 21000), "size": (1024, 1024, 2048)},
    "train07": {"start": (87000, 122000, 21500), "size": (1024, 1024, 2048)},
    "train08": {"start": (122000, 122000, 22500), "size": (1024, 1024, 2048)},
    "train09": {"start": (156000, 122000, 23500), "size": (1024, 1024, 2048)},
    "test01": {"start": (70000, 95000, 20000), "size": (1024, 1024, 2048)},
    "test02": {"start": (139000, 95000, 21000), "size": (1024, 1024, 2048)},
}

TRAIN_SPLITS = [k for k in SPLITS if k.startswith("train")]
TEST_SPLITS = [k for k in SPLITS if k.startswith("test")]


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


def make_name(
    prefix: str,
    mip: int,
    crop_size: Tuple[int, int, int],
    start: Tuple[int, int, int],
    seg_version: int = 0,
) -> str:
    """Build coordinate-based file name.

    Examples:
        make_name("volume", 0, (1024,1024,2048), (53000, 67000, 16500))
        -> "minnie65_mip0_1024x1024x2048_x53000_y67000_z16500_volume.h5"

        make_name("segmentation", 0, (1024,1024,2048), (53000, 67000, 16500), seg_version=1300)
        -> "minnie65_mip0_1024x1024x2048_x53000_y67000_z16500_v1300_segmentation.h5"
    """
    x, y, z = start
    sx, sy, sz = crop_size
    size_tag = str(sx) if sx == sy == sz else f"{sx}x{sy}x{sz}"
    base = f"minnie65_mip{mip}_{size_tag}_x{x}_y{y}_z{z}"
    if seg_version:
        base = f"{base}_v{seg_version}"
    return f"{base}_{prefix}.h5"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MICrONS minnie65 subvolume (EM + segmentation)",
    )
    parser.add_argument(
        "--output", type=str, default="data/MICRONS",
        help="Output directory (default: data/MICRONS)",
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
    parser.add_argument(
        "--split", action="store_true",
        help=(
            "Download all 11 pre-defined splits (9 train + 2 test) from "
            "disjoint 1024x1024x2048 regions. Ignores --size and --start when set."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
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

    # Build list of crops to download
    if args.split:
        crops = [
            (name, sp["start"], sp["size"])
            for name, sp in SPLITS.items()
        ]
    else:
        crops = [
            ("custom", tuple(args.start), tuple(args.size)),
        ]

    print("=" * 60)
    print("MICrONS minnie65 Download")
    print("=" * 60)
    print(f"  Output      : {out_dir}")
    print(f"  Mip level   : {mip}")
    print(f"  Resolution  : 8 x 8 x 40 nm (mip0)")
    print(f"  Seg versions: {versions}")

    total_gb_all = 0.0
    for label, start, size in crops:
        sx, sy, sz = size
        em_gb = (sx * sy * sz) / 1e9
        seg_gb = (sx * sy * sz * 8) / 1e9
        crop_gb = em_gb + seg_gb * len(versions)
        total_gb_all += crop_gb
        print(f"  {label:8s}: start={start}  size={size}  ~{crop_gb:.1f} GB")

    print(f"  Total est.  : {total_gb_all:.1f} GB (uncompressed)")
    print()

    for label, bbox_start, bbox_size in crops:
        print(f"--- {label} ---")
        print(f"  start: {bbox_start}  size: {bbox_size}")
        print()

        # -- EM imagery --
        em_file = out_dir / make_name("volume", mip, bbox_size, bbox_start)
        if em_file.exists():
            print(f"  EM: SKIP (already exists: {em_file.name})")
        else:
            print(f"  Downloading EM imagery ...")
            print(f"    source: {EM_PATH}")
            em_vol = download_subvolume(EM_PATH, bbox_start, bbox_size, mip=mip)
            print(f"    shape : {em_vol.shape}  dtype={em_vol.dtype}")
            save_h5(em_vol, em_file)
            print(f"    saved : {em_file}")
        print()

        # -- Segmentation versions --
        for ver in versions:
            seg_cloud = SEG_VERSIONS[ver]
            seg_file = out_dir / make_name("segmentation", mip, bbox_size, bbox_start, seg_version=ver)

            if seg_file.exists():
                print(f"  Seg v{ver}: SKIP (already exists: {seg_file.name})")
                continue

            print(f"  Downloading segmentation v{ver} ...")
            print(f"    source: {seg_cloud}")
            seg_vol = download_subvolume(seg_cloud, bbox_start, bbox_size, mip=mip)
            print(f"    shape : {seg_vol.shape}  dtype={seg_vol.dtype}")
            n_ids = len(np.unique(seg_vol))
            print(f"    unique: {n_ids} segment IDs")
            save_h5(seg_vol, seg_file)
            print(f"    saved : {seg_file}")
            print()

    # -- Summary --
    print("=" * 60)
    print("Download complete!")
    print(f"  Output directory: {out_dir}")
    print()
    print("  Files:")
    for f in sorted(out_dir.glob("minnie65_*.h5")):
        size_mb = f.stat().st_size / 1e6
        print(f"    {f.name}  ({size_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()

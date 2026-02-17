#!/usr/bin/env python
"""
Download MICrONS minnie65 crops at multiple mip levels (mip0 through mip2).

For each mip level the crop coordinates and size are scaled so the same
physical region of tissue is captured at every resolution.  Only mip levels
where **both** EM and segmentation are available are kept.

Naming convention:
    minnie65_mip{M}_crop_{S}_volume.h5
    minnie65_mip{M}_crop_{S}_v{V}_segmentation.h5

where M = mip level, S = XY crop size at that mip, V = seg version.

Usage:
    # Default: mip 0-2, seg v1300
    python scripts/download_microns_multimip.py

    # Specific mips
    python scripts/download_microns_multimip.py --mips 0 1

    # Multiple seg versions
    python scripts/download_microns_multimip.py --seg-version 117 1300
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

# Mip 0-2: XY halves per step, Z stays the same -> seg is always valid.
# Mip 3+: Z also downsamples and our crop origin falls outside the seg
# volume bounds, so we restrict to 0-2 by default.
MIP_SCALE: Dict[int, Tuple[int, int]] = {
    0: (1, 1),
    1: (2, 1),
    2: (4, 1),
}

VALID_MIPS = sorted(MIP_SCALE.keys())


def scale_start(
    start_mip0: Tuple[int, int, int],
    mip: int,
) -> Tuple[int, int, int]:
    """Scale only the start coordinate from mip0 to target mip.

    The crop *size* stays the same (always 1024^3 voxels at every mip)
    so that each mip level covers a progressively larger physical region
    while keeping the tensor dimensions identical.
    """
    xy_s, z_s = MIP_SCALE[mip]
    x0, y0, z0 = start_mip0
    return (x0 // xy_s, y0 // xy_s, z0 // z_s)


def download_subvolume(
    cloud_path: str,
    bbox_start: Tuple[int, int, int],
    bbox_size: Tuple[int, int, int],
    mip: int = 0,
) -> np.ndarray:
    """Download sub-volume, clamping to valid bounds. Returns (Z, Y, X)."""
    from cloudvolume import CloudVolume

    vol = CloudVolume(cloud_path, mip=mip, use_https=True, fill_missing=True,
                      bounded=False)

    x0, y0, z0 = bbox_start
    sx, sy, sz = bbox_size

    data = vol[x0 : x0 + sx, y0 : y0 + sy, z0 : z0 + sz]
    arr = np.squeeze(data)
    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 1, 0))
    return arr


def save_h5(arr: np.ndarray, path: Path) -> None:
    """Save array to gzip-compressed HDF5."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("main", data=arr, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MICrONS minnie65 crops at mip 0-2 (EM + seg)",
    )
    parser.add_argument("--output", type=str, default="/tmp/MICRONS_download")
    parser.add_argument(
        "--mips", type=int, nargs="+", default=VALID_MIPS,
        help=f"Mip levels to download (default: {VALID_MIPS})",
    )
    parser.add_argument(
        "--crop-size", type=int, nargs=3, default=[1024, 1024, 1024],
        help="Crop size at mip0 in X Y Z (default: 1024 1024 1024)",
    )
    parser.add_argument(
        "--start", type=int, nargs=3, default=[140000, 100000, 20000],
        help="Start coords at mip0 in X Y Z",
    )
    parser.add_argument(
        "--seg-version", type=str, nargs="+", default=["1300"],
        help="Seg version(s): 117, 343, 943, 1300, all (default: 1300)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_mip0 = tuple(args.start)
    size_mip0 = tuple(args.crop_size)

    if "all" in args.seg_version:
        versions: List[int] = sorted(SEG_VERSIONS.keys())
    else:
        versions = sorted(int(v) for v in args.seg_version)

    mips = sorted(m for m in args.mips if m in MIP_SCALE)
    skipped = [m for m in args.mips if m not in MIP_SCALE]
    if skipped:
        print(f"NOTE: mip {skipped} skipped (no seg coverage). Valid: {VALID_MIPS}")

    print("=" * 70)
    print("MICrONS minnie65 Multi-Mip Download")
    print("=" * 70)
    print(f"  Output       : {out_dir}")
    print(f"  Mip0 start   : {start_mip0}")
    print(f"  Mip0 crop    : {size_mip0}")
    print(f"  Mip levels   : {mips}")
    print(f"  Seg versions : {versions}")
    print()

    crop_size = tuple(args.crop_size)
    crop_label = str(crop_size[0])  # e.g. "1024"

    for mip in mips:
        start_scaled = scale_start(start_mip0, mip)

        print(f"--- mip {mip}  crop={crop_size}  start={start_scaled} ---")

        # EM
        em_name = f"minnie65_mip{mip}_crop_{crop_label}_volume.h5"
        em_file = out_dir / em_name
        if em_file.exists():
            print(f"  EM: SKIP ({em_name})")
        else:
            print(f"  EM: downloading ...")
            em_vol = download_subvolume(EM_PATH, start_scaled, crop_size, mip=mip)
            print(f"       shape={em_vol.shape}  dtype={em_vol.dtype}")
            save_h5(em_vol, em_file)
            print(f"       saved {em_name} ({em_file.stat().st_size / 1e6:.1f} MB)")

        # Segmentation
        for ver in versions:
            seg_name = f"minnie65_mip{mip}_crop_{crop_label}_v{ver}_segmentation.h5"
            seg_file = out_dir / seg_name
            if seg_file.exists():
                print(f"  Seg v{ver}: SKIP ({seg_name})")
                continue

            print(f"  Seg v{ver}: downloading ...")
            seg_vol = download_subvolume(SEG_VERSIONS[ver], start_scaled, crop_size, mip=mip)
            n_ids = len(np.unique(seg_vol))
            print(f"       shape={seg_vol.shape}  dtype={seg_vol.dtype}  ids={n_ids}")
            save_h5(seg_vol, seg_file)
            print(f"       saved {seg_name} ({seg_file.stat().st_size / 1e6:.1f} MB)")

        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    for f in sorted(out_dir.glob("minnie65_*.h5")):
        print(f"  {f.name:60s} {f.stat().st_size / 1e6:>8.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Download the CREMI3D dataset.

The CREMI (Circuit Reconstruction from Electron Microscopy Images) challenge
provides 3 EM volumes (A, B, C) with neuron and synapse annotations.

Volumes:
- sample_A, sample_B: Training (125 slices, 1250x1250, 4x4x40 nm)
- sample_C: Test (125 slices, 1250x1250, 4x4x40 nm)

Source: https://cremi.org/

Two variants per sample:
- *_20160501.hdf: Original challenge data
- *+_20160601.hdf: Padded/realigned version

Usage:
    # Download from source
    python scripts/download_cremi3d.py --output data/cremi3d

    # Symlink from existing /scratch location
    python scripts/download_cremi3d.py --output data/cremi3d --link /scratch/CREMI3D
"""

import argparse
import os
import urllib.request
from pathlib import Path


CREMI_BASE_URL = "https://cremi.org/static/data"

# Original challenge files
DOWNLOADS = {
    "sample_A_20160501.hdf": f"{CREMI_BASE_URL}/sample_A_20160501.hdf",
    "sample_B_20160501.hdf": f"{CREMI_BASE_URL}/sample_B_20160501.hdf",
    "sample_C_20160501.hdf": f"{CREMI_BASE_URL}/sample_C_20160501.hdf",
    "sample_A+_20160601.hdf": f"{CREMI_BASE_URL}/sample_A+_20160601.hdf",
    "sample_B+_20160601.hdf": f"{CREMI_BASE_URL}/sample_B+_20160601.hdf",
    "sample_C+_20160601.hdf": f"{CREMI_BASE_URL}/sample_C+_20160601.hdf",
}

EXPECTED_FILES = list(DOWNLOADS.keys())


def link_from_existing(output_dir: Path, source_dir: Path) -> None:
    """Create symlinks from an existing data directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for fname in EXPECTED_FILES:
        src = source_dir / fname
        dst = output_dir / fname

        if dst.exists() or dst.is_symlink():
            try:
                dst.unlink()
            except PermissionError:
                print(f"  SKIP {fname} (already linked, cannot overwrite)")
                continue

        if src.exists():
            os.symlink(src, dst)
            print(f"  {fname} -> {src}")
        else:
            print(f"  WARNING: {fname} not found in {source_dir}")


def _download_file(url: str, dest: Path) -> None:
    """Download a file with progress."""
    if dest.exists():
        print(f"  SKIP {dest.name} (already exists)")
        return

    print(f"  Downloading {dest.name} ...")
    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / 1e6
        print(f"  OK   {dest.name} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  FAIL {dest.name}: {e}")


def download_from_source(output_dir: Path) -> None:
    """Download CREMI HDF5 files from cremi.org."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for fname, url in DOWNLOADS.items():
        _download_file(url, output_dir / fname)


def verify(output_dir: Path) -> None:
    """Verify downloaded files."""
    print("\nVerification:")
    found = 0
    for fname in EXPECTED_FILES:
        p = output_dir / fname
        if p.exists() or p.is_symlink():
            real = p.resolve()
            size = real.stat().st_size / 1e6 if real.exists() else 0
            print(f"  OK  {fname} ({size:.1f} MB)")
            found += 1
        else:
            print(f"  MISSING  {fname}")

    print(f"\n{found}/{len(EXPECTED_FILES)} files available.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CREMI3D dataset")
    parser.add_argument(
        "--output", type=str, default="data/cremi3d",
        help="Output directory (default: data/cremi3d)",
    )
    parser.add_argument(
        "--link", type=str, default=None,
        help="Symlink from existing directory instead of downloading",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    print("=" * 50)
    print("CREMI3D Dataset Download")
    print("=" * 50)
    print(f"Output: {output_dir}")

    if args.link:
        source = Path(args.link)
        print(f"Linking from: {source}\n")
        link_from_existing(output_dir, source)
    else:
        print(f"Downloading from: {CREMI_BASE_URL}\n")
        download_from_source(output_dir)

    verify(output_dir)
    print("=" * 50)


if __name__ == "__main__":
    main()

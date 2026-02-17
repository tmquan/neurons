#!/usr/bin/env python
"""
Download the SNEMI3D dataset.

The SNEMI3D challenge provides serial-section EM images of mouse cortex
with dense neuron segmentation labels.

Volumes:
- AC3: Test volume  (100 slices, 1024x1024, 6x6x30 nm)
- AC4: Train volume (100 slices, 1024x1024, 6x6x30 nm)

Source: https://github.com/tmquan/SNEMI3D (mirrored HDF5 + TIFF)

Usage:
    # Download from source
    python scripts/download_snemi3d.py --output data/snemi3d

    # Symlink from existing /scratch location
    python scripts/download_snemi3d.py --output data/snemi3d --link /scratch/SNEMI3D
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path


SNEMI3D_REPO = "https://github.com/tmquan/SNEMI3D.git"

EXPECTED_FILES = [
    "AC3_inputs.h5",
    "AC3_labels.h5",
    "AC4_inputs.h5",
    "AC4_labels.h5",
]


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
            # Try .tiff variant
            tiff_name = fname.replace(".h5", ".tiff")
            src_tiff = source_dir / tiff_name
            if src_tiff.exists():
                os.symlink(src_tiff, output_dir / tiff_name)
                print(f"  {tiff_name} -> {src_tiff}")
            else:
                print(f"  WARNING: {fname} not found in {source_dir}")


def download_from_source(output_dir: Path) -> None:
    """Clone the SNEMI3D repository (contains HDF5 files via Git LFS)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_clone = output_dir / "_clone_tmp"

    print(f"  Cloning {SNEMI3D_REPO} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", SNEMI3D_REPO, str(tmp_clone)],
        check=True,
    )

    # Copy data files to output
    for fname in EXPECTED_FILES:
        src = tmp_clone / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"  Copied {fname} ({(output_dir / fname).stat().st_size / 1e6:.1f} MB)")

    # Also look for TIFF variants
    for f in tmp_clone.glob("AC*_inputs.tiff"):
        shutil.copy2(f, output_dir / f.name)
        print(f"  Copied {f.name}")
    for f in tmp_clone.glob("AC*_labels.tiff"):
        shutil.copy2(f, output_dir / f.name)
        print(f"  Copied {f.name}")

    # Cleanup
    shutil.rmtree(tmp_clone, ignore_errors=True)


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
            tiff = output_dir / fname.replace(".h5", ".tiff")
            if tiff.exists() or tiff.is_symlink():
                real = tiff.resolve()
                size = real.stat().st_size / 1e6 if real.exists() else 0
                print(f"  OK  {tiff.name} ({size:.1f} MB)")
                found += 1
            else:
                print(f"  MISSING  {fname}")

    print(f"\n{found}/{len(EXPECTED_FILES)} files available.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SNEMI3D dataset")
    parser.add_argument(
        "--output", type=str, default="data/snemi3d",
        help="Output directory (default: data/snemi3d)",
    )
    parser.add_argument(
        "--link", type=str, default=None,
        help="Symlink from existing directory instead of downloading",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    print("=" * 50)
    print("SNEMI3D Dataset Download")
    print("=" * 50)
    print(f"Output: {output_dir}")

    if args.link:
        source = Path(args.link)
        print(f"Linking from: {source}\n")
        link_from_existing(output_dir, source)
    else:
        print(f"Downloading from: {SNEMI3D_REPO}\n")
        download_from_source(output_dir)

    verify(output_dir)
    print("=" * 50)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Download the MitoEM2 dataset.

MitoEM2 provides 8 EM datasets for mitochondria segmentation from
different cell types. Data is in NIfTI format (nnU-Net convention).

Datasets:
- Dataset001_ME2-Beta     (beta cells)
- Dataset002_ME2-Jurkat   (Jurkat cells)
- Dataset003_ME2-Macro    (macrophages)
- Dataset004_ME2-Mossy    (mossy fibers)
- Dataset005_ME2-Podo     (podocytes)
- Dataset006_ME2-Pyra     (pyramidal neurons)
- Dataset007_ME2-Sperm    (sperm cells)
- Dataset008_ME2-Stem     (stem cells)

Source: https://mitoem.grand-challenge.org/

Usage:
    # Symlink all datasets from /scratch
    python scripts/download_mitoem2.py --output data/mitoem2 --link /scratch/MitoEM2

    # Symlink specific datasets
    python scripts/download_mitoem2.py --output data/mitoem2 --link /scratch/MitoEM2 \\
        --datasets Dataset001_ME2-Beta Dataset006_ME2-Pyra

    # Download from Zenodo (if available)
    python scripts/download_mitoem2.py --output data/mitoem2
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import List, Optional


ALL_DATASETS = [
    "Dataset001_ME2-Beta",
    "Dataset002_ME2-Jurkat",
    "Dataset003_ME2-Macro",
    "Dataset004_ME2-Mossy",
    "Dataset005_ME2-Podo",
    "Dataset006_ME2-Pyra",
    "Dataset007_ME2-Sperm",
    "Dataset008_ME2-Stem",
]

# Zenodo download URLs for the zip archives
ZENODO_URLS = {
    "Dataset001_ME2-Beta":   "https://zenodo.org/records/14595790/files/Dataset001_ME2-Beta.zip",
    "Dataset002_ME2-Jurkat": "https://zenodo.org/records/14595790/files/Dataset002_ME2-Jurkat.zip",
    "Dataset003_ME2-Macro":  "https://zenodo.org/records/14595790/files/Dataset003_ME2-Macro.zip",
    "Dataset004_ME2-Mossy":  "https://zenodo.org/records/14595790/files/Dataset004_ME2-Mossy.zip",
    "Dataset005_ME2-Podo":   "https://zenodo.org/records/14595790/files/Dataset005_ME2-Podo.zip",
    "Dataset006_ME2-Pyra":   "https://zenodo.org/records/14595790/files/Dataset006_ME2-Pyra.zip",
    "Dataset007_ME2-Sperm":  "https://zenodo.org/records/14595790/files/Dataset007_ME2-Sperm.zip",
    "Dataset008_ME2-Stem":   "https://zenodo.org/records/14595790/files/Dataset008_ME2-Stem.zip",
}


def link_from_existing(
    output_dir: Path,
    source_dir: Path,
    datasets: List[str],
) -> None:
    """Create symlinks from an existing data directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in datasets:
        src = source_dir / ds_name
        dst = output_dir / ds_name

        if dst.exists() or dst.is_symlink():
            try:
                if dst.is_symlink():
                    dst.unlink()
                else:
                    print(f"  SKIP {ds_name} (already exists as real directory)")
                    continue
            except PermissionError:
                print(f"  SKIP {ds_name} (already linked, cannot overwrite)")
                continue

        if src.exists():
            os.symlink(src, dst)
            n_files = sum(1 for _ in src.rglob("*") if _.is_file())
            print(f"  {ds_name}/ -> {src}  ({n_files} files)")
        else:
            print(f"  WARNING: {ds_name} not found in {source_dir}")


def download_from_zenodo(
    output_dir: Path,
    datasets: List[str],
) -> None:
    """Download datasets from Zenodo and extract."""
    import zipfile

    output_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in datasets:
        url = ZENODO_URLS.get(ds_name)
        if url is None:
            print(f"  WARNING: No download URL for {ds_name}")
            continue

        ds_dir = output_dir / ds_name
        if ds_dir.exists():
            print(f"  SKIP {ds_name} (already exists)")
            continue

        zip_path = output_dir / f"{ds_name}.zip"

        print(f"  Downloading {ds_name} ...")
        try:
            subprocess.run(
                ["wget", "-q", "--show-progress", "-O", str(zip_path), url],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            import urllib.request
            print(f"  wget failed, trying urllib ...")
            urllib.request.urlretrieve(url, str(zip_path))

        size_mb = zip_path.stat().st_size / 1e6
        print(f"  Extracting {ds_name}.zip ({size_mb:.0f} MB) ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)

        zip_path.unlink()
        print(f"  OK  {ds_name}")


def verify(output_dir: Path, datasets: List[str]) -> None:
    """Verify downloaded datasets."""
    print("\nVerification:")
    found = 0
    for ds_name in datasets:
        ds_dir = output_dir / ds_name
        if ds_dir.exists() or ds_dir.is_symlink():
            real = ds_dir.resolve() if ds_dir.is_symlink() else ds_dir
            if real.exists():
                has_images = (real / "imagesTr").exists()
                has_labels = (real / "labelsTr").exists()
                n_images = len(list((real / "imagesTr").glob("*"))) if has_images else 0
                n_labels = len(list((real / "labelsTr").glob("*"))) if has_labels else 0
                status = "OK" if has_images and has_labels else "INCOMPLETE"
                print(f"  {status}  {ds_name}  (images={n_images}, labels={n_labels})")
                found += 1
            else:
                print(f"  BROKEN  {ds_name} (symlink target missing)")
        else:
            print(f"  MISSING  {ds_name}")

    print(f"\n{found}/{len(datasets)} datasets available.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MitoEM2 dataset")
    parser.add_argument(
        "--output", type=str, default="data/mitoem2",
        help="Output directory (default: data/mitoem2)",
    )
    parser.add_argument(
        "--link", type=str, default=None,
        help="Symlink from existing directory instead of downloading",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="*", default=None,
        help="Specific datasets to download (default: all 8)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    datasets = args.datasets if args.datasets else ALL_DATASETS

    print("=" * 50)
    print("MitoEM2 Dataset Download")
    print("=" * 50)
    print(f"Output: {output_dir}")
    print(f"Datasets: {len(datasets)}\n")

    if args.link:
        source = Path(args.link)
        print(f"Linking from: {source}\n")
        link_from_existing(output_dir, source, datasets)
    else:
        print("Downloading from Zenodo ...\n")
        download_from_zenodo(output_dir, datasets)

    verify(output_dir, datasets)
    print("=" * 50)


if __name__ == "__main__":
    main()

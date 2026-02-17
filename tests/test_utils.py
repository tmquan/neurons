"""
Tests for utility functions.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from neurons.utils.io import find_folder, load_volume, save_volume
from neurons.utils.labels import (
    compute_ari_point,
    compute_ami_point,
    compute_axi_point,
    compute_ari_batch,
    compute_ami_batch,
    compute_axi_batch,
    relabel_after_crop,
    relabel_sequential,
)


class TestFindPath:
    """Tests for find_folder utility."""

    def test_finds_h5_file(self, tmp_path: Path) -> None:
        """Test finding an HDF5 file."""
        import h5py

        h5_file = tmp_path / "volume.h5"
        with h5py.File(h5_file, "w") as f:
            f.create_dataset("main", data=np.zeros((10, 10)))

        result = find_folder(tmp_path, "volume")
        assert result is not None
        assert result.suffix == ".h5"

    def test_finds_tiff_file(self, tmp_path: Path) -> None:
        """Test finding a TIFF file."""
        import tifffile

        tiff_file = tmp_path / "image.tiff"
        tifffile.imwrite(str(tiff_file), np.zeros((10, 10), dtype=np.uint8))

        result = find_folder(tmp_path, "image")
        assert result is not None
        assert result.suffix == ".tiff"

    def test_returns_none_when_not_found(self, tmp_path: Path) -> None:
        """Test returning None when file not found."""
        result = find_folder(tmp_path, "nonexistent")
        assert result is None

    def test_priority_order(self, tmp_path: Path) -> None:
        """Test that h5 is preferred over tiff."""
        import h5py
        import tifffile

        h5_file = tmp_path / "data.h5"
        with h5py.File(h5_file, "w") as f:
            f.create_dataset("main", data=np.zeros((5,)))

        tiff_file = tmp_path / "data.tiff"
        tifffile.imwrite(str(tiff_file), np.zeros((5, 5), dtype=np.uint8))

        result = find_folder(tmp_path, "data")
        assert result is not None
        assert result.suffix == ".h5"


class TestLoadSaveVolume:
    """Tests for load_volume and save_volume."""

    def test_h5_roundtrip(self, tmp_path: Path) -> None:
        """Test saving and loading HDF5 volume."""
        data = np.random.rand(10, 32, 32).astype(np.float32)
        path = tmp_path / "volume.h5"

        save_volume(data, path)
        loaded = load_volume(path)

        np.testing.assert_array_almost_equal(data, loaded)

    def test_tiff_roundtrip(self, tmp_path: Path) -> None:
        """Test saving and loading TIFF volume."""
        data = np.random.randint(0, 255, (10, 32, 32), dtype=np.uint8)
        path = tmp_path / "volume.tiff"

        save_volume(data, path)
        loaded = load_volume(path)

        np.testing.assert_array_equal(data, loaded)

    def test_npy_roundtrip(self, tmp_path: Path) -> None:
        """Test saving and loading NPY volume."""
        data = np.random.rand(5, 16, 16).astype(np.float32)
        path = tmp_path / "volume.npy"

        save_volume(data, path)
        loaded = load_volume(path)

        np.testing.assert_array_almost_equal(data, loaded)

    def test_tensor_input(self, tmp_path: Path) -> None:
        """Test saving from torch tensor."""
        data = torch.randn(5, 16, 16)
        path = tmp_path / "volume.h5"

        save_volume(data, path)
        loaded = load_volume(path)

        np.testing.assert_array_almost_equal(data.numpy(), loaded, decimal=5)

    def test_nonexistent_raises(self) -> None:
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_volume("/nonexistent/path.h5")


class TestRelabelSequential:
    """Tests for relabel_sequential."""

    def test_basic_relabeling(self) -> None:
        """Test sequential relabeling."""
        labels = torch.tensor([0, 5, 0, 5, 12, 12, 0])
        result = relabel_sequential(labels)
        expected = torch.tensor([0, 1, 0, 1, 2, 2, 0])
        assert torch.equal(result, expected)

    def test_all_background(self) -> None:
        """Test with all background."""
        labels = torch.zeros(10, dtype=torch.long)
        result = relabel_sequential(labels)
        assert torch.equal(result, labels)

    def test_already_sequential(self) -> None:
        """Test with already sequential labels."""
        labels = torch.tensor([0, 1, 1, 2, 2, 3, 0])
        result = relabel_sequential(labels)
        assert torch.equal(result, labels)


class TestRelabelAfterCrop:
    """Tests for relabel_after_crop."""

    def test_2d_relabeling(self) -> None:
        """Test 2D connected component relabeling."""
        labels = torch.zeros(10, 10, dtype=torch.long)
        labels[0:3, 0:3] = 5
        labels[7:10, 7:10] = 5  # Same label, disconnected

        result = relabel_after_crop(labels, spatial_dims=2)
        unique = torch.unique(result)
        assert len(unique) == 3  # background + 2 components

    def test_3d_relabeling(self) -> None:
        """Test 3D connected component relabeling."""
        labels = torch.zeros(4, 10, 10, dtype=torch.long)
        labels[:, 0:3, 0:3] = 1
        labels[:, 7:10, 7:10] = 2

        result = relabel_after_crop(labels, spatial_dims=3)
        unique = torch.unique(result)
        assert len(unique) == 3  # background + 2 instances

    def test_batch_relabeling(self) -> None:
        """Test batch relabeling."""
        labels = torch.zeros(2, 10, 10, dtype=torch.long)
        labels[0, 0:5, 0:5] = 1
        labels[1, 5:10, 5:10] = 2

        result = relabel_after_crop(labels, spatial_dims=2)
        assert result.shape == labels.shape


class TestComputeMetricsPoint:
    """Tests for per-sample ARI/AMI/AXI computation."""

    def test_perfect_ari(self) -> None:
        """Test ARI with perfect segmentation."""
        labels = torch.tensor([[1, 1, 2, 2], [1, 1, 2, 2]])
        assert compute_ari_point(labels, labels) == pytest.approx(1.0, abs=1e-6)

    def test_perfect_ami(self) -> None:
        """Test AMI with perfect segmentation."""
        labels = torch.tensor([[1, 1, 2, 2], [1, 1, 2, 2]])
        assert compute_ami_point(labels, labels) == pytest.approx(1.0, abs=1e-6)

    def test_perfect_axi(self) -> None:
        """Test AXI with perfect segmentation."""
        labels = torch.tensor([[1, 1, 2, 2], [1, 1, 2, 2]])
        assert compute_axi_point(labels, labels) == pytest.approx(1.0, abs=1e-6)

    def test_all_background(self) -> None:
        """Test with all background."""
        pred = torch.zeros(4, 4, dtype=torch.long)
        true = torch.zeros(4, 4, dtype=torch.long)
        assert compute_ari_point(pred, true) == 0.0
        assert compute_ami_point(pred, true) == 0.0
        assert compute_axi_point(pred, true) == 0.0


class TestComputeMetricsBatch:
    """Tests for batch ARI/AMI/AXI computation."""

    def test_batch_ari(self) -> None:
        """Test batch ARI."""
        pred = torch.tensor([[[1, 1, 2, 2]], [[3, 3, 4, 4]]])
        true = torch.tensor([[[1, 1, 2, 2]], [[3, 3, 4, 4]]])
        assert compute_ari_batch(pred, true) > 0.0

    def test_batch_ami(self) -> None:
        """Test batch AMI."""
        pred = torch.tensor([[[1, 1, 2, 2]], [[3, 3, 4, 4]]])
        true = torch.tensor([[[1, 1, 2, 2]], [[3, 3, 4, 4]]])
        assert compute_ami_batch(pred, true) > 0.0

    def test_batch_axi(self) -> None:
        """Test batch AXI."""
        pred = torch.tensor([[[1, 1, 2, 2]], [[3, 3, 4, 4]]])
        true = torch.tensor([[[1, 1, 2, 2]], [[3, 3, 4, 4]]])
        assert compute_axi_batch(pred, true) > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

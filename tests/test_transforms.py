"""
Tests for domain-specific connectomics transforms.
"""

import numpy as np
import pytest
import torch

from neurons.transforms.connectomics import ElasticDeformation, MissingSection, Defects


# ---------------------------------------------------------------------------
# ElasticDeformation
# ---------------------------------------------------------------------------

class TestElasticDeformation:
    """Tests for ElasticDeformation transform."""

    def test_passthrough_when_prob_zero(self) -> None:
        t = ElasticDeformation(keys=("image",), prob=0.0)
        data = {"image": np.random.rand(32, 32).astype(np.float32)}
        result = t(data)
        np.testing.assert_array_equal(result["image"], data["image"])

    def test_always_applies_when_prob_one(self) -> None:
        t = ElasticDeformation(keys=("image",), prob=1.0, sigma=5.0, alpha=50.0)
        data = {"image": np.ones((32, 32), dtype=np.float32) * 0.5}
        result = t(data)
        assert result["image"].shape == (32, 32)

    def test_preserves_other_keys(self) -> None:
        t = ElasticDeformation(keys=("image",), prob=1.0, sigma=5.0, alpha=50.0)
        data = {
            "image": np.random.rand(32, 32).astype(np.float32),
            "extra": "untouched",
        }
        result = t(data)
        assert result["extra"] == "untouched"

    def test_multi_key_transform(self) -> None:
        t = ElasticDeformation(keys=("image", "label"), prob=1.0, sigma=5.0, alpha=50.0)
        data = {
            "image": np.random.rand(32, 32).astype(np.float32),
            "label": np.random.randint(0, 5, (32, 32)).astype(np.int64),
        }
        result = t(data)
        assert result["image"].shape == (32, 32)
        assert result["label"].shape == (32, 32)

    def test_3channel_input(self) -> None:
        t = ElasticDeformation(keys=("image",), prob=1.0, sigma=5.0, alpha=50.0)
        data = {"image": np.random.rand(3, 32, 32).astype(np.float32)}
        result = t(data)
        assert result["image"].shape == (3, 32, 32)

    def test_tensor_input_output(self) -> None:
        t = ElasticDeformation(keys=("image",), prob=1.0, sigma=5.0, alpha=50.0)
        data = {"image": torch.rand(32, 32)}
        result = t(data)
        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].shape == (32, 32)

    def test_custom_parameters(self) -> None:
        t = ElasticDeformation(keys=("image",), sigma=20.0, alpha=200.0, prob=0.8)
        assert t.sigma == 20.0
        assert t.alpha == 200.0
        assert t.prob == 0.8


# ---------------------------------------------------------------------------
# MissingSection
# ---------------------------------------------------------------------------

class TestMissingSection:
    """Tests for MissingSection transform."""

    def test_passthrough_when_prob_zero(self) -> None:
        t = MissingSection(keys=("image",), prob=0.0)
        data = {"image": np.random.rand(10, 32, 32).astype(np.float32)}
        result = t(data)
        np.testing.assert_array_equal(result["image"], data["image"])

    def test_interpolate_mode(self) -> None:
        t = MissingSection(keys=("image",), prob=1.0, fill_mode="interpolate")
        data = {"image": np.random.rand(10, 16, 16).astype(np.float32)}
        original = data["image"].copy()
        result = t(data)
        assert result["image"].shape == original.shape

    def test_zero_mode(self) -> None:
        t = MissingSection(keys=("image",), prob=1.0, fill_mode="zero")
        data = {"image": np.ones((10, 8, 8), dtype=np.float32)}
        result = t(data)
        has_zero_slice = any(
            result["image"][i].sum() == 0.0 for i in range(1, 9)
        )
        assert has_zero_slice

    def test_copy_mode(self) -> None:
        t = MissingSection(keys=("image",), prob=1.0, fill_mode="copy")
        data = {"image": np.random.rand(10, 8, 8).astype(np.float32)}
        result = t(data)
        assert result["image"].shape == (10, 8, 8)

    def test_2d_input_not_modified(self) -> None:
        t = MissingSection(keys=("image",), prob=1.0)
        data = {"image": np.random.rand(8, 8).astype(np.float32)}
        result = t(data)
        np.testing.assert_array_equal(result["image"], data["image"])

    def test_multi_key(self) -> None:
        t = MissingSection(keys=("image", "label"), prob=1.0, fill_mode="zero")
        data = {
            "image": np.ones((10, 8, 8), dtype=np.float32),
            "label": np.ones((10, 8, 8), dtype=np.int64),
        }
        result = t(data)
        assert result["image"].shape == (10, 8, 8)
        assert result["label"].shape == (10, 8, 8)


# ---------------------------------------------------------------------------
# Defects
# ---------------------------------------------------------------------------

class TestDefects:
    """Tests for Defects transform."""

    def test_passthrough_when_prob_zero(self) -> None:
        t = Defects(keys=("image",), prob=0.0)
        data = {"image": np.random.rand(32, 32).astype(np.float32)}
        result = t(data)
        np.testing.assert_array_equal(result["image"], data["image"])

    def test_output_clipped_to_01(self) -> None:
        t = Defects(keys=("image",), prob=1.0, line_prob=0.5, intensity_prob=0.5)
        data = {"image": np.random.rand(32, 32).astype(np.float32)}
        result = t(data)
        assert result["image"].min() >= 0.0
        assert result["image"].max() <= 1.0

    def test_preserves_shape_2d(self) -> None:
        t = Defects(keys=("image",), prob=1.0)
        data = {"image": np.random.rand(32, 32).astype(np.float32)}
        result = t(data)
        assert result["image"].shape == (32, 32)

    def test_preserves_shape_3d(self) -> None:
        t = Defects(keys=("image",), prob=1.0)
        data = {"image": np.random.rand(3, 32, 32).astype(np.float32)}
        result = t(data)
        assert result["image"].shape == (3, 32, 32)

    def test_tensor_input_output(self) -> None:
        t = Defects(keys=("image",), prob=1.0, line_prob=1.0, intensity_prob=0.0)
        data = {"image": torch.rand(32, 32)}
        result = t(data)
        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].shape == (32, 32)

    def test_only_image_key_modified(self) -> None:
        t = Defects(keys=("image",), prob=1.0)
        label = np.random.randint(0, 5, (32, 32))
        data = {
            "image": np.random.rand(32, 32).astype(np.float32),
            "label": label.copy(),
        }
        result = t(data)
        np.testing.assert_array_equal(result["label"], label)

    def test_custom_probabilities(self) -> None:
        t = Defects(keys=("image",), prob=0.5, line_prob=0.3, intensity_prob=0.7)
        assert t.prob == 0.5
        assert t.line_prob == 0.3
        assert t.intensity_prob == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

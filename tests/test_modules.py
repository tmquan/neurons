"""
Tests for PyTorch Lightning modules.
"""

import pytest
import torch

from neurons.modules.vista2d_module import Vista2DModule
from neurons.modules.vista3d_module import Vista3DModule


# Small model config shared across tests to keep things fast
_SMALL_MODEL_CFG = {
    "in_channels": 1,
    "out_channels": 2,
    "spatial_dims": 2,
    "init_filters": 8,
    "feature_dim": 16,
    "blocks_down": (1, 1),
    "blocks_up": (1,),
}


# ---------------------------------------------------------------------------
# Vista2DModule
# ---------------------------------------------------------------------------

class TestVista2DModule:
    """Tests for Vista2DModule."""

    def _make_module(self):
        return Vista2DModule(
            model_config={"in_channels": 1, "num_classes": 16, "feature_size": 16},
        )

    def test_forward(self) -> None:
        module = self._make_module()
        x = torch.randn(1, 1, 32, 32)
        out = module(x)
        assert "semantic" in out
        assert "instance" in out

    def test_training_step(self) -> None:
        module = self._make_module()
        batch = {
            "image": torch.randn(1, 1, 32, 32),
            "label": torch.randint(0, 5, (1, 32, 32)),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.isfinite()

    def test_validation_step(self) -> None:
        module = self._make_module()
        batch = {
            "image": torch.randn(1, 1, 32, 32),
            "label": torch.randint(0, 5, (1, 32, 32)),
        }
        loss = module.validation_step(batch, batch_idx=0)
        assert loss.isfinite()

    def test_prepare_targets_defaults(self) -> None:
        module = self._make_module()
        batch = {
            "image": torch.randn(1, 1, 32, 32),
            "label": torch.randint(0, 5, (1, 32, 32)),
        }
        targets = module._prepare_targets(batch)
        assert "class_labels" in targets
        assert "labels" in targets

    def test_prepare_targets_squeezes_4d(self) -> None:
        module = self._make_module()
        batch = {
            "image": torch.randn(1, 1, 32, 32),
            "label": torch.randint(0, 5, (1, 1, 32, 32)),
        }
        targets = module._prepare_targets(batch)
        assert targets["labels"].dim() == 3  # B, H, W

    def test_training_step_with_3d_image(self) -> None:
        """Image without channel dim should be auto-expanded."""
        module = self._make_module()
        batch = {
            "image": torch.randn(1, 32, 32),
            "label": torch.randint(0, 5, (1, 32, 32)),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.isfinite()

    def test_configure_optimizers_default(self) -> None:
        module = self._make_module()
        result = module.configure_optimizers()
        assert "optimizer" in result

    def test_configure_optimizers_no_scheduler(self) -> None:
        module = Vista2DModule(
            model_config={"in_channels": 1, "num_classes": 16, "feature_size": 16},
            optimizer_config={"scheduler": {"type": "none"}},
        )
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.Optimizer)


# ---------------------------------------------------------------------------
# Vista3DModule
# ---------------------------------------------------------------------------

class TestVista3DModule:
    """Tests for Vista3DModule."""

    def _make_module(self):
        return Vista3DModule(
            model_config={"in_channels": 1, "num_classes": 16, "feature_size": 16},
        )

    def test_forward(self) -> None:
        module = self._make_module()
        x = torch.randn(1, 1, 16, 16, 16)
        out = module(x)
        assert "semantic" in out
        assert "instance" in out

    def test_training_step(self) -> None:
        module = self._make_module()
        batch = {
            "image": torch.randn(1, 1, 16, 16, 16),
            "label": torch.randint(0, 5, (1, 16, 16, 16)),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.isfinite()

    def test_validation_step(self) -> None:
        module = self._make_module()
        batch = {
            "image": torch.randn(1, 1, 16, 16, 16),
            "label": torch.randint(0, 5, (1, 16, 16, 16)),
        }
        loss = module.validation_step(batch, batch_idx=0)
        assert loss.isfinite()

    def test_prepare_targets_squeezes_5d(self) -> None:
        module = self._make_module()
        batch = {
            "image": torch.randn(1, 1, 16, 16, 16),
            "label": torch.randint(0, 5, (1, 1, 16, 16, 16)),
        }
        targets = module._prepare_targets(batch)
        assert targets["labels"].dim() == 4  # B, D, H, W

    def test_training_step_auto_expand_image(self) -> None:
        module = self._make_module()
        batch = {
            "image": torch.randn(1, 16, 16, 16),
            "label": torch.randint(0, 5, (1, 16, 16, 16)),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.isfinite()

    def test_configure_optimizers(self) -> None:
        module = self._make_module()
        result = module.configure_optimizers()
        assert "optimizer" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

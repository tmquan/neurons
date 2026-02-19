"""
Tests for PyTorch Lightning modules.
"""

import pytest
import torch

from neurons.modules.semantic_seg import SemanticSegmentationModule
from neurons.modules.instance_seg import InstanceSegmentationModule
from neurons.modules.affinity_seg import AffinitySegmentationModule
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
# SemanticSegmentationModule
# ---------------------------------------------------------------------------

class TestSemanticSegmentationModule:
    """Tests for SemanticSegmentationModule."""

    def test_instantiation_defaults(self) -> None:
        module = SemanticSegmentationModule()
        assert module.num_classes == 2

    def test_instantiation_custom(self) -> None:
        module = SemanticSegmentationModule(
            model_config=_SMALL_MODEL_CFG,
            num_classes=2,
        )
        assert module.num_classes == 2

    def test_forward_returns_dict(self) -> None:
        module = SemanticSegmentationModule(model_config=_SMALL_MODEL_CFG, num_classes=2)
        x = torch.randn(2, 1, 32, 32)
        out = module(x)
        assert "logits" in out
        assert out["logits"].shape == (2, 2, 32, 32)

    def test_training_step(self) -> None:
        module = SemanticSegmentationModule(model_config=_SMALL_MODEL_CFG, num_classes=2)
        batch = {
            "image": torch.randn(2, 1, 32, 32),
            "label": torch.randint(0, 2, (2, 1, 32, 32)),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.isfinite()
        assert loss.dim() == 0

    def test_validation_step(self) -> None:
        module = SemanticSegmentationModule(model_config=_SMALL_MODEL_CFG, num_classes=2)
        batch = {
            "image": torch.randn(2, 1, 32, 32),
            "label": torch.randint(0, 2, (2, 1, 32, 32)),
        }
        loss = module.validation_step(batch, batch_idx=0)
        assert loss.isfinite()

    def test_compute_metrics_keys(self) -> None:
        module = SemanticSegmentationModule(model_config=_SMALL_MODEL_CFG, num_classes=3)
        logits = torch.randn(2, 3, 16, 16)
        targets = torch.randint(0, 3, (2, 16, 16))
        metrics = module._compute_metrics(logits, targets)
        assert "accuracy" in metrics
        assert "mean_iou" in metrics
        assert "iou_class_0" in metrics
        assert "dice_class_2" in metrics

    def test_class_weights(self) -> None:
        module = SemanticSegmentationModule(
            model_config=_SMALL_MODEL_CFG,
            loss_config={"class_weights": [0.3, 0.7]},
            num_classes=2,
        )
        assert module.class_weights is not None

    def test_configure_optimizers_default(self) -> None:
        module = SemanticSegmentationModule(model_config=_SMALL_MODEL_CFG)
        result = module.configure_optimizers()
        assert "optimizer" in result

    def test_configure_optimizers_reduce_on_plateau(self) -> None:
        module = SemanticSegmentationModule(
            model_config=_SMALL_MODEL_CFG,
            optimizer_config={"scheduler": {"type": "reduce_on_plateau"}},
        )
        result = module.configure_optimizers()
        assert "lr_scheduler" in result

    def test_configure_optimizers_no_scheduler(self) -> None:
        module = SemanticSegmentationModule(
            model_config=_SMALL_MODEL_CFG,
            optimizer_config={"scheduler": {"type": "none"}},
        )
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.Optimizer)


# ---------------------------------------------------------------------------
# InstanceSegmentationModule
# ---------------------------------------------------------------------------

class TestInstanceSegmentationModule:
    """Tests for InstanceSegmentationModule."""

    def _make_module(self):
        cfg = dict(_SMALL_MODEL_CFG)
        cfg["use_ins_head"] = True
        cfg["emb_dim"] = 8
        return InstanceSegmentationModule(model_config=cfg)

    def test_instantiation(self) -> None:
        module = self._make_module()
        assert module.model.use_ins_head

    def test_forward(self) -> None:
        module = self._make_module()
        x = torch.randn(2, 1, 32, 32)
        out = module(x)
        assert "logits" in out
        assert "embedding" in out

    def test_training_step(self) -> None:
        module = self._make_module()
        batch = {
            "image": torch.randn(2, 1, 32, 32),
            "label": torch.randint(0, 5, (2, 32, 32)),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.isfinite()

    def test_validation_step(self) -> None:
        module = self._make_module()
        batch = {
            "image": torch.randn(2, 1, 32, 32),
            "label": torch.randint(0, 5, (2, 32, 32)),
        }
        loss = module.validation_step(batch, batch_idx=0)
        assert loss.isfinite()

    def test_compute_losses_keys(self) -> None:
        module = self._make_module()
        x = torch.randn(1, 1, 32, 32)
        outputs = module(x)
        labels = torch.randint(0, 3, (1, 32, 32))
        loss_dict = module._compute_losses(outputs, labels)
        for key in ("loss", "loss_sem", "loss_ins", "loss_var", "loss_dst", "loss_reg"):
            assert key in loss_dict

    def test_compute_losses_with_4d_labels(self) -> None:
        module = self._make_module()
        x = torch.randn(1, 1, 32, 32)
        outputs = module(x)
        labels = torch.randint(0, 3, (1, 1, 32, 32))
        loss_dict = module._compute_losses(outputs, labels)
        assert loss_dict["loss"].isfinite()

    def test_compute_metrics(self) -> None:
        module = self._make_module()
        x = torch.randn(1, 1, 32, 32)
        outputs = module(x)
        labels = torch.randint(0, 3, (1, 32, 32))
        metrics = module._compute_metrics(outputs, labels)
        assert "iou" in metrics
        assert "accuracy" in metrics

    def test_configure_optimizers(self) -> None:
        module = self._make_module()
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.Optimizer)


# ---------------------------------------------------------------------------
# AffinitySegmentationModule
# ---------------------------------------------------------------------------

class TestAffinitySegmentationModule:
    """Tests for AffinitySegmentationModule."""

    def _make_module(self, spatial_dims=2):
        cfg = dict(_SMALL_MODEL_CFG)
        cfg["spatial_dims"] = spatial_dims
        cfg["init_filters"] = 8
        cfg["feature_dim"] = 16
        cfg["blocks_down"] = (1, 1)
        cfg["blocks_up"] = (1,)
        return AffinitySegmentationModule(model_config=cfg, spatial_dims=spatial_dims)

    def test_default_2d_offsets(self) -> None:
        module = self._make_module(spatial_dims=2)
        assert module.num_affinities == 2
        assert module.affinity_offsets == [(0, 1), (1, 0)]

    def test_default_3d_offsets(self) -> None:
        module = self._make_module(spatial_dims=3)
        assert module.num_affinities == 3

    def test_custom_offsets(self) -> None:
        module = AffinitySegmentationModule(
            model_config=dict(_SMALL_MODEL_CFG),
            spatial_dims=2,
            affinity_offsets=[(0, 1), (1, 0), (1, 1)],
        )
        assert module.num_affinities == 3

    def test_forward(self) -> None:
        module = self._make_module()
        x = torch.randn(2, 1, 32, 32)
        out = module(x)
        assert "affinity_logits" in out
        assert "affinity" in out
        assert out["affinity"].min() >= 0.0
        assert out["affinity"].max() <= 1.0

    def test_compute_affinity_targets_2d(self) -> None:
        module = self._make_module(spatial_dims=2)
        labels = torch.zeros(2, 16, 16, dtype=torch.long)
        labels[0, :8, :8] = 1
        labels[0, :8, 8:] = 2
        targets = module._compute_affinity_targets(labels)
        assert targets.shape == (2, 2, 16, 16)
        assert targets.min() >= 0.0
        assert targets.max() <= 1.0

    def test_compute_affinity_targets_3d(self) -> None:
        module = self._make_module(spatial_dims=3)
        labels = torch.zeros(1, 4, 8, 8, dtype=torch.long)
        labels[:, :, :4, :4] = 1
        targets = module._compute_affinity_targets(labels)
        assert targets.shape == (1, 3, 4, 8, 8)

    def test_compute_metrics(self) -> None:
        module = self._make_module()
        pred = torch.rand(2, 2, 16, 16)
        target = (torch.rand(2, 2, 16, 16) > 0.5).float()
        metrics = module._compute_metrics(pred, target)
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in metrics
            assert metrics[key].isfinite()

    def test_training_step(self) -> None:
        module = self._make_module()
        batch = {
            "image": torch.randn(2, 1, 32, 32),
            "label": torch.randint(0, 5, (2, 32, 32)),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.isfinite()

    def test_validation_step(self) -> None:
        module = self._make_module()
        batch = {
            "image": torch.randn(2, 1, 32, 32),
            "label": torch.randint(0, 5, (2, 32, 32)),
        }
        loss = module.validation_step(batch, batch_idx=0)
        assert loss.isfinite()

    def test_configure_optimizers_cosine(self) -> None:
        module = AffinitySegmentationModule(
            model_config=dict(_SMALL_MODEL_CFG),
            optimizer_config={"scheduler": {"type": "cosine", "T_max": 50}},
        )
        result = module.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result


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

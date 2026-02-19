"""
Tests for model architectures.
"""

import pytest
import torch

from neurons.models.base import BaseModel
from neurons.models.segresnet import SegResNetWrapper
from neurons.models.vista2d_model import Vista2DWrapper
from neurons.models.vista3d_model import Vista3DWrapper


# ---------------------------------------------------------------------------
# BaseModel (abstract)
# ---------------------------------------------------------------------------

class TestBaseModel:
    """Tests for the abstract BaseModel."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseModel(in_channels=1, out_channels=2)  # type: ignore[abstract]

    def test_concrete_subclass_requires_forward_and_get_output_channels(self) -> None:
        class IncompleteModel(BaseModel):
            def forward(self, x):
                return {"logits": x}

        with pytest.raises(TypeError):
            IncompleteModel(in_channels=1, out_channels=2)  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        class TinyModel(BaseModel):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.linear = torch.nn.Linear(4, kwargs["out_channels"])

            def forward(self, x):
                return {"logits": self.linear(x)}

            def get_output_channels(self) -> int:
                return self.out_channels

        model = TinyModel(in_channels=4, out_channels=2, spatial_dims=2)
        assert model.in_channels == 4
        assert model.out_channels == 2
        assert model.spatial_dims == 2
        assert model.get_output_channels() == 2

    def test_get_num_parameters(self) -> None:
        class TinyModel(BaseModel):
            def __init__(self):
                super().__init__(in_channels=1, out_channels=1)
                self.w = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, x):
                return {"logits": x}

            def get_output_channels(self):
                return 1

        m = TinyModel()
        assert m.get_num_parameters(trainable_only=True) == 9
        assert m.get_num_parameters(trainable_only=False) == 9

        m.w.requires_grad_(False)
        assert m.get_num_parameters(trainable_only=True) == 0
        assert m.get_num_parameters(trainable_only=False) == 9

    def test_repr(self) -> None:
        class TinyModel(BaseModel):
            def __init__(self):
                super().__init__(in_channels=1, out_channels=2, spatial_dims=3)

            def forward(self, x):
                return {"logits": x}

            def get_output_channels(self):
                return 2

        r = repr(TinyModel())
        assert "TinyModel" in r
        assert "in_channels=1" in r


# ---------------------------------------------------------------------------
# SegResNetWrapper
# ---------------------------------------------------------------------------

class TestSegResNetWrapper:
    """Tests for SegResNetWrapper."""

    def test_2d_forward_shape(self) -> None:
        model = SegResNetWrapper(
            in_channels=1, out_channels=2, spatial_dims=2,
            init_filters=8, feature_dim=16, blocks_down=(1, 1), blocks_up=(1,),
        )
        x = torch.randn(2, 1, 32, 32)
        out = model(x)
        assert "logits" in out
        assert "features" in out
        assert out["logits"].shape == (2, 2, 32, 32)

    def test_3d_forward_shape(self) -> None:
        model = SegResNetWrapper(
            in_channels=1, out_channels=3, spatial_dims=3,
            init_filters=8, feature_dim=16, blocks_down=(1, 1), blocks_up=(1,),
        )
        x = torch.randn(1, 1, 8, 16, 16)
        out = model(x)
        assert out["logits"].shape == (1, 3, 8, 16, 16)

    def test_instance_head(self) -> None:
        model = SegResNetWrapper(
            in_channels=1, out_channels=2, spatial_dims=2,
            init_filters=8, feature_dim=16, emb_dim=8,
            blocks_down=(1, 1), blocks_up=(1,),
            use_ins_head=True,
        )
        x = torch.randn(2, 1, 32, 32)
        out = model(x)
        assert "embedding" in out
        assert out["embedding"].shape == (2, 8, 32, 32)

    def test_boundary_head(self) -> None:
        model = SegResNetWrapper(
            in_channels=1, out_channels=2, spatial_dims=2,
            init_filters=8, feature_dim=16,
            blocks_down=(1, 1), blocks_up=(1,),
            use_boundary_head=True,
        )
        x = torch.randn(1, 1, 32, 32)
        out = model(x)
        assert "boundary" in out
        assert out["boundary"].shape[1] == 1

    def test_no_optional_heads_by_default(self) -> None:
        model = SegResNetWrapper(
            in_channels=1, out_channels=2, spatial_dims=2,
            init_filters=8, feature_dim=16,
            blocks_down=(1, 1), blocks_up=(1,),
        )
        x = torch.randn(1, 1, 32, 32)
        out = model(x)
        assert "embedding" not in out
        assert "boundary" not in out

    def test_get_output_channels(self) -> None:
        model = SegResNetWrapper(
            in_channels=1, out_channels=5, spatial_dims=2,
            init_filters=8, feature_dim=16,
            blocks_down=(1, 1), blocks_up=(1,),
        )
        assert model.get_output_channels() == 5

    def test_freeze_unfreeze_encoder(self) -> None:
        model = SegResNetWrapper(
            in_channels=1, out_channels=2, spatial_dims=2,
            init_filters=8, feature_dim=16,
            blocks_down=(1, 1), blocks_up=(1,),
        )
        model.freeze_encoder()
        frozen = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        assert frozen > 0

        model.unfreeze_encoder()
        unfrozen = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        assert unfrozen == 0

    def test_backward_pass(self) -> None:
        model = SegResNetWrapper(
            in_channels=1, out_channels=2, spatial_dims=2,
            init_filters=8, feature_dim=16,
            blocks_down=(1, 1), blocks_up=(1,),
        )
        x = torch.randn(1, 1, 32, 32, requires_grad=True)
        out = model(x)
        out["logits"].sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Vista2DWrapper
# ---------------------------------------------------------------------------

class TestVista2DWrapper:
    """Tests for Vista2DWrapper."""

    def test_forward_output_keys(self) -> None:
        model = Vista2DWrapper(in_channels=1, num_classes=16, feature_size=16)
        x = torch.randn(1, 1, 32, 32)
        out = model(x)
        assert "semantic" in out
        assert "instance" in out

    def test_output_shapes(self) -> None:
        model = Vista2DWrapper(in_channels=1, num_classes=16, feature_size=16)
        x = torch.randn(2, 1, 32, 32)
        out = model(x)
        for key in ("semantic", "instance"):
            assert out[key].shape == (2, 16, 32, 32)

    def test_different_input_channels(self) -> None:
        model = Vista2DWrapper(in_channels=3, num_classes=16, feature_size=16)
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out["semantic"].shape[1] == 16

    def test_backward_pass(self) -> None:
        model = Vista2DWrapper(in_channels=1, num_classes=16, feature_size=16)
        x = torch.randn(1, 1, 32, 32, requires_grad=True)
        out = model(x)
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Vista3DWrapper
# ---------------------------------------------------------------------------

class TestVista3DWrapper:
    """Tests for Vista3DWrapper."""

    def test_forward_output_keys(self) -> None:
        model = Vista3DWrapper(in_channels=1, num_classes=16, feature_size=16)
        x = torch.randn(1, 1, 8, 16, 16)
        out = model(x)
        assert "semantic" in out
        assert "instance" in out

    def test_output_shapes(self) -> None:
        model = Vista3DWrapper(in_channels=1, num_classes=16, feature_size=16)
        x = torch.randn(1, 1, 8, 16, 16)
        out = model(x)
        for key in ("semantic", "instance"):
            assert out[key].shape == (1, 16, 8, 16, 16)

    def test_backward_pass(self) -> None:
        model = Vista3DWrapper(in_channels=1, num_classes=16, feature_size=16)
        x = torch.randn(1, 1, 8, 16, 16, requires_grad=True)
        out = model(x)
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        assert x.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

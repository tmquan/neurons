"""
Tests for model architectures.
"""

import pytest
import torch

from neurons.models.base import BaseModel
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
        x = torch.randn(1, 1, 32, 32, 32)
        out = model(x)
        assert "semantic" in out
        assert "instance" in out

    def test_output_shapes(self) -> None:
        model = Vista3DWrapper(in_channels=1, num_classes=16, feature_size=16)
        x = torch.randn(1, 1, 32, 32, 32)
        out = model(x)
        for key in ("semantic", "instance"):
            assert out[key].shape == (1, 16, 32, 32, 32)

    def test_backward_pass(self) -> None:
        model = Vista3DWrapper(in_channels=1, num_classes=16, feature_size=16)
        x = torch.randn(1, 1, 32, 32, 32, requires_grad=True)
        out = model(x)
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        assert x.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for loss functions.
"""

import pytest
import torch

from neurons.losses.discriminative import DiscriminativeLoss, DiscriminativeLossVectorized
from neurons.losses.boundary import BoundaryLoss, BoundaryAwareCrossEntropy
from neurons.losses.weighted_boundary import WeightedBoundaryLoss


class TestDiscriminativeLoss:
    """Tests for DiscriminativeLoss."""

    def test_zero_instances(self) -> None:
        """Test loss with no foreground instances (all background)."""
        loss_fn = DiscriminativeLoss()
        embedding = torch.randn(2, 8, 16, 16)
        labels = torch.zeros(2, 16, 16, dtype=torch.long)

        total, l_var, l_dst, l_reg = loss_fn(embedding, labels)
        assert total.item() == 0.0
        assert l_var.item() == 0.0
        assert l_dst.item() == 0.0
        assert l_reg.item() == 0.0

    def test_single_instance(self) -> None:
        """Test loss with a single foreground instance."""
        loss_fn = DiscriminativeLoss()
        embedding = torch.randn(1, 8, 16, 16)
        labels = torch.ones(1, 16, 16, dtype=torch.long)

        total, l_var, l_dst, l_reg = loss_fn(embedding, labels)
        assert total.isfinite()
        assert l_dst.item() == 0.0  # Only one instance, no distance loss

    def test_multiple_instances(self) -> None:
        """Test loss with multiple instances."""
        loss_fn = DiscriminativeLoss(delta_var=0.5, delta_dst=1.5)
        embedding = torch.randn(2, 8, 32, 32)
        labels = torch.zeros(2, 32, 32, dtype=torch.long)
        labels[:, :16, :16] = 1
        labels[:, :16, 16:] = 2
        labels[:, 16:, :16] = 3

        total, l_var, l_dst, l_reg = loss_fn(embedding, labels)
        assert total.isfinite()
        assert total.item() >= 0.0

    def test_3d_input(self) -> None:
        """Test loss with 3D input."""
        loss_fn = DiscriminativeLoss()
        embedding = torch.randn(1, 8, 4, 16, 16)
        labels = torch.zeros(1, 4, 16, 16, dtype=torch.long)
        labels[:, :, :8, :8] = 1
        labels[:, :, :8, 8:] = 2

        total, l_var, l_dst, l_reg = loss_fn(embedding, labels)
        assert total.isfinite()

    def test_backward_pass(self) -> None:
        """Test that gradients flow correctly."""
        loss_fn = DiscriminativeLoss()
        embedding = torch.randn(1, 8, 16, 16, requires_grad=True)
        labels = torch.zeros(1, 16, 16, dtype=torch.long)
        labels[:, :8, :] = 1
        labels[:, 8:, :] = 2

        total, _, _, _ = loss_fn(embedding, labels)
        total.backward()
        assert embedding.grad is not None
        assert embedding.grad.isfinite().all()


class TestDiscriminativeLossVectorized:
    """Tests for DiscriminativeLossVectorized."""

    def test_matches_naive_implementation(self) -> None:
        """Test that vectorized gives similar results to naive."""
        torch.manual_seed(42)
        embedding = torch.randn(1, 8, 16, 16)
        labels = torch.zeros(1, 16, 16, dtype=torch.long)
        labels[:, :8, :8] = 1
        labels[:, :8, 8:] = 2
        labels[:, 8:, :] = 3

        naive = DiscriminativeLoss(delta_var=0.5, delta_dst=1.5)
        vectorized = DiscriminativeLossVectorized(delta_var=0.5, delta_dst=1.5)

        total_naive, _, _, _ = naive(embedding, labels)
        total_vec, _, _, _ = vectorized(embedding, labels)

        assert torch.allclose(total_naive, total_vec, atol=1e-4)


class TestBoundaryLoss:
    """Tests for BoundaryLoss."""

    def test_output_shape(self) -> None:
        """Test that output is a scalar."""
        loss_fn = BoundaryLoss(boundary_weight=5.0)
        logits = torch.randn(2, 2, 32, 32)
        labels = torch.randint(0, 2, (2, 32, 32))

        loss = loss_fn(logits, labels)
        assert loss.dim() == 0
        assert loss.isfinite()

    def test_boundary_weight_effect(self) -> None:
        """Test that higher boundary weight increases loss at boundaries."""
        logits = torch.randn(1, 2, 32, 32)
        labels = torch.zeros(1, 32, 32, dtype=torch.long)
        labels[:, :, 16:] = 1

        loss_low = BoundaryLoss(boundary_weight=1.0)(logits, labels)
        loss_high = BoundaryLoss(boundary_weight=10.0)(logits, labels)

        assert loss_high >= loss_low


class TestBoundaryAwareCrossEntropy:
    """Tests for BoundaryAwareCrossEntropy."""

    def test_output_shape(self) -> None:
        """Test that output is a scalar."""
        loss_fn = BoundaryAwareCrossEntropy()
        logits = torch.randn(2, 2, 32, 32)
        labels = torch.randint(0, 2, (2, 32, 32))

        loss = loss_fn(logits, labels)
        assert loss.dim() == 0
        assert loss.isfinite()


class TestWeightedBoundaryLoss:
    """Tests for WeightedBoundaryLoss."""

    def test_2d_output(self) -> None:
        """Test 2D weighted boundary loss."""
        loss_fn = WeightedBoundaryLoss(num_classes=2, boundary_weight=5.0)
        logits = torch.randn(2, 2, 32, 32)
        labels = torch.randint(0, 2, (2, 32, 32))

        loss = loss_fn(logits, labels)
        assert loss.dim() == 0
        assert loss.isfinite()

    def test_3d_output(self) -> None:
        """Test 3D weighted boundary loss."""
        loss_fn = WeightedBoundaryLoss(num_classes=4, boundary_weight=3.0)
        logits = torch.randn(1, 4, 8, 16, 16)
        labels = torch.randint(0, 4, (1, 8, 16, 16))

        loss = loss_fn(logits, labels)
        assert loss.dim() == 0
        assert loss.isfinite()

    def test_manual_class_weights(self) -> None:
        """Test with manually specified class weights."""
        loss_fn = WeightedBoundaryLoss(
            num_classes=3,
            boundary_weight=5.0,
            class_weights=[0.1, 0.5, 0.4],
        )
        logits = torch.randn(2, 3, 32, 32)
        labels = torch.randint(0, 3, (2, 32, 32))

        loss = loss_fn(logits, labels)
        assert loss.isfinite()

    def test_backward_pass(self) -> None:
        """Test gradient computation."""
        loss_fn = WeightedBoundaryLoss(num_classes=2)
        logits = torch.randn(1, 2, 16, 16, requires_grad=True)
        labels = torch.randint(0, 2, (1, 16, 16))

        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

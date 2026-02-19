"""
Tests for loss functions.
"""

import pytest
import torch

from neurons.losses.discriminative import (
    CentroidEmbeddingLoss,
    SkeletonEmbeddingLoss,
    DiscriminativeLoss,
    DiscriminativeLossVectorized,
)
from neurons.losses.vista2d_losses import Vista2DLoss
from neurons.losses.vista3d_losses import Vista3DLoss


# ---------------------------------------------------------------------------
# CentroidEmbeddingLoss  (+ backward-compat aliases)
# ---------------------------------------------------------------------------

class TestCentroidEmbeddingLoss:
    """Tests for CentroidEmbeddingLoss (discriminative loss + projection heads)."""

    def test_return_dict_keys(self) -> None:
        loss_fn = CentroidEmbeddingLoss(emb_dim=8)
        embedding = torch.randn(1, 8, 16, 16)
        labels = torch.ones(1, 16, 16, dtype=torch.long)
        result = loss_fn(embedding, labels)
        for key in ("loss", "l_var", "l_dst", "l_reg", "l_dir", "l_cov", "l_raw"):
            assert key in result

    def test_zero_instances(self) -> None:
        loss_fn = CentroidEmbeddingLoss(emb_dim=8)
        embedding = torch.randn(2, 8, 16, 16)
        labels = torch.zeros(2, 16, 16, dtype=torch.long)
        result = loss_fn(embedding, labels)
        assert result["loss"].item() == 0.0
        assert result["l_var"].item() == 0.0
        assert result["l_dst"].item() == 0.0
        assert result["l_reg"].item() == 0.0

    def test_single_instance(self) -> None:
        loss_fn = CentroidEmbeddingLoss(emb_dim=8)
        embedding = torch.randn(1, 8, 16, 16)
        labels = torch.ones(1, 16, 16, dtype=torch.long)
        result = loss_fn(embedding, labels)
        assert result["loss"].isfinite()
        assert result["l_dst"].item() == 0.0

    def test_multiple_instances(self) -> None:
        loss_fn = CentroidEmbeddingLoss(delta_var=0.5, delta_dst=1.5, emb_dim=8)
        embedding = torch.randn(2, 8, 32, 32)
        labels = torch.zeros(2, 32, 32, dtype=torch.long)
        labels[:, :16, :16] = 1
        labels[:, :16, 16:] = 2
        labels[:, 16:, :16] = 3
        result = loss_fn(embedding, labels)
        assert result["loss"].isfinite()
        assert result["loss"].item() >= 0.0

    def test_3d_input(self) -> None:
        loss_fn = CentroidEmbeddingLoss(emb_dim=8, spatial_dims=3)
        embedding = torch.randn(1, 8, 4, 16, 16)
        labels = torch.zeros(1, 4, 16, 16, dtype=torch.long)
        labels[:, :, :8, :8] = 1
        labels[:, :, :8, 8:] = 2
        result = loss_fn(embedding, labels)
        assert result["loss"].isfinite()

    def test_backward_pass(self) -> None:
        loss_fn = CentroidEmbeddingLoss(emb_dim=8)
        embedding = torch.randn(1, 8, 16, 16, requires_grad=True)
        labels = torch.zeros(1, 16, 16, dtype=torch.long)
        labels[:, :8, :] = 1
        labels[:, 8:, :] = 2
        result = loss_fn(embedding, labels)
        result["loss"].backward()
        assert embedding.grad is not None
        assert embedding.grad.isfinite().all()

    def test_repr(self) -> None:
        loss_fn = CentroidEmbeddingLoss(delta_var=0.3, delta_dst=2.0, A=2.0)
        r = repr(loss_fn)
        assert "CentroidEmbeddingLoss" in r
        assert "delta_var=0.3" in r

    def test_backward_compat_aliases(self) -> None:
        assert DiscriminativeLoss is CentroidEmbeddingLoss
        assert DiscriminativeLossVectorized is CentroidEmbeddingLoss

    def test_alias_produces_same_results(self) -> None:
        torch.manual_seed(42)
        embedding = torch.randn(1, 8, 16, 16)
        labels = torch.zeros(1, 16, 16, dtype=torch.long)
        labels[:, :8, :8] = 1
        labels[:, :8, 8:] = 2
        labels[:, 8:, :] = 3

        a = DiscriminativeLoss(delta_var=0.5, delta_dst=1.5, emb_dim=8)
        b = DiscriminativeLossVectorized(delta_var=0.5, delta_dst=1.5, emb_dim=8)
        ra = a(embedding, labels)
        rb = b(embedding, labels)
        assert torch.allclose(ra["loss"], rb["loss"], atol=1e-6)

    # -- projection loss: backward compat (w_*=0 by default) --

    def test_projections_disabled_by_default(self) -> None:
        loss_fn = CentroidEmbeddingLoss(emb_dim=8)
        embedding = torch.randn(1, 8, 16, 16)
        labels = torch.ones(1, 16, 16, dtype=torch.long)
        result = loss_fn(embedding, labels)
        assert result["l_dir"].item() == 0.0
        assert result["l_cov"].item() == 0.0
        assert result["l_raw"].item() == 0.0

    # -- L_dir with centroid target --

    def test_l_dir_centroid_finite(self) -> None:
        loss_fn = CentroidEmbeddingLoss(
            emb_dim=8, dir_target="centroid", w_dir=1.0,
        )
        embedding = torch.randn(1, 8, 16, 16)
        labels = torch.zeros(1, 16, 16, dtype=torch.long)
        labels[:, :8, :8] = 1
        labels[:, 8:, :] = 2
        result = loss_fn(embedding, labels)
        assert result["l_dir"].isfinite()
        assert result["l_dir"].item() >= 0.0

    def test_l_dir_centroid_backward(self) -> None:
        loss_fn = CentroidEmbeddingLoss(emb_dim=8, w_dir=1.0)
        embedding = torch.randn(1, 8, 16, 16, requires_grad=True)
        labels = torch.zeros(1, 16, 16, dtype=torch.long)
        labels[:, :8, :] = 1
        labels[:, 8:, :] = 2
        result = loss_fn(embedding, labels)
        result["loss"].backward()
        assert embedding.grad is not None
        assert loss_fn.proj_dir.weight.grad is not None

    # -- L_dir with skeleton target --

    def test_l_dir_skeleton_finite(self) -> None:
        loss_fn = CentroidEmbeddingLoss(
            emb_dim=8, dir_target="skeleton", w_dir=1.0,
        )
        embedding = torch.randn(1, 8, 16, 16)
        labels = torch.zeros(1, 16, 16, dtype=torch.long)
        labels[:, 2:14, 2:7] = 1
        labels[:, 2:14, 9:14] = 2
        result = loss_fn(embedding, labels)
        assert result["l_dir"].isfinite()
        assert result["l_dir"].item() >= 0.0

    # -- L_cov --

    def test_l_cov_finite(self) -> None:
        loss_fn = CentroidEmbeddingLoss(emb_dim=8, w_cov=1.0)
        embedding = torch.randn(1, 8, 16, 16)
        labels = torch.zeros(1, 16, 16, dtype=torch.long)
        labels[:, :8, :8] = 1
        labels[:, 8:, :] = 2
        result = loss_fn(embedding, labels)
        assert result["l_cov"].isfinite()
        assert result["l_cov"].item() >= 0.0

    def test_l_cov_backward(self) -> None:
        loss_fn = CentroidEmbeddingLoss(emb_dim=8, w_cov=1.0)
        embedding = torch.randn(1, 8, 16, 16, requires_grad=True)
        labels = torch.zeros(1, 16, 16, dtype=torch.long)
        labels[:, :8, :] = 1
        labels[:, 8:, :] = 2
        result = loss_fn(embedding, labels)
        result["loss"].backward()
        assert loss_fn.proj_cov.weight.grad is not None

    # -- L_raw --

    def test_l_raw_finite(self) -> None:
        loss_fn = CentroidEmbeddingLoss(emb_dim=8, w_raw=1.0)
        embedding = torch.randn(1, 8, 16, 16)
        labels = torch.zeros(1, 16, 16, dtype=torch.long)
        labels[:, :8, :] = 1
        gt_image = torch.rand(1, 1, 16, 16)
        result = loss_fn(embedding, labels, raw_image=gt_image)
        assert result["l_raw"].isfinite()
        assert result["l_raw"].item() >= 0.0

    def test_l_raw_zero_without_gt_image(self) -> None:
        loss_fn = CentroidEmbeddingLoss(emb_dim=8, w_raw=1.0)
        embedding = torch.randn(1, 8, 16, 16)
        labels = torch.ones(1, 16, 16, dtype=torch.long)
        result = loss_fn(embedding, labels)
        assert result["l_raw"].item() == 0.0

    def test_l_raw_backward(self) -> None:
        loss_fn = CentroidEmbeddingLoss(emb_dim=8, w_raw=1.0)
        embedding = torch.randn(1, 8, 16, 16, requires_grad=True)
        labels = torch.ones(1, 16, 16, dtype=torch.long)
        gt_image = torch.rand(1, 1, 16, 16)
        result = loss_fn(embedding, labels, raw_image=gt_image)
        result["loss"].backward()
        assert loss_fn.proj_raw.weight.grad is not None

    # -- all projections enabled together --

    def test_all_projections_together(self) -> None:
        loss_fn = CentroidEmbeddingLoss(
            emb_dim=8, w_dir=1.0, w_cov=1.0, w_raw=1.0,
        )
        embedding = torch.randn(2, 8, 16, 16, requires_grad=True)
        labels = torch.zeros(2, 16, 16, dtype=torch.long)
        labels[:, :8, :8] = 1
        labels[:, :8, 8:] = 2
        labels[:, 8:, :] = 3
        gt_image = torch.rand(2, 1, 16, 16)
        result = loss_fn(embedding, labels, raw_image=gt_image)
        assert result["loss"].isfinite()
        result["loss"].backward()
        assert embedding.grad is not None
        assert embedding.grad.isfinite().all()


# ---------------------------------------------------------------------------
# SkeletonEmbeddingLoss
# ---------------------------------------------------------------------------

def _make_skeleton_gt(H, W, device="cpu"):
    """Build synthetic GT tensors for SkeletonEmbeddingLoss tests."""
    masks = torch.zeros(1, H, W, dtype=torch.long, device=device)
    masks[0, :H // 2, :W // 2] = 1
    masks[0, :H // 2, W // 2:] = 2
    masks[0, H // 2:, :] = 3

    nearest_skel = torch.zeros(1, 2, H, W, device=device)
    nearest_skel[0, 0] = W / 4
    nearest_skel[0, 1] = H / 4

    dt_norm = torch.ones(1, 1, H, W, device=device) * 0.5
    dt_grad = torch.zeros(1, 2, H, W, device=device)
    dt_grad[0, 0] = 0.0
    dt_grad[0, 1] = 1.0
    return masks, nearest_skel, dt_norm, dt_grad


class TestSkeletonEmbeddingLoss:
    """Tests for SkeletonEmbeddingLoss."""

    def test_output_keys(self) -> None:
        loss_fn = SkeletonEmbeddingLoss()
        H, W = 16, 16
        offsets = torch.randn(1, 2, H, W)
        masks, skel, dt_n, dt_g = _make_skeleton_gt(H, W)
        result = loss_fn(offsets, masks, skel, dt_n, dt_g)
        for key in ("loss", "l_pull", "l_push", "l_penalty", "l_benefit"):
            assert key in result

    def test_all_finite(self) -> None:
        loss_fn = SkeletonEmbeddingLoss()
        H, W = 16, 16
        offsets = torch.randn(1, 2, H, W)
        masks, skel, dt_n, dt_g = _make_skeleton_gt(H, W)
        result = loss_fn(offsets, masks, skel, dt_n, dt_g)
        for v in result.values():
            assert v.isfinite(), f"{v} is not finite"

    def test_backward_pass(self) -> None:
        loss_fn = SkeletonEmbeddingLoss()
        H, W = 16, 16
        offsets = torch.randn(1, 2, H, W, requires_grad=True)
        masks, skel, dt_n, dt_g = _make_skeleton_gt(H, W)
        result = loss_fn(offsets, masks, skel, dt_n, dt_g)
        result["loss"].backward()
        assert offsets.grad is not None
        assert offsets.grad.isfinite().all()

    def test_zero_instances(self) -> None:
        loss_fn = SkeletonEmbeddingLoss()
        H, W = 8, 8
        offsets = torch.randn(1, 2, H, W)
        masks = torch.zeros(1, H, W, dtype=torch.long)
        skel = torch.zeros(1, 2, H, W)
        dt_n = torch.ones(1, 1, H, W)
        dt_g = torch.zeros(1, 2, H, W)
        result = loss_fn(offsets, masks, skel, dt_n, dt_g)
        assert result["loss"].isfinite()

    def test_single_instance_no_push(self) -> None:
        loss_fn = SkeletonEmbeddingLoss()
        H, W = 16, 16
        offsets = torch.randn(1, 2, H, W)
        masks = torch.ones(1, H, W, dtype=torch.long)
        skel = torch.zeros(1, 2, H, W)
        skel[0, 0] = W / 2
        skel[0, 1] = H / 2
        dt_n = torch.ones(1, 1, H, W)
        dt_g = torch.ones(1, 2, H, W)
        result = loss_fn(offsets, masks, skel, dt_n, dt_g)
        assert result["l_push"].item() == 0.0

    def test_pull_decreases_with_better_offsets(self) -> None:
        """Offsets that exactly reach the skeleton should have lower pull."""
        loss_fn = SkeletonEmbeddingLoss()
        H, W = 16, 16
        masks, skel, dt_n, dt_g = _make_skeleton_gt(H, W)

        yg, xg = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )
        coords = torch.stack([xg, yg], dim=0).unsqueeze(0)

        perfect_offsets = skel - coords
        random_offsets = torch.randn(1, 2, H, W) * 10.0

        res_perfect = loss_fn(perfect_offsets, masks, skel, dt_n, dt_g)
        res_random = loss_fn(random_offsets, masks, skel, dt_n, dt_g)
        assert res_perfect["l_pull"].item() < res_random["l_pull"].item()

    def test_batch_size_2(self) -> None:
        loss_fn = SkeletonEmbeddingLoss()
        H, W = 12, 12
        offsets = torch.randn(2, 2, H, W)
        masks = torch.zeros(2, H, W, dtype=torch.long)
        masks[0, :6, :6] = 1
        masks[0, :6, 6:] = 2
        masks[1, 6:, :] = 3
        skel = torch.zeros(2, 2, H, W)
        dt_n = torch.ones(2, 1, H, W) * 0.5
        dt_g = torch.randn(2, 2, H, W)
        result = loss_fn(offsets, masks, skel, dt_n, dt_g)
        assert result["loss"].isfinite()

    def test_custom_weights(self) -> None:
        loss_fn = SkeletonEmbeddingLoss(
            delta_push=10.0, w_pull=2.0, w_push=3.0,
            w_penalty=0.5, w_benefit=10.0,
        )
        assert loss_fn.delta_push == 10.0
        assert loss_fn.w_pull == 2.0
        assert loss_fn.w_benefit == 10.0

    def test_3d_input(self) -> None:
        loss_fn = SkeletonEmbeddingLoss()
        D, H, W = 4, 8, 8
        offsets = torch.randn(1, 3, D, H, W)
        masks = torch.zeros(1, D, H, W, dtype=torch.long)
        masks[0, :, :4, :4] = 1
        masks[0, :, :4, 4:] = 2
        masks[0, :, 4:, :] = 3
        skel = torch.zeros(1, 3, D, H, W)
        skel[0, 0] = W / 4
        skel[0, 1] = H / 4
        skel[0, 2] = D / 2
        dt_n = torch.ones(1, 1, D, H, W) * 0.5
        dt_g = torch.randn(1, 3, D, H, W)
        result = loss_fn(offsets, masks, skel, dt_n, dt_g)
        for v in result.values():
            assert v.isfinite()

    def test_3d_backward_pass(self) -> None:
        loss_fn = SkeletonEmbeddingLoss()
        D, H, W = 4, 8, 8
        offsets = torch.randn(1, 3, D, H, W, requires_grad=True)
        masks = torch.zeros(1, D, H, W, dtype=torch.long)
        masks[0, :, :4, :] = 1
        masks[0, :, 4:, :] = 2
        skel = torch.zeros(1, 3, D, H, W)
        dt_n = torch.ones(1, 1, D, H, W) * 0.5
        dt_g = torch.randn(1, 3, D, H, W)
        result = loss_fn(offsets, masks, skel, dt_n, dt_g)
        result["loss"].backward()
        assert offsets.grad is not None
        assert offsets.grad.isfinite().all()

    def test_3d_single_instance_no_push(self) -> None:
        loss_fn = SkeletonEmbeddingLoss()
        D, H, W = 4, 8, 8
        offsets = torch.randn(1, 3, D, H, W)
        masks = torch.ones(1, D, H, W, dtype=torch.long)
        skel = torch.zeros(1, 3, D, H, W)
        dt_n = torch.ones(1, 1, D, H, W)
        dt_g = torch.ones(1, 3, D, H, W)
        result = loss_fn(offsets, masks, skel, dt_n, dt_g)
        assert result["l_push"].item() == 0.0

    def test_repr(self) -> None:
        loss_fn = SkeletonEmbeddingLoss(delta_push=15.0)
        r = repr(loss_fn)
        assert "SkeletonEmbeddingLoss" in r
        assert "delta_push=15.0" in r


class TestVista2DLoss:
    """Tests for Vista2DLoss."""

    @pytest.fixture()
    def loss_fn(self) -> Vista2DLoss:
        return Vista2DLoss(
            weight_pull=1.0,
            weight_push=1.0,
            weight_norm=0.001,
            weight_edge=1.0,
            weight_bone=1.0,
            delta_v=0.5,
            delta_d=1.5,
        )

    @pytest.fixture()
    def sample_inputs(self):
        B, H, W = 2, 16, 16
        predictions = {
            "semantic": torch.randn(B, 16, H, W),
            "instance": torch.randn(B, 16, H, W),
        }
        labels = torch.zeros(B, H, W, dtype=torch.long)
        labels[:, :8, :8] = 1
        labels[:, :8, 8:] = 2
        targets = {
            "class_labels": (labels > 0).long(),
            "labels": labels,
        }
        return predictions, targets

    def test_forward_returns_required_keys(self, loss_fn, sample_inputs) -> None:
        predictions, targets = sample_inputs
        result = loss_fn(predictions, targets)
        for key in ("loss", "loss_sem", "loss_ins"):
            assert key in result

    def test_total_loss_is_finite(self, loss_fn, sample_inputs) -> None:
        predictions, targets = sample_inputs
        result = loss_fn(predictions, targets)
        assert result["loss"].isfinite()

    def test_sub_losses_non_negative(self, loss_fn, sample_inputs) -> None:
        predictions, targets = sample_inputs
        result = loss_fn(predictions, targets)
        assert result["loss_sem"].item() >= 0.0

    def test_backward_pass(self, loss_fn, sample_inputs) -> None:
        predictions, targets = sample_inputs
        for v in predictions.values():
            v.requires_grad_(True)
        result = loss_fn(predictions, targets)
        result["loss"].backward()
        for v in predictions.values():
            assert v.grad is not None

    def test_zero_instances_no_error(self, loss_fn) -> None:
        B, H, W = 1, 8, 8
        predictions = {
            "semantic": torch.randn(B, 16, H, W),
            "instance": torch.randn(B, 16, H, W),
        }
        targets = {
            "class_labels": torch.zeros(B, H, W, dtype=torch.long),
            "labels": torch.zeros(B, H, W, dtype=torch.long),
        }
        result = loss_fn(predictions, targets)
        assert result["loss"].isfinite()

    def test_boundary_weight_activates(self) -> None:
        loss_fn = Vista2DLoss(weight_edge=10.0, weight_bone=1.0)
        B, H, W = 1, 16, 16
        labels = torch.zeros(B, H, W, dtype=torch.long)
        labels[:, :8, :] = 1
        w = loss_fn._get_weight_boundary(labels)
        assert w.shape == (B, H, W)
        assert w.max() > 1.0

    def test_custom_hyperparameters(self) -> None:
        loss_fn = Vista2DLoss(
            weight_pull=2.0, weight_push=3.0, weight_norm=0.01,
            delta_v=0.3, delta_d=2.0,
        )
        assert loss_fn.weight_pull == 2.0
        assert loss_fn.weight_push == 3.0
        assert loss_fn.delta_v == 0.3
        assert loss_fn.delta_d == 2.0


class TestVista3DLoss:
    """Tests for Vista3DLoss."""

    @pytest.fixture()
    def loss_fn(self) -> Vista3DLoss:
        return Vista3DLoss(
            weight_pull=1.0,
            weight_push=1.0,
            weight_norm=0.001,
            weight_edge=1.0,
            weight_bone=1.0,
            delta_v=0.5,
            delta_d=1.5,
        )

    @pytest.fixture()
    def sample_inputs(self):
        B, D, H, W = 1, 4, 8, 8
        predictions = {
            "semantic": torch.randn(B, 16, D, H, W),
            "instance": torch.randn(B, 16, D, H, W),
        }
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        labels[:, :, :4, :4] = 1
        labels[:, :, :4, 4:] = 2
        targets = {
            "class_labels": (labels > 0).long(),
            "labels": labels,
        }
        return predictions, targets

    def test_forward_returns_required_keys(self, loss_fn, sample_inputs) -> None:
        predictions, targets = sample_inputs
        result = loss_fn(predictions, targets)
        for key in ("loss", "loss_sem", "loss_ins"):
            assert key in result

    def test_total_loss_is_finite(self, loss_fn, sample_inputs) -> None:
        predictions, targets = sample_inputs
        result = loss_fn(predictions, targets)
        assert result["loss"].isfinite()

    def test_backward_pass(self, loss_fn, sample_inputs) -> None:
        predictions, targets = sample_inputs
        for v in predictions.values():
            v.requires_grad_(True)
        result = loss_fn(predictions, targets)
        result["loss"].backward()
        for v in predictions.values():
            assert v.grad is not None

    def test_boundary_weight_3d(self) -> None:
        loss_fn = Vista3DLoss(weight_edge=10.0, weight_bone=1.0)
        B, D, H, W = 1, 4, 8, 8
        labels = torch.zeros(B, D, H, W, dtype=torch.long)
        labels[:, :, :4, :] = 1
        w = loss_fn._get_weight_boundary(labels)
        assert w.shape == (B, D, H, W)
        assert w.max() > 1.0

    def test_zero_instances_no_error(self, loss_fn) -> None:
        B, D, H, W = 1, 4, 8, 8
        predictions = {
            "semantic": torch.randn(B, 16, D, H, W),
            "instance": torch.randn(B, 16, D, H, W),
        }
        targets = {
            "class_labels": torch.zeros(B, D, H, W, dtype=torch.long),
            "labels": torch.zeros(B, D, H, W, dtype=torch.long),
        }
        result = loss_fn(predictions, targets)
        assert result["loss"].isfinite()

    def test_single_instance(self, loss_fn) -> None:
        B, D, H, W = 1, 4, 8, 8
        predictions = {
            "semantic": torch.randn(B, 16, D, H, W),
            "instance": torch.randn(B, 16, D, H, W),
        }
        labels = torch.ones(B, D, H, W, dtype=torch.long)
        targets = {
            "class_labels": torch.ones(B, D, H, W, dtype=torch.long),
            "labels": labels,
        }
        result = loss_fn(predictions, targets)
        assert result["loss"].isfinite()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

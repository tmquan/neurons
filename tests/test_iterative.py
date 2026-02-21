"""
Tests for iterative/interactive training components.
"""

import pytest
import torch

from neurons.utils.point_sampling import sample_point_prompts
from neurons.models.point_prompt_encoder import PointPromptEncoder


# ---------------------------------------------------------------------------
# sample_point_prompts
# ---------------------------------------------------------------------------

class TestSamplePointPrompts:

    def test_class_mode_returns_correct_keys(self) -> None:
        sem = torch.zeros(2, 8, 8, dtype=torch.long)
        sem[0, :4, :] = 1
        sem[1, 4:, :] = 2
        inst = torch.zeros(2, 8, 8, dtype=torch.long)
        inst[0, :4, :] = 1
        inst[1, 4:, :] = 2
        result = sample_point_prompts(sem, inst, num_pos=3, num_neg=3, sample_mode="class")
        assert "pos_points" in result
        assert "neg_points" in result
        assert "target_semantic_ids" in result
        assert "target_instance_ids" in result

    def test_class_mode_shapes(self) -> None:
        sem = torch.zeros(2, 8, 8, dtype=torch.long)
        sem[:, :4, :] = 1
        inst = sem.clone()
        result = sample_point_prompts(sem, inst, num_pos=5, num_neg=3, sample_mode="class")
        assert len(result["pos_points"]) == 2
        assert len(result["neg_points"]) == 2
        assert result["pos_points"][0].shape == (5, 2)
        assert result["neg_points"][0].shape == (3, 2)
        assert result["target_semantic_ids"].shape == (2,)
        assert result["target_instance_ids"].shape == (2,)

    def test_instance_mode_shapes(self) -> None:
        sem = torch.zeros(1, 8, 8, dtype=torch.long)
        sem[:, :4, :4] = 1
        inst = torch.zeros(1, 8, 8, dtype=torch.long)
        inst[:, :4, :4] = 5
        result = sample_point_prompts(sem, inst, num_pos=4, num_neg=4, sample_mode="instance")
        assert result["pos_points"][0].shape == (4, 2)
        assert result["neg_points"][0].shape == (4, 2)
        assert result["target_instance_ids"][0].item() == 5
        assert result["target_semantic_ids"][0].item() == 1

    def test_all_background_returns_empty(self) -> None:
        sem = torch.zeros(1, 8, 8, dtype=torch.long)
        inst = torch.zeros(1, 8, 8, dtype=torch.long)
        result = sample_point_prompts(sem, inst, num_pos=5, num_neg=5)
        assert result["pos_points"][0].shape[0] == 0
        assert result["neg_points"][0].shape[0] == 0

    def test_3d_input(self) -> None:
        sem = torch.zeros(1, 4, 8, 8, dtype=torch.long)
        sem[:, :, :4, :4] = 1
        inst = torch.zeros(1, 4, 8, 8, dtype=torch.long)
        inst[:, :, :4, :4] = 1
        result = sample_point_prompts(sem, inst, num_pos=3, num_neg=3)
        assert result["pos_points"][0].shape == (3, 3)
        assert result["neg_points"][0].shape == (3, 3)

    def test_pos_points_inside_target(self) -> None:
        sem = torch.zeros(1, 16, 16, dtype=torch.long)
        sem[:, :8, :] = 1
        sem[:, 8:, :] = 2
        inst = sem.clone()
        result = sample_point_prompts(sem, inst, num_pos=10, num_neg=10, sample_mode="class")
        target_cls = result["target_semantic_ids"][0].item()
        for pt in result["pos_points"][0]:
            y, x = pt[0].item(), pt[1].item()
            assert sem[0, y, x].item() == target_cls


# ---------------------------------------------------------------------------
# PointPromptEncoder
# ---------------------------------------------------------------------------

class TestPointPromptEncoder:

    @pytest.fixture()
    def encoder_3d(self):
        return PointPromptEncoder(num_classes=4, feature_size=16, spatial_dims=3)

    @pytest.fixture()
    def encoder_2d(self):
        return PointPromptEncoder(num_classes=4, feature_size=16, spatial_dims=2)

    def test_output_shape_3d(self, encoder_3d) -> None:
        pos = [torch.tensor([[1, 2, 3], [2, 3, 4]])]
        neg = [torch.tensor([[0, 0, 0]])]
        sem_ids = torch.tensor([1])
        ins_ids = torch.tensor([5])
        out = encoder_3d(pos, neg, sem_ids, ins_ids, spatial_shape=(8, 8, 8))
        assert out.shape == (1, 16, 8, 8, 8)

    def test_output_shape_2d(self, encoder_2d) -> None:
        pos = [torch.tensor([[2, 3], [4, 5]])]
        neg = [torch.tensor([[0, 0]])]
        sem_ids = torch.tensor([2])
        ins_ids = torch.tensor([3])
        out = encoder_2d(pos, neg, sem_ids, ins_ids, spatial_shape=(8, 8))
        assert out.shape == (1, 16, 8, 8)

    def test_empty_points_near_zero(self, encoder_3d) -> None:
        """With no points the sparse volume is all zeros; output is small (BN bias only)."""
        encoder_3d.eval()
        pos = [torch.zeros(0, 3, dtype=torch.long)]
        neg = [torch.zeros(0, 3, dtype=torch.long)]
        sem_ids = torch.tensor([0])
        ins_ids = torch.tensor([0])
        out = encoder_3d(pos, neg, sem_ids, ins_ids, spatial_shape=(4, 4, 4))
        assert out.abs().max().item() < 0.2

    def test_backward_pass(self, encoder_3d) -> None:
        pos = [torch.tensor([[1, 2, 3]])]
        neg = [torch.tensor([[0, 0, 0]])]
        sem_ids = torch.tensor([1])
        ins_ids = torch.tensor([1])
        out = encoder_3d(pos, neg, sem_ids, ins_ids, spatial_shape=(4, 4, 4))
        out.sum().backward()
        for p in encoder_3d.parameters():
            assert p.grad is not None

    def test_batch_size_2(self, encoder_2d) -> None:
        pos = [torch.tensor([[1, 2]]), torch.tensor([[3, 4], [5, 6]])]
        neg = [torch.tensor([[0, 0]]), torch.tensor([[7, 7]])]
        sem_ids = torch.tensor([1, 2])
        ins_ids = torch.tensor([10, 20])
        out = encoder_2d(pos, neg, sem_ids, ins_ids, spatial_shape=(8, 8))
        assert out.shape == (2, 16, 8, 8)


# ---------------------------------------------------------------------------
# Module-level helpers (import the 3D module for testing)
# ---------------------------------------------------------------------------

class TestModuleHelpers:
    """Test _get_proofread_sub_mode and _resolve_fractionary_labels."""

    @pytest.fixture()
    def module(self):
        from neurons.modules.vista3d_module import Vista3DModule
        return Vista3DModule(
            loss_config={"ignore_index": -100},
            training_config={"training_modes": ["automatic"]},
        )

    def test_sub_mode_interactive_for_full_annotation(self, module) -> None:
        targets = {
            "labels": torch.tensor([[[1, 2], [3, 0]]]),
            "semantic_labels": torch.tensor([[[1, 1], [1, 0]]]),
        }
        assert module._get_proofread_sub_mode(targets) == "interactive"

    def test_sub_mode_fractionary_for_partial_annotation(self, module) -> None:
        labels = torch.tensor([[[1, -100], [2, 0]]])
        targets = {
            "labels": labels,
            "semantic_labels": torch.tensor([[[1, -100], [1, 0]]]),
        }
        assert module._get_proofread_sub_mode(targets) == "fractionary"

    def test_resolve_fractionary_labels(self, module) -> None:
        labels = torch.tensor([[[1, -100], [5, 0]]])
        sem = torch.tensor([[[1, -100], [1, 0]]])
        targets = {"labels": labels, "semantic_labels": sem}
        resolved = module._resolve_fractionary_labels(targets)
        assert (resolved["semantic_labels"][0, 0, 1] == -100)
        assert resolved["labels"][0, 0, 1] == 0
        assert resolved["semantic_ids"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

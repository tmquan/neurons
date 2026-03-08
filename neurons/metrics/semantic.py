"""
Semantic segmentation metrics.

Provides per-point and per-batch variants of:
- Dice coefficient  (2|P∩G| / (|P|+|G|))
- IoU / Jaccard     (|P∩G| / |P∪G|)

Both support multi-class evaluation with an optional ``ignore_index``.
"""

from typing import Optional

import torch


# ======================================================================
# Dice
# ======================================================================

def compute_per_point_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    ignore_index: int = -100,
    eps: float = 1e-7,
) -> float:
    """Mean Dice coefficient across classes for a single sample.

    Args:
        pred: Predicted class labels [H, W] or [D, H, W] (int/long).
        target: Ground truth class labels, same shape.
        num_classes: Number of classes.
        ignore_index: Label value to ignore (default -100).
        eps: Smoothing to avoid division by zero.

    Returns:
        Mean Dice in [0, 1].
    """
    from einops import rearrange as _r
    pred_flat = _r(pred.cpu().long(), "... -> (...)")
    tgt_flat = _r(target.cpu().long(), "... -> (...)")

    valid = tgt_flat != ignore_index
    pred_flat = pred_flat[valid]
    tgt_flat = tgt_flat[valid]

    classes = torch.arange(num_classes, device=pred_flat.device)
    pred_onehot = pred_flat.unsqueeze(0) == classes.unsqueeze(1)
    tgt_onehot = tgt_flat.unsqueeze(0) == classes.unsqueeze(1)

    intersection = (pred_onehot & tgt_onehot).sum(dim=1).float()
    pred_sum = pred_onehot.sum(dim=1).float()
    tgt_sum = tgt_onehot.sum(dim=1).float()

    present = (pred_sum > 0) | (tgt_sum > 0)
    if not present.any():
        return 0.0

    dice = (2.0 * intersection + eps) / (pred_sum + tgt_sum + eps)
    return dice[present].mean().item()


def compute_per_batch_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    ignore_index: int = -100,
    eps: float = 1e-7,
) -> float:
    """Mean Dice averaged over a batch [B, ...]."""
    total, count = 0.0, 0
    for b in range(pred.shape[0]):
        total += compute_per_point_dice(pred[b], target[b], num_classes, ignore_index, eps)
        count += 1
    return total / count if count > 0 else 0.0


# ======================================================================
# IoU (Jaccard)
# ======================================================================

def compute_per_point_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    ignore_index: int = -100,
    eps: float = 1e-7,
) -> float:
    """Mean IoU (Jaccard) across classes for a single sample.

    Args:
        pred: Predicted class labels [H, W] or [D, H, W] (int/long).
        target: Ground truth class labels, same shape.
        num_classes: Number of classes.
        ignore_index: Label value to ignore (default -100).
        eps: Smoothing to avoid division by zero.

    Returns:
        Mean IoU in [0, 1].
    """
    from einops import rearrange as _r
    pred_flat = _r(pred.cpu().long(), "... -> (...)")
    tgt_flat = _r(target.cpu().long(), "... -> (...)")

    valid = tgt_flat != ignore_index
    pred_flat = pred_flat[valid]
    tgt_flat = tgt_flat[valid]

    classes = torch.arange(num_classes, device=pred_flat.device)
    pred_onehot = pred_flat.unsqueeze(0) == classes.unsqueeze(1)
    tgt_onehot = tgt_flat.unsqueeze(0) == classes.unsqueeze(1)

    intersection = (pred_onehot & tgt_onehot).sum(dim=1).float()
    union = (pred_onehot | tgt_onehot).sum(dim=1).float()

    present = (pred_onehot.sum(dim=1) > 0) | (tgt_onehot.sum(dim=1) > 0)
    if not present.any():
        return 0.0

    iou = (intersection + eps) / (union + eps)
    return iou[present].mean().item()


def compute_per_batch_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    ignore_index: int = -100,
    eps: float = 1e-7,
) -> float:
    """Mean IoU averaged over a batch [B, ...]."""
    total, count = 0.0, 0
    for b in range(pred.shape[0]):
        total += compute_per_point_iou(pred[b], target[b], num_classes, ignore_index, eps)
        count += 1
    return total / count if count > 0 else 0.0

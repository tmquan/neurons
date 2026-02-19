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
    pred_flat = pred.detach().cpu().long().reshape(-1)
    tgt_flat = target.detach().cpu().long().reshape(-1)

    valid = tgt_flat != ignore_index
    pred_flat = pred_flat[valid]
    tgt_flat = tgt_flat[valid]

    dice_sum, n = 0.0, 0
    for c in range(num_classes):
        p = pred_flat == c
        t = tgt_flat == c
        if t.sum() == 0 and p.sum() == 0:
            continue
        intersection = (p & t).sum().float()
        dice = (2.0 * intersection + eps) / (p.sum().float() + t.sum().float() + eps)
        dice_sum += dice.item()
        n += 1

    return dice_sum / n if n > 0 else 0.0


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
    pred_flat = pred.detach().cpu().long().reshape(-1)
    tgt_flat = target.detach().cpu().long().reshape(-1)

    valid = tgt_flat != ignore_index
    pred_flat = pred_flat[valid]
    tgt_flat = tgt_flat[valid]

    iou_sum, n = 0.0, 0
    for c in range(num_classes):
        p = pred_flat == c
        t = tgt_flat == c
        if t.sum() == 0 and p.sum() == 0:
            continue
        intersection = (p & t).sum().float()
        union = (p | t).sum().float()
        iou = (intersection + eps) / (union + eps)
        iou_sum += iou.item()
        n += 1

    return iou_sum / n if n > 0 else 0.0


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

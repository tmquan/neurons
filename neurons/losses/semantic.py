"""
Semantic segmentation loss: CE + IoU + Dice.

Dimension-agnostic — works for both 2-D and 3-D inputs.

Supports two activation modes:

- **sigmoid**: independent per-channel binary CE (multi-label).
- **softmax**: mutually-exclusive CE via ``nn.CrossEntropyLoss``.

Both modes correctly handle ``ignore_index`` in all three sub-losses.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SemanticLoss(nn.Module):
    """Semantic segmentation loss.

    ``loss = w_ce * CE + w_iou * (1 - SoftIoU) + w_dice * (1 - SoftDice)``

    Args:
        mode: ``"sigmoid"`` for multi-label or ``"softmax"`` for exclusive.
        weight_ce: Weight for the cross-entropy term.
        weight_iou: Weight for the IoU term (0 to disable).
        weight_dice: Weight for the Dice term (0 to disable).
        class_weights: Per-class weights for the CE loss.
        ignore_index: Label value to exclude from all loss terms.
        active_classes: Number of leading channels to include in the loss.
            ``None`` means use all channels.  Set to e.g. 2 when the model
            outputs 16 channels but only classes 0-1 have labels today;
            channels beyond ``active_classes`` receive zero gradient.
    """

    def __init__(
        self,
        mode: str = "sigmoid",
        weight_ce: float = 1.0,
        weight_iou: float = 0.0,
        weight_dice: float = 0.0,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -100,
        active_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mode not in ("sigmoid", "softmax"):
            raise ValueError(f"mode must be 'sigmoid' or 'softmax', got '{mode}'")
        self.mode = mode
        self.weight_ce = weight_ce
        self.weight_iou = weight_iou
        self.weight_dice = weight_dice
        self.ignore_index = ignore_index
        self.active_classes = active_classes

        cw = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        if mode == "softmax":
            self.ce_loss = nn.CrossEntropyLoss(weight=cw, ignore_index=ignore_index)
        else:
            self.ce_loss = nn.BCEWithLogitsLoss(pos_weight=cw, reduction="none")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _slice_active(self, logits, class_labels):
        """Restrict logits to the first ``active_classes`` channels.

        In softmax mode the label values are clamped to [0, active_classes).
        Returns (logits_sliced, class_labels_adjusted).
        """
        if self.active_classes is None or self.active_classes >= logits.shape[1]:
            return logits, class_labels
        C = self.active_classes
        logits = logits[:, :C]
        if self.mode == "softmax":
            class_labels = class_labels.clone()
            valid = class_labels != self.ignore_index
            class_labels[valid] = class_labels[valid].clamp(0, C - 1)
        return logits, class_labels

    def _to_probs_and_target(self, logits, class_labels):
        """Convert logits + labels -> (probs [B,C,*], target [B,C,*], valid [B,1,*]).

        Returns a valid mask so that callers can exclude ``ignore_index``
        pixels from soft overlap losses regardless of mode.
        """
        C = logits.shape[1]

        if self.mode == "softmax":
            probs = F.softmax(logits, dim=1)
            valid = class_labels != self.ignore_index
            safe = class_labels.clone()
            safe[~valid] = 0
            one_hot = rearrange(
                F.one_hot(safe.long(), C).float(),
                "b ... c -> b c ...",
            )
            valid_mask = rearrange(valid.float(), "b ... -> b 1 ...")
            return probs * valid_mask, one_hot * valid_mask, valid_mask

        # sigmoid mode
        probs = torch.sigmoid(logits)

        if class_labels.dim() == logits.dim():
            valid_mask = torch.ones(
                logits.shape[0], 1, *logits.shape[2:],
                device=logits.device, dtype=torch.float32,
            )
            return probs, class_labels.float(), valid_mask

        safe = class_labels.clone().long()
        neg = safe < 0
        safe[neg] = 0
        safe = safe.clamp(0, C - 1)
        target = rearrange(
            F.one_hot(safe, C).float(),
            "b ... c -> b c ...",
        )
        neg_mask = rearrange(neg, "b ... -> b 1 ...")
        target[neg_mask.expand_as(target)] = 0.0

        valid_mask = rearrange((~neg).float(), "b ... -> b 1 ...")
        return probs * valid_mask, target * valid_mask, valid_mask

    def _compute_ce(self, logits, class_labels):
        logits, class_labels = self._slice_active(logits, class_labels)
        if self.mode == "softmax":
            return self.ce_loss(logits, class_labels)

        _, target, valid_mask = self._to_probs_and_target(logits, class_labels)
        per_pixel = self.ce_loss(logits, target)                   # [B, C, *spatial]
        per_pixel = per_pixel * valid_mask
        n_valid = valid_mask.sum().clamp(min=1.0) * per_pixel.shape[1]
        return per_pixel.sum() / n_valid

    def _iou_loss(self, logits, class_labels, eps=1e-5):
        """1 - mean(IoU) over present classes, ignoring invalid pixels.

        Only classes with non-zero target mass in the batch contribute
        to the average, avoiding gradient dilution from absent classes.
        """
        logits, class_labels = self._slice_active(logits, class_labels)
        probs, target, _ = self._to_probs_and_target(logits, class_labels)
        spatial = tuple(range(2, probs.dim()))
        inter = (probs * target).sum(dim=spatial)                  # [B, C]
        union = probs.sum(dim=spatial) + target.sum(dim=spatial) - inter
        iou_per_class = (inter + eps) / (union + eps)              # [B, C]
        present = target.sum(dim=spatial) > 0                      # [B, C]
        n_present = present.sum().clamp(min=1.0)
        return 1.0 - (iou_per_class * present).sum() / n_present

    def _dice_loss(self, logits, class_labels, eps=1e-5):
        """1 - mean(Dice) over present classes, ignoring invalid pixels.

        Only classes with non-zero target mass in the batch contribute
        to the average, avoiding gradient dilution from absent classes.
        """
        logits, class_labels = self._slice_active(logits, class_labels)
        probs, target, _ = self._to_probs_and_target(logits, class_labels)
        spatial = tuple(range(2, probs.dim()))
        inter = (probs * target).sum(dim=spatial)                  # [B, C]
        card = probs.sum(dim=spatial) + target.sum(dim=spatial)
        dice_per_class = (2.0 * inter + eps) / (card + eps)       # [B, C]
        present = target.sum(dim=spatial) > 0                      # [B, C]
        n_present = present.sum().clamp(min=1.0)
        return 1.0 - (dice_per_class * present).sum() / n_present

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, logits, class_labels) -> Dict[str, torch.Tensor]:
        dev = logits.device
        ce = self._compute_ce(logits, class_labels)
        iou = self._iou_loss(logits, class_labels) if self.weight_iou > 0 else torch.tensor(0.0, device=dev)
        dice = self._dice_loss(logits, class_labels) if self.weight_dice > 0 else torch.tensor(0.0, device=dev)
        loss = self.weight_ce * ce + self.weight_iou * iou + self.weight_dice * dice
        return {"loss": loss, "ce": ce, "iou": iou, "dice": dice}

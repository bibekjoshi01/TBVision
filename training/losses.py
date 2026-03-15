import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)

        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        return self.bce(preds, targets) + self.dice(preds, targets)


class FocalLoss(nn.Module):
    """Multi-class focal loss built on top of CrossEntropy."""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean", ignore_index=None):
        super().__init__()
        if alpha is not None:
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha_tensor)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        weight = self.alpha
        if weight is not None and weight.device != logits.device:
            weight = weight.to(logits.device)

        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=weight,
            reduction="none",
            ignore_index=self.ignore_index,
        )

        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1 - pt) ** self.gamma
        loss = focal_factor * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

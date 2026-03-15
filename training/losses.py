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
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.unsqueeze(1), reduction="none"
        )

        prob = torch.sigmoid(logits)
        pt = targets.unsqueeze(1) * prob + (1 - targets.unsqueeze(1)) * (1 - prob)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        
        return loss.mean()

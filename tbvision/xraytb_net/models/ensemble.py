import torch
import torch.nn as nn
from typing import List, Optional

from .classifier import TBClassifier


# Ensemble Model
# -------------------------------
class TBEnsemble(nn.Module):
    """Ensemble of multiple backbones that averages logits before applying the loss."""

    def __init__(
        self,
        backbones: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        use_mc_dropout: bool = False,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        if backbones is None:
            backbones = ["densenet121", "efficientnet_b3", "resnet50"]
        self.backbones = backbones
        self.models = nn.ModuleList(
            [TBClassifier(b, pretrained=True, dropout=dropout_rate) for b in backbones]
        )
        self.num_classes = self.models[0].num_classes

        if weights is None:
            weights = [1 / len(backbones)] * len(backbones)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", weight_tensor)

        self.use_mc_dropout = use_mc_dropout
        if use_mc_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = [model(x) for model in self.models]
        stacked = torch.stack(logits, dim=0)
        weighted = stacked * self.weights.view(-1, 1, 1)
        ensemble_logits = weighted.sum(dim=0)
        return ensemble_logits

    # MC Dropout for uncertainty
    # -------------------------------
    def _forward_with_dropout(self, x: torch.Tensor) -> torch.Tensor:
        logits = []
        for model in self.models:
            out = model(x)
            out = self.dropout(out)
            logits.append(out)
        stacked = torch.stack(logits, dim=0)
        weighted = stacked * self.weights.view(-1, 1, 1)
        return weighted.sum(dim=0)

    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 20):
        """Estimate predictive mean and STD over MC dropout forward passes."""
        assert self.use_mc_dropout, "MC Dropout must be enabled to estimate uncertainty."
        self.eval()
        self.dropout.train()

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self._forward_with_dropout(x)
                preds = torch.softmax(logits, dim=1)
                predictions.append(preds)

        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        return mean_pred, std_pred


# Utility: Load ensemble from checkpoint
# -------------------------------
def load_ensemble(
    checkpoint_path: Optional[str] = None, device: str = "cuda"
) -> TBEnsemble:
    model = TBEnsemble(use_mc_dropout=True)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Test ensemble
# -------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TBEnsemble(use_mc_dropout=True).to(device)

    x = torch.randn(2, 1, 224, 224).to(device)

    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

    # Uncertainty estimation
    mean, std = model.predict_with_uncertainty(x, n_samples=10)
    print(f"\nMean prediction: {mean}")
    print(f"Std prediction: {std}")

    print("\n✅ Ensemble test passed")

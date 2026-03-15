import torch
import torch.nn as nn
import torchvision.models as models
import timm


# Single Backbone Classifier
# -------------------------------
class TBClassifier(nn.Module):
    """Single backbone classifier for TB detection"""

    def __init__(
        self,
        backbone: str = "densenet121",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone

        # Initialize backbone
        if backbone == "densenet121":
            model = models.densenet121(pretrained=pretrained)
            in_features = model.classifier.in_features
            model.classifier = nn.Identity()
            self.encoder = model

        elif backbone == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            self.encoder = model

        elif backbone == "efficientnet_b3":
            model = timm.create_model(
                "efficientnet_b3", pretrained=pretrained, num_classes=0
            )
            in_features = model.num_features
            self.encoder = model

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)

        # Global average pool if 4D tensor
        if len(features.shape) == 4:
            features = torch.mean(features, dim=[2, 3])

        logits = self.classifier(features)
        return logits.squeeze(-1)  # [B]



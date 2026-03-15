import torch
import torch.nn as nn
import torchvision.models as models
import timm


class TBClassifier(nn.Module):
    """Single backbone classifier for multi-class (Normal / OtherDisease / TB)"""

    def __init__(
        self,
        backbone: str = "densenet121",
        pretrained: bool = True,
        dropout: float = 0.3,
        num_classes: int = 3,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        # Initialize backbone
        if backbone == "densenet121":
            model = models.densenet121(pretrained=pretrained)
            in_features = model.classifier.in_features
            model.classifier = nn.Identity()  # remove original classifier
            self.encoder = model

        elif backbone == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            in_features = model.fc.in_features
            model.fc = nn.Identity()  # remove original classifier
            self.encoder = model

        elif backbone.startswith("efficientnet"):
            # EfficientNet via timm
            model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            in_features = model.num_features
            self.encoder = model

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Classifier head for multi-class
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),  # 3 classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)

        # Global average pool if 4D tensor (B, C, H, W)
        if len(features.shape) == 4:
            features = features.mean(dim=[2, 3])

        logits = self.classifier(features)
        return logits

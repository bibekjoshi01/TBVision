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
            model.classifier = nn.Identity()
            self._adapt_input_conv(model.features, "conv0", in_channels=1)
            self.encoder = model

        elif backbone == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            self._adapt_input_conv(model, "conv1", in_channels=1)
            self.encoder = model

        elif backbone.startswith("efficientnet"):
            model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            in_features = model.num_features
            self._adapt_input_conv(model, "conv_stem", in_channels=1)
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

    def _adapt_input_conv(self, parent_module: nn.Module, attr_name: str, in_channels: int):
        conv_layer = getattr(parent_module, attr_name)
        if conv_layer.in_channels == in_channels:
            return

        new_conv = nn.Conv2d(
            in_channels,
            conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
            bias=conv_layer.bias is not None,
        )
        with torch.no_grad():
            weight = conv_layer.weight
            if weight.shape[1] == 3 and in_channels == 1:
                mean_weight = weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
            else:
                new_conv.weight.copy_(weight[:, :in_channels].clone())
            if conv_layer.bias is not None:
                new_conv.bias.copy_(conv_layer.bias.clone())

        parent_module._modules[attr_name] = new_conv
        setattr(parent_module, attr_name, new_conv)

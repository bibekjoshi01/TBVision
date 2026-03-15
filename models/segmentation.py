import segmentation_models_pytorch as smp
import torch.nn as nn


class LungUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = smp.Unet(
            encoder_name="timm-efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    def forward(self, x):
        return self.model(x)

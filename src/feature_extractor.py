import torch
import torch.nn as nn
from importlib import import_module


_resnet_models = import_module("torchvision.models")
resnet18 = _resnet_models.resnet18
ResNet18_Weights = _resnet_models.ResNet18_Weights


class ResNet18FeatureExtractor(nn.Module):
    """
    ResNet-18 feature extractor for PatchCore.

    Extracts intermediate feature maps from:
    - layer2
    - layer3
    """

    def __init__(self):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT

        backbone = resnet18(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # PatchCore baseline uses intermediate features.
        for parameter in self.parameters():
            parameter.requires_grad = False

        self.eval()

    @torch.no_grad()
    def forward(self, x):
        """
        Returns intermediate feature maps.
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        feature_layer2 = self.layer2(x)

        feature_layer3 = self.layer3(
            feature_layer2
        )

        return {
            "layer2": feature_layer2,
            "layer3": feature_layer3,
        }
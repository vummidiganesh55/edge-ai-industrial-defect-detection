import torch

from src.dataset import MVTecDataset
from src.preprocessing import get_train_transform
from src.feature_extractor import ResNet18FeatureExtractor


def test_feature_extractor_output():
    dataset = MVTecDataset(
        root_dir="data/mvtec_anomaly_detection",
        category="bottle",
        split="train",
        transform=get_train_transform(),
    )

    sample = dataset[0]
    image = sample["image"].unsqueeze(0)

    model = ResNet18FeatureExtractor()

    with torch.no_grad():
        features = model(image)

    # Required feature levels
    assert "layer2" in features
    assert "layer3" in features

    # Features must contain tensors
    assert isinstance(features["layer2"], torch.Tensor)
    assert isinstance(features["layer3"], torch.Tensor)

    # Batch dimension must match input batch size
    assert features["layer2"].shape[0] == image.shape[0]
    assert features["layer3"].shape[0] == image.shape[0]

    # Feature maps must have spatial dimensions
    assert features["layer2"].ndim == 4
    assert features["layer3"].ndim == 4
import torch

from src.dataset import MVTecDataset
from src.preprocessing import get_train_transform


def test_preprocessing_output():
    dataset = MVTecDataset(
        root_dir="data/mvtec_anomaly_detection",
        category="bottle",
        split="train",
        transform=get_train_transform(),
    )

    sample = dataset[0]
    image = sample["image"]

    # Output must be a PyTorch tensor
    assert isinstance(image, torch.Tensor)

    # ResNet preprocessing should produce 3-channel 224x224 input
    assert image.shape == (3, 224, 224)

    # Tensor should contain valid numeric values
    assert torch.isfinite(image).all()

    # Image must not be empty
    assert image.numel() > 0
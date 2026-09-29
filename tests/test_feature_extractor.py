import torch

from src.dataset import MVTecDataset
from src.preprocessing import get_train_transform
from src.feature_extractor import ResNet18FeatureExtractor


def main():

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

    print("=" * 60)
    print("Feature Extractor Test")
    print("=" * 60)

    print(f"Input shape : {image.shape}")

    for name, feature in features.items():
        print(
            f"{name:10s}: "
            f"shape={tuple(feature.shape)}, "
            f"dtype={feature.dtype}"
        )

    assert "layer2" in features
    assert "layer3" in features

    print("\nFeature extractor test PASSED.")


if __name__ == "__main__":
    main()
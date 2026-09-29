import torch

from src.dataset import MVTecDataset
from src.preprocessing import get_train_transform
from src.feature_extractor import ResNet18FeatureExtractor
from src.patchcore import PatchCore


def main():

    dataset = MVTecDataset(
        root_dir="data/mvtec_anomaly_detection",
        category="bottle",
        split="train",
        transform=get_train_transform(),
    )

    feature_extractor = ResNet18FeatureExtractor()

    patchcore = PatchCore(
        sampling_ratio=0.1
    )

    feature_batches = []

    print("=" * 60)
    print("PatchCore Test")
    print("=" * 60)

    # Use a few images first.
    for index in range(3):

        image = dataset[index]["image"]

        image = image.unsqueeze(0)

        with torch.no_grad():

            features = feature_extractor(
                image
            )

        feature_batches.append(
            features
        )

        print(
            f"Processed training image "
            f"{index + 1}/3"
        )

    patchcore.fit(
        feature_batches
    )

    print(
        "\nMemory bank shape:",
        patchcore.memory_bank.shape
    )

    print("\nPatchCore test PASSED.")


if __name__ == "__main__":
    main()
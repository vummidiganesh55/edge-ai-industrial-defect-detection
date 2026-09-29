import torch

from src.dataset import MVTecDataset
from src.preprocessing import get_train_transform


def main():

    dataset = MVTecDataset(
        root_dir="data/mvtec_anomaly_detection",
        category="bottle",
        split="train",
        transform=get_train_transform(),
    )

    sample = dataset[0]

    image = sample["image"]

    print("=" * 60)
    print("Preprocessing Test")
    print("=" * 60)

    print(f"Tensor type : {type(image)}")
    print(f"Shape       : {image.shape}")
    print(f"Dtype       : {image.dtype}")
    print(f"Min value   : {image.min().item():.4f}")
    print(f"Max value   : {image.max().item():.4f}")

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)

    print("\nPreprocessing test PASSED.")


if __name__ == "__main__":
    main()
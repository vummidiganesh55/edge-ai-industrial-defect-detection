import torch

from src.dataset import MVTecDataset
from src.preprocessing import get_train_transform
from src.feature_extractor import ResNet18FeatureExtractor
from src.patchcore import PatchCore


def test_patchcore_fit():
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

    # Use a few normal training images
    for index in range(3):
        image = dataset[index]["image"]
        image = image.unsqueeze(0)

        with torch.no_grad():
            features = feature_extractor(image)

        feature_batches.append(features)

    # Fit PatchCore memory bank
    patchcore.fit(feature_batches)

    # Memory bank must be created
    assert patchcore.memory_bank is not None

    # Memory bank must contain feature vectors
    assert patchcore.memory_bank.ndim == 2

    # Sampling should produce at least one feature
    assert patchcore.memory_bank.shape[0] > 0

    # Feature dimension should match PatchCore feature representation
    assert patchcore.memory_bank.shape[1] > 0
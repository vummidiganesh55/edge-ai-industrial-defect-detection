from src.dataset import MVTecDataset


DATASET_ROOT = "data/mvtec_anomaly_detection"


def test_mvtec_dataset_loading():
    train_dataset = MVTecDataset(
        root_dir=DATASET_ROOT,
        category="bottle",
        split="train",
    )

    test_dataset = MVTecDataset(
        root_dir=DATASET_ROOT,
        category="bottle",
        split="test",
    )

    # Dataset should contain samples
    assert len(train_dataset) > 0
    assert len(test_dataset) > 0

    # Inspect one training sample
    sample = train_dataset[0]

    # Required fields
    assert "image" in sample
    assert "label" in sample
    assert "defect_type" in sample
    assert "path" in sample

    # Basic image validation
    assert sample["image"] is not None
    assert sample["image"].size[0] > 0
    assert sample["image"].size[1] > 0

    # Label should be valid
    assert sample["label"] in [0, 1]

    # Path should exist
    assert sample["path"]
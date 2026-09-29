from src.dataset import MVTecDataset


DATASET_ROOT = "data/mvtec_anomaly_detection"


def main():

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

    print("=" * 60)
    print("MVTec AD Dataset Test")
    print("=" * 60)

    print(f"Category      : bottle")
    print(f"Train samples : {len(train_dataset)}")
    print(f"Test samples  : {len(test_dataset)}")

    sample = train_dataset[0]

    print("\nFirst training sample:")
    print(f"Image type    : {type(sample['image'])}")
    print(f"Image size    : {sample['image'].size}")
    print(f"Label         : {sample['label']}")
    print(f"Defect type   : {sample['defect_type']}")
    print(f"Path          : {sample['path']}")

    print("\nDataset test PASSED.")


if __name__ == "__main__":
    main()
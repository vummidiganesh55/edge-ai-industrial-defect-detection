import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import MVTecDataset
from src.preprocessing import get_train_transform
from src.feature_extractor import ResNet18FeatureExtractor
from src.patchcore import PatchCore


DATASET_ROOT = "data/mvtec_anomaly_detection"


def build_memory_bank(
    category="bottle",
    batch_size=8,
    sampling_ratio=0.1,
    indices=None,
):
    device = torch.device("cpu")

    print("=" * 60)
    print("PATCHCORE MEMORY BANK BUILD")
    print("=" * 60)

    print(f"Device          : {device}")
    print(f"Category        : {category}")
    print(f"Batch size      : {batch_size}")
    print(f"Sampling ratio  : {sampling_ratio}")

    dataset = MVTecDataset(
        root_dir=DATASET_ROOT,
        category=category,
        split="train",
        transform=get_train_transform(),
    )

    # --------------------------------------------------------
    # Optional training subset
    # --------------------------------------------------------
    if indices is not None:
        dataset_for_bank = Subset(
            dataset,
            indices,
        )

        print(
            f"Using training subset : "
            f"{len(dataset_for_bank)}/{len(dataset)}"
        )
    else:
        dataset_for_bank = dataset

        print(
            f"Using full training set : "
            f"{len(dataset)} images"
        )

    loader = DataLoader(
        dataset_for_bank,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    feature_extractor = (
        ResNet18FeatureExtractor()
        .to(device)
    )

    patchcore = PatchCore(
        sampling_ratio=sampling_ratio
    )

    feature_batches = []

    total_batches = len(loader)

    for batch_index, batch in enumerate(loader):

        images = batch["image"].to(device)

        with torch.no_grad():

            features = feature_extractor(
                images
            )

        feature_batches.append(
            features
        )

        print(
            f"Batch "
            f"{batch_index + 1}/{total_batches}"
        )

    patchcore.fit(
        feature_batches
    )

    print("\n" + "=" * 60)
    print("MEMORY BANK COMPLETE")
    print("=" * 60)

    print(
        f"Training images : "
        f"{len(dataset_for_bank)}"
    )

    print(
        f"Memory bank     : "
        f"{patchcore.memory_bank.shape}"
    )

    return patchcore


if __name__ == "__main__":

    build_memory_bank(
        category="bottle",
        batch_size=8,
        sampling_ratio=0.1,
    )
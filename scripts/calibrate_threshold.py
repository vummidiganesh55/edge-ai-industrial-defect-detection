import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import MVTecDataset
from src.preprocessing import get_test_transform
from src.feature_extractor import ResNet18FeatureExtractor
from src.patchcore_pipeline import build_memory_bank
from src.patchcore import PatchCore


DATASET_ROOT = "data/mvtec_anomaly_detection"
CATEGORY = "bottle"

CALIBRATION_RATIO = 0.20
TARGET_FPR = 0.05

RANDOM_SEED = 42

OUTPUT_PATH = Path(
    "outputs/threshold_calibration.json"
)


def main():

    print("=" * 60)
    print("PATCHCORE THRESHOLD CALIBRATION")
    print("=" * 60)

    device = torch.device("cpu")

    print(f"Device       : {device}")
    print(f"Category     : {CATEGORY}")
    print(f"Calibration  : {CALIBRATION_RATIO * 100:.0f}%")
    print(f"Target FPR   : {TARGET_FPR * 100:.0f}%")
    print(f"Random seed  : {RANDOM_SEED}")

    # ---------------------------------------------------------
    # 1. Load normal training dataset
    # ---------------------------------------------------------

    dataset = MVTecDataset(
        root_dir=DATASET_ROOT,
        category=CATEGORY,
        split="train",
        transform=get_test_transform(),
    )

    total_images = len(dataset)

    print(f"\nTotal normal images : {total_images}")

    # ---------------------------------------------------------
    # 2. Deterministic 80/20 split
    # ---------------------------------------------------------

    indices = list(range(total_images))

    random.Random(
        RANDOM_SEED
    ).shuffle(indices)

    calibration_size = int(
        total_images * CALIBRATION_RATIO
    )

    calibration_indices = indices[
        :calibration_size
    ]

    memory_indices = indices[
        calibration_size:
    ]

    print(
        f"Memory-bank images   : "
        f"{len(memory_indices)}"
    )

    print(
        f"Calibration images   : "
        f"{len(calibration_indices)}"
    )

    # ---------------------------------------------------------
    # 3. Build PatchCore memory bank
    #    IMPORTANT:
    #    calibration images are excluded.
    # ---------------------------------------------------------

    patchcore = build_memory_bank(
        category=CATEGORY,
        batch_size=8,
        sampling_ratio=0.1,
        indices=memory_indices,
    )

    # ---------------------------------------------------------
    # 4. Feature extractor
    # ---------------------------------------------------------

    feature_extractor = (
        ResNet18FeatureExtractor()
        .to(device)
    )

    feature_extractor.eval()

    # ---------------------------------------------------------
    # 5. Score calibration images
    # ---------------------------------------------------------

    scores = []

    print("\n" + "=" * 60)
    print("CALIBRATION SCORING")
    print("=" * 60)

    for counter, index in enumerate(
        calibration_indices,
        start=1,
    ):

        sample = dataset[index]

        image = sample["image"].unsqueeze(0)
        image = image.to(device)

        image_path = sample["path"]

        with torch.no_grad():

            features = feature_extractor(
                image
            )

        anomaly_score, _ = patchcore.predict(
            features
        )

        scores.append(
            {
                "path": image_path,
                "score": float(anomaly_score),
            }
        )

        print(
            f"{counter:02d}/{len(calibration_indices)} "
            f"{Path(image_path).name:<15} "
            f"score={anomaly_score:.6f}"
        )

    # ---------------------------------------------------------
    # 6. Calculate threshold
    # ---------------------------------------------------------

    score_values = np.array(
        [item["score"] for item in scores],
        dtype=np.float64,
    )

    percentile = (
        100.0 * (1.0 - TARGET_FPR)
    )

    threshold = float(
        np.percentile(
            score_values,
            percentile,
        )
    )

    # ---------------------------------------------------------
    # 7. Calibration statistics
    # ---------------------------------------------------------

    predictions = (
        score_values >= threshold
    )

    false_positives = int(
        predictions.sum()
    )

    empirical_fpr = (
        false_positives /
        len(score_values)
    )

    print("\n" + "=" * 60)
    print("THRESHOLD CALIBRATION RESULT")
    print("=" * 60)

    print(
        f"Calibration samples : "
        f"{len(score_values)}"
    )

    print(
        f"Minimum score       : "
        f"{score_values.min():.6f}"
    )

    print(
        f"Maximum score       : "
        f"{score_values.max():.6f}"
    )

    print(
        f"Mean score          : "
        f"{score_values.mean():.6f}"
    )

    print(
        f"Median score        : "
        f"{np.median(score_values):.6f}"
    )

    print(
        f"Percentile          : "
        f"{percentile:.1f}"
    )

    print(
        f"Calibrated threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"Empirical FPR       : "
        f"{empirical_fpr:.4f}"
    )

    # ---------------------------------------------------------
    # 8. Save calibration artifact
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "category": CATEGORY,
        "random_seed": RANDOM_SEED,
        "total_training_images": total_images,
        "memory_bank_images": len(memory_indices),
        "calibration_images": len(
            calibration_indices
        ),
        "target_fpr": TARGET_FPR,
        "percentile": percentile,
        "threshold": threshold,
        "empirical_fpr": empirical_fpr,
        "min_score": float(score_values.min()),
        "max_score": float(score_values.max()),
        "mean_score": float(score_values.mean()),
        "median_score": float(
            np.median(score_values)
        ),
        "scores": scores,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    print(
        f"\nSaved calibration: "
        f"{OUTPUT_PATH}"
    )

    print("\nCalibration complete.")


if __name__ == "__main__":
    main()
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import MVTecDataset
from src.preprocessing import get_test_transform
from src.feature_extractor import ResNet18FeatureExtractor
from src.patchcore_pipeline import build_memory_bank
from src.config import DEFAULT_THRESHOLD


DATASET_ROOT = "data/mvtec_anomaly_detection"
OUTPUT_DIR = Path("outputs/evaluation")

CATEGORY = "bottle"
BATCH_SIZE = 1
SAMPLING_RATIO = 0.1


def run_failure_analysis(category=CATEGORY):
    device = torch.device("cpu")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("PATCHCORE FAILURE ANALYSIS")
    print("=" * 60)
    print(f"Device   : {device}")
    print(f"Category : {category}")

    # --------------------------------------------------
    # Build PatchCore
    # --------------------------------------------------

    patchcore = build_memory_bank(
        category=category,
        batch_size=8,
        sampling_ratio=SAMPLING_RATIO,
    )

    # --------------------------------------------------
    # Test dataset
    # --------------------------------------------------

    dataset = MVTecDataset(
        root_dir=DATASET_ROOT,
        category=category,
        split="test",
        transform=get_test_transform(),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    feature_extractor = (
        ResNet18FeatureExtractor()
        .to(device)
    )

    feature_extractor.eval()

    results = []

    # --------------------------------------------------
    # Run inference
    # --------------------------------------------------

    with torch.no_grad():

        for index, batch in enumerate(loader):

            image = batch["image"].to(device)

            features = feature_extractor(image)

            anomaly_score, _ = patchcore.predict(
                features
            )

            label = int(
                batch["label"].item()
            )

            defect_type = batch[
                "defect_type"
            ][0]

            image_path = batch[
                "path"
            ][0]

            results.append(
                {
                    "index": index,
                    "path": image_path,
                    "label": label,
                    "defect_type": defect_type,
                    "score": float(anomaly_score),
                }
            )

    # --------------------------------------------------
    # Production threshold
    #
    # IMPORTANT:
    # Use the calibrated production threshold
    # loaded from src.config.
    #
    # The value is controlled by:
    #
    # configs/production.yaml
    #
    # Current calibrated value:
    # 17.364517
    # --------------------------------------------------

    threshold = DEFAULT_THRESHOLD

    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS")
    print("=" * 60)

    print(
        f"Decision threshold : {threshold:.4f}"
    )

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    false_positives = []
    false_negatives = []

    normal_images = []
    defect_images = []

    for result in results:

        predicted = (
            1
            if result["score"] >= threshold
            else 0
        )

        result["prediction"] = predicted

        if result["label"] == 0:

            normal_images.append(result)

            if predicted == 1:
                false_positives.append(result)

        else:

            defect_images.append(result)

            if predicted == 0:
                false_negatives.append(result)

    # --------------------------------------------------
    # False positives
    # --------------------------------------------------

    print("\n" + "-" * 60)
    print("FALSE POSITIVES")
    print("-" * 60)

    if false_positives:

        for result in false_positives:

            print(
                f"Index={result['index']:02d} | "
                f"Type={result['defect_type']:20s} | "
                f"Score={result['score']:.4f} | "
                f"Path={result['path']}"
            )

    else:

        print("No false positives.")

    # --------------------------------------------------
    # False negatives
    # --------------------------------------------------

    print("\n" + "-" * 60)
    print("FALSE NEGATIVES")
    print("-" * 60)

    if false_negatives:

        for result in false_negatives:

            print(
                f"Index={result['index']:02d} | "
                f"Type={result['defect_type']:20s} | "
                f"Score={result['score']:.4f} | "
                f"Path={result['path']}"
            )

    else:

        print("No false negatives.")

    # --------------------------------------------------
    # Hardest normal images
    # --------------------------------------------------

    hardest_normals = sorted(
        normal_images,
        key=lambda x: x["score"],
        reverse=True,
    )

    print("\n" + "-" * 60)
    print("HARDEST NORMAL IMAGES")
    print("-" * 60)

    for result in hardest_normals[:10]:

        print(
            f"Index={result['index']:02d} | "
            f"Score={result['score']:.4f} | "
            f"Path={result['path']}"
        )

    # --------------------------------------------------
    # Hardest defect images
    # --------------------------------------------------

    hardest_defects = sorted(
        defect_images,
        key=lambda x: x["score"],
    )

    print("\n" + "-" * 60)
    print("HARDEST DEFECT IMAGES")
    print("-" * 60)

    for result in hardest_defects[:10]:

        print(
            f"Index={result['index']:02d} | "
            f"Type={result['defect_type']:20s} | "
            f"Score={result['score']:.4f} | "
            f"Path={result['path']}"
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FAILURE SUMMARY")
    print("=" * 60)

    print(
        f"Total images      : {len(results)}"
    )

    print(
        f"Normal images     : {len(normal_images)}"
    )

    print(
        f"Defect images     : {len(defect_images)}"
    )

    print(
        f"False positives   : "
        f"{len(false_positives)}"
    )

    print(
        f"False negatives   : "
        f"{len(false_negatives)}"
    )

    # --------------------------------------------------
    # Per-defect-type analysis
    # --------------------------------------------------

    print("\n" + "-" * 60)
    print("DEFECT-TYPE ANALYSIS")
    print("-" * 60)

    defect_types = sorted(
        set(
            result["defect_type"]
            for result in defect_images
        )
    )

    for defect_type in defect_types:

        type_results = [
            result
            for result in defect_images
            if result["defect_type"]
            == defect_type
        ]

        missed = [
            result
            for result in type_results
            if result["prediction"] == 0
        ]

        scores = [
            result["score"]
            for result in type_results
        ]

        print(
            f"{defect_type:20s} | "
            f"count={len(type_results):02d} | "
            f"min={min(scores):.4f} | "
            f"max={max(scores):.4f} | "
            f"mean={np.mean(scores):.4f} | "
            f"FN={len(missed)}"
        )

    # --------------------------------------------------
    # Save report
    # --------------------------------------------------

    report_path = (
        OUTPUT_DIR
        / "failure_analysis.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "PATCHCORE FAILURE ANALYSIS\n"
        )

        file.write(
            "=" * 60 + "\n"
        )

        file.write(
            f"Category: {category}\n"
        )

        file.write(
            f"Threshold: {threshold:.6f}\n\n"
        )

        file.write(
            f"Total images: {len(results)}\n"
        )

        file.write(
            f"Normal images: {len(normal_images)}\n"
        )

        file.write(
            f"Defect images: {len(defect_images)}\n"
        )

        file.write(
            f"False positives: "
            f"{len(false_positives)}\n"
        )

        file.write(
            f"False negatives: "
            f"{len(false_negatives)}\n\n"
        )

        file.write(
            "FALSE POSITIVES\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        for result in false_positives:

            file.write(
                f"Index={result['index']:02d} | "
                f"Score={result['score']:.6f} | "
                f"Path={result['path']}\n"
            )

        file.write(
            "\nFALSE NEGATIVES\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        for result in false_negatives:

            file.write(
                f"Index={result['index']:02d} | "
                f"Type={result['defect_type']} | "
                f"Score={result['score']:.6f} | "
                f"Path={result['path']}\n"
            )

        file.write(
            "\nHARDEST NORMAL IMAGES\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        for result in hardest_normals[:10]:

            file.write(
                f"Index={result['index']:02d} | "
                f"Score={result['score']:.6f} | "
                f"Path={result['path']}\n"
            )

        file.write(
            "\nHARDEST DEFECT IMAGES\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        for result in hardest_defects[:10]:

            file.write(
                f"Index={result['index']:02d} | "
                f"Type={result['defect_type']} | "
                f"Score={result['score']:.6f} | "
                f"Path={result['path']}\n"
            )

    print(
        f"\nSaved: {report_path}"
    )

    print("\n" + "=" * 60)
    print(
        "FAILURE ANALYSIS COMPLETE"
    )
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_failure_analysis(
        category=CATEGORY
    )
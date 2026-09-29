from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from src.config import DEFAULT_THRESHOLD
from src.dataset import MVTecDataset
from src.preprocessing import get_test_transform
from src.feature_extractor import ResNet18FeatureExtractor
from src.patchcore_pipeline import build_memory_bank


DATASET_ROOT = "data/mvtec_anomaly_detection"
OUTPUT_DIR = Path("outputs/evaluation")

CATEGORY = "bottle"

DECISION_THRESHOLD = DEFAULT_THRESHOLD


def evaluate_unseen_defect_type(
    dataset,
    defect_type,
    patchcore,
    feature_extractor,
    device,
):
    """
    Evaluate one defect type against normal images.

    Normal images:
        label = 0

    Selected unseen defect:
        label = 1
    """

    scores = []
    labels = []
    paths = []

    # --------------------------------------------------
    # Select normal + target defect samples
    # --------------------------------------------------

    selected_samples = []

    for index in range(len(dataset)):

        sample = dataset[index]

        if sample["defect_type"] == "good":

            selected_samples.append(
                sample
            )

        elif sample["defect_type"] == defect_type:

            selected_samples.append(
                sample
            )

    print(
        f"\nSelected samples for "
        f"'{defect_type}': "
        f"{len(selected_samples)}"
    )

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------

    with torch.no_grad():

        for sample in selected_samples:

            image = (
                sample["image"]
                .unsqueeze(0)
                .to(device)
            )

            features = (
                feature_extractor(
                    image
                )
            )

            anomaly_score, _ = (
                patchcore.predict(
                    features
                )
            )

            original_label = int(
                sample["label"]
            )

            if sample[
                "defect_type"
            ] == defect_type:

                label = 1

            else:

                label = 0

            scores.append(
                float(anomaly_score)
            )

            labels.append(
                label
            )

            paths.append(
                sample["path"]
            )

    y_true = np.asarray(
        labels,
        dtype=np.int32,
    )

    y_scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    y_pred = (
        y_scores
        >= DECISION_THRESHOLD
    ).astype(np.int32)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    if len(np.unique(y_true)) < 2:

        print(
            "WARNING: Both normal and "
            "defect classes are required."
        )

        return None

    auroc = roc_auc_score(
        y_true,
        y_scores,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    false_positives = int(
        cm[0, 1]
    )

    false_negatives = int(
        cm[1, 0]
    )

    result = {
        "defect_type": defect_type,
        "samples": len(y_true),
        "auroc": float(auroc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "confusion_matrix": cm,
    }

    print(
        "\n" + "-" * 60
    )

    print(
        f"UNSEEN DEFECT: "
        f"{defect_type}"
    )

    print(
        "-" * 60
    )

    print(
        f"Samples         : "
        f"{len(y_true)}"
    )

    print(
        f"AUROC           : "
        f"{auroc:.4f}"
    )

    print(
        f"Precision       : "
        f"{precision:.4f}"
    )

    print(
        f"Recall          : "
        f"{recall:.4f}"
    )

    print(
        f"F1              : "
        f"{f1:.4f}"
    )

    print(
        f"False positives : "
        f"{false_positives}"
    )

    print(
        f"False negatives : "
        f"{false_negatives}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    return result


def run_unseen_defect_testing(
    category=CATEGORY,
):
    device = torch.device(
        "cpu"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("PATCHCORE UNSEEN-DEFECT TESTING")
    print("=" * 60)

    print(
        f"Device    : {device}"
    )

    print(
        f"Category  : {category}"
    )

    print(
        f"Threshold : "
        f"{DECISION_THRESHOLD:.4f}"
    )

    # --------------------------------------------------
    # Build PatchCore from NORMAL training images only
    # --------------------------------------------------

    patchcore = build_memory_bank(
        category=category,
        batch_size=8,
        sampling_ratio=0.1,
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

    print(
        f"Total test images: "
        f"{len(dataset)}"
    )

    # --------------------------------------------------
    # Feature extractor
    # --------------------------------------------------

    feature_extractor = (
        ResNet18FeatureExtractor()
        .to(device)
    )

    feature_extractor.eval()

    # --------------------------------------------------
    # Discover defect types
    # --------------------------------------------------

    defect_types = sorted(
        set(
            sample["defect_type"]
            for sample in dataset
            if sample["defect_type"]
            != "good"
        )
    )

    print(
        "\nAvailable defect types:"
    )

    for defect_type in defect_types:

        print(
            f"  - {defect_type}"
        )

    # --------------------------------------------------
    # Evaluate every defect type independently
    # --------------------------------------------------

    results = []

    for defect_type in defect_types:

        result = (
            evaluate_unseen_defect_type(
                dataset=dataset,
                defect_type=defect_type,
                patchcore=patchcore,
                feature_extractor=feature_extractor,
                device=device,
            )
        )

        if result is not None:

            results.append(
                result
            )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("UNSEEN-DEFECT SUMMARY")
    print("=" * 60)

    print(
        f"{'Defect Type':20s}"
        f"{'AUROC':>10s}"
        f"{'Precision':>12s}"
        f"{'Recall':>10s}"
        f"{'F1':>10s}"
        f"{'FP':>6s}"
        f"{'FN':>6s}"
    )

    print(
        "-" * 80
    )

    for result in results:

        print(
            f"{result['defect_type']:20s}"
            f"{result['auroc']:10.4f}"
            f"{result['precision']:12.4f}"
            f"{result['recall']:10.4f}"
            f"{result['f1']:10.4f}"
            f"{result['false_positives']:6d}"
            f"{result['false_negatives']:6d}"
        )

    # --------------------------------------------------
    # Save report
    # --------------------------------------------------

    report_path = (
        OUTPUT_DIR
        / "unseen_defect_results.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "PATCHCORE UNSEEN-DEFECT TESTING\n"
        )

        file.write(
            "=" * 60 + "\n"
        )

        file.write(
            f"Category: {category}\n"
        )

        file.write(
            f"Threshold: "
            f"{DECISION_THRESHOLD:.6f}\n\n"
        )

        for result in results:

            file.write(
                f"Defect Type: "
                f"{result['defect_type']}\n"
            )

            file.write(
                f"Samples: "
                f"{result['samples']}\n"
            )

            file.write(
                f"AUROC: "
                f"{result['auroc']:.6f}\n"
            )

            file.write(
                f"Precision: "
                f"{result['precision']:.6f}\n"
            )

            file.write(
                f"Recall: "
                f"{result['recall']:.6f}\n"
            )

            file.write(
                f"F1: "
                f"{result['f1']:.6f}\n"
            )

            file.write(
                f"False positives: "
                f"{result['false_positives']}\n"
            )

            file.write(
                f"False negatives: "
                f"{result['false_negatives']}\n"
            )

            file.write(
                "Confusion matrix:\n"
            )

            file.write(
                f"{result['confusion_matrix']}\n"
            )

            file.write(
                "\n"
            )

    print(
        f"\nSaved: {report_path}"
    )

    print("\n" + "=" * 60)
    print(
        "UNSEEN-DEFECT TESTING COMPLETE"
    )
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_unseen_defect_testing(
        category=CATEGORY
    )

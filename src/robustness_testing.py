from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from src.config import DEFAULT_THRESHOLD

from src.dataset import MVTecDataset
from src.preprocessing import get_test_transform
from src.feature_extractor import ResNet18FeatureExtractor
from src.patchcore_pipeline import build_memory_bank


DATASET_ROOT = "data/mvtec_anomaly_detection"
OUTPUT_DIR = Path("outputs/robustness")

CATEGORY = "bottle"

DECISION_THRESHOLD = DEFAULT_THRESHOLD

IMAGE_SIZE = 224


# ---------------------------------------------------------
# Image corruption functions
# ---------------------------------------------------------

def adjust_brightness(image, factor):
    """
    factor < 1.0 -> darker
    factor > 1.0 -> brighter
    """

    image = image.astype(np.float32)

    image = image * factor

    image = np.clip(
        image,
        0,
        255,
    )

    return image.astype(np.uint8)


def adjust_contrast(image, factor):
    """
    factor < 1.0 -> lower contrast
    factor > 1.0 -> higher contrast
    """

    image = image.astype(np.float32)

    mean = np.mean(
        image,
        axis=(0, 1),
        keepdims=True,
    )

    image = (
        (image - mean)
        * factor
        + mean
    )

    image = np.clip(
        image,
        0,
        255,
    )

    return image.astype(np.uint8)


def add_gaussian_noise(
    image,
    sigma=15,
):
    """
    Add Gaussian sensor noise.
    """

    noise = np.random.normal(
        loc=0.0,
        scale=sigma,
        size=image.shape,
    )

    noisy = (
        image.astype(np.float32)
        + noise
    )

    noisy = np.clip(
        noisy,
        0,
        255,
    )

    return noisy.astype(np.uint8)


def apply_blur(
    image,
    kernel_size=7,
):
    """
    Simulate camera motion / focus blur.
    """

    return cv2.GaussianBlur(
        image,
        (
            kernel_size,
            kernel_size,
        ),
        0,
    )


def reduce_resolution(
    image,
    scale=0.5,
):
    """
    Downsample and resize back to
    the model input resolution.
    """

    height, width = image.shape[:2]

    small_width = max(
        1,
        int(width * scale),
    )

    small_height = max(
        1,
        int(height * scale),
    )

    small = cv2.resize(
        image,
        (
            small_width,
            small_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    restored = cv2.resize(
        small,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    return restored


# ---------------------------------------------------------
# Corruption configurations
# ---------------------------------------------------------

CORRUPTIONS = {
    "clean": lambda image: image,

    "brightness_low": lambda image:
        adjust_brightness(
            image,
            factor=0.70,
        ),

    "brightness_high": lambda image:
        adjust_brightness(
            image,
            factor=1.30,
        ),

    "contrast_low": lambda image:
        adjust_contrast(
            image,
            factor=0.70,
        ),

    "contrast_high": lambda image:
        adjust_contrast(
            image,
            factor=1.30,
        ),

    "gaussian_noise": lambda image:
        add_gaussian_noise(
            image,
            sigma=15,
        ),

    "blur": lambda image:
        apply_blur(
            image,
            kernel_size=7,
        ),

    "low_resolution": lambda image:
        reduce_resolution(
            image,
            scale=0.50,
        ),
}


# ---------------------------------------------------------
# Convert image to model tensor
# ---------------------------------------------------------

def image_to_tensor(
    image,
):
    """
    Convert uint8 RGB image to the same
    normalized tensor used during testing.
    """

    pil_image = Image.fromarray(
        image
    )

    transform = get_test_transform()

    tensor = transform(
        pil_image
    )

    return tensor.unsqueeze(0)


# ---------------------------------------------------------
# Evaluate one corruption condition
# ---------------------------------------------------------

def evaluate_condition(
    condition_name,
    corruption_function,
    dataset,
    feature_extractor,
    patchcore,
    device,
):
    scores = []
    labels = []

    print(
        "\n"
        + "-" * 60
    )

    print(
        f"CONDITION: {condition_name}"
    )

    print(
        "-" * 60
    )

    with torch.no_grad():

        for index in range(
            len(dataset)
        ):

            sample = dataset[index]

            image_path = sample[
                "path"
            ]

            label = int(
                sample["label"]
            )

            # ---------------------------------------------
            # Load original RGB image
            # ---------------------------------------------

            original = Image.open(
                image_path
            ).convert("RGB")

            original = original.resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                Image.Resampling.BILINEAR,
            )

            image_np = np.asarray(
                original,
                dtype=np.uint8,
            )

            # ---------------------------------------------
            # Apply corruption
            # ---------------------------------------------

            corrupted = (
                corruption_function(
                    image_np
                )
            )

            # ---------------------------------------------
            # Convert to model tensor
            # ---------------------------------------------

            tensor = image_to_tensor(
                corrupted
            ).to(device)

            # ---------------------------------------------
            # Feature extraction
            # ---------------------------------------------

            features = (
                feature_extractor(
                    tensor
                )
            )

            # ---------------------------------------------
            # PatchCore
            # ---------------------------------------------

            anomaly_score, _ = (
                patchcore.predict(
                    features
                )
            )

            scores.append(
                float(anomaly_score)
            )

            labels.append(
                label
            )

            if (
                index + 1
            ) % 20 == 0:

                print(
                    f"Processed "
                    f"{index + 1}/"
                    f"{len(dataset)}"
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

    # ---------------------------------------------
    # Metrics
    # ---------------------------------------------

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

    false_positives = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 1)
        )
    )

    false_negatives = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 0)
        )
    )

    result = {
        "condition": condition_name,
        "auroc": float(auroc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

    print(
        f"AUROC          : "
        f"{auroc:.4f}"
    )

    print(
        f"Precision      : "
        f"{precision:.4f}"
    )

    print(
        f"Recall         : "
        f"{recall:.4f}"
    )

    print(
        f"F1             : "
        f"{f1:.4f}"
    )

    print(
        f"False positives: "
        f"{false_positives}"
    )

    print(
        f"False negatives: "
        f"{false_negatives}"
    )

    return result


# ---------------------------------------------------------
# Main robustness experiment
# ---------------------------------------------------------

def run_robustness_testing(
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
    print("PATCHCORE ROBUSTNESS TESTING")
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

    # -----------------------------------------------------
    # Build memory bank
    # -----------------------------------------------------

    patchcore = build_memory_bank(
        category=category,
        batch_size=8,
        sampling_ratio=0.1,
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    dataset = MVTecDataset(
        root_dir=DATASET_ROOT,
        category=category,
        split="test",
        transform=None,
    )

    print(
        f"Test images: {len(dataset)}"
    )

    # -----------------------------------------------------
    # Feature extractor
    # -----------------------------------------------------

    feature_extractor = (
        ResNet18FeatureExtractor()
        .to(device)
    )

    feature_extractor.eval()

    # -----------------------------------------------------
    # Evaluate conditions
    # -----------------------------------------------------

    results = []

    for condition_name, function in (
        CORRUPTIONS.items()
    ):

        result = evaluate_condition(
            condition_name=condition_name,
            corruption_function=function,
            dataset=dataset,
            feature_extractor=feature_extractor,
            patchcore=patchcore,
            device=device,
        )

        results.append(
            result
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY")
    print("=" * 60)

    print(
        f"{'Condition':20s} "
        f"{'AUROC':>8s} "
        f"{'Precision':>10s} "
        f"{'Recall':>8s} "
        f"{'F1':>8s} "
        f"{'FP':>5s} "
        f"{'FN':>5s}"
    )

    print("-" * 75)

    for result in results:

        print(
            f"{result['condition']:20s} "
            f"{result['auroc']:8.4f} "
            f"{result['precision']:10.4f} "
            f"{result['recall']:8.4f} "
            f"{result['f1']:8.4f} "
            f"{result['false_positives']:5d} "
            f"{result['false_negatives']:5d}"
        )

    # -----------------------------------------------------
    # Save CSV-style report
    # -----------------------------------------------------

    report_path = (
        OUTPUT_DIR
        / "robustness_results.csv"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "condition,"
            "auroc,"
            "precision,"
            "recall,"
            "f1,"
            "false_positives,"
            "false_negatives\n"
        )

        for result in results:

            file.write(
                f"{result['condition']},"
                f"{result['auroc']:.6f},"
                f"{result['precision']:.6f},"
                f"{result['recall']:.6f},"
                f"{result['f1']:.6f},"
                f"{result['false_positives']},"
                f"{result['false_negatives']}\n"
            )

    print(
        f"\nSaved: {report_path}"
    )

    print("\n" + "=" * 60)
    print(
        "ROBUSTNESS TESTING COMPLETE"
    )
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_robustness_testing(
        category=CATEGORY
    )

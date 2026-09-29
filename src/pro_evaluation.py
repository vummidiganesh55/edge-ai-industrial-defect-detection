from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from src.dataset import MVTecDataset
from src.preprocessing import get_test_transform
from src.feature_extractor import ResNet18FeatureExtractor
from src.patchcore_pipeline import build_memory_bank
from src.anomaly_map import (
    create_anomaly_map,
    normalize_anomaly_map,
)


DATASET_ROOT = "data/mvtec_anomaly_detection"
OUTPUT_DIR = Path("outputs/evaluation")

IMAGE_SIZE = 224

# Number of thresholds used to construct the PRO curve.
NUM_THRESHOLDS = 100

# PRO is commonly evaluated in the low false-positive region.
MAX_FPR = 0.30


def load_ground_truth_mask(
    image_path,
    category,
    defect_type,
):
    """
    Load MVTec ground-truth mask.

    Good images have no defect region.
    """

    if defect_type == "good":
        return np.zeros(
            (IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.uint8,
        )

    image_path = Path(image_path)

    mask_path = (
        Path(DATASET_ROOT)
        / category
        / "ground_truth"
        / defect_type
        / f"{image_path.stem}_mask.png"
    )

    if not mask_path.exists():
        print(
            f"WARNING: Ground-truth mask not found: "
            f"{mask_path}"
        )

        return np.zeros(
            (IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.uint8,
        )

    mask = Image.open(
        mask_path
    ).convert("L")

    mask = mask.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST,
    )

    mask = np.asarray(
        mask,
        dtype=np.uint8,
    )

    return (
        mask > 0
    ).astype(np.uint8)


def get_connected_components(mask):
    """
    Extract connected components from a binary mask.

    Uses OpenCV if available.
    """

    import cv2

    mask = (
        np.asarray(mask)
        .astype(np.uint8)
    )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    components = []

    for label_index in range(
        1,
        num_labels,
    ):
        component = (
            labels == label_index
        )

        area = int(
            stats[
                label_index,
                cv2.CC_STAT_AREA,
            ]
        )

        if area > 0:
            components.append(
                (
                    component,
                    area,
                )
            )

    return components


def calculate_pro_for_threshold(
    anomaly_maps,
    ground_truth_masks,
    threshold,
):
    """
    Calculate:

    PRO:
        Per-Region Overlap

    FPR:
        False Positive Rate

    for a single threshold.
    """

    region_overlaps = []

    total_negative_pixels = 0
    false_positive_pixels = 0

    for anomaly_map, ground_truth in zip(
        anomaly_maps,
        ground_truth_masks,
    ):

        predicted = (
            anomaly_map >= threshold
        ).astype(np.uint8)

        # --------------------------------------------------
        # Region overlap
        # --------------------------------------------------

        components = get_connected_components(
            ground_truth
        )

        for component, area in components:

            intersection = np.logical_and(
                predicted == 1,
                component,
            ).sum()

            overlap = (
                intersection / area
            )

            region_overlaps.append(
                overlap
            )

        # --------------------------------------------------
        # False positive pixels
        # --------------------------------------------------

        negative_region = (
            ground_truth == 0
        )

        false_positive_pixels += (
            np.logical_and(
                predicted == 1,
                negative_region,
            ).sum()
        )

        total_negative_pixels += (
            negative_region.sum()
        )

    if region_overlaps:
        pro = float(
            np.mean(region_overlaps)
        )
    else:
        pro = 0.0

    if total_negative_pixels > 0:
        fpr = float(
            false_positive_pixels
            / total_negative_pixels
        )
    else:
        fpr = 0.0

    return pro, fpr


def calculate_pro_auc(
    anomaly_maps,
    ground_truth_masks,
):
    """
    Calculate area under the PRO-FPR curve.

    Only the region:

        FPR <= MAX_FPR

    is used.
    """

    thresholds = np.linspace(
        1.0,
        0.0,
        NUM_THRESHOLDS,
    )

    pro_values = []
    fpr_values = []

    for threshold in thresholds:

        pro, fpr = (
            calculate_pro_for_threshold(
                anomaly_maps,
                ground_truth_masks,
                threshold,
            )
        )

        pro_values.append(pro)
        fpr_values.append(fpr)

    pro_values = np.asarray(
        pro_values,
        dtype=np.float64,
    )

    fpr_values = np.asarray(
        fpr_values,
        dtype=np.float64,
    )

    # ------------------------------------------------------
    # Sort by FPR
    # ------------------------------------------------------

    order = np.argsort(
        fpr_values
    )

    fpr_values = fpr_values[
        order
    ]

    pro_values = pro_values[
        order
    ]

    # ------------------------------------------------------
    # Remove duplicate FPR values
    # ------------------------------------------------------

    unique_fpr = []
    unique_pro = []

    for fpr, pro in zip(
        fpr_values,
        pro_values,
    ):

        if not unique_fpr:

            unique_fpr.append(fpr)
            unique_pro.append(pro)

        elif fpr == unique_fpr[-1]:

            unique_pro[-1] = max(
                unique_pro[-1],
                pro,
            )

        else:

            unique_fpr.append(fpr)
            unique_pro.append(pro)

    fpr_values = np.asarray(
        unique_fpr,
        dtype=np.float64,
    )

    pro_values = np.asarray(
        unique_pro,
        dtype=np.float64,
    )

    # ------------------------------------------------------
    # Restrict to maximum FPR
    # ------------------------------------------------------

    valid = (
        fpr_values <= MAX_FPR
    )

    fpr_valid = fpr_values[
        valid
    ]

    pro_valid = pro_values[
        valid
    ]

    # ------------------------------------------------------
    # Add boundary at MAX_FPR
    # ------------------------------------------------------

    if len(fpr_valid) == 0:

        return 0.0, fpr_values, pro_values

    if fpr_valid[-1] < MAX_FPR:

        interpolated_pro = np.interp(
            MAX_FPR,
            fpr_values,
            pro_values,
        )

        fpr_valid = np.append(
            fpr_valid,
            MAX_FPR,
        )

        pro_valid = np.append(
            pro_valid,
            interpolated_pro,
        )

    # ------------------------------------------------------
    # Normalize area
    # ------------------------------------------------------

    pro_auc = np.trapezoid(
        pro_valid,
        fpr_valid,
    ) / MAX_FPR

    return (
        float(pro_auc),
        fpr_values,
        pro_values,
    )


def evaluate_pro(
    category="bottle",
):
    """
    Complete PatchCore PRO evaluation.
    """

    device = torch.device("cpu")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("PATCHCORE PRO EVALUATION")
    print("=" * 60)

    print(
        f"Device       : {device}"
    )

    print(
        f"Category     : {category}"
    )

    print(
        f"Image size   : {IMAGE_SIZE}"
    )

    print(
        f"Max FPR      : {MAX_FPR}"
    )

    # ------------------------------------------------------
    # Build memory bank
    # ------------------------------------------------------

    patchcore = build_memory_bank(
        category=category,
        batch_size=8,
        sampling_ratio=0.1,
    )

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------

    dataset = MVTecDataset(
        root_dir=DATASET_ROOT,
        category=category,
        split="test",
        transform=get_test_transform(),
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Test images : {len(dataset)}"
    )

    # ------------------------------------------------------
    # Feature extractor
    # ------------------------------------------------------

    feature_extractor = (
        ResNet18FeatureExtractor()
        .to(device)
    )

    feature_extractor.eval()

    anomaly_maps = []
    ground_truth_masks = []

    # ------------------------------------------------------
    # Generate maps
    # ------------------------------------------------------

    with torch.no_grad():

        for index, batch in enumerate(
            loader
        ):

            image = batch[
                "image"
            ].to(device)

            features = (
                feature_extractor(
                    image
                )
            )

            _, patch_distances = (
                patchcore.predict(
                    features
                )
            )

            anomaly_map = (
                create_anomaly_map(
                    features["layer2"],
                    features["layer3"],
                    patch_distances,
                )
            )

            normalized_map = (
                normalize_anomaly_map(
                    anomaly_map
                )
            )

            image_path = batch[
                "path"
            ][0]

            defect_type = batch[
                "defect_type"
            ][0]

            ground_truth = (
                load_ground_truth_mask(
                    image_path=image_path,
                    category=category,
                    defect_type=defect_type,
                )
            )

            anomaly_maps.append(
                normalized_map
            )

            ground_truth_masks.append(
                ground_truth
            )

            print(
                f"{index + 1:02d}/"
                f"{len(dataset)} | "
                f"{defect_type:20s}"
            )

    # ------------------------------------------------------
    # PRO evaluation
    # ------------------------------------------------------

    pro_auc, fpr_values, pro_values = (
        calculate_pro_auc(
            anomaly_maps,
            ground_truth_masks,
        )
    )

    print("\n" + "=" * 60)
    print("PRO RESULTS")
    print("=" * 60)

    print(
        f"PRO-AUC          : "
        f"{pro_auc:.4f}"
    )

    print(
        f"FPR range tested : "
        f"0.00 - {MAX_FPR:.2f}"
    )

    # ------------------------------------------------------
    # Save results
    # ------------------------------------------------------

    results_path = (
        OUTPUT_DIR
        / "pro_metrics.txt"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "PATCHCORE PRO EVALUATION\n"
        )

        file.write(
            "=" * 60 + "\n"
        )

        file.write(
            f"Category: {category}\n"
        )

        file.write(
            f"Image size: {IMAGE_SIZE}\n"
        )

        file.write(
            f"Max FPR: {MAX_FPR}\n"
        )

        file.write(
            f"PRO-AUC: {pro_auc:.6f}\n"
        )

    print(
        f"\nSaved: {results_path}"
    )

    print("\n" + "=" * 60)
    print(
        "PRO EVALUATION COMPLETE"
    )
    print("=" * 60)

    return {
        "pro_auc": pro_auc,
        "fpr": fpr_values,
        "pro": pro_values,
    }


if __name__ == "__main__":
    evaluate_pro(
        category="bottle"
    )

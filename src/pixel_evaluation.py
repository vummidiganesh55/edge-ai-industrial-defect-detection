from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
)

from src.dataset import MVTecDataset
from src.preprocessing import get_test_transform
from src.feature_extractor import ResNet18FeatureExtractor
from src.patchcore_pipeline import build_memory_bank
from src.anomaly_map import (
    create_anomaly_map,
    normalize_anomaly_map,
    get_anomaly_region,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = "data/mvtec_anomaly_detection"

OUTPUT_DIR = Path(
    "outputs/evaluation"
)

IMAGE_SIZE = 224

ANOMALY_THRESHOLD = 0.5


# ============================================================
# LOAD GROUND-TRUTH MASK
# ============================================================

def load_ground_truth_mask(
    mask_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
):
    """
    Load and resize an MVTec ground-truth mask.

    For normal ('good') images, MVTec does not provide
    a ground-truth mask. In that case, return an all-zero mask.
    """

    # --------------------------------------------------------
    # Normal image has no ground-truth mask
    # --------------------------------------------------------

    if mask_path is None:
        return np.zeros(
            target_size,
            dtype=np.uint8,
        )

    # --------------------------------------------------------
    # Convert path
    # --------------------------------------------------------

    mask_path = Path(mask_path)

    # --------------------------------------------------------
    # Missing mask protection
    # --------------------------------------------------------

    if not mask_path.exists():
        print(
            f"WARNING: Ground-truth mask not found: "
            f"{mask_path}"
        )

        return np.zeros(
            target_size,
            dtype=np.uint8,
        )

    # --------------------------------------------------------
    # Load mask
    # --------------------------------------------------------

    mask = Image.open(
        mask_path
    ).convert("L")

    # --------------------------------------------------------
    # Resize using NEAREST
    #
    # Important:
    # Ground-truth masks are categorical/binary.
    # Therefore NEAREST is preferable to bilinear here.
    # --------------------------------------------------------

    mask = mask.resize(
        target_size,
        Image.Resampling.NEAREST,
    )

    # --------------------------------------------------------
    # Convert PIL -> NumPy
    # --------------------------------------------------------

    mask = np.asarray(
        mask,
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Convert grayscale mask -> binary mask
    # --------------------------------------------------------

    mask = (
        mask > 0
    ).astype(np.uint8)

    return mask


# ============================================================
# GET GROUND-TRUTH MASK PATH
# ============================================================

def get_ground_truth_path(
    image_path,
    category,
    defect_type,
):
    """
    Convert an MVTec test image path into its corresponding
    ground-truth mask path.

    Example:

    test/broken_large/000.png

    becomes:

    ground_truth/broken_large/000_mask.png

    Normal ('good') images do not have masks.
    """

    # --------------------------------------------------------
    # Normal image
    # --------------------------------------------------------

    if defect_type == "good":
        return None

    # --------------------------------------------------------
    # Image path
    # --------------------------------------------------------

    image_path = Path(
        image_path
    )

    image_name = image_path.stem

    # --------------------------------------------------------
    # Ground-truth mask path
    # --------------------------------------------------------

    mask_path = (
        Path(DATASET_ROOT)
        / category
        / "ground_truth"
        / defect_type
        / f"{image_name}_mask.png"
    )

    return mask_path


# ============================================================
# PIXEL-LEVEL EVALUATION
# ============================================================

def evaluate_pixel_level(
    category="bottle",
    batch_size=1,
):
    """
    Run PatchCore pixel-level evaluation.

    Metrics:
        - Pixel AUROC
        - Pixel Precision
        - Pixel Recall
        - Pixel F1
        - Pixel IoU
    """

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cpu"
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 60)
    print(
        "PATCHCORE PIXEL-LEVEL EVALUATION"
    )
    print("=" * 60)

    print(
        f"Device   : {device}"
    )

    print(
        f"Category : {category}"
    )

    print(
        f"Threshold: {ANOMALY_THRESHOLD}"
    )

    # ========================================================
    # 1. BUILD PATCHCORE MEMORY BANK
    # ========================================================

    print("\n" + "-" * 60)
    print("BUILDING PATCHCORE MEMORY BANK")
    print("-" * 60)

    patchcore = build_memory_bank(
        category=category,
        batch_size=8,
        sampling_ratio=0.1,
    )

    # ========================================================
    # 2. LOAD TEST DATASET
    # ========================================================

    print("\n" + "-" * 60)
    print("LOADING TEST DATASET")
    print("-" * 60)

    dataset = MVTecDataset(
        root_dir=DATASET_ROOT,
        category=category,
        split="test",
        transform=get_test_transform(),
    )

    print(
        f"Test images : {len(dataset)}"
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ========================================================
    # 3. FEATURE EXTRACTOR
    # ========================================================

    feature_extractor = (
        ResNet18FeatureExtractor()
        .to(device)
    )

    feature_extractor.eval()

    # ========================================================
    # 4. STORAGE
    # ========================================================

    all_ground_truth = []

    all_scores = []

    image_results = []

    # ========================================================
    # 5. PATCHCORE INFERENCE
    # ========================================================

    print("\n" + "-" * 60)
    print("GENERATING ANOMALY MAPS")
    print("-" * 60)

    with torch.no_grad():

        for index, batch in enumerate(
            loader
        ):

            # ------------------------------------------------
            # Image
            # ------------------------------------------------

            image = batch[
                "image"
            ].to(device)

            # ------------------------------------------------
            # Feature extraction
            # ------------------------------------------------

            features = feature_extractor(
                image
            )

            # ------------------------------------------------
            # PatchCore prediction
            # ------------------------------------------------

            (
                anomaly_score,
                patch_distances,
            ) = patchcore.predict(
                features
            )

            # ------------------------------------------------
            # Create anomaly map
            # ------------------------------------------------

            anomaly_map = create_anomaly_map(
                features["layer2"],
                features["layer3"],
                patch_distances,
            )

            # ------------------------------------------------
            # Normalize anomaly map
            # ------------------------------------------------

            normalized_map = (
                normalize_anomaly_map(
                    anomaly_map
                )
            )

            # ------------------------------------------------
            # Dataset metadata
            # ------------------------------------------------

            image_path = batch[
                "path"
            ][0]

            defect_type = batch[
                "defect_type"
            ][0]

            label = int(
                batch["label"].item()
            )

            # ------------------------------------------------
            # Ground-truth path
            # ------------------------------------------------

            mask_path = get_ground_truth_path(
                image_path=image_path,
                category=category,
                defect_type=defect_type,
            )

            # ------------------------------------------------
            # Ground-truth mask
            # ------------------------------------------------

            ground_truth = (
                load_ground_truth_mask(
                    mask_path=mask_path,
                    target_size=(
                        IMAGE_SIZE,
                        IMAGE_SIZE,
                    ),
                )
            )

            # ------------------------------------------------
            # Store pixel-level values
            # ------------------------------------------------

            all_ground_truth.extend(
                ground_truth.flatten().tolist()
            )

            all_scores.extend(
                normalized_map.flatten().tolist()
            )

            # ------------------------------------------------
            # Binary prediction
            # ------------------------------------------------

            predicted_mask = (
                get_anomaly_region(
                    anomaly_map,
                    threshold=ANOMALY_THRESHOLD,
                )
            )

            predicted_flat = (
                predicted_mask.flatten()
            )

            ground_truth_flat = (
                ground_truth.flatten()
            )

            # ------------------------------------------------
            # Per-image IoU
            # ------------------------------------------------

            image_iou = jaccard_score(
                ground_truth_flat,
                predicted_flat,
                zero_division=0,
            )

            # ------------------------------------------------
            # Per-image precision
            # ------------------------------------------------

            image_precision = precision_score(
                ground_truth_flat,
                predicted_flat,
                zero_division=0,
            )

            # ------------------------------------------------
            # Per-image recall
            # ------------------------------------------------

            image_recall = recall_score(
                ground_truth_flat,
                predicted_flat,
                zero_division=0,
            )

            # ------------------------------------------------
            # Per-image F1
            # ------------------------------------------------

            image_f1 = f1_score(
                ground_truth_flat,
                predicted_flat,
                zero_division=0,
            )

            # ------------------------------------------------
            # Store image result
            # ------------------------------------------------

            image_results.append(
                {
                    "path": image_path,
                    "label": label,
                    "defect_type": defect_type,
                    "anomaly_score": float(
                        anomaly_score
                    ),
                    "iou": float(
                        image_iou
                    ),
                    "precision": float(
                        image_precision
                    ),
                    "recall": float(
                        image_recall
                    ),
                    "f1": float(
                        image_f1
                    ),
                }
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            print(
                f"{index + 1:02d}/"
                f"{len(dataset)} | "
                f"{defect_type:20s} | "
                f"score="
                f"{anomaly_score:.4f} | "
                f"IoU="
                f"{image_iou:.4f}"
            )

    # ========================================================
    # 6. CONVERT TO NUMPY
    # ========================================================

    y_true = np.asarray(
        all_ground_truth,
        dtype=np.uint8,
    )

    y_scores = np.asarray(
        all_scores,
        dtype=np.float32,
    )

    # ========================================================
    # 7. BINARY PREDICTION
    # ========================================================

    y_pred = (
        y_scores >= ANOMALY_THRESHOLD
    ).astype(np.uint8)

    # ========================================================
    # 8. PIXEL AUROC
    # ========================================================

    pixel_auroc = roc_auc_score(
        y_true,
        y_scores,
    )

    # ========================================================
    # 9. PIXEL PRECISION
    # ========================================================

    pixel_precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    # ========================================================
    # 10. PIXEL RECALL
    # ========================================================

    pixel_recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    # ========================================================
    # 11. PIXEL F1
    # ========================================================

    pixel_f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    # ========================================================
    # 12. PIXEL IoU
    # ========================================================

    pixel_iou = jaccard_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    # ========================================================
    # 13. PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print(
        "PIXEL-LEVEL RESULTS"
    )
    print("=" * 60)

    print(
        f"Pixel AUROC       : "
        f"{pixel_auroc:.4f}"
    )

    print(
        f"Pixel Precision   : "
        f"{pixel_precision:.4f}"
    )

    print(
        f"Pixel Recall      : "
        f"{pixel_recall:.4f}"
    )

    print(
        f"Pixel F1          : "
        f"{pixel_f1:.4f}"
    )

    print(
        f"Pixel IoU         : "
        f"{pixel_iou:.4f}"
    )

    # ========================================================
    # 14. SAVE METRICS
    # ========================================================

    results_path = (
        OUTPUT_DIR
        / "pixel_metrics.txt"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "PATCHCORE PIXEL-LEVEL EVALUATION\n"
        )

        file.write(
            "=" * 60 + "\n"
        )

        file.write(
            f"Category: {category}\n"
        )

        file.write(
            f"Threshold: "
            f"{ANOMALY_THRESHOLD}\n"
        )

        file.write(
            f"Pixel AUROC: "
            f"{pixel_auroc:.6f}\n"
        )

        file.write(
            f"Pixel Precision: "
            f"{pixel_precision:.6f}\n"
        )

        file.write(
            f"Pixel Recall: "
            f"{pixel_recall:.6f}\n"
        )

        file.write(
            f"Pixel F1: "
            f"{pixel_f1:.6f}\n"
        )

        file.write(
            f"Pixel IoU: "
            f"{pixel_iou:.6f}\n"
        )

    # ========================================================
    # 15. COMPLETE
    # ========================================================

    print(
        f"\nSaved: {results_path}"
    )

    print("\n" + "=" * 60)
    print(
        "PIXEL EVALUATION COMPLETE"
    )
    print("=" * 60)

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "pixel_auroc": float(
            pixel_auroc
        ),
        "pixel_precision": float(
            pixel_precision
        ),
        "pixel_recall": float(
            pixel_recall
        ),
        "pixel_f1": float(
            pixel_f1
        ),
        "pixel_iou": float(
            pixel_iou
        ),
        "image_results": image_results,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluate_pixel_level(
        category="bottle"
    )

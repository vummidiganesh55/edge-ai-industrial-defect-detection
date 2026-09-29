import torch
from torch.utils.data import DataLoader
from src.config import DEFAULT_THRESHOLD
from src.dataset import MVTecDataset
from src.preprocessing import get_test_transform
from src.feature_extractor import ResNet18FeatureExtractor
from src.patchcore_pipeline import build_memory_bank
from src.evaluate import evaluate_image_level


DATASET_ROOT = "data/mvtec_anomaly_detection"


def run_test(category="bottle"):
    device = torch.device("cpu")

    print("=" * 60)
    print("PATCHCORE TEST INFERENCE")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Build PatchCore memory bank
    # --------------------------------------------------
    patchcore = build_memory_bank(
        category=category,
        batch_size=8,
        sampling_ratio=0.1,
    )

    # --------------------------------------------------
    # 2. Load test dataset
    # --------------------------------------------------
    test_dataset = MVTecDataset(
        root_dir=DATASET_ROOT,
        category=category,
        split="test",
        transform=get_test_transform(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------
    # 3. Create feature extractor
    # --------------------------------------------------
    feature_extractor = ResNet18FeatureExtractor().to(device)
    feature_extractor.eval()

    results = []

    # --------------------------------------------------
    # 4. Run inference
    # --------------------------------------------------
    with torch.no_grad():

        for index, batch in enumerate(test_loader):

            image = batch["image"].to(device)

            # Extract CNN features
            features = feature_extractor(image)

            # PatchCore prediction
            anomaly_score, patch_distances = patchcore.predict(
                features
            )

            label = int(batch["label"].item())
            defect_type = batch["defect_type"][0]
            image_path = batch["path"][0]

            # Store result
            results.append(
                {
                    "path": image_path,
                    "label": label,
                    "defect_type": defect_type,
                    "anomaly_score": anomaly_score,
                }
            )

            print(
                f"{index + 1:02d}/{len(test_dataset)} | "
                f"{defect_type:20s} | "
                f"label={label} | "
                f"score={anomaly_score:.4f}"
            )

    # --------------------------------------------------
    # 5. Separate normal and defect scores
    # --------------------------------------------------
    normal_scores = [
        result["anomaly_score"]
        for result in results
        if result["label"] == 0
    ]

    defect_scores = [
        result["anomaly_score"]
        for result in results
        if result["label"] == 1
    ]

    # --------------------------------------------------
    # 6. Print inference summary
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("PATCHCORE INFERENCE SUMMARY")
    print("=" * 60)

    print(f"Total test images : {len(results)}")
    print(f"Normal images     : {len(normal_scores)}")
    print(f"Defect images     : {len(defect_scores)}")

    if normal_scores:

        print(
            f"Normal score      : "
            f"min={min(normal_scores):.4f}, "
            f"max={max(normal_scores):.4f}, "
            f"mean={sum(normal_scores) / len(normal_scores):.4f}"
        )

    if defect_scores:

        print(
            f"Defect score      : "
            f"min={min(defect_scores):.4f}, "
            f"max={max(defect_scores):.4f}, "
            f"mean={sum(defect_scores) / len(defect_scores):.4f}"
        )

    # --------------------------------------------------
    # 7. Image-level evaluation
    # --------------------------------------------------
    metrics = evaluate_image_level(results, threshold=DEFAULT_THRESHOLD)

    # --------------------------------------------------
    # 8. Final metrics summary
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL PATCHCORE METRICS")
    print("=" * 60)

    print(
        f"Image AUROC       : "
        f"{metrics['auroc']:.4f}"
    )

    print(
    f"Production threshold : "
    f"{metrics['threshold']:.6f}"
)

    print(
    f"Youden threshold     : "
    f"{metrics['best_threshold']:.4f}"
    )

    print(
        f"Precision         : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall            : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score          : "
        f"{metrics['f1']:.4f}"
    )
    print(
            f"Accuracy             : "
            f"{metrics['accuracy']:.4f}"
        )

    print(
            f"FPR                  : "
            f"{metrics['fpr']:.4f}"
        )

    print(
            f"TN                   : "
            f"{metrics['tn']}"
        )

    print(
            f"FP                   : "
            f"{metrics['fp']}"
        )

    print(
            f"FN                   : "
            f"{metrics['fn']}"
        )

    print(
            f"TP                   : "
            f"{metrics['tp']}"
        )


    print("\nConfusion Matrix:")

    print(
        metrics["confusion_matrix"]
    )

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    return results, metrics


if __name__ == "__main__":
    run_test(category="bottle")

